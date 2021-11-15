import logging
import random
from copy import deepcopy
import sys
import json

import torch
import torch.backends
from tqdm import tqdm

import transformers

from dataclasses import dataclass, field

from datasets import load_dataset

from transformers import HfArgumentParser, TrainingArguments

from baal.active import get_heuristic
from baal.active import ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal.transformers_trainer_wrapper import BaalTransformersTrainer

from deepquestpy.commands.cli_args import DataArguments, ModelArguments
from deepquestpy.commands.utils import DATASETS_LOADERS_DIR, get_deepquest_model
from deepquestpy.models.base import DeepQuestModelWord


logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningArguments:
    """
    Arguments pertaining to the active learning loop
    """

    epoch: int = field(
        default=100, metadata={"help": ""},
    )
    batch_size: int = field(
        default=32, metadata={"help": ""},
    )
    initial_pool: int = field(
        default=1000, metadata={"help": ""},
    )
    n_data_to_label: int = field(
        default=100, metadata={"help": ""},
    )
    heuristic: str = field(
        default="bald", metadata={"help": ""},
    )
    iterations: int = field(
        default=20, metadata={"help": ""},
    )
    shuffle_prop: float = field(
        default=0.05, metadata={"help": ""},
    )
    reduction: str = field(
        default="mean", metadata={"help": ""},
    )


def main():
    # Read the arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ActiveLearningArguments))
    model_args, data_args, training_args, activelearning_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    random.seed(1337)
    torch.manual_seed(1337)
    if not use_cuda:
        print("warning, the experiments would take ages to run on cpu")

    # Load the dataset splits
    if data_args.dataset_name in ["mqm_google"]:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        raw_datasets = load_dataset(
            f"{DATASETS_LOADERS_DIR}/{data_args.dataset_name}",
            name=f"{data_args.src_lang}-{data_args.tgt_lang}",
            data_files=data_files,
            # download_mode="force_redownload",
        )
    else:
        raise ValueError("Invalid dataset name")

    # Create an instance of a DeepQuestModel
    deepquest_model = get_deepquest_model(model_args.arch_name, model_args, data_args, training_args)
    if isinstance(deepquest_model, DeepQuestModelWord):
        for split in ["train", "validation", "test"]:
            if split in raw_datasets:
                features = raw_datasets[split].features
                break
        label_list = features[data_args.label_column_name_tgt].feature.names
        deepquest_model.set_label_list(label_list)

    # Preprocess the datasets
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = deepquest_model.tokenize_datasets(raw_datasets["train"], remove_columns=list(features.keys()))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = deepquest_model.tokenize_datasets(raw_datasets["validation"], remove_columns=list(features.keys()))
        deepquest_model.set_evaluation_dataset_for_metrics(eval_dataset)

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = deepquest_model.tokenize_datasets(raw_datasets["test"], remove_columns=list(features.keys()))

    # Initialise the heuristic for active learning loop
    heuristic = get_heuristic(name=activelearning_args.heuristic, shuffle_prop=activelearning_args.shuffle_prop, reduction=activelearning_args.reduction)

    # change dropout layer to MCDropout
    model = patch_module(deepquest_model.get_model())

    if use_cuda:
        model.cuda()

    init_weights = deepcopy(model.state_dict())

    deepquest_model.set_model(model)

    # We turn our training dataset into one for active learning
    columns_to_remove = ["ids_words", "length_source", "length_target"]
    active_set = ActiveLearningDataset(train_dataset.remove_columns(columns_to_remove))
    # We start labeling randomly
    active_set.label_randomly(activelearning_args.initial_pool)
    # Initialize Trainer
    trainer = BaalTransformersTrainer(
        model=deepquest_model.get_model(),
        args=training_args,
        train_dataset=active_set,
        eval_dataset=eval_dataset.remove_columns(columns_to_remove),
        tokenizer=deepquest_model.get_tokenizer(),
        data_collator=deepquest_model.get_data_collator(),
        compute_metrics=deepquest_model.compute_metrics,
    )

    logs = []

    active_loop = ActiveLearningLoop(
        dataset=active_set,
        get_probabilities=trainer.predict_on_dataset,
        heuristic=heuristic,
        ndata_to_label=activelearning_args.n_data_to_label,
        max_sample=50,
        iterations=activelearning_args.iterations,
    )

    for epoch in tqdm(range(activelearning_args.epoch)):
        # we use the default setup of HuggingFace for training (ex: epoch=1).
        # The setup is adjustable when BaalHuggingFaceTrainer is defined.
        trainer.train()

        # Validation!
        eval_metrics = trainer.evaluate()

        # We reorder the unlabelled pool at the frequency of learning_epoch
        # This helps with speed while not changing the quality of uncertainty estimation.
        should_continue = active_loop.step()

        # We reset the model weights to relearn from the new trainset.
        trainer.load_state_dict(init_weights)
        trainer.lr_scheduler = None
        if not should_continue:
            break
        active_logs = {
            "epoch": epoch,
            # "labeled_data": active_set._labelled,
            "Next Training set size": len(active_set),
        }

        train_logs = {**eval_metrics, **active_logs}
        print(train_logs)
        logs.append(train_logs)
    trainer.log_metrics("validation", eval_metrics)
    trainer.save_metrics("validation", eval_metrics)

    with open(f"{training_args.output_dir}/logs.json", "w") as f:
        json.dump(logs, f)


if __name__ == "__main__":
    main()
