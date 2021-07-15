import logging
import os
import sys
import transformers

from datasets import load_dataset

from transformers.trainer_utils import get_last_checkpoint, set_seed
from transformers import HfArgumentParser, TrainingArguments, Trainer

from deepquestpy.commands.cli_args import DataArguments, ModelArguments
from deepquestpy.commands.utils import DATASETS_LOADERS_DIR, get_deepquest_model

logger = logging.getLogger(__name__)


def main():
    # Read the arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
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

    # Set seed before initializing model
    set_seed(training_args.seed)

    # Load the dataset splits
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(data_args.dataset_name, name=f"{data_args.src_lang}-{data_args.tgt_lang}")
    elif data_args.dataset_name in ["mlqe_pe"]:
        raw_datasets = load_dataset(
            f"{DATASETS_LOADERS_DIR}/{data_args.dataset_name}.py", name=f"{data_args.src_lang}-{data_args.tgt_lang}"
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        raw_datasets = load_dataset(
            f"{DATASETS_LOADERS_DIR}/custom.py", data_files=data_files, download_mode="force_redownload"
        )

    # Create an instance of a DeepQuestModel
    deepquest_model = get_deepquest_model(model_args.arch_name, model_args, data_args, training_args)

    # Preprocess the dataset
    tokenized_datasets = deepquest_model.tokenize_datasets(raw_datasets)

    # Initialize Trainer
    trainer = Trainer(
        model=deepquest_model.get_model(),
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=deepquest_model.get_tokenizer(),
        data_collator=deepquest_model.get_data_collator(),
        compute_metrics=deepquest_model.compute_metrics,
    )

    # Train the model
    if training_args.do_train:
        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate the model
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        trainer.log_metrics("validation", metrics)
        trainer.save_metrics("validation", metrics)

    # Predict with the model
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(tokenized_datasets["test"])

        predictions = deepquest_model.postprocess_predictions(predictions, labels)

        if trainer.is_world_process_zero():
            deepquest_model.save_output(
                output_file_path=os.path.join(training_args.output_dir, "test"), predictions=predictions
            )

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
