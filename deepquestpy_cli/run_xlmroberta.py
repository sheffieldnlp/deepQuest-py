import logging
import os
import sys

import numpy as np
from functools import partial
from datasets import ClassLabel, load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from deepquestpy.models.xlmroberta.xlmroberta_word import XLMRobertaForWordQualityEstimation
from deepquestpy.models.xlmroberta.xlmroberta_utils import preprocess_wordlevel, preprocess_sentencelevel
from deepquestpy.data.data_collator import DataCollatorForJointClassification

from deepquestpy_cli.model_args import ModelArguments
from deepquestpy_cli.data_args import DataTrainingArguments
from deepquestpy_cli.utils import DATASETS_LOADERS_DIR, METRICS_DIR


logger = logging.getLogger(__name__)


def setup_logging(training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if transformers.trainer_utils.is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if transformers.trainer_utils.is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)


def get_true_predictions_for_source_and_target(tokenized_eval_dataset, raw_predictions, raw_labels, labels_in_gaps):
    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(raw_predictions, raw_labels)
    ]
    # Split in src and tgt
    preds_src, preds_tgt = [], []
    for preds, src_length in zip(true_predictions, tokenized_eval_dataset["length_source"]):
        preds_src.append(preds[:src_length])
        # For the target side we predict OK by default for the GAPS
        preds_tgt_no_gaps = preds[src_length:]
        if labels_in_gaps:
            preds_tgt_with_gaps = [0] * (2 * len(preds_tgt_no_gaps) + 1)
            for i, tag in enumerate(preds_tgt_no_gaps):
                preds_tgt_with_gaps[2 * i + 1] = tag
            preds_tgt.append(preds_tgt_with_gaps)
        else:
            preds_tgt.append(preds_tgt_no_gaps)

    return preds_src, preds_tgt


def compute_metrics_wordlevel(metric, tokenized_eval_dataset, labels_in_gaps, p):
    raw_predictions, raw_labels = p
    raw_predictions = np.argmax(raw_predictions, axis=2)
    preds_src, preds_tgt = get_true_predictions_for_source_and_target(
        tokenized_eval_dataset, raw_predictions, raw_labels, labels_in_gaps
    )
    metrics = metric.compute(
        references_src=[[tag for tag in tags] for tags in tokenized_eval_dataset["src_tags"]],
        predictions_src=preds_src,
        references_tgt=[[tag for tag in tags] for tags in tokenized_eval_dataset["mt_tags"]],
        predictions_tgt=preds_tgt,
    )
    return metrics


def compute_metrics_sentencelevel(metric, p):
    predictions, labels = p
    predictions = np.squeeze(predictions)
    metrics = metric.compute(references=labels, predictions=predictions)
    return metrics


def save_output_wordlevel(label_list, output_eval_predictions_file, predictions_src, predictions_tgt):
    # Save predictions
    with open(f"{output_eval_predictions_file}.src.preds", "w") as writer:
        for prediction in predictions_src:
            writer.write(" ".join([label_list[p] for p in prediction]) + "\n")

    with open(f"{output_eval_predictions_file}.tgt.preds", "w") as writer:
        for prediction in predictions_tgt:
            writer.write(" ".join([label_list[p] for p in prediction]) + "\n")


def save_output_sentencelevel(output_eval_predictions_file, predictions):
    with open(f"{output_eval_predictions_file}.preds", "w") as writer:
        for item in predictions:
            writer.write(f"{item:3.3f}\n")


def main(model_args, data_args, training_args):
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
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

    # Set seed before initializing model
    set_seed(training_args.seed)

    # Load the dataset
    if data_args.dataset_name is not None:
        datasets = load_dataset(data_args.dataset_name, name=f"{data_args.src_lang}-{data_args.tgt_lang}")
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        datasets = load_dataset(
            f"{DATASETS_LOADERS_DIR}/customqe.py", data_files=data_files, download_mode="force_redownload"
        )

    if model_args.model_type == "sentence":
        num_labels = 1  # regression
    elif model_args.model_type == "word":
        if training_args.do_train:
            features = datasets["train"].features
        elif training_args.do_eval:
            features = datasets["validation"].features
        else:
            features = datasets["test"].features

        label_list = features[data_args.label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
        num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
    )

    if model_args.model_type == "sentence":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )
    elif model_args.model_type == "word":
        model = XLMRobertaForWordQualityEstimation.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    if model_args.model_type == "word":
        tokenized_datasets = datasets.map(
            preprocess_wordlevel,
            fn_kwargs={
                "src_lang": data_args.src_lang,
                "tgt_lang": data_args.tgt_lang,
                "tokenizer": tokenizer,
                "padding": padding,
                "label_to_id": label_to_id,
                "label_all_tokens": data_args.label_all_tokens,
                "labels_in_gaps": data_args.labels_in_gaps,
            },
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        data_collator = DataCollatorForJointClassification(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
        metric = load_metric(f"{METRICS_DIR}/questeval_word")
        compute_metrics_fn = partial(
            compute_metrics_wordlevel,
            metric,
            tokenized_datasets["validation"] if training_args.do_eval else None,
            data_args.labels_in_gaps,
        )
    elif model_args.model_type == "sentence":
        tokenized_datasets = datasets.map(
            preprocess_sentencelevel,
            fn_kwargs={
                "src_lang": data_args.src_lang,
                "tgt_lang": data_args.tgt_lang,
                "label_column_name": data_args.label_column_name,
                "tokenizer": tokenizer,
                "padding": padding,
            },
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        if data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif training_args.fp16:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None
        metric = load_metric(f"{METRICS_DIR}/questeval_sentence")
        compute_metrics_fn = partial(compute_metrics_sentencelevel, metric)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        results = trainer.evaluate()

        trainer.log_metrics("valid", results)
        trainer.save_metrics("valid", results)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(tokenized_datasets["test"])

        if model_args.model_type == "word":
            predictions = np.argmax(predictions, axis=2)
            preds_src, preds_tgt = get_true_predictions_for_source_and_target(
                tokenized_datasets["test"], predictions, labels, data_args.labels_in_gaps
            )
            if trainer.is_world_process_zero():
                save_output_wordlevel(
                    label_list=label_list,
                    output_eval_predictions_file=os.path.join(training_args.output_dir, "test"),
                    predictions_src=preds_src,
                    predictions_tgt=preds_tgt,
                )
        elif model_args.model_type == "sentence":
            predictions = np.squeeze(predictions)
            if trainer.is_world_process_zero():
                save_output_sentencelevel(
                    output_eval_predictions_file=os.path.join(training_args.output_dir, "test"),
                    predictions=predictions,
                )

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    return


def cli_main():
    # See all possible arguments by passing the --help flag to this script.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    setup_logging(training_args)

    main(model_args, data_args, training_args)


if __name__ == "__main__":
    cli_main()
