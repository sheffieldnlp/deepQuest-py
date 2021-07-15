import numpy as np

from datasets import load_metric
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    default_data_collator,
)

from deepquestpy.models.base import DeepQuestModelSent
from deepquestpy.commands.utils import METRICS_DIR


class TransformerDeepQuestModelSent(DeepQuestModelSent):
    def __init__(self, model_args, data_args, training_args) -> None:
        super().__init__()
        self.num_labels = 1  # regression

        self.config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=self.num_labels,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
        )

        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    def tokenize_datasets(self, datasets):
        return datasets.map(
            self._preprocess_examples,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )

    def _preprocess_examples(self, examples):
        src_lang = self.data_args.src_lang
        tgt_lang = self.data_args.tgt_lang
        label_column_name = self.data_args.label_column_name
        tokenized_inputs = self.tokenizer(
            text=[e[src_lang] for e in examples["translation"]],
            text_pair=[e[tgt_lang] for e in examples["translation"]],
            padding="max_length" if self.data_args.pad_to_max_length else False,
            truncation=True,
        )
        tokenized_inputs["labels"] = examples[label_column_name]
        return tokenized_inputs

    def get_data_collator(self):
        if self.data_args.pad_to_max_length:
            return default_data_collator
        elif self.training_args.fp16:
            return DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        else:
            return None

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=self.config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
        )

    def compute_metrics(self, p):
        metric = load_metric(f"{METRICS_DIR}/questeval_sentence")
        predictions, labels = p
        predictions = np.squeeze(predictions)
        metrics = metric.compute(references=labels, predictions=predictions)
        return metrics

    def postprocess_predictions(self, predictions, *args):
        return {"predictions": np.squeeze(predictions)}
