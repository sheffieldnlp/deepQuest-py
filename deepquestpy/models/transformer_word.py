import numpy as np

from datasets import load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification

from deepquestpy.models.base import DeepQuestModelWord
from deepquestpy.commands.utils import METRICS_DIR


class TransformerDeepQuestModelWord(DeepQuestModelWord):
    def __init__(self, model_args, data_args, training_args):
        super().__init__()
        for split in ["train", "validation", "test"]:
            if split in datasets:
                features = datasets[split].features

        self.label_list = features[data_args.label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        self.label_to_id = {i: i for i in range(len(self.label_list))}
        self.num_labels = len(self.label_list)

        # Load pretrained model and tokenizer
        self.config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=self.num_labels,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )

        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    def tokenize_datasets(self, datasets):
        tokenized_datasets = datasets.map(
            self._preprocess_examples,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )
        if self.training_args.do_eval:
            self.evaluation_dataset = tokenized_datasets["validation"]

        return tokenized_datasets

    def _preprocess_examples(self, examples):
        src_lang = self.data_args.src_lang
        tgt_lang = self.data_args.tgt_lang
        label_all_tokens = self.data_args.label_all_tokens
        labels_in_gaps = self.data_args.labels_in_gaps
        tokenized_inputs = self.tokenizer(
            text=[e[src_lang].split() for e in examples["translation"]],
            text_pair=[e[tgt_lang].split() for e in examples["translation"]],
            padding="max_length" if self.data_args.pad_to_max_length else False,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        tokenized_inputs["length_source"] = [len(label_src) for label_src in examples["src_tags"]]
        labels_word = []
        labels_sent = []
        for i, (hter, label_src, label_tgt) in enumerate(
            zip(examples["hter"], examples["src_tags"], examples["mt_tags"])
        ):
            # remove the labels for GAPS in target
            if labels_in_gaps:
                label_tgt = [l for j, l in enumerate(label_tgt) if j % 2 != 0]
            label = label_src
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            count_special = 0  # this variable helps keep track on when to change from src labels to tgt labels
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    count_special += 1
                    if count_special == 3:  # two from start and end of src, and one from start of tgt
                        label = label_tgt
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(self.label_to_id[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels_word.append(label_ids)
            labels_sent.append(hter)
        tokenized_inputs["labels"] = labels_word
        tokenized_inputs["score_sent"] = labels_sent
        return tokenized_inputs

    def get_model(self):
        return AutoModelForTokenClassification.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=self.config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
        )

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=True,
            revision=self.model_args.model_revision,
        )

    def get_data_collator(self):
        return DataCollatorForTokenClassification(
            self.tokenizer, pad_to_multiple_of=8 if self.training_args.fp16 else None
        )

    def compute_metrics(self, p):
        metric = load_metric(f"{METRICS_DIR}/questeval_word")
        labels_in_gaps = self.data_args.labels_in_gaps
        raw_predictions, raw_labels = p
        raw_predictions = np.argmax(raw_predictions, axis=2)
        preds_src, preds_tgt = self._get_true_predictions_for_source_and_target(
            self.evaluation_dataset, raw_predictions, raw_labels, labels_in_gaps
        )
        metrics = metric.compute(
            references_src=[[tag for tag in tags] for tags in self.evaluation_dataset["src_tags"]],
            predictions_src=preds_src,
            references_tgt=[[tag for tag in tags] for tags in self.evaluation_dataset["mt_tags"]],
            predictions_tgt=preds_tgt,
        )
        return metrics

    def _get_true_predictions_for_source_and_target(self, tokenized_eval_dataset, raw_predictions, raw_labels):
        # Remove ignored index (special tokens)
        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(raw_predictions, raw_labels)
        ]
        # Split in src and tgt
        preds_src, preds_tgt = [], []
        for preds, src_length in zip(true_predictions, tokenized_eval_dataset["length_source"]):
            preds_src.append(preds[:src_length])
            # For the target side we predict OK by default for the GAPS
            preds_tgt_no_gaps = preds[src_length:]
            if self.data_args.labels_in_gaps:
                preds_tgt_with_gaps = [0] * (2 * len(preds_tgt_no_gaps) + 1)
                for i, tag in enumerate(preds_tgt_no_gaps):
                    preds_tgt_with_gaps[2 * i + 1] = tag
                preds_tgt.append(preds_tgt_with_gaps)
            else:
                preds_tgt.append(preds_tgt_no_gaps)

        return preds_src, preds_tgt

    def postprocess_predictions(self, predictions, labels):
        predictions = np.argmax(predictions, axis=2)
        preds_src, preds_tgt = self._get_true_predictions_for_source_and_target(
            tokenized_datasets["test"], predictions, labels
        )
        return {"predictions_src": preds_src, "predictions_tgt": preds_tgt}

