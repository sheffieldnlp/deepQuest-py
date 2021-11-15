import numpy as np

from datasets import load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification

from deepquestpy.models.base import DeepQuestModelWord
from deepquestpy.commands.utils import METRICS_DIR


class TransformerDeepQuestModelWord(DeepQuestModelWord):
    def __init__(self, model_args, data_args, training_args):
        super().__init__()
        self.tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
        self.tokenizer = None

        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    def set_label_list(self, label_list):
        self.label_list = label_list
        self.label_to_id = {i: i for i in range(len(self.label_list))}
        self.num_labels = len(self.label_list)

    def set_evaluation_dataset_for_metrics(self, evaluation_dataset_for_metrics):
        self.evaluation_dataset_for_metrics = evaluation_dataset_for_metrics

    def _load_tokenizer(self):
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, cache_dir=self.model_args.cache_dir, use_fast=True, revision=self.model_args.model_revision,
            )

    def get_tokenizer(self):
        self._load_tokenizer()
        return self.tokenizer

    def tokenize_datasets(self, datasets, *args, **kwargs):
        self._load_tokenizer()
        tokenized_datasets = datasets.map(
            self._preprocess_examples,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            *args,
            **kwargs,
        )
        return tokenized_datasets

    def _preprocess_examples(self, examples):
        src_lang = self.data_args.src_lang
        tgt_lang = self.data_args.tgt_lang
        label_all_tokens = self.data_args.label_all_tokens
        labels_in_gaps = self.data_args.labels_in_gaps
        label_column_name_tgt = self.data_args.label_column_name_tgt
        label_column_name_src = self.data_args.label_column_name_src

        tokenized_inputs = self.tokenizer(
            text=[e[src_lang].split() for e in examples["translation"]],
            text_pair=[e[tgt_lang].split() for e in examples["translation"]],
            padding="max_length" if self.data_args.pad_to_max_length else False,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        tokenized_inputs["length_source"] = [len(e[src_lang].split()) for e in examples["translation"]]
        tokenized_inputs["length_target"] = [len(e[tgt_lang].split()) for e in examples["translation"]]

        if len(examples[label_column_name_tgt][0]) > 0:  # to verify that there are labels
            ids_words, labels_words = self._preprocess_src_and_tgt_labels(
                examples, tokenized_inputs, label_column_name_src, label_column_name_tgt, labels_in_gaps, label_all_tokens,
            )
            tokenized_inputs["labels"] = labels_words
        else:
            ids_words = [tokenized_inputs.word_ids(batch_index=i) for i in range(len(examples["translation"]))]

        tokenized_inputs["ids_words"] = ids_words

        return tokenized_inputs

    def _preprocess_src_and_tgt_labels(self, examples, tokenized_inputs, label_column_name_src, label_column_name_tgt, labels_in_gaps, label_all_tokens):
        if label_column_name_src in examples:
            all_tokens_labels = zip(examples[label_column_name_src], examples[label_column_name_tgt])
        else:
            all_tokens_labels = examples[label_column_name_tgt]
        ids_words = []
        labels_word = []
        for i, batch_tokens_labels in enumerate(all_tokens_labels):
            if label_column_name_src in examples:
                label_src, label_tgt = batch_tokens_labels
            else:
                label_src = None
                label_tgt = batch_tokens_labels
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
                    if label is not None:
                        label_ids.append(self.label_to_id[label[word_idx]])
                    else:
                        label_ids.append(-100)
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if label is not None:
                        label_ids.append(self.label_to_id[label[word_idx]] if label_all_tokens else -100)
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            labels_word.append(label_ids)
            ids_words.append(word_ids)
        return ids_words, labels_word

    def set_model(self, model):
        self.model = model

    def get_model(self):
        # Load pretrained model
        self.config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            num_labels=self.num_labels,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=self.config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
        )
        return self.model

    def get_data_collator(self):
        return DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8 if self.training_args.fp16 else None)

    def compute_metrics(self, p):
        metric = load_metric(f"{METRICS_DIR}/questeval_word")
        raw_predictions, raw_labels = p
        raw_predictions = np.argmax(raw_predictions, axis=2)
        preds_src, preds_tgt = self._get_true_predictions_for_source_and_target(self.evaluation_dataset_for_metrics, raw_predictions, raw_labels)

        label_column_name_tgt = self.data_args.label_column_name_tgt
        label_column_name_src = self.data_args.label_column_name_src

        refs_tgt = [[tag for tag in tags] for tags in self.evaluation_dataset_for_metrics[label_column_name_tgt]]
        if label_column_name_src in self.evaluation_dataset_for_metrics:
            refs_src = [[tag for tag in tags] for tags in self.evaluation_dataset_for_metrics[label_column_name_src]]
        else:
            refs_src = [[]] * len(preds_src)

        metrics = metric.compute(
            references=[{"src": ref_src, "tgt": ref_tgt} for ref_src, ref_tgt in zip(refs_src, refs_tgt)],
            predictions=[{"src": pred_src, "tgt": pred_tgt} for pred_src, pred_tgt in zip(preds_src, preds_tgt)],
        )

        return metrics

    def _get_true_predictions_for_source_and_target(self, tokenized_eval_dataset, raw_predictions, raw_labels):
        # Remove ignored index (special tokens)
        if raw_labels is not None:
            true_predictions = [[p for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(raw_predictions, raw_labels)]
        else:
            true_predictions = []
            for word_ids, pred_labels in zip(tokenized_eval_dataset["ids_words"], raw_predictions):
                label_ids = []
                previous_word_idx = None
                for word_idx, pred_label_idx in zip(word_ids, pred_labels):
                    if word_idx is None:
                        continue
                    elif word_idx != previous_word_idx:
                        label_ids.append(pred_label_idx)
                    elif self.data_args.label_all_tokens:
                        label_ids.append(pred_label_idx)
                    previous_word_idx = word_idx
                true_predictions.append(label_ids)

        # Split in src and tgt
        preds_src, preds_tgt = [], []
        for i, (preds, src_length, tgt_length) in enumerate(
            zip(true_predictions, tokenized_eval_dataset["length_source"], tokenized_eval_dataset["length_target"])
        ):
            if src_length + tgt_length == len(preds):
                preds_src.append(preds[:src_length])
                preds_tgt_no_gaps = preds[src_length:]
            else:
                # There are no predictions for the source, only for the target
                preds_src.append([])
                assert tgt_length == len(preds)
                preds_tgt_no_gaps = preds
            assert len(preds_tgt_no_gaps) == tgt_length, f"{i}: {len(preds_tgt_no_gaps)} != {tgt_length}"
            if self.data_args.labels_in_gaps:
                preds_tgt_with_gaps = [self.label_to_id[self.label_list.index("OK")]] * (2 * len(preds_tgt_no_gaps) + 1)
                for i, tag in enumerate(preds_tgt_no_gaps):
                    preds_tgt_with_gaps[2 * i + 1] = tag
                preds_tgt.append(preds_tgt_with_gaps)
            else:
                preds_tgt.append(preds_tgt_no_gaps)

        return preds_src, preds_tgt

    def postprocess_predictions(self, predictions, labels):
        predictions = np.argmax(predictions, axis=2)
        preds_src, preds_tgt = self._get_true_predictions_for_source_and_target(self.evaluation_dataset_for_metrics, predictions, labels)
        return {"predictions_src": preds_src, "predictions_tgt": preds_tgt}

