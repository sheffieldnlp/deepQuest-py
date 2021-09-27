import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaClassificationHead,
)
from transformers.modeling_outputs import TokenClassifierOutput

from deepquestpy.models.transformer_word import TransformerDeepQuestModelWord
from deepquestpy.data.data_collator import DataCollatorForJointClassification


class RobertaForQualityEstimationWord(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_word_labels = config.num_labels  # word-level tags

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # word-level
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_wordlevel = nn.Linear(config.hidden_size, config.num_labels)

        # sentence-level
        config.num_labels = 1  # regression
        self.classifier_sentlevel = RobertaClassificationHead(config)
        config.num_labels = self.num_word_labels

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        score_sent=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # sentence-level
        logits_sentlevel = self.classifier_sentlevel(sequence_output)
        # word-level
        sequence_output = self.dropout(sequence_output)
        logits_wordlevel = self.classifier_wordlevel(sequence_output)

        loss = None
        if score_sent is not None and labels is not None:
            # sentence-level (regression)
            loss_fct_sent = MSELoss()
            loss_sentlevel = loss_fct_sent(logits_sentlevel.view(-1), score_sent.view(-1))

            # word-level
            weights = [3.0, 1.0]  # additional weight for the BAD tags ({0: 'BAD', 1: 'OK'})
            class_weights = torch.FloatTensor(weights).to(logits_wordlevel.device)
            loss_fct_word = CrossEntropyLoss(weight=class_weights)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits_wordlevel.view(-1, self.num_word_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct_word.ignore_index).type_as(labels)
                )
                loss_wordlevel = loss_fct_word(active_logits, active_labels)
            else:
                loss_wordlevel = loss_fct_word(logits_wordlevel.view(-1, self.num_word_labels), labels.view(-1))

            # joint
            loss = loss_sentlevel + loss_wordlevel

        if not return_dict:
            output = (logits_wordlevel,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=logits_wordlevel, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )


class XLMRobertaForQualityEstimationWord(RobertaForQualityEstimationWord):
    config_class = XLMRobertaConfig


class BeringLabWord(TransformerDeepQuestModelWord):
    def load_pretrained_model(self):
        return XLMRobertaForQualityEstimationWord.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=self.config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
        )

    def get_data_collator(self):
        return DataCollatorForJointClassification(
            self.tokenizer, pad_to_multiple_of=8 if self.training_args.fp16 else None
        )

    def _preprocess_examples(self, examples):
        src_lang = self.data_args.src_lang
        tgt_lang = self.data_args.tgt_lang
        label_all_tokens = self.data_args.label_all_tokens
        labels_in_gaps = self.data_args.labels_in_gaps
        label_column_name = self.data_args.label_column_name

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
        ids_words = []
        if len(examples[label_column_name][0]) > 0:  # to verify that there are labels
            labels_word = []
            labels_sent = []
            for i, (hter, label_src, label_tgt) in enumerate(
                zip(examples["hter"], examples["src_tags"], examples[label_column_name])
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
                ids_words.append(word_ids)
            tokenized_inputs["labels"] = labels_word
            tokenized_inputs["score_sent"] = labels_sent
        else:
            ids_words = [tokenized_inputs.word_ids(batch_index=i) for i in range(len(examples["translation"]))]
        tokenized_inputs["ids_words"] = ids_words
        return tokenized_inputs

