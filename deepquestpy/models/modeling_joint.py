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


class RobertaForJointQualityEstimation(RobertaPreTrainedModel):
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
            weights = [1.0, 3.0]  # additional weight for the BAD tags
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


class XLMRobertaForJointQualityEstimation(RobertaForJointQualityEstimation):
    config_class = XLMRobertaConfig
