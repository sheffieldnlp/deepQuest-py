from typing import Dict
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.attention import DotProductAttention
from allennlp.nn.util import get_text_field_mask, weighted_sum
from allennlp.training.metrics import PearsonCorrelation


@Model.register("birnn")
class BiRNN(Model):
    """
    BiRNN QE model [0] using attention like [1] over words in the source and target
    [0] https://www.aclweb.org/anthology/C18-1266.pdf
    [1] https://www.aclweb.org/anthology/N16-1174.pdf
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder_src: TextFieldEmbedder,
        text_field_embedder_tgt: TextFieldEmbedder,
        seq2seq_encoder_src: Seq2SeqEncoder,
        seq2seq_encoder_tgt: Seq2SeqEncoder,
        attention: DotProductAttention,
        dropout: float = None,
        kd_without_gold_data: bool = False,
        kd_with_gold_data: bool = False,
        alpha: float = None,
        label_namespace: str = "labels",
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder_src = text_field_embedder_src
        self._text_field_embedder_tgt = text_field_embedder_tgt
        self.seq2seq_encoder_src = seq2seq_encoder_src
        self.seq2seq_encoder_tgt = seq2seq_encoder_tgt
        self.attention = attention

        src_out_dim = seq2seq_encoder_src.get_output_dim()
        tgt_out_dim = seq2seq_encoder_tgt.get_output_dim()

        self._linear_layer_src = torch.nn.Linear(src_out_dim, src_out_dim)
        self._linear_layer_tgt = torch.nn.Linear(tgt_out_dim, tgt_out_dim)

        self.context_weights_src = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand((1, src_out_dim))))
        self.context_weights_tgt = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand((1, tgt_out_dim))))

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self.kd_without_gold_data = kd_without_gold_data
        self.kd_with_gold_data = kd_with_gold_data
        self.alpha = alpha

        self._label_namespace = label_namespace
        self._classifier_input_dim = src_out_dim + tgt_out_dim
        self._linear_layer = torch.nn.Linear(self._classifier_input_dim, 1)
        self._pearson = PearsonCorrelation()
        self._loss = torch.nn.MSELoss()

    def forward(
        self, tokens_src: TextFieldTensors, tokens_tgt: TextFieldTensors, labels: torch.FloatTensor = None, t_pred: torch.FloatTensor = None
    ) -> Dict[str, torch.Tensor]:

        embedded_text_src = self._text_field_embedder_src(tokens_src)
        embedded_text_tgt = self._text_field_embedder_tgt(tokens_tgt)

        mask_src = get_text_field_mask(tokens_src)
        mask_tgt = get_text_field_mask(tokens_tgt)

        encoded_text_src = self.seq2seq_encoder_src(embedded_text_src, mask=mask_src)
        encoded_text_tgt = self.seq2seq_encoder_tgt(embedded_text_tgt, mask=mask_tgt)

        if self._dropout:
            encoded_text_src = self._dropout(encoded_text_src)
            encoded_text_tgt = self._dropout(encoded_text_tgt)

        encoded_text_src = self._linear_layer_src(encoded_text_src)
        encoded_text_tgt = self._linear_layer_tgt(encoded_text_tgt)

        attention_dist_src = self.attention(self.context_weights_src.expand(encoded_text_src.size()[0], -1), encoded_text_src)
        encoded_text_src = weighted_sum(encoded_text_src, attention_dist_src)

        attention_dist_tgt = self.attention(self.context_weights_tgt.expand(encoded_text_tgt.size()[0], -1), encoded_text_tgt)
        encoded_text_tgt = weighted_sum(encoded_text_tgt, attention_dist_tgt)

        encoded_text = torch.cat([encoded_text_src, encoded_text_tgt], dim=-1)
        scores = torch.sigmoid(self._linear_layer(encoded_text).squeeze())

        output_dict = {"scores": scores}

        # The changes for adding knowledge distillation are included here.
        if labels is not None:

            if self.kd_without_gold_data and self.kd_with_gold_data:
                raise ValueError("kd_without_gold_data and kd_with_gold_data options cannot be true simultaneously; Please set only one in the config file.")

            if self.kd_with_gold_data and not self.kd_without_gold_data:
                if self.alpha == 0:
                    raise ValueError("When kd_with_gold_data flag is true, please specify a non-zero value for  alpha in config file")
                else:
                    print("Calling Distillation With GOLD data")

                    distill_loss = self._loss(scores, t_pred.view(-1))
                    output_dict["d_loss"] = distill_loss

                    normal_loss = self._loss(scores, labels.view(-1))
                    output_dict["normal_loss"] = normal_loss

                    loss = self.alpha * distill_loss + (1.0 - self.alpha) * normal_loss
                    output_dict["loss"] = loss

                    self._pearson(scores, labels)

            elif self.kd_without_gold_data and not self.kd_with_gold_data:

                print("Calling Distillation WITHOUT GOLD data")

                loss = self._loss(scores, t_pred.view(-1))

                output_dict["loss"] = loss
                self._pearson(scores, labels)

            else:
                print("Calling Normal Loss; Without Any Distillation")
                loss = self._loss(scores, labels.view(-1))

                output_dict["loss"] = loss
                self._pearson(scores, labels)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"pearson": self._pearson.get_metric(reset)}
        return metrics
