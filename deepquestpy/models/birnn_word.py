import torch
from typing import Dict

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import FBetaMeasure


@Model.register("birnn_word")
class BiRNNWord(Model):
    """
    Word-level BiRNN reader
    """
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder_src: TextFieldEmbedder,
        text_field_embedder_tgt: TextFieldEmbedder,
        seq2seq_encoder_src: Seq2SeqEncoder,
        seq2seq_encoder_tgt: Seq2SeqEncoder,
        dropout: float = None,
        tag_label_namespace: str = "tag_labels",
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder_src = text_field_embedder_src
        self._text_field_embedder_tgt = text_field_embedder_tgt
        self.seq2seq_encoder_src = seq2seq_encoder_src
        self.seq2seq_encoder_tgt = seq2seq_encoder_tgt

        src_out_dim = seq2seq_encoder_src.get_output_dim()
        tgt_out_dim = seq2seq_encoder_tgt.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self.tag_label_namespace = tag_label_namespace
        num_tag_labels = vocab.get_vocab_size(tag_label_namespace)

        token_index_vocab = self.vocab.get_token_to_index_vocabulary(namespace=self.tag_label_namespace)

        # label indices for WMT20 QE task 2 (word-level)
        self.index_OK = token_index_vocab["OK"]
        self.index_BAD = token_index_vocab["BAD"]

        self._linear_layer_src = torch.nn.Linear(src_out_dim, num_tag_labels)
        self._linear_layer_tgt = torch.nn.Linear(tgt_out_dim, num_tag_labels)

        self._f1_src = FBetaMeasure()
        self._f1_tgt = FBetaMeasure()

        self._loss = torch.nn.MSELoss()

    def forward(
        self, tokens_src: TextFieldTensors, tokens_tgt: TextFieldTensors,
            tags_src: torch.LongTensor=None, tags_tgt: torch.LongTensor=None,
            sent_label: torch.FloatTensor = None
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

        logits_src = self._linear_layer_src(encoded_text_src)
        logits_tgt = self._linear_layer_tgt(encoded_text_tgt)
        output_dict = {"logits_src": logits_src, "logits_tgt": logits_tgt}

        if (tags_src is not None and tags_tgt is not None):
            loss_src = sequence_cross_entropy_with_logits(logits_src,tags_src,mask_src)
            loss_tgt = sequence_cross_entropy_with_logits(logits_tgt,tags_tgt,mask_tgt)
            loss = loss_src + loss_tgt
            output_dict["loss"] = loss
            self._f1_src(logits_src,tags_src,mask_src)
            self._f1_tgt(logits_tgt,tags_tgt,mask_tgt)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_src_metrics = self._f1_src.get_metric(reset)
        f1_tgt_metrics = self._f1_tgt.get_metric(reset)
        return {'f1_src_OK': f1_src_metrics['fscore'][self.index_OK],'f1_src_BAD': f1_src_metrics['fscore'][self.index_BAD],
                'f1_tgt_OK': f1_tgt_metrics['fscore'][self.index_OK],'f1_tgt_BAD': f1_tgt_metrics['fscore'][self.index_BAD]}