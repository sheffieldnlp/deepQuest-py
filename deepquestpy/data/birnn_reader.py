from typing import Dict, Optional
import os
import numpy as np
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField,TensorField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer


@DatasetReader.register("birnn_reader")
class BiRNNReader(DatasetReader):
    """
    Reader for WMT20 QE Task 2
    """
    def __init__(
        self,
        data_path:str=None,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers_src: Dict[str, TokenIndexer] = None,
        token_indexers_tgt: Dict[str, TokenIndexer] = None,
        sentence_level:bool = True,
        tag_label_namespace = "tag_labels",
        **kwargs,
    ) -> None:
        super().__init__(manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs)
        self.data_path = data_path
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers_src = token_indexers_src
        self._token_indexers_tgt = token_indexers_tgt
        self.sentence_level = sentence_level
        self._tag_label_namespace: str = tag_label_namespace

    @overrides
    def _read(self, path_name: str):
        src_filename = os.path.join(self.data_path,path_name,path_name+".src")
        src_tags_filename = os.path.join(self.data_path,path_name,path_name+".source_tags")
        tgt_filename = os.path.join(self.data_path,path_name,path_name+".mt")
        tgt_tags_filename = os.path.join(self.data_path,path_name,path_name+".tags")
        hter_filename = os.path.join(self.data_path,path_name,path_name+".hter")
        with open(src_filename, "r") as src_file, open(src_tags_filename, "r") as src_tags_file, open(tgt_filename, "r") as tgt_file,\
                open(tgt_tags_filename, "r") as tgt_tags_file,\
                open(hter_filename, "r") as hter_file:
            for src,src_tags,tgt,tgt_tags,hter in zip(self.shard_iterable(src_file), self.shard_iterable(src_tags_file),
                self.shard_iterable(tgt_file), self.shard_iterable(tgt_tags_file), self.shard_iterable(hter_file)):
                yield self.text_to_instance(src, src_tags, tgt, tgt_tags,np.asarray(hter,dtype=np.float32))

    @overrides
    def text_to_instance(
        self,
        src: str,
        src_tags: str,
        tgt: str,
        tgt_tags:str,
        sent_label: np.ndarray = None,
    ) -> Instance:

        fields: Dict[str, Field] = {}
        src = self._tokenizer.tokenize(src.strip())
        tgt = self._tokenizer.tokenize(tgt.strip())
        src_tags = src_tags.strip().split(" ")
        tgt_tags = tgt_tags.strip().split(" ")

        # filter out GAP labels
        tgt_tags = [tt for i, tt in enumerate(tgt_tags) if i % 2 != 0]

        src_tokens = self._tokenizer.add_special_tokens(src)
        tgt_tokens = self._tokenizer.add_special_tokens(tgt)
        tokens_src= TextField(src_tokens, token_indexers=self._token_indexers_src)
        tokens_tgt= TextField(tgt_tokens, token_indexers=self._token_indexers_tgt)

        fields["tokens_src"] = tokens_src
        fields["tokens_tgt"] = tokens_tgt
        if not self.sentence_level:
            fields["tags_src"] = SequenceLabelField(src_tags, tokens_src,self._tag_label_namespace)
            fields["tags_tgt"] = SequenceLabelField(tgt_tags, tokens_tgt,self._tag_label_namespace)
        else:
            fields["labels"] = TensorField(sent_label)

        return Instance(fields)