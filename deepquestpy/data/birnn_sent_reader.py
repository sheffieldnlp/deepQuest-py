from typing import Dict, Optional
import os
import numpy as np
import pickle
import pandas as pd
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, TensorField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer
from deepquestpy_cli.utils import one_hot_encoding


@DatasetReader.register("birnn_sent_reader")
class BiRNNSentReader(DatasetReader):
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
        tgt_filename = os.path.join(self.data_path,path_name,path_name+".mt")
        score_filename = os.path.join(self.data_path,path_name,path_name+".score")
        t_logits_filename = os.path.join(self.data_path,path_name,path_name+".logits")

        print ("++++++++++")
        print ("In BiRNN Sent Reader")
        print ("++++++++++")

        scores = pd.read_csv(score_filename, sep="\n", header=None, names=["score"])
        print ("scores.info() :", scores.info())

        print ("ONE-HOT-SCORES")
        one_hot_class_labels = one_hot_encoding(scores["score"].values)
        print ("Shape :", one_hot_class_labels.shape)
        print (one_hot_class_labels)

        with open(src_filename, "r") as src_file, open(tgt_filename, "r") as tgt_file,\
                 open(t_logits_filename, "rb") as t_logits_pkl_file:

            for src, tgt, one_hot, t_logits in zip(self.shard_iterable(src_file),
                self.shard_iterable(tgt_file), self.shard_iterable(one_hot_class_labels), self.shard_iterable(pickle.load(t_logits_pkl_file))):
                
                print()    
                print ("one_hot: ", one_hot)

                yield self.text_to_instance(src, tgt, np.asarray(one_hot, dtype=np.float32), np.asarray(t_logits, dtype=np.float32))
        

    @overrides
    def text_to_instance(
        self,
        src: str,
        tgt: str,
        sent_label: np.ndarray = None,
        t_logits: np.ndarray = None,
    ) -> Instance:

        fields: Dict[str, Field] = {}
        src = self._tokenizer.tokenize(src.strip())
        tgt = self._tokenizer.tokenize(tgt.strip())

        src_tokens = self._tokenizer.add_special_tokens(src)
        tgt_tokens = self._tokenizer.add_special_tokens(tgt)
        tokens_src= TextField(src_tokens, token_indexers=self._token_indexers_src)
        tokens_tgt= TextField(tgt_tokens, token_indexers=self._token_indexers_tgt)

        fields["tokens_src"] = tokens_src
        fields["tokens_tgt"] = tokens_tgt
        
        fields["labels"] = TensorField(sent_label)
            
        fields["t_logits"] = TensorField(t_logits)

        print ("sent_label :", sent_label)
        print ("tokens_src :", tokens_src)
        print ("tokens_tgt :", tokens_tgt)
        print ("t_logits :", t_logits)

        return Instance(fields)