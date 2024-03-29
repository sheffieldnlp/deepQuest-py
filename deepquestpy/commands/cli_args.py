from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library or from deepquest-py)."},
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Local folder that has the dataset to use."},
    )
    src_lang: str = field(default="src", metadata={"help": "Two letter identifier of the source language."})
    tgt_lang: str = field(default="tgt", metadata={"help": "Two letter identifier of the target language."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input training data file (a csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input evaluation data file to evaluate on (a csv file)."},
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input test data file to predict on (a csv file)."},
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. "
            "More efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    label_column_name_src: str = field(
        default="src_tags",
        metadata={"help": "The name of the column in the dataset that contains the word-level labels for the source."},
    )
    label_column_name_tgt: str = field(
        default="mt_tags",
        metadata={"help": "The name of the column in the dataset that contains the word-level labels for target."},
    )
    label_column_name_sent: str = field(
        default="sent_label",
        metadata={"help": "The name of the column in the dataset that contains the labels for the sentence-pair."},
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "For word-level only. Whether to put the label for one word on all tokens generated by that word "
            "or just on the one (in which case the other tokens will have a padding index)."
        },
    )
    labels_in_gaps: bool = field(
        default=False, metadata={"help": "For word-level only. Whether to use labels for gaps in the target sentence."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    arch_name: str = field(metadata={"help": "Specifies the architecture to use. "})  # TODO: show valid options
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
