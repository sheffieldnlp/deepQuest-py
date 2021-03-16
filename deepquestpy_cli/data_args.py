from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library or from deepquest-py)."},
    )
    src_lang: str = field(default=None, metadata={"help": "Two letter identifier of the source language."})
    tgt_lang: str = field(default=None, metadata={"help": "Two letter identifier of the target language."})
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
    label_column_name: str = field(
        default="label", metadata={"help": "The name of the column in the dataset that contains the labels."},
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

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv"], "`train_file` should be a csv file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv"], "`validation_file` should be a csv file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv"], "`test_file` should be a csv file."
