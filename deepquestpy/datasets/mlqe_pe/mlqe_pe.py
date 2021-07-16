# Lint as: python3
"""Custom Quality Estimation Dataset"""

import os

import datasets


_CITATION = """
@article{fomicheva2020mlqepe,
      title={{MLQE-PE}: A Multilingual Quality Estimation and Post-Editing Dataset}, 
      author={Marina Fomicheva 
        and Shuo Sun and Erick Fonseca 
        and Fr\'ed\'eric Blain 
        and Vishrav Chaudhary 
        and Francisco Guzm\'an 
        and Nina Lopatina 
        and Lucia Specia 
        and Andr\'e F.~T.~Martins},
      year={2020},
      journal={arXiv preprint arXiv:2010.04480}
}
"""

_DESCRIPTION = """
Multilingual Quality Estimation and Automatic Post-editing Dataset. 
This is an updated version of the MLQE dataset to include post-editing data, as well Ru-En data. 
Please refer to the MLQE repo for the NMT models that generated the data.
"""

_HOMEPAGE = "https://github.com/sheffieldnlp/mlqe-pe"

_LICENSE = "CC0 1.0 Universal"

_LANGUAGE_PAIRS = [
    ("en", "de"),
    ("en", "zh"),
    ("et", "en"),
    ("ne", "en"),
    ("ro", "en"),
    ("ru", "en"),
    ("si", "en"),
]

_MAIN_URL = "https://github.com/sheffieldnlp/mlqe-pe/raw/master/data/post-editing"


def inject_to_link(src_lg, tgt_lg):
    links = {
        "train": f"{_MAIN_URL}/train/{src_lg}-{tgt_lg}-train.tar.gz",
        "dev": f"{_MAIN_URL}/dev/{src_lg}-{tgt_lg}-dev.tar.gz",
        "test": f"{_MAIN_URL}/test/{src_lg}-{tgt_lg}-test.tar.gz",
    }
    return links


_URLs = {f"{src_lg}-{tgt_lg}": inject_to_link(src_lg, tgt_lg) for (src_lg, tgt_lg) in _LANGUAGE_PAIRS}


class MlqePeConfig(datasets.BuilderConfig):
    def __init__(self, src_lg, tgt_lg, **kwargs):
        super(MlqePeConfig, self).__init__(**kwargs)
        self.src_lg = src_lg
        self.tgt_lg = tgt_lg


class MlqePePostEditing(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        MlqePeConfig(
            name=f"{src_lg}-{tgt_lg}",
            version=datasets.Version("1.0.0"),
            description=f"Post-Editing: {src_lg} - {tgt_lg}",
            src_lg=src_lg,
            tgt_lg=tgt_lg,
        )
        for (src_lg, tgt_lg) in _LANGUAGE_PAIRS
    ]
    BUILDER_CONFIG_CLASS = MlqePeConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "translation": datasets.Translation(languages=(self.config.src_lg, self.config.tgt_lg)),
                    "src_tags": datasets.Sequence(datasets.ClassLabel(names=["BAD", "OK"])),
                    "mt_tags": datasets.Sequence(datasets.ClassLabel(names=["BAD", "OK"])),
                    "pe": datasets.Value("string"),
                    "hter": datasets.Value("float32"),
                    "alignments": datasets.Sequence(datasets.Sequence(datasets.Value("int32"))),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLs[self.config.name]
        data_dir = dl_manager.download_and_extract(my_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir["train"], f"{self.config.src_lg}-{self.config.tgt_lg}-train"),
                    "split": "train",
                    "source_lg": self.config.src_lg,
                    "target_lg": self.config.tgt_lg,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir["dev"], f"{self.config.src_lg}-{self.config.tgt_lg}-dev"),
                    "split": "dev",
                    "source_lg": self.config.src_lg,
                    "target_lg": self.config.tgt_lg,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir["test"], f"{self.config.src_lg}-{self.config.tgt_lg}-test20"),
                    "split": "test20",
                    "source_lg": self.config.src_lg,
                    "target_lg": self.config.tgt_lg,
                },
            ),
        ]

    def _generate_examples(self, filepath, split, source_lg, target_lg):
        """Yields examples."""

        def open_and_read(fp):
            with open(fp, encoding="utf-8") as f:
                return f.read().splitlines()

        srcs = open_and_read(os.path.join(filepath, f"{split}.src"))
        mts = open_and_read(os.path.join(filepath, f"{split}.mt"))
        alignments = [
            [idx_.split("-") for idx_ in t.split(" ")]
            for t in open_and_read(os.path.join(filepath, f"{split}.src-mt.alignments"))
        ]
        src_tags = [t.split(" ") for t in open_and_read(os.path.join(filepath, f"{split}.source_tags"))]
        mt_tags = [t.split(" ") for t in open_and_read(os.path.join(filepath, f"{split}.tags"))]
        pes = open_and_read(os.path.join(filepath, f"{split}.pe"))
        hters = open_and_read(os.path.join(filepath, f"{split}.hter"))

        for id_, (src_, src_tags_, mt_, mt_tags_, pe_, hter_, alignments_) in enumerate(
            zip(srcs, src_tags, mts, mt_tags, pes, hters, alignments)
        ):
            yield id_, {
                "translation": {source_lg: src_, target_lg: mt_},
                "src_tags": src_tags_,
                "mt_tags": mt_tags_,
                "pe": pe_,
                "hter": hter_,
                "alignments": alignments_,
            }
