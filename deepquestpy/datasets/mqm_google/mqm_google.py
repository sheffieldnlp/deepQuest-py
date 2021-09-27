# Lint as: python3
"""newstest2020 dataset annoted with MQM labels by Google"""

import os

import logging

import datasets
import pandas as pd

_CITATION = """
"""

_DESCRIPTION = """
"""

_LANGUAGE_PAIRS = [
    ("en", "de"),
]


class MQMGoogleConfig(datasets.BuilderConfig):
    def __init__(self, src_lg, tgt_lg, **kwargs):
        super(MQMGoogleConfig, self).__init__(**kwargs)
        self.src_lg = src_lg
        self.tgt_lg = tgt_lg


class MQMGoogle(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MQMGoogleConfig(
            name=f"{src_lg}-{tgt_lg}",
            version=datasets.Version("0.0.1"),
            description=f"Google MQM: {src_lg} - {tgt_lg}",
            src_lg=src_lg,
            tgt_lg=tgt_lg,
        )
        for (src_lg, tgt_lg) in _LANGUAGE_PAIRS
    ]

    BUILDER_CONFIG_CLASS = MQMGoogleConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "translation": datasets.Translation(languages=(self.config.src_lg, self.config.tgt_lg)),
                    "src_tags": datasets.Sequence(datasets.ClassLabel(names=["BAD", "OK"])),
                    "bad_labels": datasets.Sequence(datasets.ClassLabel(names=["BAD", "OK"])),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        generators = []
        for name, split in [
            (datasets.Split.TRAIN, "train"),
            (datasets.Split.VALIDATION, "validation"),
            (datasets.Split.TEST, "test"),
        ]:
            if split in self.config.data_files:
                generators.append(
                    datasets.SplitGenerator(
                        name=name,
                        gen_kwargs={
                            "filepath": self.config.data_files[split],
                            "split": split,
                            "source_lg": self.config.src_lg,
                            "target_lg": self.config.tgt_lg,
                        },
                    )
                )
        return generators

    def _generate_examples(self, filepath, split, source_lg, target_lg):
        logging.info("Generating examples")
        df_translations = pd.read_csv(filepath)
        for id, row in df_translations.iterrows():
            yield id, {
                "translation": {source_lg: str(row["original"]).strip(), target_lg: str(row["translation"]).strip()},
                "src_tags": ["OK"] * len(str(row["original"]).strip().split()),
                "bad_labels": [] if split == "test" else str(row["bad_labels"]).strip().split(),
            }
