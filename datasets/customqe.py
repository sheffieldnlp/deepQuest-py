# Lint as: python3
"""Custom Quality Estimation Dataset"""

import os

import logging

import datasets
import pandas as pd

_CITATION = """
"""

_DESCRIPTION = """
"""


class CustomQEConfig(datasets.BuilderConfig):
    def __init__(self, src_lg="src", tgt_lg="tgt", **kwargs):
        super(CustomQEConfig, self).__init__(**kwargs)
        self.src_lg = src_lg
        self.tgt_lg = tgt_lg


class CustomQE(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CustomQEConfig(
            name="customqe", version=datasets.Version("1.0.0"), description="Custom Quality Estimation dataset"
        ),
    ]

    BUILDER_CONFIG_CLASS = CustomQEConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "translation": datasets.Translation(languages=(self.config.src_lg, self.config.tgt_lg)),
                    # "src_tags": datasets.Sequence(datasets.features.ClassLabel(num_classes=2)),
                    # "mt_tags": datasets.Sequence(datasets.features.ClassLabel(num_classes=2)),
                    "sent_score": datasets.Value("float32"),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": self.config.data_files["train"],
                    "split": "train",
                    "source_lg": self.config.src_lg,
                    "target_lg": self.config.tgt_lg,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": self.config.data_files["validation"],
                    "split": "valid",
                    "source_lg": self.config.src_lg,
                    "target_lg": self.config.tgt_lg,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": self.config.data_files["test"],
                    "split": "test",
                    "source_lg": self.config.src_lg,
                    "target_lg": self.config.tgt_lg,
                },
            ),
        ]

    def _generate_examples(self, filepath, split, source_lg, target_lg):
        logging.info(f"Generating for {split}...")
        df_translations = pd.read_csv(filepath)
        for id, row in df_translations.iterrows():
            yield id, {
                "translation": {source_lg: str(row["src"]), target_lg: str(row["tgt"])},
                "sent_score": float(row["sent_score"]),
                # "src_tags": str(row["src_tags"]).split(),
                # "mt_tags": row["tgt_tags"].split(),
            }
