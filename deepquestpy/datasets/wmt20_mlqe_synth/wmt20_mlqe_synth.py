# coding=utf-8
import os

import logging

import datasets


_CITATION = """
"""

_DESCRIPTION = """
"""

_LANGUAGE_PAIRS = [
    ("en", "de"),
]


class WMT20QESynthConfig(datasets.BuilderConfig):
    def __init__(self, src_lg, tgt_lg, **kwargs):
        super(WMT20QESynthConfig, self).__init__(**kwargs)
        self.src_lg = src_lg
        self.tgt_lg = tgt_lg


class WMT20QESynth(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        WMT20QESynthConfig(
            name=f"{src_lg}-{tgt_lg}",
            version=datasets.Version("0.0.1"),
            description=f"WMT20QESynth: {src_lg} - {tgt_lg}",
            src_lg=src_lg,
            tgt_lg=tgt_lg,
        )
        for (src_lg, tgt_lg) in _LANGUAGE_PAIRS
    ]
    BUILDER_CONFIG_CLASS = WMT20QESynthConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "translation": datasets.Translation(languages=(self.config.src_lg, self.config.tgt_lg)),
                    "src_tags": datasets.Sequence(datasets.ClassLabel(names=["BAD", "OK"])),
                    "mt_tags": datasets.Sequence(datasets.ClassLabel(names=["BAD", "OK"])),
                    "hter": datasets.Value("float32"),
                }
            ),
            supervised_keys=None,
            homepage="",
            license="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if not self.config.data_dir:
            raise ValueError(f"Must specify the folder where the files are, but got data_dir={self.config.data_dir}")
        data_dir = self.config.data_dir
        generators = []
        generators.append(
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "source_lg": self.config.src_lg,
                    "target_lg": self.config.tgt_lg,
                    "src_path": os.path.join(data_dir, "train", "train.src"),
                    "tgt_path": os.path.join(data_dir, "train", "train.mt"),
                    "src_tags_path": os.path.join(data_dir, "train", "train.source_tags"),
                    "tgt_tags_path": os.path.join(data_dir, "train", "train.tags"),
                    "hter_path": os.path.join(data_dir, "train", "train.hter"),
                },
            )
        )

        return generators

    def _generate_examples(self, source_lg, target_lg, src_path, tgt_path, src_tags_path, tgt_tags_path, hter_path):
        logging.info("Generating examples")
        with open(src_path, encoding="utf-8") as src_file, open(tgt_path, encoding="utf-8") as mt_file, open(
            src_tags_path, encoding="utf-8"
        ) as src_tags_file, open(tgt_tags_path, encoding="utf-8") as mt_tags_file, open(
            hter_path, encoding="utf-8"
        ) as hter_file:
            for id, (src, mt, src_tags, mt_tags, hter) in enumerate(
                zip(src_file, mt_file, src_tags_file, mt_tags_file, hter_file)
            ):
                yield id, {
                    "translation": {source_lg: src.strip(), target_lg: mt.strip()},
                    "src_tags": src_tags.strip().split(),
                    "mt_tags": mt_tags.strip().split(),
                    "hter": float(hter.strip()),
                }
