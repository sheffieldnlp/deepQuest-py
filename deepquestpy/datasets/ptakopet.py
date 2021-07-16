# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bergamot Ptakopet."""

import os

import datasets


_CITATION = """
Not available.
"""

_DESCRIPTION = """\
Not available.
"""

_HOMEPAGE = "https://browser.mt/data"

_LICENSE = "Unknown"

_LANGUAGE_PAIRS = [
    ("en", "et"),
    ("en", "cs"),
]


class BergamotPtakopetConfig(datasets.BuilderConfig):
    def __init__(self, src_lang, tgt_lang, use_binary_tags, **kwargs):
        super(BergamotPtakopetConfig, self).__init__(**kwargs)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.use_binary_tags = use_binary_tags


class BergamotPtakopet(datasets.GeneratorBasedBuilder):
    """Bergamot Ptakopet dataset."""

    BUILDER_CONFIGS = [
        BergamotPtakopetConfig(
            name=f"{src_lang}-{tgt_lang}",
            version=datasets.Version("1.0.0"),
            description=f"Bergamot Ptakopet: {src_lang} - {tgt_lang}",
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        for (src_lang, tgt_lang) in _LANGUAGE_PAIRS
    ]

    BUILDER_CONFIG_CLASS = BergamotPtakopetConfig

    def _info(self):
        if self.config.use_binary_tags:
            features = (
                datasets.Features(
                    {
                        "translation": datasets.Translation(languages=(self.config.src_lang, self.config.tgt_lang)),
                        "src_tags": datasets.Sequence(datasets.ClassLabel(names=["OK", "BAD"])),
                        "tgt_tags": datasets.Sequence(datasets.ClassLabel(names=["OK", "BAD"])),
                        "sent_tag": datasets.Sequence(datasets.ClassLabel(names=["OK", "BAD"])),
                    }
                ),
            )
        else:
            features = (
                datasets.Features(
                    {
                        "translation": datasets.Translation(languages=(self.config.src_lang, self.config.tgt_lang)),
                        "src_tags": datasets.Sequence(datasets.ClassLabel(names=["OK", "Minor", "Major", "Critical"])),
                        "tgt_tags": datasets.Sequence(datasets.ClassLabel(names=["OK", "Minor", "Major", "Critical"])),
                        "sent_tag": datasets.Sequence(datasets.ClassLabel(names=["OK", "Minor", "Major", "Critical"])),
                    }
                ),
            )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if not self.config.data_dir:
            raise ValueError(f"Must specify the folder where the files are, but got data_dir={self.config.data_dir}")
        data_dir = self.config.data_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "src_path": os.path.join(data_dir, "test.src"),
                    "tgt_path": os.path.join(data_dir, "test.mt"),
                    "tgt_tags_path": os.path.join(data_dir, "test.word_tags"),
                    "src_lang": self.config.src_lang,
                    "tgt_lang": self.config.tgt_lang,
                },
            ),
        ]

    def _generate_examples(self, src_path, mt_path, mt_tags_path):
        with open(src_path, encoding="utf-8") as src_file, open(mt_path, encoding="utf-8") as mt_file, open(
            mt_tags_path, encoding="utf-8"
        ) as mt_tags_file:
            for id, (src, mt, mt_tags) in enumerate(zip(src_file, mt_file, mt_tags_file)):
                raw_mt_tags = mt_tags.strip().split()
                mt_tags = ["OK" if tag == "0" else "BAD" for tag in raw_mt_tags]
                src_tokens = src.strip().split()
                yield id, {
                    "src": src_tokens,
                    "tgt": mt.strip().split(),
                    "src_tags": ["OK"] * len(src_tokens),
                    "tgt_tags": mt_tags,
                }
