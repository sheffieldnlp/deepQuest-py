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

# Lint as: python3
"""WMT20: Quality Estimation Task"""

import os

import logging

import datasets


_CITATION = """\
@InProceedings{specia-EtAl:2020:WMTQE,
  author    = {Specia, Lucia  and  Blain, Frederic  and  Fomicheva, Marina  and  Fonseca, Erick  and  Chaudhary, Vishrav  and  GuzmÃ¡n, Francisco  and  Martins, AndrÃ© F. T.},
  title     = {Findings of the WMT 2020 Shared Task on Quality Estimation},
  booktitle      = {Proceedings of the Fifth Conference on Machine Translation},
  month          = {November},
  year           = {2020},
  address        = {Online},
  publisher      = {Association for Computational Linguistics},
  pages     = {652--673},
  url       = {https://www.aclweb.org/anthology/2020.wmt-1.79}
}
"""

_DESCRIPTION = """\
For details see http://www.statmt.org/wmt20/quality-estimation-task.html
"""


class WMT2020QEConfig(datasets.BuilderConfig):
    """BuilderConfig for WMT2020QE"""

    def __init__(self, **kwargs):
        """BuilderConfig for WMT2020 QE.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WMT2020QEConfig, self).__init__(**kwargs)


class WMT2020QE(datasets.GeneratorBasedBuilder):
    """ dataset."""

    BUILDER_CONFIGS = [
        WMT2020QEConfig(
            name="wmt2020qe", version=datasets.Version("1.0.0"), description="WMT2020 Quality Estimation dataset"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "src": datasets.Sequence(datasets.Value("string")),
                    "tgt": datasets.Sequence(datasets.Value("string")),
                    "src_tags": datasets.Sequence(datasets.features.ClassLabel(names=["OK", "BAD"])),
                    "tgt_tags": datasets.Sequence(datasets.features.ClassLabel(names=["OK", "BAD"])),
                    "hter": datasets.Value("float32"),
                }
            ),
            supervised_keys=None,
            homepage="http://www.statmt.org/wmt20/quality-estimation-task.html",
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
                    "src_path": os.path.join(data_dir, "train", "train.src"),
                    "tgt_path": os.path.join(data_dir, "train", "train.mt"),
                    "src_tags_path": os.path.join(data_dir, "train", "train.source_tags"),
                    "tgt_tags_path": os.path.join(data_dir, "train", "train.tags"),
                    "hter_path": os.path.join(data_dir, "train", "train.hter"),
                },
            )
        )

        if os.path.exists(os.path.join(data_dir, "dev")):
            generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "src_path": os.path.join(data_dir, "dev", "dev.src"),
                        "tgt_path": os.path.join(data_dir, "dev", "dev.mt"),
                        "src_tags_path": os.path.join(data_dir, "dev", "dev.source_tags"),
                        "tgt_tags_path": os.path.join(data_dir, "dev", "dev.tags"),
                        "hter_path": os.path.join(data_dir, "dev", "dev.hter"),
                    },
                )
            )

        if os.path.exists(os.path.join(data_dir, "test")):
            generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "src_path": os.path.join(data_dir, "test", "test.src"),
                        "tgt_path": os.path.join(data_dir, "test", "test.mt"),
                        "src_tags_path": os.path.join(data_dir, "test", "test.source_tags"),
                        "tgt_tags_path": os.path.join(data_dir, "test", "test.tags"),
                        "hter_path": os.path.join(data_dir, "test", "test.hter"),
                    },
                )
            )

        return generators

    def _generate_examples(self, src_path, tgt_path, src_tags_path, tgt_tags_path, hter_path):
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
                    "src": src.strip().split(),
                    "tgt": mt.strip().split(),
                    "src_tags": src_tags.strip().split(),
                    "tgt_tags": mt_tags.strip().split(),
                    "hter": float(hter.strip()),
                }
