import logging
import sys

import tarfile
from pathlib import Path

import transformers

# Paths
PACKAGE_DIR = Path(__file__).resolve().parent.parent
DATASETS_LOADERS_DIR = PACKAGE_DIR / "datasets"
METRICS_DIR = PACKAGE_DIR / "deepquestpy" / "metrics"



def disk_footprint(model_file):
    total_bytes = 0
    model_tar = tarfile.open(model_file, "r:gz")
    for model_file in model_tar:
        total_bytes += model_file.size
    return total_bytes