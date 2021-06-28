import logging
import sys

from pathlib import Path

import transformers

# Paths
PACKAGE_DIR = Path(__file__).resolve().parent.parent
DATASETS_LOADERS_DIR = PACKAGE_DIR / "datasets"
METRICS_DIR = PACKAGE_DIR / "deepquestpy" / "metrics"
