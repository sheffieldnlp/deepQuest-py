from pathlib import Path
import numpy as np
import torch

# Paths
PACKAGE_DIR = Path(__file__).resolve().parent.parent
DATASETS_LOADERS_DIR = PACKAGE_DIR / "datasets"

def one_hot_encoding(A):  # A is Input array
    a = np.unique(A, return_inverse=1)[1]
    return torch.tensor((a.ravel()[:,None] == np.arange(a.max()+1)).astype(int))
