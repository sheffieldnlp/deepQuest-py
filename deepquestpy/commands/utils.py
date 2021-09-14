import importlib
from pathlib import Path

# Paths
PACKAGE_DIR = Path(__file__).resolve().parent.parent
DATASETS_LOADERS_DIR = PACKAGE_DIR / "datasets"
METRICS_DIR = PACKAGE_DIR / "metrics"


# maps lower-cased model names to model class names
ARCHITECTURE_MAP = {
    "transformer-sent": "TransformerDeepQuestModelSent",
    "transformer-word": "TransformerDeepQuestModelWord",
    "beringlab-word": "BeringLabWord",
    "birnn-sent": "BiRNNSent",
    "birnn-word": "BiRNNWord",
}

# QE MODEL FACTORY-like
def get_deepquest_model(architecture_name, *args, **kwargs):
    try:
        deepquest_model = getattr(
            importlib.import_module("deepquestpy.models"), ARCHITECTURE_MAP[architecture_name.lower()]
        )
        return deepquest_model(*args, **kwargs)

    except ValueError as e:
        print(f"/!\ {e}")

