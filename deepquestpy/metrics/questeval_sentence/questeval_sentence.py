import numpy as np

import datasets

from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr

_CITATION = """
"""

_DESCRIPTION = """\
Implements metrics for quality estimation evaluation. Based on the official scripts from WMT20 QE Shared Task
https://github.com/sheffieldnlp/qe-eval-scripts
"""

_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.
    >>> questeval_sentence = datasets.load_metric("questeval_sentence")
    >>> results = questeval_sentence.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""


# @datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class QuEstEvalSentence(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {"predictions": datasets.Value("float32"), "references": datasets.Value("float32")}
            ),
        )

    def _compute(self, predictions, references):
        assert len(references) == len(
            predictions
        ), f"Incorrect number of predicted scores, expecting {len(references)}, given {len(predictions)}."

        pearson = pearsonr(references, predictions)[0]
        diff = np.subtract(references, predictions)
        mae = np.abs(diff).mean()
        rmse = (diff ** 2).mean() ** 0.5

        scores = {"pearson": pearson, "mae": mae, "rmse": rmse}
        return scores
