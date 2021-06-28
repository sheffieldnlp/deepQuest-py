import numpy as np
from collections import Counter

from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr

import datasets

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
    >>> questeval_word = datasets.load_metric("questeval_word")
    >>> results = questeval_word.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""


def list_of_lists(a_list):
    """
        check if <a_list> is a list of lists
        """
    if (
        isinstance(a_list, (list, tuple, np.ndarray))
        and len(a_list) > 0
        and all([isinstance(l, (list, tuple, np.ndarray)) for l in a_list])
    ):
        return True
    return False


def flatten(lofl):
    """
    convert list of lists into a flat list
    """
    if list_of_lists(lofl):
        return [item for sublist in lofl for item in sublist]
    elif type(lofl) == dict:
        return lofl.values()


def compute_scores(references, predictions):
    # Verify that there's the same number of tags for each instance
    for idx, (gold, pred) in enumerate(zip(references, predictions)):
        assert len(gold) == len(pred), f"Numbers of tags don't match in sequence {idx}: {len(gold)} and {len(pred)}"

    flat_references = flatten(references)
    flat_predictions = flatten(predictions)

    f1_good, f1_bad = f1_score(flat_references, flat_predictions, average=None, pos_label=None)
    mcc = matthews_corrcoef(flat_references, flat_predictions)

    scores = {"f1_good": f1_good, "f1_bad": f1_bad, "mcc": mcc}

    return scores


# @datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class QuEstEvalWord(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "references_src": datasets.Sequence(datasets.Value("int")),
                    "predictions_src": datasets.Sequence(datasets.Value("int")),
                    "references_tgt": datasets.Sequence(datasets.Value("int")),
                    "predictions_tgt": datasets.Sequence(datasets.Value("int")),
                }
            ),
        )

    def _compute(self, references_src, predictions_src, references_tgt, predictions_tgt):
        scores_src = compute_scores(references_src, predictions_src)
        scores_tgt = compute_scores(references_tgt, predictions_tgt)
        return {"src": scores_src, "tgt": scores_tgt}
