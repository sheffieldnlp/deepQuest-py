import numpy as np

from sklearn.metrics import f1_score, matthews_corrcoef

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

    # {0: 'BAD', 1: 'OK'}
    f1_bad, f1_good = f1_score(flat_references, flat_predictions, average=None, pos_label=None)
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
                    "references": {
                        "src": datasets.Sequence(datasets.Value("int32")),
                        "tgt": datasets.Sequence(datasets.Value("int32")),
                    },
                    "predictions": {
                        "src": datasets.Sequence(datasets.Value("int32")),
                        "tgt": datasets.Sequence(datasets.Value("int32")),
                    },
                }
            ),
        )

    def _compute(self, references, predictions):
        scores = {}
        if len(references[0]["tgt"]) > 0:
            scores_tgt = compute_scores(
                references=[ref["tgt"] for ref in references], predictions=[pred["tgt"] for pred in predictions]
            )
            scores.update(
                {
                    "tgt_f1_good": scores_tgt["f1_good"],
                    "tgt_f1_bad": scores_tgt["f1_bad"],
                    "tgt_mcc": scores_tgt["mcc"],
                }
            )
        if len(references[0]["src"]) > 0:
            scores_src = compute_scores(
                references=[ref["src"] for ref in references], predictions=[pred["src"] for pred in predictions]
            )
            scores.update(
                {
                    "src_f1_good": scores_src["f1_good"],
                    "src_f1_bad": scores_src["f1_bad"],
                    "src_mcc": scores_src["mcc"],
                }
            )
        return scores
