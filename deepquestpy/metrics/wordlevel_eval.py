from __future__ import division, print_function

import numpy as np
import argparse
from collections import Counter

from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import KFold

"""
Scoring programme for WMT'20 Task 2 HTER **word-level**
"""


# -------------PREPROCESSING----------------
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


def compute_scores(gold_tags, pred_tags):
    flat_gold = flatten(gold_tags)
    flat_pred = flatten(pred_tags)

    # Main Metrics
    def to_single_severity_label(tags):
        return tags
        # return np.array([1 if t > 0 else 0 for t in tags])

    f1_good, f1_bad = f1_score(
        to_single_severity_label(flat_gold), to_single_severity_label(flat_pred), average=None, pos_label=None
    )
    mcc = matthews_corrcoef(flat_gold, flat_pred)
    scores = {"f1_good": f1_good, "f1_bad": f1_bad, "mcc": mcc}

    # Severity Analysis

    # stats = dict.fromkeys(set(flat_gold), {"tp": 0, "fp": 0})
    # for gold_tag, pred_tag in zip(flat_gold, flat_pred):
    #     if gold_tag == 0:
    #         if gold_tag == pred_tag:
    #             stats[gold_tag]["tp"] += 1
    #         else:
    #             stats[pred_tag]["fp"] += 1
    #     else:
    #         if pred_tag > 0:
    #             stats[gold_tag]["tp"] += 1
    #         else:
    #             stats[pred_tag]["fp"] += 1

    # counts = Counter(flat_gold)
    # for tag_name, tag_stats in stats.items():
    #     tag_stats["fn"] = counts[tag_name] - tag_stats["tp"]
    #     tag_stats["tn"] = len(flat_gold) - counts[tag_name] - tag_stats["fp"]
    #     p = tag_stats["tp"] / (tag_stats["tp"] + tag_stats["fp"])
    #     r = tag_stats["tp"] / (tag_stats["tp"] + tag_stats["fn"])
    #     f1_tag = 2 * p * r / (p + r)
    #     # mcc_num = (tag_stats["tp"] * tag_stats["tn"]) - (tag_stats["fp"] * tag_stats["fn"])
    #     # mcc_den = (
    #     #     (tag_stats["tp"] + tag_stats["fp"])
    #     #     * (tag_stats["tp"] + tag_stats["fn"])
    #     #     * (tag_stats["tn"] + tag_stats["fp"])
    #     #     * (tag_stats["tn"] + tag_stats["fn"])
    #     # ) ** 0.5
    #     # mcc_tag = mcc_num / mcc_den
    #     scores.update({f"f1-{tag_name}": f1_tag, f"mcc-{tag_name}": 0.0})

    return scores


def read_tags(file_path):
    with open(file_path) as f:
        tags = []
        for line in f:
            raw_tags = line.strip().split()
            cleaned_tags = []
            for tag in raw_tags:
                if tag == "OK":
                    tag = 0
                elif tag == "BAD":
                    tag = 1
                else:
                    tag = int(tag)
                cleaned_tags.append(tag)
            tags.append(np.array(cleaned_tags))
    return np.array(tags, dtype=object)


def compute_crossval_scores(true_tags, test_tags, n_folds):
    kf = KFold(n_splits=n_folds)
    accum_scores = {}
    for _, test_index in kf.split(true_tags):
        scores = compute_scores(true_tags[test_index], test_tags[test_index])
        for k, v in scores.items():
            if k in accum_scores:
                accum_scores[k].append(v)
            else:
                accum_scores[k] = [v]

    crossval_scores = {}
    for k, v in accum_scores.items():
        crossval_scores[f"{k}_mean"] = np.mean(v)
        crossval_scores[f"{k}_std"] = np.std(v)
    return crossval_scores


def compute_and_print_scores(true_tags_file, test_tags_file, n_folds, prefix=""):
    gold = read_tags(true_tags_file)
    pred = read_tags(test_tags_file)
    metrics = compute_crossval_scores(gold, pred, n_folds)
    for k, v in metrics.items():
        print(f"{prefix}_{k} = {v:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_tags_gold", required=True)
    parser.add_argument("--tgt_tags_pred", required=True)
    parser.add_argument("--src_tags_gold", required=False)
    parser.add_argument("--src_tags_pred", required=False)
    parser.add_argument("--k_fold", required=False, default=5)
    args = parser.parse_args()

    compute_and_print_scores(
        args.tgt_tags_gold, args.tgt_tags_pred, args.k_fold, prefix="tgt",
    )
    if args.src_tags_gold and args.src_tags_pred:
        compute_and_print_scores(
            args.src_tags_gold, args.src_tags_pred, args.k_fold, prefix="src",
        )
