import argparse
import numpy as np

from sklearn.model_selection import KFold

from datasets import load_metric


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
        questeval_word = load_metric("quest_eval", "word")
        scores = questeval_word.compute(true_tags[test_index], test_tags[test_index])
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
