# TransQuest at WMT2020: Sentence-Level Direct Assessment

[Ranasinghe et al. (2020)](https://aclanthology.org/2020.wmt-1.122/) propose models for **sentence-level** Quality Estimation based on fine-tuning XLM-R on gold data. In particular, deepQuest-py provides an implementation of the **MonoTransQuest** architecture.

## WMT20 Shared Task on Quality Estimation - Task 1

1. Run `train.sh` to train a model using the data from the Shared Task.
3. Run `predict.sh` to generate predictions using the fine-tuned model on test data, and compute evaluation metrics.

Make sure to change the paths in each scripts accordingly.
