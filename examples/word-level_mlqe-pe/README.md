# Word-Level Quality Estimation

## MLQE-PE

1. Run `simple_train.sh` to train a basic model.
2. Run `simple_predict.sh` to generate predictions.

## WMT20QE-Task2
Steps to reproduce the results from [Lee (2020)](https://www.aclweb.org/anthology/2020.wmt-1.118/) in the WMT20 Shared Task on Quality Estimation - Task 2.

1. Run `pretrain.sh` to train a model using synthetic data.
2. Run `finetune.sh` to fine-tune the previous model on gold data.
3. Run `predict.sh` to generate predictions using the fine-tuned model on dev or test data, and compute evaluation metrics.

Make sure to change the paths in each scripts accordingly.
