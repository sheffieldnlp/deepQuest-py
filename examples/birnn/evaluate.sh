#!/bin/bash
set -e

deepquestpy_dir="."
eval_model="./data/output/model/model.tar.gz"
eval_data_path="test"
config_file="/Users/hppx88/MT-QualityEstimation/code/deepQuest-py/deepquestpy/config/birnn_sent.jsonnet"

python "${deepquestpy_dir}/deepquestpy_cli/run_birnn.py" \
    --do_eval \
    --eval_model "${eval_model}" \
    --eval_data_path "${eval_data_path}" \
    --config_file "${config_file}"
