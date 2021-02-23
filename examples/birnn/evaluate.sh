#!/bin/bash
set -e

deepquestpy_dir="."
eval_model="./data/output/model/model.tar.gz"
eval_data_path="test"

python "${deepquestpy_dir}/deepquestpy_cli/run_birnn.py" \
    --do_eval \
    --eval_model "${eval_model}" \
    --eval_data_path "${eval_data_path}" \