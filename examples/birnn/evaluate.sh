#!/bin/bash
set -e

deepquestpy_dir="."

eval_model="data/output/model/model.tar.gz"
eval_data_path="test"
lang_pair="ro-en" # The language pair, for example, ro-en, the model is trained on and is getting evaluated. 

eval_output_file="data/output/model/eval_results.json"
pred_output_file="data/output/model/predictions.txt"

python "${deepquestpy_dir}/deepquestpy_cli/run_birnn.py" \
    --do_eval \
    --lang_pair "${lang_pair}" \
    --eval_model "${eval_model}" \
    --eval_data_path "${eval_data_path}" \
    --eval_output_file "${eval_output_file}" \
    --pred_output_file "${pred_output_file}"
