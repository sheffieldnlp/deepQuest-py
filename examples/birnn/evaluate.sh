#!/bin/bash
set -e

deepquestpy_dir="."
#eval_model="./data/output/model/model.tar.gz"

eval_model="data/any_en_wiki_matrix_122k/model.tar.gz"
eval_data_path="neen_test20"
lang_pair="ne-en" # The language pair, for example, ro-en, the model is trained on and is getting evaluated. 

eval_output_file="data/any_en_wiki_matrix_122k/neen_test20/eval_results_test20.json"
pred_output_file="data/any_en_wiki_matrix_122k/neen_test20/predictions_test20.txt"


python "${deepquestpy_dir}/deepquestpy_cli/run_birnn.py" \
    --do_eval \
    --lang_pair "${lang_pair}" \
    --eval_model "${eval_model}" \
    --eval_data_path "${eval_data_path}" \
    --eval_output_file "${eval_output_file}" \
    --pred_output_file "${pred_output_file}"
