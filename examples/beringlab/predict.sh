#!/bin/bash
set -e

deepquestpy_dir="/experiments/falva/tools/deepQuest-py"
data_dir="/data/falva/wmt20qe_hter/gold_data/for_qe/en-de"
model_dir="/experiments/falva/wordlevel_qe/wmt20qe-hter.joint+synt+gold/model"
output_dir="/experiments/falva/wordlevel_qe/wmt20qe-hter.joint+synt+gold/output"

CUDA_VISIBLE_DEVICES=0,1,2,3 python "${deepquestpy_dir}/deepquestpy_cli/run_model.py" \
    --task_name "mlqe" \
    --model_name_or_path "${model_dir}" \
    --model_type "joint" \
    --dataset_name "wmt20qe_hter" \
    --data_dir "${data_dir}" \
    --output_dir "${output_dir}" \
    --do_eval --do_predict \
