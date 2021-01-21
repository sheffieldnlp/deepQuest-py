#!/bin/bash
set -e

deepquestpy_dir="/experiments/falva/tools/deepQuest-py"
data_dir="/data/falva/wmt20qe_hter/gold_data/for_qe/en-de"
pretrained_model_dir="/experiments/falva/wordlevel_qe/wmt20qe-hter.joint-synth/model"
output_dir="/experiments/falva/wordlevel_qe/wmt20qe-hter.joint-synt-gold/model"

CUDA_VISIBLE_DEVICES=0,1,2,3 python "${deepquestpy_dir}/deepquestpy_cli/run_model.py" \
    --task_name "mlqe" \
    --model_name_or_path "${pretrained_model_dir}" \
    --model_type "joint" \
    --dataset_name "wmt20qe_hter" \
    --data_dir "${data_dir}" \
    --output_dir "${output_dir}" \
    --do_train --num_train_epochs 5 --save_steps 1000 --save_total_limit 1 \
    --overwrite_output_dir \
