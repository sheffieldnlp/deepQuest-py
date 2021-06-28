#!/bin/bash
set -e

deepquestpy_dir="/experiments/falva/tools/deepQuest-py"
synthetic_train_dir="/data/falva/wmt20qe_hter/synthetic_data/for_qe"
output_dir="/experiments/falva/wordlevel_qe/wmt20qe-hter.joint-synth/model"

CUDA_VISIBLE_DEVICES=0,1,2,3 python "${deepquestpy_dir}/deepquestpy_cli/run_model.py" \
    --task_name "mlqe" \
    --model_name_or_path "xlm-roberta-large" \
    --model_type "joint" \
    --dataset_name "wmt20qe_hter_synth" \
    --synthetic_train_dir "${synthetic_train_dir}" --synthetic_train_perc 100 \
    --output_dir "${output_dir}" \
    --do_train --num_train_epochs 2 --save_steps 10000 --save_total_limit 1 \
    --learning_rate=5e-6 \
    --overwrite_output_dir