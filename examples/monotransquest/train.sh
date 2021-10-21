#!/bin/bash
set -e

deepquestpy_dir="/experiments/falva/tools/deepQuest-py"
output_dir="/data/falva/deepquestpyweb_models/transformer-sent/en-de"

CUDA_VISIBLE_DEVICES=0,1,2,3 python "${deepquestpy_dir}/deepquestpy_cli/run_transformer.py" \
    --model_name_or_path "xlm-roberta-base" --arch_name "transformer-sent" \
    --dataset_name "wmt20_mlqe_task1" --src_lang "en" --tgt_lang "de" --label_column_name "z_mean" \
    --do_train --num_train_epochs 5 --save_steps 10000 --save_total_limit 1 \
    --learning_rate 1e-5 --warmup_ratio 0.1 --max_grad_norm 1.0 --adam_epsilon 1e-8 \
    --do_eval \
    --output_dir "${output_dir}" --overwrite_output_dir \