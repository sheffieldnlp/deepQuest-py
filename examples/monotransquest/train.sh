#!/bin/bash
set -e

deepquestpy_dir="/experiments/falva/tools/deepQuest-py"
output_dir="/data/falva/deepquestpyweb_models/transformer-sent/et-en"

CUDA_VISIBLE_DEVICES=0,1,2,3 python "${deepquestpy_dir}/deepquestpy_cli/run_transformer.py" \
    --model_name_or_path "xlm-roberta-base" --arch_name "transformer-sent" \
    --dataset_name "mlqe_da" --src_lang "et" --tgt_lang "en" --label_column_name_sent "z_mean" \
    --do_train --num_train_epochs 20 --save_steps 10000 --save_total_limit 1 \
    --learning_rate 1e-5 --warmup_ratio 0.1 --max_grad_norm 1.0 --adam_epsilon 1e-8 \
    --do_eval \
    --output_dir "${output_dir}" --overwrite_output_dir \
