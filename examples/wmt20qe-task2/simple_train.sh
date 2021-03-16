#!/bin/bash
set -e

deepquestpy_dir="/experiments/falva/tools/deepQuest-py"
output_dir="./model"

CUDA_VISIBLE_DEVICES=0,1,2,3 python "${deepquestpy_dir}/deepquestpy_cli/run_xlmroberta.py" \
    --model_name_or_path "xlm-roberta-base" --model_type "word" \
    --dataset_name "wmt20_mlqe_task2" --src_lang "en" --tgt_lang "de" --label_column_name "mt_tags" \
    --do_train --num_train_epochs 1 --save_steps 10000 --save_total_limit 1 \
    --do_eval \
    --output_dir "${output_dir}" --overwrite_output_dir \