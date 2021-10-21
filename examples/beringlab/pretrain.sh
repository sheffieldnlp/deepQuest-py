#!/bin/bash
set -e

deepquestpy_dir="/experiments/falva/tools/deepQuest-py"
synthetic_train_dir="/data/falva/wmt20qe_hter/synthetic_data/for_qe"
output_dir="/experiments/falva/wordlevel_qe/wmt20qe-hter.joint-synth/model"

CUDA_VISIBLE_DEVICES=0,1,2,3 python "${deepquestpy_dir}/deepquestpy_cli/run_transformer.py" \
    --model_name_or_path "xlm-roberta-base" --arch_name "beringlab-word" \
    --dataset_name "wmt20_mlqe_synth" --data_dir "${synthetic_train_dir}" \
    --src_lang "en" --tgt_lang "de" --label_column_name "mt_tags" --labels_in_gaps \
    --do_train --num_train_epochs 2 --save_steps 10000 --save_total_limit 1 \
    --learning_rate=5e-6 \
    --output_dir "${output_dir}" --overwrite_output_dir \