#!/bin/bash
set -e

deepquestpy_dir="/experiments/falva/tools/deepQuest-py"
model_dir="/data/falva/deepquestpyweb_models/transformer-sent/en-de"
log_path="./log"

python "${deepquestpy_dir}/deepquestpy_web/run_transformer.py" \
    --port 5003 --host 0.0.0.0 --log_path "${log_path}"\
    --model_name_or_path "${model_dir}" --arch_name "transformer-sent" \
    --src_lang "en" --tgt_lang "de" \
    --output_dir . \
