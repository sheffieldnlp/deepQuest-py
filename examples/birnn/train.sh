#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

set -e

deepquestpy_dir="."

config_file="deepquestpy/config/birnn_sent.jsonnet" # Use for sentence level experiments
#config_file="deepquestpy/config/birnn_word.jsonnet" # use for word level experiments

output_dir="data/output/model"

python "${deepquestpy_dir}/deepquestpy_cli/run_birnn.py" \
  --do_train \
  --config_file "${config_file}" \
  --output_dir "${output_dir}" \
  --overwrite_output_dir
