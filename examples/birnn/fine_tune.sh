#!/bin/bash
set -e

deepquestpy_dir="."
config_file="deepquestpy/config/birnn_sent_fine_tune.jsonnet" # Config file for loading the model saved with pre-training.

#config_file="deepquestpy/config/birnn_word.jsonnet" # use for word level experiments

output_dir="data/output/fine_tunned_model"

python "${deepquestpy_dir}/deepquestpy_cli/run_birnn.py" \
  --do_train \
  --config_file "${config_file}" \
  --output_dir "${output_dir}" \
  --overwrite_output_dir
