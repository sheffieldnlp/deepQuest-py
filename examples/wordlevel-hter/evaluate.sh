#!/bin/bash
set -e

deepquestpy_dir="/experiments/falva/tools/deepQuest-py"
data_dir="/data/falva/wmt20qe_hter/gold_data/for_qe/en-de/dev"
# data_dir="/data/falva/wmt20qe_hter/gold_data/for_qe/en-de/test"
output_dir="/experiments/falva/wordlevel_qe/wmt20qe-hter.joint+synt+gold/output"

# do not forget to change 'dev' and 'valid' accordingly
python "${deepquestpy_dir}/deepquestpy/metrics/wordlevel_eval.py" \
	--tgt_tags_gold "${data_dir}/dev.tags" \
	--src_tags_gold "${data_dir}/dev.source_tags" \
	--tgt_tags_pred "${output_dir}/valid.mlqe.tgt.preds"\
	--src_tags_pred "${output_dir}/valid.mlqe.src.preds"\