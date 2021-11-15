deepquestpy_dir="/experiments/falva/tools/deepQuest-py"
pretrained_model_dir="xlm-roberta-base"
dataset_name="mqm_google"
train_file="/data/falva/mqm-google/mqm-google_rows-merged_newstest2020_ende.train.csv"
validation_file="/data/falva/mqm-google/mqm-google_rows-merged_newstest2020_ende.test.csv"
output_dir="/experiments/falva/word-level/al.mqm-google.transformerword-mqm/model"

python "${deepquestpy_dir}/examples/active_learning/run_al.py" \
    --model_name_or_path "${pretrained_model_dir}" --arch_name "transformer-word" \
    --dataset_name "${dataset_name}" --src_lang "en" --tgt_lang "de" \
    --label_column_name_src "none" --label_column_name_tgt "bad_labels" --pad_to_max_length \
    --do_train --train_file "${train_file}" --per_device_train_batch_size=8 \
    --num_train_epochs 20 --save_steps 10000 --save_total_limit 1 --weight_decay=0.01 \
    --learning_rate=5e-6 \
    --do_eval --validation_file "${validation_file}" --per_device_eval_batch_size=8 \
    --output_dir "${output_dir}" --overwrite_output_dir \

