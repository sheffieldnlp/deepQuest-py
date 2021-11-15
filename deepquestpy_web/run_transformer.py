import logging
import os
import sys
import pandas as pd
import transformers

from dataclasses import dataclass, field

from flask import Flask, request, jsonify

from transformers import HfArgumentParser, TrainingArguments, Trainer
import datasets

from sacremoses import MosesTokenizer

from deepquestpy.commands.cli_args import DataArguments, ModelArguments
from deepquestpy.commands.utils import get_deepquest_model

app = Flask(__name__)
logger = logging.getLogger(__name__)


@dataclass
class ServerArguments:
    """
    Arguments pertaining to how we are running the server application.
    """

    port: int = field(
        default=None, metadata={"help": "Port number"},
    )
    host: str = field(
        default=None, metadata={"help": "Host address"},
    )
    log_path: str = field(
        default="predictor.log", metadata={"help": "File to output the log"},
    )


def create_dataset_from_json(data_json, data_args):
    formatted_data = [
        {"translation": {data_args.src_lang: json_line["text_a"], data_args.tgt_lang: json_line["text_b"]}, "sent_label": -10_000,} for json_line in data_json
    ]
    return datasets.Dataset.from_pandas(pd.DataFrame(formatted_data))


def format_predictions_for_web(data_json, predictions, data_args):
    if "predictions" in predictions:
        # it is a sentence-level model
        predictions_scores = predictions["predictions"].astype("float64")
        result = [[pred] for pred in predictions_scores]
    else:
        # it is a word-level model
        # TODO: how does the front-end deal with tags in the source
        result = [pred for pred in predictions["predictions_tgt"]]

    def tokenize(data_json, column_name, lang):
        tokenizer = MosesTokenizer(lang=lang)
        tokenized = [tokenizer.tokenize(item[column_name]) for item in data_json]
        return tokenized

    response = {
        "predictions": result,
        "source_tokens": tokenize(data_json, "text_a", data_args.src_lang),
        "target_tokens": tokenize(data_json, "text_b", data_args.tgt_lang),
    }
    logger.info(response)
    response = jsonify(response)
    return response


def main():
    # Read the arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ServerArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, server_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, server_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", handlers=[logging.StreamHandler(server_args.log_path)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Prediction parameters {training_args}")

    deepquest_model = get_deepquest_model(model_args.arch_name, model_args, data_args, training_args)

    # Initialize Trainer
    trainer = Trainer(
        model=deepquest_model.get_model(), args=training_args, tokenizer=deepquest_model.get_tokenizer(), data_collator=deepquest_model.get_data_collator(),
    )

    @app.route("/predict", methods=["POST"])
    def predict():
        input_json = request.json
        print(input_json)
        # check that the language selected correspond to the model loaded
        if input_json["language"] == f"{data_args.src_lang}-{data_args.tgt_lang}":
            raw_dataset = create_dataset_from_json(input_json["data"], data_args)
            predict_dataset = deepquest_model.tokenize_datasets(raw_dataset)
            predictions, labels, _ = trainer.predict(predict_dataset)
            predictions = deepquest_model.postprocess_predictions(predictions, labels)
            return format_predictions_for_web(input_json["data"], predictions, data_args)

    app.run(host=server_args.host, port=server_args.port)


if __name__ == "__main__":
    main()
