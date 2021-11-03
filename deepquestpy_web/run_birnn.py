from dataclasses import dataclass, field

from flask import Flask, request, jsonify

from transformers import HfArgumentParser
from allennlp.models.archival import load_archive

from deepquestpy.commands.utils import get_deepquest_model

app = Flask(__name__)


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


@dataclass
class ModelArguments:
    """
    Arguments pertaining to the model that will be used for predictions.
    """

    arch_name: str = field(metadata={"help": "Specifies the architecture to use. "})
    lang_pair: str = field(
        default=None, metadata={"help": "language pair on which the model is trained and evaluated on."},
    )
    model_path: str = field(
        default="data/output/model/model.tar.gz", metadata={"help": "trained model."},
    )


def create_dataset_from_json(data_json, data_args):
    formatted_data = [
        {"translation": {data_args.src_lang: json_line["text_a"], data_args.tgt_lang: json_line["text_b"]}, "label": -10_000,} for json_line in data_json
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
    print(response)
    response = jsonify(response)
    return response


def main():
    # Read the arguments
    parser = HfArgumentParser((ModelArguments, ServerArguments))
    model_args, server_args = parser.parse_args_into_dataclasses()

    predictor = Predictor.from_path(model_args.model_path)

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
