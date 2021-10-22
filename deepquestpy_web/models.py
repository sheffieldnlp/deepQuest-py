import torch

from abc import ABCMeta, abstractmethod
from flask import jsonify
from sacremoses import MosesTokenizer


class DeepQuestModelServer(metaclass=ABCMeta):
    def __init__(self, args, logger, data_loader):
        self.args = args
        self.logger = logger
        self.config = load_config(args)
        self.model = None
        self.load_model()
        self.data_loader = data_loader
        self.target_tokenizer = MosesTokenizer(lang=args.lang_pair[-2:])
        self.source_tokenizer = MosesTokenizer(lang=args.lang_pair[:2])

    def load_model(self):
        use_cuda = False if self.args.cpu else torch.cuda.is_available()
        self.model = QuestModel(self.config["model_type"], self.args.model_dir, use_cuda=use_cuda, args=self.config)

    def predict(self, input_json):
        self.logger.info(input_json)
        try:
            test_set = self.load_data(input_json)
        except Exception:
            self.logger.exception("Exception occurred when processing input data")
            raise
        try:
            model_output = self.predict_from_model(test_set)
        except Exception:
            self.logger.exception("Exception occurred when generating predictions!")
            raise
        try:
            output = self.prepare_output(input_json, model_output)
        except Exception:
            self.logger.exception("Exception occurred when building response!")
            raise
        return output

    @staticmethod
    def tokenize(input_json, text_name, tokenizer):
        tokenized = []
        for item in input_json["data"]:
            tokenized.append(tokenizer.tokenize(item[text_name]))
        return tokenized

    def load_data(self, input_json):
        test_set = self.data_loader(self.config, evaluate=True, serving_mode=True)
        test_set.make_dataset(input_json["data"])
        return test_set

    @abstractmethod
    def format_output_for_server(self, predictions):
        raise NotImplementedError()


class DeepQuestModelSentenceServer(DeepQuestModelServer):
    def predict_from_model(self, test_set):
        _, model_output = self.model.eval_model(test_set.tensor_dataset, serving=True)
        return model_output

    def prepare_output(self, input_json, model_output):
        predictions = model_output.tolist()
        if not type(predictions) is list:
            predictions = [predictions]
        result = [[pred] for pred in predictions]  # return list of lists to be consistent with word-level serving
        response = {
            "predictions": result,
            "source_tokens": self.tokenize(input_json, "text_a", self.source_tokenizer),
            "target_tokens": self.tokenize(input_json, "text_b", self.target_tokenizer),
        }
        self.logger.info(response)
        response = jsonify(response)
        return response


class DeepQuestModelWordServer(DeepQuestModelServer):
    def prepare_output(self, input_json, model_output):
        response = {
            "predictions": model_output,
            "source_tokens": self.tokenize(input_json, "text_a", self.source_tokenizer),
            "target_tokens": self.tokenize(input_json, "text_b", self.target_tokenizer),
        }
        return jsonify(response)

    def predict_from_model(self, test_set):
        preds = self.model.predict(test_set.tensor_dataset, serving=True)
        res = []
        for i, preds_i in enumerate(preds):
            input_ids = test_set.tensor_dataset.tensors[0][i]
            input_mask = test_set.tensor_dataset.tensors[1][i]
            preds_i = [p for j, p in enumerate(preds_i) if input_mask[j] and input_ids[j] not in (0, 2)]
            bpe_pieces = test_set.tokenizer.tokenize(test_set.examples[i].text_a)
            mt_tokens = self.target_tokenizer.tokenize(test_set.examples[i].text_a)
            mapped = map_pieces(bpe_pieces, mt_tokens, preds_i, "average", from_sep="▁")
            res.append([int(v) for v in mapped])
        return res
