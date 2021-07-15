class DeepQuestModel:
    def __init__(self) -> None:
        return

    def predict(self):
        raise NotImplementedError()

    def postprocess_predictions(self, predictions=None, labels=None):
        raise NotImplementedError()

    def save_output(self, output_file_path, predictions):
        raise NotImplementedError()


class DeepQuestModelWord(DeepQuestModel):
    def __init__(self):
        return

    def save_output(self, output_file_path, predictions):
        with open(f"{output_file_path}.src.preds", "w") as writer:
            for prediction in predictions["predictions_src"]:
                writer.write(" ".join([self.label_list[p] for p in prediction]) + "\n")

        with open(f"{output_file_path}.tgt.preds", "w") as writer:
            for prediction in predictions["predictions_tgt"]:
                writer.write(" ".join([self.label_list[p] for p in prediction]) + "\n")


class DeepQuestModelSent(DeepQuestModel):
    def __init__(self):
        return

    def save_output(self, output_file_path, predictions):
        with open(f"{output_file_path}.preds", "w") as writer:
            for item in predictions["predictions"]:
                writer.write(f"{item:3.3f}\n")
