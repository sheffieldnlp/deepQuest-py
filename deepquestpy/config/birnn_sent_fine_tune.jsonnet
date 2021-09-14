{
  "dataset_reader": {
    "type": "birnn_sent_reader",
    "data_path": "datasets/et_en_mlqe",
    "token_indexers_src": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": false,
        "namespace": "tokens_src"
      }
    },
    "token_indexers_tgt": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": false,
        "namespace": "tokens_tgt"
      }
    }
  },
  "train_data_path": "train",
  "validation_data_path": "dev",
  "model": {
    "type": "from_archive",
    "archive_file": "data/et_en_wiki25k_fine_tuned_10M_model/model.tar.gz"},
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 32
    }
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 15,
    "validation_metric": "+pearson",
    "optimizer": {
      "type": "adagrad",
      "lr": 0.0001
    }
  }

}
