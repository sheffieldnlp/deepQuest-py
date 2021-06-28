{
  "dataset_reader": {
    "type": "birnn_sent_reader",
    "data_path": "/experiments/agajbhiye/deepQuest-py-ForPrediction/datasets/ro_en_mlqe",
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
    "type": "birnn",
   "text_field_embedder_src": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 50,
          "trainable": true,
          "vocab_namespace": "tokens_src"
        }
      }
    },
    "text_field_embedder_tgt": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 50,
          "trainable": true,
          "vocab_namespace": "tokens_tgt"
        }
      }
    },
    "seq2seq_encoder_src": {
      "type": "gru",
      "input_size": 50,
      "hidden_size": 50,
      "bidirectional": true
    },
    "seq2seq_encoder_tgt": {
      "type": "gru",
      "input_size": 50,
      "hidden_size": 50,
      "bidirectional": true
    },
    "attention": {
    },
    "dropout": 0.5,
    "kd_without_gold_data": true,
    "kd_with_gold_data": false,
    "alpha": 0.0
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 32
    }
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 15,
    "validation_metric": "+pearson",
    "optimizer": {
      "type": "adagrad",
      "lr": 0.001
    }
  }
}

