# deepQuest-py. 

- Uses the newest versions of the HuggingFace Transformers and Datasets libraries.
- Inclues models for Word-Level Quality Estimation.

## Installation from Source

```
git clone https://github.com/sheffieldnlp/deepQuest-py.git
cd deepQuest-py
pip install --editable ./
```
### deepQuest-py BiRNN Model - Training and Evaluation (Gold Data)

- **Download the data**, run the script `data_download.sh`. This will download the MLQE and Wikipedia data into the directory `datasets/`. To download a particular dataset, please use the links provided below in the `Datasets` section. 
- **Update the config file**  - `deepquestpy/config/birnn_sent.jsonnet` as follows:
	- Set the `data_path` parameter to the path of the language pair directory on which the model needs to be trained. For example, to train the model on the MLQE language pair, `ro-en`, set `data_path` to the previously downloaded`datasets/ro_en_mlqe` directory.
	- Set the knowledge distillation flags - `kd_without_gold_data` and `kd_with_gold_data` to `false` and parameter `alpha` to `0.0`
- **Train the model** - Execute the script `examples/birnn/train.sh`
- **Evaluate the model** - Execute the evaluate script - `examples/birnn/evaluate.sh` 


## Knowledge Distillation

### Teacher Model, MonoTransQuest - Training and Evaluation
- Knowledge distillation framework in the `deepQuest-py` can utilise predictions from any teacher model.
- We used the predictions from the state-of-the-art [MonoTransQuest ](https://aclanthology.org/2020.coling-main.445/) pre-trained models available [here](https://tharindu.co.uk/TransQuest/models/sentence_level_pretrained.html). We include these predictions in the data provided for training the student model in the `Datasets` section.    
- The MonoTransQuest model can also be trained from scratch as detailed [here](https://tharindu.co.uk/TransQuest/architectures/sentence_level_architectures.html).


### Student Model, deepQuest-py BiRNN - Training and Evaluation

- **Download the data**, run the script `data_download.sh`. This will download the MLQE and Wikipedia data into the directory `datasets/`. To download a particular dataset, please use the links provided below in the `Datasets` section. 
- **Update the config file**  - `deepquestpy/config/birnn_sent.jsonnet` as follows:
	- Set the `data_path` parameter to the path of the language pair directory on which the model needs to be trained. For example, to train the model on the MLQE language pair, `ro-en`, set `data_path` to the previously downloaded`datasets/ro_en_mlqe` directory.
	- **Set the knowledge distillation flags** - `kd_without_gold_data` to `true` and `kd_with_gold_data` to `false` and parameter `alpha` to `0.0`
- **Train the model** - Execute the script `examples/birnn/train.sh`
- **Evaluate the model** - Execute the evaluate script - `examples/birnn/evaluate.sh` 

**NOTE** : With default parameter settings the BiRNN student model is ready to be trained on MLQE `ro-en` language pair. First, please download the data by running `data_download.sh` script and then execute the `train.sh` and `evaluate.sh` scripts as mentioned above.

### Datasets
- MLQE
  - [Et-En](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/et_en_mlqe.tar.gz)
   - [Ro-En](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/ro_en_mlqe.tar.gz)
   - [Si-En](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/si_en_mlqe.tar.gz)
   - [Ne-En](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/ne_en_mlqe.tar.gz)
   - [En-Zh](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/en_zh_mlqe.tar.gz)

	
- Wikipedia
	- [Et-En](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/et_en_25k_wiki.tar.gz)
	- [Ro-En](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/ro_en_100k_wiki.tar.gz)
	- [Si-En](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/si_en_100k_wiki.tar.gz)
	- [Ne-EN](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/ne_en_100k_wiki.tar.gz)
	- [En-Zh](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/en_zh_100k_wiki.tar.gz)

### Trained Student Models
- **Download the data**, as mentioned above by executing the script `data_download.sh`. To download a particular dataset, please use the links provided above in the `Datasets` section.  
- **Download the trained student models** by executing the script - `model_download.sh`. This will create a directory called `trained_models` in the `deepQuest-py` directory and will download all the trained models in the directory. The links to download the individual trained model for a particular language pair are also provided below.

To evaluate a trained model on the test data for a particular language pair:
- Update the `eval_model` path in the evaluate script `examples/birnn/evaluate.sh` to the path to the trained model `tar.gz`file and execute the script. For example, to evalute on the MLQE `ro-en` downloaded trained model update as `eval_model=trained_models/birnn_mlqe_ro_en.tar.gz`  and execute the script.

Links to trained BiRNN Student models:
- MLQE
	- [Et-En](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_mlqe_et_en.tar.gz)
	- [Ro-En](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_mlqe_ro_en.tar.gz)
	- [Si-En](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_mlqe_si_en.tar.gz)
	- [Ne-EN](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_mlqe_ne_en.tar.gz)
	- [En-Zh](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_mlqe_en_zh.tar.gz)
	
- Wikipedia
	- [Et-En](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_wiki25k_et_en.tar.gz)
	- [Ro-En](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_wiki100k_ro_en.tar.gz)
	- [Si-En](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_wiki100k_si_en.tar.gz)
	- [Ne-En](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_wiki100k_ne_en.tar.gz)
	- [En-Zh](https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_wiki100k_en_zh.tar.gz)
