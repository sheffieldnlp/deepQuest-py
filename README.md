# deepQuest-py. 

- Uses the newest versions of the HuggingFace Transformers and Datasets libraries.
- Inclues models for Word-Level Quality Estimation.

## Instalation from Source

```
git clone https://github.com/sheffieldnlp/deepQuest-py.git
cd deepQuest-py
pip install --editable ./
```
### deepQuest-py BiRNN Model - Training and Evaluation (Gold Data)

- **Download the MLQE data**, for the language pair the model needs to be trained. The links to the datasets are provided below. Copy the downloaded dataset directory in the `deepQuest-py/datasets/` directory.
- **Update the config file**  - `deepQuest-py/deepquestpy/config/birnn_sent.jsonnet` as follows:
	- Set the `data_path` parameter to the path to the downloaded dataset.
	- Set the knowledge distillation flags - `kd_without_gold_data` and `kd_with_gold_data` to `false` and parameter `alpha` to `0.0`
- **Train the model** - Update the `config_file` parameter of the train script`deepQuest-py/examples/birnn/train.sh` to the path to the updated `birnn_sent.jsonnet` config file and execute the script to train the model.
- **Evaluate the model** by executing the evaluate script - `deepQuest-py/examples/birnn/evaluate.sh` 

## Knowledge Distillation

### Teacher Model, MonoTransQuest - Training and Evaluation
- Knowledge distillation framework in the `deepQuest-py` can utilise predictions from any teacher model.
- We used the predictions from the state-of-the-art [MonoTransQuest ](https://aclanthology.org/2020.coling-main.445/) pre-trained models available [here](https://tharindu.co.uk/TransQuest/models/sentence_level_pretrained.html). We include these predictions in the data provided for training the student model in the `Datasets` section.    
- The MonoTransQuest model can also be trained from scratch as detailed [here](https://tharindu.co.uk/TransQuest/architectures/sentence_level_architectures.html).


### Student Model, deepQuest-py BiRNN - Training and Evaluation

- **Download the data** - MLQE or Wikipedia from the links provided below.
- **Update the config file** - Set the flag `kd_without_gold_data` in the config file, `deepQuest-py/deepquestpy/config/birnn_sent.jsonnet` to `true`
- **Train the model** - Execute the script `deepQuest-py/examples/birnn/train.sh` with the updated config file. 
- **Evaluate the model** by executing the evaluate script - `deepQuest-py/examples/birnn/evaluate.sh`

### Datasets
Following are the links to download MLQE and Wikipedia datasets for training the student BiRNN model.
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
Following are the trained BiRNN student models. The models are trained on MLQE and Wikipedia data.
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

To evaluate a trained model on the test data. Run the following command:

```python
python deepQuest-py/deepquestpy_cli/run_birnn.py --do_eval --eval_model "path to the saved model.tar.gz file" --eval_data_path "path to the test data directory" + "test"
