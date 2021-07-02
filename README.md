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

### deepQuest-py BiRNN Student Model - Training and Evaluation (Distilled MLQE and Wikipedia Data)

- **Download the data** - MLQE or Wikipedia from the links provided below.
- **Update the config file** - Set the flag `kd_without_gold_data` in the config file, `deepQuest-py/deepquestpy/config/birnn_sent.jsonnet` to `true`
- **Train the model** - Execute the script `deepQuest-py/examples/birnn/train.sh` with the updated config file. 
- **Evaluate the model** by executing the evaluate script - `deepQuest-py/examples/birnn/evaluate.sh`

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
```
