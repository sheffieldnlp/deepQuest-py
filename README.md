# deepQuest-py
deepQuest-py is a framework for training and evaluation of models for Quality Estimation of Machine Translation. 
This is a new version of [deepQuest](https://github.com/sheffieldnlp/deepQuest) - the first framework for neural Quality Estimation. 

deepQuest-py provides:

- **high performing** sentence-level and word-level models based on finetuning pre-trained Transformers;
- **light-weight and efficient** sentence-level models implemented via **knowledge distillation**.

deepQuest-py includes implementations of several approaches for Quality Estimation proposed in recent research:

- [Knowledge Distillation for Quality Estimation (Gajbhiye et al., 2021)](https://github.com/sheffieldnlp/deepQuest-py/tree/main/examples/knowledge_distillation)
- [TransQuest at WMT2020: Sentence-Level Direct Assessment (Ranasinghe et al., 2020)](https://github.com/sheffieldnlp/deepQuest-py/tree/main/examples/monotransquest)
- [Two-Phase Cross-Lingual Language Model Fine-Tuning for Machine Translation Quality Estimation (Lee, 2020)](https://github.com/sheffieldnlp/deepQuest-py/tree/main/examples/beringlab)
- [deepQuest: A Framework for Neural-based Quality Estimation (Ive et al., 2018)](https://github.com/sheffieldnlp/deepQuest-py/tree/main/examples/birnn)

See our [examples](https://github.com/sheffieldnlp/deepQuest-py/tree/main/examples) for instructions on how to train and test specific models.

## Online Demo
Check out our [web tool](https://dq.fredblain.org/) to try out most of our trained models on your own data!

## Installation
deepQuest-py requires Python 3.6 or later. 

```
git clone https://github.com/sheffieldnlp/deepQuest-py.git
cd deepQuest-py
pip install -e .
```
## Licence
deepQuest-py is licenced under a CC BY-NC-SA licence.

## Citation
If you use deepQuest-py in your research, please cite our [EMNLP 2021 Demo paper](https://aclanthology.org/2021.emnlp-demo.42.pdf):

```
@inproceedings{alva-manchego-etal-2021-deepquest,
    title = "deep{Q}uest-py: {L}arge and Distilled Models for Quality Estimation",
    author = "Alva-Manchego, Fernando  and
      Obamuyide, Abiola  and
      Gajbhiye, Amit  and
      Blain, Fr{\'e}d{\'e}ric  and
      Fomicheva, Marina  and
      Specia, Lucia",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-demo.42",
    pages = "382--389",
}
```
