#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import argparse

import torch
import pandas as pd

from transquest.algo.transformers.evaluation import pearson_corr, spearman_corr, rmse

from sklearn.metrics import mean_absolute_error
from transquest.algo.transformers.run_model import QuestModel


# In[ ]:


src_file = "/experiments/agajbhiye/deepQuestPY/deepQuest-py/datasets/small_et_en_mlqe/train/train.src" 
mt_file = "/experiments/agajbhiye/deepQuestPY/deepQuest-py/datasets/small_et_en_mlqe/train/train.mt"

bestPreTrainedModelDir = "/experiments/agajbhiye/knowDistill/preTtransQuest/bestPreTrainedModel"
transformer_config = "/experiments/agajbhiye/knowDistill/preTtransQuest/bestPreTrainedModel/config.json"


# In[ ]:


def main():
    
    src_list = []
    mt_list = []
    
    with open (src_file) as f:
        for line in f:
            src_list.append(line.strip())
            
    with open (mt_file) as f:
        for line in f:
            mt_list.append(line.strip())
            
    input_data = list(map(list, zip(src_list, mt_list)))
                
    model = QuestModel("xlmroberta", bestPreTrainedModelDir, num_labels=1, 
                       use_cuda=torch.cuda.is_available(), args=transformer_config)

    predictions, _ = model.predict(input_data)
    
    print ("predictions")
    print (type(predictions))
    
    with open ("/experiments/agajbhiye/deepQuestPY/deepQuest-py/datasets/small_et_en_mlqe/train/train.tpred", "w") as f:
        for pred in predictions:
            f.write("%f\n" % (pred))
    

if __name__ == '__main__':
    main()


