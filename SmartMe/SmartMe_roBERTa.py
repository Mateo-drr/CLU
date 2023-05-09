# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:45:25 2023

@author: Mateo-drr
"""

from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd


path = 'D:/MachineLearning/datasets/SmartME/personachat/personality.csv'
ds = pd.read_csv(path)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


class CustomDataset(Dataset):

    def __init__(self, data):
    #WHAT DO WE PUT HERE?
        self.data = data

    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)

    def __getitem__(self, idx):
    #TAKE ONE ITEM FROM THE DATASET
        dic = self.data[idx].to_dict()
        persona = dic['Persona']
        chat = 
        

        return {'img':img[0],
              'label':label,
              'blabel':blabel[0],
              'bbox':bbox[0]}