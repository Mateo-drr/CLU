# -*- coding: utf-8 -*-
"""
Created on Fri May 19 20:12:29 2023

@author: Mateo-drr
"""


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from tqdm import tqdm
import wandb
import numpy as np
import copy
import random
import torch.nn as nn
import pickle

torch.backends.cudnn.benchmark = True 
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
#https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json
path = 'D:/MachineLearning/datasets/SmartME/personachat/personality.csv'
spath = 'D:/Universidades/Trento/2S/NLP/epochs/'
ds = 'D:/MachineLearning/datasets/SmartME/'

device = "cuda" if torch.cuda.is_available() else "cpu"
init_lr = 0.001
batch_size = 4
msg_num = 6
n_epochs = 5
clipping_value=1
save_freq = 1
wb = True

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq").to(device)

def generate(instruction, knowledge, dialog):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

# Instruction for a chitchat task
instruction = f'Instruction: given a dialog context, you need to response empathically.'
# Leave the knowldge empty
knowledge = ''
dialog = [
    'Does money buy happiness?',
    'It is a question. Money buys you a lot of things, but not enough to buy happiness.',
    'What is the best way to buy happiness ?'
]
response = generate(instruction, knowledge, dialog)
print(response)



class CustomDataset(Dataset):

    def __init__(self, data):
    #WHAT DO WE PUT HERE?
        self.data = copy.deepcopy(data)
        
    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)

    def __getitem__(self, idx):
    #TAKE ONE ITEM FROM THE DATASET
        tkdata = tokenizer.encode(self.data[idx], truncation=True, max_length=512, padding='max_length',  return_tensors='pt')
        return tkdata
    

with open(ds+'dialog.p', "rb") as file:
    # Load the object from the file
    sepit = pickle.load(file)

#8939 items -> 20% = 1788 -> /2 = 894    
#train_ds, val_ds, test_ds = sepit[:7151], sepit[7151:8045], sepit[8045:]
train_ds, val_ds, test_ds = sepit[:8137] + sepit[8749:], sepit[8137:8749], sepit[8137:8749]
train_ds, val_ds, test_ds = CustomDataset(train_ds),CustomDataset(val_ds),CustomDataset(test_ds)
train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True)
val_dl = DataLoader(val_ds,batch_size=batch_size,shuffle=False)
test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False)
    
    
#model = GPT2LMHeadModel.from_pretrained("gpt2")


#Modify dropout
#model.transformer.drop = nn.Dropout(p=0.5, inplace=False)



optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)  

for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    val_loss= 0.0
    print(epoch)
    
    model.train()
    i=0
    bcounter = 0
    for bi, data in tqdm(enumerate(train_dl), total=int(len(train_ds)/train_dl.batch_size)):
    
        data = data.to(device)
        data = data.squeeze(1)
        outputs = model(data)
        loss = outputs[0]
        
        if wb:
            wandb.log({'tloss': loss})
        loss.backward()        
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
        optimizer.step()
        train_loss += loss.item()*batch_size
        optimizer.zero_grad()
        
        i+=1
        #'''
        # if i == 100:
        #break
    
    train_loss = train_loss/len(train_dl)
    print('E: {} T Loss: {:.3f}'.format(epoch, train_loss) + " %" + "{:.3}".format(np.exp(-abs(train_loss))*100))
    if epoch%save_freq == 0:
        try:
            torch.save(model.state_dict(), spath + 'epoch{0:05d}.pth'.format(epoch))
        except Exception as e:
            print("An error occurred:", e)
            
        if wb:
            wandb.save(path + 'wandb/wandb{0:05d}.pth'.format(epoch))


    model.eval()
    #compress.eval()
    temp =0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(val_dl), total=int(len(val_ds)/val_dl.batch_size)):
        #for data in val_dl:
            #'''
            data = data.to(device)
            loss = model(data,labels=data)
            loss = loss[0]
            
            val_loss += loss.item()*batch_size
            #'''
            
        val_loss = val_loss/len(val_dl)
        print('E: {} V Loss: {:.3f}'.format(epoch, val_loss) + " %" + "{:.3}".format(np.exp(-abs(val_loss))*100))
        
        
        #Sample generation
        random_index = random.randint(8137, 8748)
        data = tokenizer.encode(sepit[random_index].split('\t')[0] + tokenizer.eos_token, return_tensors='pt')
        data = data.to(device)
        chat_history_ids = model.generate(data, max_length=100, pad_token_id=tokenizer.eos_token_id)
        print(sepit[random_index].split('\t')[0])
        print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, data.shape[-1]:][0], skip_special_tokens=False)))
        
        if wb:
            wandb.log({'Train loss': train_loss, 'Validation Loss': val_loss})