# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:45:25 2023

@author: Mateo-drr
"""

from torch.utils.data import Dataset, DataLoader
#from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
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
batch_size = 16
msg_num = 6
n_epochs = 5
clipping_value=1
save_freq = 1
wb = False

if wb:
    wandb.init(name='GPTNeo',project="SmartMe")
    config = {
        "learning_rate": init_lr,
        "batch_size": batch_size,
        "num_epochs": n_epochs,
    }
    wandb.config.update(config)

#This dataset has a persona and a chat between two people
#The persona describes the person that sends the seconde message
#From there it simply alternates
#All chats have 11 lines
#smallest conversation is #2 6 messages (including persona in the first message)
#largest conversation is #688 25 messages ''
#do + 1 now since im using the persona description as its own line

#Data preprocessing:
    #Im goign to structure the data like this:
        #<start> <persona> Persona info <endofpersona> l1 <bot> l2 <endofbot> l3 <bot> l4 <endofbot> ... <|endoftext|>
'''
ds = pd.read_csv(path)
dic = ds.transpose().to_dict()

items = []
for j in range(len(dic)):
    sample = dic[j]
    temp = ' <|start|>'
    temp = temp + ' <|persona|>' + sample['Persona'] + ' <|endofbot|> '
    for i,line in enumerate(sample['chat'].split('\n')[:-1]):
        if i%2 != 0: #bot lines
            temp = temp + ' <|bot|>: ' + line + ' <|endoftext|> <|endofbot|> '
        else: 
            temp = temp + '<|start|> <|user|> ' + line
            
    #temp = temp + ' <|end|>'
    items.append(temp)

sepit = []
for item in items:
    sepit.append(item.split('<|endofbot|>'))

for i,_ in enumerate(sepit):
    line = sepit[i]
    last_two = line[-2] + line[-1].replace(" ", "")
    sepit[i] = line[:-2] + [last_two]
    
    line = sepit[i]
    for j in range(len(line)):
        line[j] = line[j][1:]
        
    sepit[i] = line
'''

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#tokenizer.add_special_tokens({'pad_token': '<|pad|>','bos_token': '<|start|>','eos_token': '<|endoftext|>'})
#tokenizer.add_tokens(['<|bot|>:', '<|endofbot|>', '<|persona|>', '<|endofpersona|>'])
#tokenizer.add_tokens(['<|bot|>:', '<|persona|>', '<|user|>'])

#encdi = tokenizer(items, truncation=True, padding=True)


class CustomDataset(Dataset):

    def __init__(self, data):
    #WHAT DO WE PUT HERE?
        self.data = copy.deepcopy(data)
        
    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)

    def __getitem__(self, idx):
    #TAKE ONE ITEM FROM THE DATASET
        if len(self.data[idx]) > 7:
            fdata = []
            fdata.append(self.data[idx][0]) #get the persona
            random_index = random.randint(1, len(self.data[idx]) - 7)
            for line in self.data[idx][random_index:random_index+6]:
                fdata.append(line)
        else:
            fdata = self.data[idx]
        
        if len(fdata)> 6:
            merged = fdata[0] + fdata[1].split('<|start|> ')[1] #rmmove start token from merged persona and message
            fdata[0:2] = [merged]# + fdata[2:]
        
        #else:
        #    print('hey')

        tkdata = []
        for line in fdata:
            tkdata.append(tokenizer(line, truncation=True,max_length=192, padding='max_length',return_tensors='pt'))
        
            
        return {'tkdata':tkdata}
    

with open(ds+'dialog.p', "rb") as file:
    # Load the object from the file
    sepit = pickle.load(file)

#8939 items -> 20% = 1788 -> /2 = 894    
#train_ds, val_ds, test_ds = sepit[:7151], sepit[7151:8045], sepit[8045:]
train_ds, val_ds, test_ds = sepit[:8137] + sepit[8749:], sepit[8137:8749], sepit[8137:8749]
train_ds, val_ds, test_ds = CustomDataset(train_ds),CustomDataset(val_ds),CustomDataset(test_ds)
train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=False)
val_dl = DataLoader(val_ds,batch_size=batch_size,shuffle=False)
test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False)
    
    
#model = GPT2LMHeadModel.from_pretrained("gpt2")
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125m')
model.resize_token_embeddings(len(tokenizer))
model.to(device)

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
    
        #'''
        for chat in data['tkdata']:
            inid,attm = chat['input_ids'], chat['attention_mask']
            inid = inid.to(device)
            attm = attm.to(device)
            loss = model(inid, attention_mask=attm,labels=inid).loss / 6
            if wb:
                wandb.log({'tloss': loss*6})
            loss.backward()        
         
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
        optimizer.step()
        train_loss += loss.item()*batch_size
        optimizer.zero_grad()
        
        i+=1
        #'''
        # if i == 100:
        #     break
    
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
            for chat in data['tkdata']:
                inid,attm = chat['input_ids'], chat['attention_mask']
                inid = inid.to(device)
                attm = attm.to(device)
                loss = model(inid, attention_mask=attm,labels=inid).loss/6
                temp += loss
                
            val_loss += loss.item()*batch_size
            #'''
            
        val_loss = val_loss/len(val_dl)
        print('E: {} V Loss: {:.3f}'.format(epoch, val_loss) + " %" + "{:.3}".format(np.exp(-abs(val_loss))*100))
        
        
        #Sample generation
        random_index = random.randint(7151, 8044)
        test = copy.deepcopy(sepit[random_index])
        merged = test[0] + test[1].split('<|start|> ')[1] #rmmove start token from merged persona and message
        test[0:2] = [merged]
        for line in test:
            
            inp = line.split('<|bot|>:')[0] + '<|bot|>:'
            inp = tokenizer.encode(inp, return_tensors='pt').to('cuda')
            beam_output = model.generate(inp, 
                                         max_new_tokens=30,
                                         # num_beams = 10,
                                         # temperature= 0.1,
                                         # no_repeat_ngram_size=1,
                                         # num_return_sequences=1,
                                         pad_token_id=tokenizer.pad_token_id)
            # output = tokenizer.decode(output[0])
            # print(output)
            for beam in beam_output:
                out = tokenizer.decode(beam)
                fout = out.replace("<N>", "\n")
                
                print(str(fout))
            print('\n')
        
        if wb:
            wandb.log({'Train loss': train_loss, 'Validation Loss': val_loss})

    