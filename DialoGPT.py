# -*- coding: utf-8 -*-
"""
Created on Fri May 19 05:39:56 2023

@author: Mateo-drr
"""

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import pickle

# ds = 'D:/MachineLearning/datasets/SmartME/'
# context = 8

# with open(ds+'dialog.p', "rb") as file:
#     # Load the object from the file
#     data = pickle.load(file)




# #print(sequences)


# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# # Let's chat for 5 lines
# for step in range(5):
#     # encode the new user input, add the eos_token and return a tensor in Pytorch
#     new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt').to(device)

#     # append the new user input tokens to the chat history
#     bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

#     # generated a response while limiting the total chat history to 1000 tokens, 
#     chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

#     # pretty print last ouput tokens from bot
#     print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=False)))



# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:45:25 2023

@author: Mateo-drr
"""

from torch.utils.data import Dataset, DataLoader
#from transformers import GPT2LMHeadModel, GPT2Tokenizer
#from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
init_lr = 1e-7
batch_size =4
msg_num = 6
n_epochs = 25
clipping_value=1
save_freq = 1
wb = True
tklen =500

#tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

config = AutoConfig.from_pretrained('microsoft/DialoGPT-small', cache_dir='cached')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small', cache_dir='cached', padding_side='left')
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small',config=config)
tokenizer.pad_token=tokenizer.eos_token #id is 50256
tokenizer.add_tokens(['<|user|>:', '<|bot|>:'])
model.resize_token_embeddings(len(tokenizer))


model.to(device)

class CustomDataset(Dataset):

    def __init__(self, data):
    #WHAT DO WE PUT HERE?
        self.data = copy.deepcopy(data)
        
    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)

    def __getitem__(self, idx):
    #TAKE ONE ITEM FROM THE DATASET
        split = np.random.randint(3) #7 split gives an extra emptry line so from 0 to 6
        if split%2 !=0: #so that it always starts with the user message
            split = 0
            
        lines = self.data[idx].split('<|endoftext|>')[split:]
        lines = [item + '<|endoftext|>' for item in lines]
        lines[-1] = lines[-1].split('<|endoftext|>')[0] #remove the eos token from last line
        
        #attention masks
        attm=[]
        tkid=[]
        for i,line in enumerate(lines):
            idk = tokenizer.encode(line, return_tensors='pt') #tokenize
            if i%2==0: #user lines
                temp = torch.tensor([0] * len(idk[0]))
                attm.append(temp)
            else: #bot lines
                temp = torch.tensor([1] * len(idk[0]))
                attm.append(temp)
            tkid.append(idk[0])
        
        
        
        #padding
        attmsk = torch.cat(attm)
        #actual attention mask only ignoring padding:
        actatt = torch.tensor([1] *len(attmsk)) 
        
        padding = tklen - len(attmsk)
        padding = torch.tensor([0] * padding)
        attmsk = torch.cat([padding, attmsk])
        
        actatt = torch.cat([padding, actatt])
        
        tkdata = torch.cat(tkid)
        padding = tklen - len(tkdata)
        padding = torch.tensor([50256] * padding)
        try:
            tkdata = torch.cat([padding, tkdata]).to(torch.int64) #model requieres int 64 idk why
            attmsk = attmsk.to(torch.int64)
        except Exception as e:
            print(e)
            print(padding, tkdata, lines)
            return 0
        
        #tkdata = tokenizer.encode(data, truncation=True, max_length=tklen, padding='max_length',  return_tensors='pt')
        #print(len(tkdata),len(attmsk))
        if len(tkdata) > tklen:
            tkdata = tkdata[-tklen:]
            attmsk = attmsk[-tklen:]
            actatt = actatt[-tklen:]
        #print(len(tkdata),len(attmsk))    
        
        return tkdata, attmsk, actatt
    


    
def main():
#if True:    
    with open(ds+'dialog.p', "rb") as file:
        # Load the object from the file
        sepit = pickle.load(file)
    
    if wb:
        wandb.init(name='DGPT-v1d2+',project="SmartMe")
        config = {
            "learning_rate": init_lr,
            "batch_size": batch_size,
            
            "num_epochs": n_epochs,
        }
        wandb.config.update(config)
    
    m=0
    for i,item in enumerate(sepit):
        if len(item) > m:
            print(i)
            m = len(item)
            
    sepit = [string for string in sepit if len(string) <= 4450]
    
    # sepit = sepit[:19278] + sepit[19280:]
    # sepit = sepit[:11311] + sepit[11313:]
    # sepit = sepit[:6809] + sepit[6810:]
    # #second round
    # sepit = sepit[:11397] + sepit[11399:]
    # sepit = sepit[:9098] + sepit[9099:]
    #sepit = sepit[:17247] + sepit[17251:] #remove messages that are too long
    #x[4465] = '<|endoftext|>'.join(x[4465].split('<|endoftext|>')[6:]) #now its 463 tokens
    
    
    #19360 total -> ~20% = 3462 ~75% = 2596 & 866
    #train_ds, val_ds, test_ds = sepit[:7151], sepit[7151:8045], sepit[8045:]
    #train_ds, val_ds, test_ds = sepit[:8137] + sepit[8749:], sepit[8137:8749], sepit[8137:8749]
    train_ds, val_ds, test_ds = sepit[:-3464], sepit[-3462:], sepit[-3462:]
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
            data,tkids,attmsk = data
            data = data.to(device)
            attmsk = attmsk.to(device)
            tkids = tkids.to(device)
            outputs = model(data,labels=data,attention_mask=attmsk, token_type_ids=tkids)#.loss
            loss = outputs[0]
            train_loss += loss.item()*data.shape[0]
            
            if wb:
                wandb.log({'tloss': loss})
            loss.backward()        
            
            if i == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
                optimizer.step()
                
                optimizer.zero_grad()
                i=0
            else:
                i+=1
            #'''
            #if i == 1:
            #    break
        
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
                data,_,_ = data
                data = data.to(device)
                outputs = model(data,labels=data)#.loss
                loss = outputs[0]
                
                val_loss += loss.item()*data.shape[0]
                #'''
                
            val_loss = val_loss/len(val_dl)
            print('E: {} V Loss: {:.3f}'.format(epoch, val_loss) + " %" + "{:.3}".format(np.exp(-abs(val_loss))*100))
            
            
            #Sample generation
            random_index = random.randint(8137, 8748)
            data = tokenizer.encode(sepit[random_index].split('\t')[0] + '\t ', return_tensors='pt')
            data = data.to(device)
            chat_history_ids = model.generate(data, max_length=512, pad_token_id=tokenizer.eos_token_id)
            print(sepit[random_index].split('\t')[0])
            print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, data.shape[-1]:][0], skip_special_tokens=False)))
            
            if wb:
                wandb.log({'Train loss': train_loss, 'Validation Loss': val_loss})
            
            #break
    
        
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        inp = input("User: ")+ " "+tokenizer.eos_token + ' \t '
        print(inp)
        new_user_input_ids = tokenizer.encode(inp,
                                              return_tensors='pt').to(device)
    
        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    
        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
        # pretty print last ouput tokens from bot
        print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=False)))
        
        
if __name__ == "__main__":
    main()