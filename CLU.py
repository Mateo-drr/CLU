# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:45:25 2023

@author: Mateo-drr
"""

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer#, AutoConfig
import torch
from tqdm import tqdm
import wandb
import numpy as np
import copy
import random
import pickle
import gc
from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True 
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
path = 'D:/MachineLearning/datasets/SmartME/personachat/personality.csv'
spath = 'D:/Universidades/Trento/2S/NLP/epochs/'
ds = 'D:/MachineLearning/datasets/SmartME/'

device = "cuda" if torch.cuda.is_available() else "cpu"
init_lr = 1e-7
batch_size =6
msg_num = 6
n_epochs = 50
clipping_value=1
save_freq = 1
wb = True
tklen =500

#SMALL
tokenizer = AutoTokenizer.from_pretrained("emre/spanish-dialoGPT")
model = AutoModelForCausalLM.from_pretrained("emre/spanish-dialoGPT")
#Medium
model = AutoModelForCausalLM.from_pretrained('ITG/DialoGPT-medium-spanish-chitchat')
tokenizer = AutoTokenizer.from_pretrained('ITG/DialoGPT-medium-spanish-chitchat')

tokenizer.pad_token=tokenizer.eos_token #id is 50256
model.to(device)

class CustomDataset(Dataset):

    def __init__(self, data):
        self.data = copy.deepcopy(data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        split = np.random.randint(3) #7 split gives an extra emptry line so from 0 to 6
        if split%2 !=0: #so that it always starts with the user message
            split = 0
            
        lines = self.data[idx].split('<|endoftext|>')[split:]
        if split==0:
            lines[0] = " " + lines[0]
        lines = [item + '<|endoftext|>' for item in lines]
        lines[-1] = lines[-1].split('<|endoftext|>')[0] 
        
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
            tkdata = torch.cat([padding, tkdata]).to(torch.int64) #model requieres int 64
            attmsk = attmsk.to(torch.int64)
        except Exception as e:
            print(e)
            print(padding, tkdata, lines)
            return 0
        
        if len(tkdata) > tklen:
            tkdata = tkdata[-tklen:]
            #attmsk = attmsk[-tklen:]
            #actatt = actatt[-tklen:] 
        
        return tkdata, 0,0#attmsk, actatt
    
def usemodel():
    ldr = 'D:/Universidades/Trento/2S/NLP/epoch00025M.pth'
    model.load_state_dict(torch.load(ldr, map_location=torch.device(device)))
    chat_history_ids=torch.tensor([])
    for step in range(10):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        inp = input("User: ")+ " "+tokenizer.eos_token+' '#+ ' \t '
        print(inp)
        new_user_input_ids = tokenizer.encode(inp, return_tensors='pt').to(device)
    
        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
        print(bot_input_ids.shape, chat_history_ids.shape, new_user_input_ids.shape)
        if bot_input_ids.size(1) > 900:
            bot_input_ids = bot_input_ids[:,-1000:]
        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids,
                                     max_length=1000,
                                     num_beams = 10,
                                     temperature= 0.2,
                                     no_repeat_ngram_size=3,
                                     num_return_sequences=1,
                                     pad_token_id=tokenizer.eos_token_id
                                     )
    
        # pretty print last ouput tokens from bot
        print("CLU: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=False)))

#DATA STATISTICS
####################
# import pandas as pd
# from collections import Counter
# import matplotlib.pyplot as plt
# import statistics
# import os

# trn_df = pd.read_csv(ds+'vl_dataf2.csv')
# df=trn_df

# def get_counter_and_lens(data, tokenizer):
#     flatten = lambda l: [item for sublist in l for item in sublist]
#     toks = [tokenizer.tokenize(x) for x in data]

#     return list(map(len, toks)), Counter(flatten(toks)), Counter(' '.join(data).split())

# lens, tok_cnt, word_cnt = get_counter_and_lens(trn_df[df.columns].apply(lambda x: ' '.join(x.astype(str)), axis = 1), tokenizer)

# def plot_counts(counts, top_k = 30):
#     labels, values = zip(*counts.most_common()[:top_k])

#     indexes = np.arange(len(labels))
#     width = 1
#     plt.figure(num=None, figsize=(22, 4), dpi=60, facecolor='w', edgecolor='k')
#     plt.bar(indexes, values, width)
#     plt.xticks(indexes + width * 0.5, labels)
#     plt.show()

# plot_counts(tok_cnt, top_k = 30)
# plot_counts(word_cnt, top_k = 30)

# def plot_hist(lens, n_bins = 50):
#     n, bins, patches = plt.hist(lens, n_bins, facecolor='blue', alpha=0.9)
#     plt.show()

# print(f'Mean: {np.mean(lens)}, Median: {np.median(lens)}, Standard Deviation: {statistics.stdev(lens)}, 90th Percentile: {np.percentile(lens, 100)}')
# plot_hist(lens)
##################
#CODE USED TO VERIFY DATASET FROMATTING WAS THE SAME AS THE TUTORIALS
# def construct_conv(row, tokenizer, eos = True):
#     # from: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
#     flatten = lambda l: [item for sublist in l for item in sublist]
#     conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
#     conv = flatten(conv)
#     return conv

# class ConversationDataset(Dataset):
#     def __init__(self, tokenizer: tokenizer, df, block_size=512):

#         block_size = block_size - (1024 - tokenizer.max_len_single_sentence)

#         # directory = args.cache_dir
#         # cached_features_file = os.path.join(
#         #     directory, args.model_type + "_cached_lm_" + str(block_size)
#         # )

#         # if os.path.exists(cached_features_file) and not args.overwrite_cache:
#         #     with open(cached_features_file, "rb") as handle:
#         #         self.examples = pickle.load(handle)
#         if True:

#             self.examples = []
#             for _, row in df.iterrows():
#                 conv = construct_conv(row, tokenizer)
#                 if len(conv) > block_size: continue
#                 self.examples.append(conv)

#             # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
#             # If your dataset is small, first you should loook for a bigger one :-) and second you
#             # can change this behavior by adding (model specific) padding.

            
#             # with open(cached_features_file, "wb") as handle:
#             #     pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, item):
#         return torch.tensor(self.examples[item], dtype=torch.long)
    
# ds = ConversationDataset(tokenizer, df)
# print(ds[0])
    
def main():
#if True:    
    with open(ds+'dialog.p', "rb") as file:
        # Load the dataset
        sepit = pickle.load(file)
    
    #Remove windows that have more than 500 tokens
    clean =[]
    maxl=[]
    m=0
    for i,c in enumerate(sepit):
        a = len(tokenizer.encode(c))
        maxl.append(a)
        if a>m:
            print(a, i)
            m=a
        if a <=tklen:
            clean.append(c)
    sepit=copy.deepcopy(clean)
    maxl=0
    
    if wb:
        wandb.init(name='CLU-S',project="SmartMe", resume='allow')
        config = {
            "learning_rate": init_lr,
            "batch_size": batch_size,
            
            "num_epochs": n_epochs,
        }
        wandb.config.update(config)
    
    #Make the datasets and dataloaders
    sp = int(len(clean)*0.2)
    train_ds, val_ds, test_ds = sepit[:-sp], sepit[-(sp-2):-int(sp*0.25)], sepit[-int(sp*0.25):]
    train_ds, val_ds, test_ds = CustomDataset(train_ds),CustomDataset(val_ds),CustomDataset(test_ds)
    train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,pin_memory=True)
    val_dl = DataLoader(val_ds,batch_size=batch_size,shuffle=False)
    test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False)
    
    #Optimizer and scaler for fp16
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)  
    scaler = GradScaler()
    clean=0
    gc.collect()
    
    #Load model to continue training
    ldr = 'D:/Universidades/Trento/2S/NLP/epoch00036S.pth'
    model.load_state_dict(torch.load(ldr, map_location=torch.device(device)))
    
    #Train and Validation loops
    for epoch in range(26, n_epochs+1):
        train_loss = 0.0
        val_loss= 0.0
        print(epoch)
        
        #TRAIN
        model.train()
        i=0
        for bi, data in tqdm(enumerate(train_dl), total=int(len(train_ds)/train_dl.batch_size)):
            data,_,_ = data
            data = data.to(device)
            with autocast():
                loss = model(data,labels=data)[0]
            train_loss += loss.item()*data.shape[0]
            
            if wb:
                wandb.log({'tloss': loss})
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        train_loss = train_loss/len(train_dl)
        print('E: {} T Loss: {:.3f}'.format(epoch, train_loss) + " %" + "{:.3}".format(np.exp(-abs(train_loss))*100))
        
        #Save weights
        if epoch%save_freq == 0:
            try:
                torch.save(model.state_dict(), spath + 'epoch{0:05d}.pth'.format(epoch))
            except Exception as e:
                print("An error occurred:", e)
                
            if wb:
                wandb.save(path + 'wandb/wandb{0:05d}.pth'.format(epoch))
                
        #VALIDATION
        model.eval()
        with torch.no_grad():
            for bi, data in tqdm(enumerate(val_dl), total=int(len(val_ds)/val_dl.batch_size)):
                data,_,_ = data
                data = data.to(device)
                loss = model(data,labels=data)[0]
                val_loss += loss.item()*data.shape[0]
                
            val_loss = val_loss/len(val_dl)
            print('E: {} V Loss: {:.3f}'.format(epoch, val_loss) + " %" + "{:.3}".format(np.exp(-abs(val_loss))*100))
            
            
            #Sample generation
            random_index = random.randint(len(train_ds),len(train_ds)+len(val_ds)+len(test_ds))
            data = tokenizer.encode(sepit[random_index].rsplit("<|endoftext|>", 2)[0] + '<|endoftext|> ', return_tensors='pt')
            data = data.to(device)
            chat_history_ids = model.generate(data,
                                         max_length=tklen,
                                         num_beams = 10,
                                         temperature= 0.5,
                                         no_repeat_ngram_size=2,
                                         num_return_sequences=1,
                                         pad_token_id=tokenizer.eos_token_id
                                         )
            print('\n',sepit[random_index])
            print('',"CLU: {}".format(tokenizer.decode(chat_history_ids[:, data.shape[-1]:][0], skip_special_tokens=False)),'\n')
            
            if wb:
                wandb.log({'Train loss': train_loss, 'Validation Loss': val_loss})
            chat_history_ids=0

        gc.collect()
    
    #TEST LOOP
    model.eval()
    tst_loss=0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(test_dl), total=int(len(test_ds)/test_dl.batch_size)):
            data,_,_ = data
            data = data.to(device)
            loss = model(data,labels=data)[0]
            tst_loss += loss.item()*data.shape[0]
        tst_loss = tst_loss/len(test_dl)
        print('Test Loss:', tst_loss)

if __name__ == "__main__":
    main()
    
    
    
    