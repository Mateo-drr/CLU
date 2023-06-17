
import os
from tokenizers import ByteLevelBPETokenizer
import copy
import numpy as np
import itertools
import pickle
import pandas as pd

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ds = 'D:/MachineLearning/datasets/SmartME/'
path = 'D:/MachineLearning/datasets/SmartME/WhatsappFlat/vocab.txt'
batch_size =8
TRAIN = False

if not TRAIN:
    

    # Define some training data
    def read_file(file):
        '''Reads Whatsapp text file into a list of strings'''
        x = open(file,'r', encoding = 'utf-8') #Opens the text file into variable x but the variable cannot be explored yet
        y = x.read() #By now it becomes a huge chunk of string that we need to separate line by line
        content = y.splitlines() #The splitline method converts the chunk of string into a list of strings
        return content
    
    #chat = read_file(ds+'gloria.txt')[1:]
    
    files =  os.listdir(ds+'WhatsappRaw/')
    text = []
    f = []
    sequences = []
    #from checking the chat data doenst have new line separators so no need to replace them
    
    for file in files:
        chat = read_file(ds+'WhatsappRaw/'+file)[1:]
        print(len(chat))
        
        for line in chat:
            #print(line)
            if line != "":
                if line[-1] != '>': #get rid of file messages
                    if (len(line) > 2):
                        if (line[0] == '0' or line[0] == '1' or line[0] == '2' or line[0] == '3') and line[2] == '/':
                            splits = line.split(':', maxsplit=2) #avoid messing messages that use :
    
                            if splits[1] == splits[-1]: #avoid new phone number messages
                                continue
                            text.append({'author':splits[1].split('-')[-1].strip(),
                                     'msg':splits[-1].strip()})
                            #print(text)
                        else:
                           text.append({'author':"",
                                        'msg':line})
                    else:
                        text.append({'author':"",
                                     'msg':line})
                    #print(line)

        #assign the last author as the author of next line messages
        for i in range(0,len(text)):
            if text[i]['author'] == '':
                text[i]['author'] = text[i-1]['author']
            if text[i]['author'] == '.':
                text[i]['author'] = 'Me'

        #Merge continuous messages from same author into one
        merged_data = []
        prev_author = None
        prev_msg = None
    
        for item in text:
            author = item['author']
            msg = item['msg']
            
            if author == prev_author:
                merged_data[-1]['msg'] += '\n ' + msg
            else:
                merged_data.append({'author': author, 'msg': msg})
            
            prev_author = author
            prev_msg = msg
        
        if merged_data[0]['author'] == 'Me':
            merged_data = merged_data[1:]
    
        #rechecking all messages start with the other sender
        
        sequences = []
        for i in range(0,len(merged_data) - 3,2):
            temp = merged_data[i:i+4]
            #temp = " ".join([f"{d['msg']} <|endoftext|>" for d in temp])
            #temp = " ".join([f"{'0.0 ' if i % 2 == 0 else '1.0 '}{d['msg']} <|endoftext|>" for i, d in enumerate(temp)])
            #temp = " ".join([f"{'<|user|>: ' if i % 2 == 0 else '<|bot|>: '}{d['msg']} <|endoftext|>" for i, d in enumerate(temp)])
            temp = " ".join([f"{'' if i % 2 == 0 else ''}{d['msg']} <|endoftext|>" for i, d in enumerate(temp)])
            temp = temp #+ '\n'
            #temp2 = temp.rsplit("<|endoftext|>", 2) 
            #temp = temp2[0] + '<|endoftext|> \t' + temp2[1] + temp2[2]
            #temp = temp2[0] + '<|endoftext|>' + temp2[1] + temp2[2]
            sequences.append(temp)
        
        # for i in range(0, len(merged_data), 8):
        #     if merged_data[i:i+8][0]['author'] == 'Me':
        #         print(merged_data[i:i+8], i)
        #         print(merged_data[i-1:i-1+8], i)
        #     sequences.append(merged_data[i:i+8])
        
        # if len(sequences[-1]) != 8:
        #     sequences = sequences[:-1]
        
        f.append(sequences)    
        text=[]
        sequences=[]
    
    text = list(itertools.chain(*f))
    

    with open(ds+'dialog6.p', 'wb') as file:
        pickle.dump(text, file)        
    
    
    columns=[
        'response', 'context', 'context/0', 'context/1',
        #'context/2', 'context/3', 'context/4', 'context/5',
        #'context/6', 'context/7', 'context/8', 'context/9'
    ]
    
    #sepit = [string for string in text if len(string) <= 4450]
    from transformers import AutoModelForCausalLM, AutoTokenizer#, AutoConfig
    clean =[]
    maxl=[]
    m=0
    tokenizer = AutoTokenizer.from_pretrained('ITG/DialoGPT-medium-spanish-chitchat')
    for i,c in enumerate(text):
        a = len(tokenizer.encode(c))
        maxl.append(a)
        if a>m:
            print(a, i)
            m=a
        if a <=500:
            clean.append(c)
    sepit=copy.deepcopy(clean)
    
    train_ds, val_ds, test_ds = sepit[:-3464], sepit[-3462:-866], sepit[-866:]
    
    split_text = [item.split('<|endoftext|>')[::-1][1:] for item in val_ds]
    ttext = list(zip(*split_text))
    
    df = pd.DataFrame(split_text, columns=columns)
    df.to_csv(ds+'vl_dataf2.csv', index = False)
    
    
    #write the messages in one single file with no authors    
    '''
    flat = ""
    #l = 0
    ll = []
    for msg in text:
        #if msg['author'] == 'Me':
            flat = flat + msg['msg'] + '\n'
            ll.append(len(msg['msg']))
            
    print(max(ll),min(ll), int(np.mean(ll)), np.median(ll), st.mode(ll), np.var(ll), st.skew(ll))
        
    with open(ds+'WhatsappFlat/vocab.txt', 'w', encoding='utf-8') as f:
        f.write(flat)
    '''
            
#     tokenizer = ByteLevelBPETokenizer()
#     tokenizer.train(files=[ds+'WhatsappFlat/vocab.txt'], vocab_size=52_000, min_frequency=2, special_tokens=[
#         "<s>",
#         "<pad>",
#         "</s>",
#         "<unk>",
#         "<mask>",
#         ])
    
#     tokenizer.save_model(ds+'tokenizer')
    
#     x = 0
#     for idk in text:
#         if idk['author'] == 'Me':
#              x +=1
#     print(x*100/len(text))
    
    
    
# else:   
##############################################################################
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # gpt2m = GPT2LMHeadModel.from_pretrained('gpt2')
    # #tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    # def loadData(path):
    #     dataset = []
    #     with open(path, 'r') as file:
    #         for line in file:
    #             dataset.append(line)
    #     return dataset
    
    # class CustomDataset(Dataset):
    
    #     def __init__(self, data, tokenizer):
    #         #copy the data
    #         self.data = copy.deepcopy(data)
    #         #load the given tokenizer
    #         self.tokenizer = tokenizer
    
    #     def __len__(self):
    #     #JUST THE LENGTH OF THE DATASET
    #         return len(self.data)
    
    
    #     def __getitem__(self, idx):
    #         #TAKE ONE ITEM FROM THE DATASET
    #         text = self.data[idx]
            
    #         encoding_text = self.tokenizer(
    #             text,
    #             max_length=1024,
    #             add_special_tokens=True,
    #             padding='max_length',
    #             return_attention_mask=True,
    #             return_token_type_ids=False,
    #             return_tensors='pt',
    #             truncation=True
    #         )
            
    #         decode =  self.tokenizer.convert_ids_to_tokens(encoding_text['input_ids'].flatten())
    #         #print(decode)
    #         return {
    #             'text': text,
    #             'decoded': decode,  
    #             'text_input_ids': encoding_text['input_ids'].flatten(),
    #             'text_attention_mask': encoding_text['attention_mask'].flatten(),
    #         }
        
    # #CREATE THE DATALOADER
    # def create_data_loader_CustomDataset(data, batch_size, eval=False):
    #     ds = CustomDataset(data=data)
    
    #     if not eval:
    #         return DataLoader(ds, batch_size=batch_size, shuffle=True), len(ds)
    
    #     else:
    #         return DataLoader(ds, batch_size=batch_size, shuffle=False), len(ds)
        
    # dataset = loadData(path)
    # train_dl, train_length = create_data_loader_CustomDataset(dataset, batch_size, eval=False)


##############################################################################
    '''    
    
#to use the gpt2 tokenizer it requeires already a vocab file, so we use the bpe tokenizer like we did above
    
inp = "que estas haciendo"

tokenizer = GPT2Tokenizer.from_pretrained(ds+'tokenizer')
#We still need to tell the tokenizer these simbols:
tokenizer.add_special_tokens({
    'eos_token':'</s>',
    'bos_token':'<s>',
    'unk_token':'<unk>',
    'pad_token':'<pad>',
    'mask_token':'<mask>'
    })

t = tokenizer.encode(inp)
print(t, tokenizer.decode(t))

#config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)


config = GPT2Config(
    vocab_size = tokenizer.vocab_size,
    bos_token = tokenizer.bos_token_id,
    eos_token = tokenizer.eos_token_id
    )


model = GPT2LMHeadModel(config)

#model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
#model.resize_token_embeddings(len(tokenizer))

dataset = load_dataset('text', data_files=[ds+'WhatsappFlat/vocab.txt'])

def encode(lines):
    return tokenizer(lines['text'], add_special_tokens=True, truncation=True, max_length=512)

dataset.set_transform(encode)
dataset = dataset['train']
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=ds+'trained',
    overwrite_output_dir=True,
    num_train_epochs=1,
    optim='adamw_torch',
    #evaluate_during_training=True,
    per_device_train_batch_size=5,
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    prediction_loss_only=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(ds+'trained')


'''




































