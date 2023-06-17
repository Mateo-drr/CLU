import torch
import torch.nn as nn
import torch.optim as optim
import os
from transformers import Trainer, TrainingArguments
from collections import defaultdict
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ds = 'D:/MachineLearning/datasets/SmartME/'
TRAIN = True

model = GPT2LMHeadModel.from_pretrained(ds+'trained').to('cuda')

tokenizer = GPT2Tokenizer.from_pretrained(ds+'tokenizer')
#We still need to tell the tokenizer these simbols:
tokenizer.add_special_tokens({
    'eos_token':'</s>',
    'bos_token':'<s>',
    'unk_token':'<unk>',
    'pad_token':'<pad>',
    'mask_token':'<mask>'
    })


while True:
    inp = input(">>> ")
    input_ids = tokenizer.encode(inp, return_tensors='pt').to('cuda')
    beam_output = model.generate(input_ids,
                                 max_length=128,
                                 num_beams = 10,
                                 temperature= 0.1,
                                 no_repeat_ngram_size=1,
                                 num_return_sequences=1
                                 )
    for beam in beam_output:
        out = tokenizer.decode(beam)
        fout = out.replace("<N>", "\n")
        
        print(str(fout))
    
    









































