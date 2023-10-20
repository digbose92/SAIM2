import os 
import pandas as pd 
import numpy as np 
from transformers import BertTokenizer, BertModel, BertConfig
import torch

filename="/data/Multi_Domain_Data/parsed_csv_data/kitchen_&_housewares/negative_review.csv"
csv_data=pd.read_csv(filename)

##instantiating BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
model = BertModel.from_pretrained("bert-base-uncased", config=config)
model.eval()
max_length_token=512
input_ids=[]


for i in np.arange(csv_data.shape[0]):

    text_sample=csv_data['Text'].iloc[i]
    dictionary = tokenizer.encode_plus(
                        text_sample,                      
                        add_special_tokens = True,
                        max_length = max_length_token,
                        padding = 'max_length',
                        truncation=True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                   )
    with torch.no_grad():
        outputs = model(dictionary['input_ids'], attention_mask=dictionary['attention_mask'])
    
    hidden_states=outputs[2][1:] #hidden state from 1st layer onwards (from embeddings)
    
    #print(type(outputs))
    #input_ids.append(dictionary['input_ids'])
# asin_list=list(csv_data['Asin'])
# print(len(set(asin_list)))
#print(csv_data['Asin'].iloc[0])