import pandas as pd 
import numpy as np 
import os 
from collections import Counter 

filename="/data/Multi_Domain_Data/parsed_csv_data/kitchen_&_housewares/negative_review.csv"
csv_data=pd.read_csv(filename)

text_tokens_length=[]
for i in np.arange(csv_data.shape[0]):
    text_sample=csv_data['Text'].iloc[i]
    text_sample_list=text_sample.split(" ")
    text_len=len(text_sample_list)
    text_tokens_length.append(text_len)

text_len_greater_than_512=[t for t in text_tokens_length if t > 512]
print((text_len_greater_than_512))