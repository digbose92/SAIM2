import os 
import pandas as pd 
import numpy
from random import sample 
from collections import Counter

train_split=0.8
source_folder="/data/Multi_Domain_Data/parsed_csv_data/kitchen_&_housewares/"
domain_name=source_folder.split("/")[-2]
dest_file=domain_name+"_review_splits_combined.csv"
pos_review_file=os.path.join(source_folder,'positive_review.csv')
neg_review_file=os.path.join(source_folder,'negative_review.csv')

pos_review_data=pd.read_csv(pos_review_file)
neg_review_data=pd.read_csv(neg_review_file)


#0 for positive and 1 for negative

review_data=pd.concat([pos_review_data,neg_review_data])
review_scores=[0]*(pos_review_data.shape[0])+[1]*(neg_review_data.shape[0])
review_data['Label']=review_scores
#print(review_data['Rating'].iloc[0])
train_id_list=sample(range(review_data.shape[0]),int(train_split*review_data.shape[0]))
val_id_list=[i for i in range(review_data.shape[0]) if i not in train_id_list]
id_split=['no split']*review_data.shape[0]

for id in train_id_list:
    id_split[id]='train'
for id in val_id_list:
    id_split[id]='val'

review_data['split']=id_split
review_data.to_csv(os.path.join(source_folder,dest_file))
#print(len(train_id_list),len(val_id_list))
#print(train_id_list)