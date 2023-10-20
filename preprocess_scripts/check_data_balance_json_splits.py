import json 
from random import shuffle 
from collections import Counter

json_file="/scratch1/dbose/Multi_Domain_data_gcp/redownloaded_data/parsed_data/electronics_review.json"
with open(json_file) as f:
    data = json.load(f)

split_train=['train']*1600
split_val=['val']*400
split_tot=split_train+split_val
shuffle(split_tot)

train_indices=[i for i in range(len(split_tot)) if split_tot[i]=='train']
val_indices=[i for i in range(len(split_tot)) if split_tot[i]=='val']    

train_data=[data[i] for i in train_indices]
val_data=[data[i] for i in val_indices]

#distribution of labels in train and val
train_labels=[i['Label'][0] for i in train_data]
val_labels=[i['Label'][0] for i in val_data]

print(Counter(train_labels))
print(Counter(val_labels))