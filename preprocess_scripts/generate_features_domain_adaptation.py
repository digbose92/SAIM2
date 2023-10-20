import os 
import pandas as pd 
import pickle 
import numpy as np 
from random import sample 
def to_one_hot(a):
    b = np.zeros((len(a), 2))
    b[np.arange(len(a)), a] = 1
    return b


def extract_source_target_feature_set(source,target,base_folder,source_csv_file,target_csv_file,layer_name="Last_encoder_w_cls",split=0.8):
    
    source_csv_data=pd.read_csv(source_csv_file)
    target_csv_data=pd.read_csv(target_csv_file)

    source_label_data=[source_csv_data['Label'].iloc[i] for i in range((source_csv_data.shape[0]))]
    target_label_data=[target_csv_data['Label'].iloc[i] for i in range((target_csv_data.shape[0]))]
    #print(source_label_data)

    feature_folder_source=os.path.join(base_folder,source)
    source_file=os.path.join(feature_folder_source,source+"_review_splits_combined.pkl")
    target_file=os.path.join(feature_folder_source,target+"_review_splits_combined.pkl")

    with open(source_file,"rb") as f:
        source_embeddings=pickle.load(f)

    with open(target_file, "rb") as f:
        target_embeddings=pickle.load(f)
    
    #split 80-20 for the source and target 
    #print(source_embeddings.keys())    
    source_embed=source_embeddings[layer_name]
    target_embed=target_embeddings[layer_name]

    #source domain feature splits
    train_id_list_source=sample(range(source_embed.shape[0]),int(split*source_embed.shape[0]))
    test_id_list_source=[i for i in range(source_embed.shape[0]) if i not in train_id_list_source]
    
    dataX_train=np.array([source_embed[i] for i in train_id_list_source])
    labelX_train=np.array([source_label_data[i] for i in train_id_list_source])
    dataX_test=np.array([source_embed[i] for i in test_id_list_source])
    labelX_test=np.array([source_label_data[i] for i in test_id_list_source])

    #target domain feature splits 
    train_id_list_target=sample(range(target_embed.shape[0]),int(split*target_embed.shape[0]))
    test_id_list_target=[i for i in range(target_embed.shape[0]) if i not in train_id_list_target]
    
    dataY_train=np.array([target_embed[i] for i in train_id_list_target])
    labelY_train=np.array([target_label_data[i] for i in train_id_list_target])
    dataY_test=np.array([target_embed[i] for i in test_id_list_target])
    labelY_test=np.array([target_label_data[i] for i in test_id_list_target])

    labelX_train=to_one_hot(labelX_train)
    labelX_test=to_one_hot(labelX_test)
    labelY_train=to_one_hot(labelY_train)
    labelY_test=to_one_hot(labelY_test)
    
    return(dataX_train,labelX_train,dataX_test,labelX_test,dataY_train,labelY_train,dataY_test,labelY_test)


# source="dvd"
# target="electronics"
# base_folder="/data/Multi_Domain_Data/BERT_features/"
# source_csv_file="/data/Multi_Domain_Data/parsed_csv_data/dvd/dvd_review_splits_combined.csv"
# target_csv_file="/data/Multi_Domain_Data/parsed_csv_data/electronics/electronics_review_splits_combined.csv"
# dataX_train,labelX_train,dataX_test,labelX_test,dataY_train,labelY_train,dataY_test,labelY_test=extract_source_target_feature_set(source,target,base_folder,source_csv_file,target_csv_file)

# print(dataX_train.shape)
# print(labelX_train.shape)
# print(dataX_test.shape)
# print(labelX_test.shape)
# print(dataY_train.shape)
# print(labelY_train.shape)
# print(dataY_test.shape)
# print(labelY_test.shape)


    

