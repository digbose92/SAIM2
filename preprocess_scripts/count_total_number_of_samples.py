import os 
import pandas as pd 
import numpy as np 

base_folder="/data/Multi_Domain_Data/parsed_csv_data/"
folder_list=['books','dvd','electronics','kitchen_&_housewares']
file_name=['positive_review.csv','unlabeled_review.csv','negative_review.csv']
num_file_list=0

for folder in folder_list:
    folder_path=os.path.join(base_folder,folder)
    for file in file_name:
        file_path=os.path.join(folder_path,file)
        if os.path.exists(file_path):
            data=pd.read_csv(file_path)
            print(folder,data.shape)
            num_file_list=num_file_list+data.shape[0]

print(num_file_list)