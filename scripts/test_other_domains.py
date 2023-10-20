#code for using a source finetuned model to test on other target domains 
import pandas as pd 
import numpy as np 
import os 
import sys 
import time 
#append path of datasets and models 
sys.path.append(os.path.join('..', 'datasets'))
sys.path.append(os.path.join('..', 'models'))
sys.path.append(os.path.join('..', 'configs'))
sys.path.append(os.path.join('..', 'losses'))
sys.path.append(os.path.join('..', 'optimizers'))
sys.path.append(os.path.join('..', 'utils'))

from ast import literal_eval
import torch
import yaml
import torchvision
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset, DataLoader
from text_dataset import *
from text_models import *
from loss_functions import *
from eval_text_models import *
import wandb
from tqdm import tqdm 
import argparse
from optimizer import *
from statistics import mean
from log_file_generate import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
#torch.autograd.set_detect_anomaly(True)
# seed_value=123457
# np.random.seed(seed_value) # cpu vars
# torch.manual_seed(seed_value) # cpu  vars
# random.seed(seed_value) # Python
# torch.cuda.manual_seed(seed_value)
# torch.cuda.manual_seed_all(seed_value)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def load_config(config_file):

    with open(config_file,'r') as f:
        config_data=yaml.safe_load(f)

    return(config_data)

#source domain here is kitchen

#model_file_list=[]
# config_file_list=['/home/dbose_usc_edu/data/domain_adapt_data/multiple_runs_log_finetuning/bert-base-uncased_electronics/20221109-064128_bert-base-uncased/20221109-064128_bert-base-uncased.yaml',
# '/home/dbose_usc_edu/data/domain_adapt_data/multiple_runs_log_finetuning/bert-base-uncased_electronics/20221109-065735_bert-base-uncased/20221109-065735_bert-base-uncased.yaml',
# '/home/dbose_usc_edu/data/domain_adapt_data/multiple_runs_log_finetuning/bert-base-uncased_electronics/20221109-070852_bert-base-uncased/20221109-070852_bert-base-uncased.yaml',
# '/home/dbose_usc_edu/data/domain_adapt_data/multiple_runs_log_finetuning/bert-base-uncased_electronics/20221109-071912_bert-base-uncased/20221109-071912_bert-base-uncased.yaml',
# '/home/dbose_usc_edu/data/domain_adapt_data/multiple_runs_log_finetuning/bert-base-uncased_electronics/20221109-073319_bert-base-uncased/20221109-073319_bert-base-uncased.yaml']

# model_file_list=['/home/dbose_usc_edu/data/domain_adapt_data/multiple_runs_model_finetuning/bert-base-uncased_electronics/20221109-064128_bert-base-uncased/20221109-064128_bert-base-uncased_best_model.pt',
# '/home/dbose_usc_edu/data/domain_adapt_data/multiple_runs_model_finetuning/bert-base-uncased_electronics/20221109-065735_bert-base-uncased/20221109-065735_bert-base-uncased_best_model.pt',
# '/home/dbose_usc_edu/data/domain_adapt_data/multiple_runs_model_finetuning/bert-base-uncased_electronics/20221109-070852_bert-base-uncased/20221109-070852_bert-base-uncased_best_model.pt',
# '/home/dbose_usc_edu/data/domain_adapt_data/multiple_runs_model_finetuning/bert-base-uncased_electronics/20221109-071912_bert-base-uncased/20221109-071912_bert-base-uncased_best_model.pt',
# '/home/dbose_usc_edu/data/domain_adapt_data/multiple_runs_model_finetuning/bert-base-uncased_electronics/20221109-073319_bert-base-uncased/20221109-073319_bert-base-uncased_best_model.pt'
# ]
model_folder="/scratch1/dbose/Multi_Domain_data_gcp/domain_adapt_data/multiple_runs_hpc_model_finetuning_updated_splits/bert-base-uncased_electronics/20221129-155356_bert-base-uncased"
#"/scratch1/dbose/Multi_Domain_data_gcp/domain_adapt_data/multiple_runs_hpc_model_finetuning/bert-base-uncased_electronics/20221125-154219_bert-base-uncased"
config_file_list=["/scratch1/dbose/Multi_Domain_data_gcp/domain_adapt_data/multiple_runs_hpc_log_finetuning_updated_splits/bert-base-uncased_electronics/20221129-155356_bert-base-uncased/20221129-155356_bert-base-uncased.yaml"]*5
model_file_list=[os.path.join(model_folder,f) for f in os.listdir(model_folder)]

#model_file_list=['/home/dbose_usc_edu/data/domain_adapt_data/model_dir_updated_finetuning/bert-base-uncased_electronics/20221109-041537_bert-base-uncased/20221109-041537_bert-base-uncased_best_model.pt',
'/home/dbose_usc_edu/data/domain_adapt_data/model_dir_updated_finetuning/bert-base-uncased_electronics/20221109-042717_bert-base-uncased/20221109-042717_bert-base-uncased_best_model.pt',
'/home/dbose_usc_edu/data/domain_adapt_data/model_dir_updated_finetuning/bert-base-uncased_electronics/20221109-043852_bert-base-uncased/20221109-043852_bert-base-uncased_best_model.pt',
'/home/dbose_usc_edu/data/domain_adapt_data/model_dir_updated_finetuning/bert-base-uncased_electronics/20221109-045026_bert-base-uncased/20221109-045026_bert-base-uncased_best_model.pt',
'/home/dbose_usc_edu/data/domain_adapt_data/model_dir_updated_finetuning/bert-base-uncased_electronics/20221109-045941_bert-base-uncased/20221109-045941_bert-base-uncased_best_model.pt'#]
#config_file_list=['/home/dbose_usc_edu/data/domain_adapt_data/log_dir_updated_finetuning/bert-base-uncased_electronics/20221109-041537_bert-base-uncased/20221109-041537_bert-base-uncased.yaml',
'/home/dbose_usc_edu/data/domain_adapt_data/log_dir_updated_finetuning/bert-base-uncased_electronics/20221109-042717_bert-base-uncased/20221109-042717_bert-base-uncased.yaml',
'/home/dbose_usc_edu/data/domain_adapt_data/log_dir_updated_finetuning/bert-base-uncased_electronics/20221109-043852_bert-base-uncased/20221109-043852_bert-base-uncased.yaml',
'/home/dbose_usc_edu/data/domain_adapt_data/log_dir_updated_finetuning/bert-base-uncased_electronics/20221109-045026_bert-base-uncased/20221109-045026_bert-base-uncased.yaml',
'/home/dbose_usc_edu/data/domain_adapt_data/log_dir_updated_finetuning/bert-base-uncased_electronics/20221109-045941_bert-base-uncased/20221109-045941_bert-base-uncased.yaml'#]
csv_file="/scratch1/dbose/Multi_Domain_data_gcp/Multi_Domain_Data/parsed_csv_data/kitchen_&_housewares/kitchen_&_housewares_review_splits_combined.csv"
csv_data=pd.read_csv(csv_file)
test_acc=[]
test_f1=[]

for i,model_file in enumerate(model_file_list):
    #target domain here is electronics/kitchen/dvd
   
    config_data=load_config(config_file_list[i])
    tokenizer=bert_tokenizer(config_data['model']['model_type'],do_lower_case=config_data['parameters']['do_lower_case'])

    test_ds=Bert_dataset(df=csv_data,
                        tokenizer=tokenizer,
                        max_len=config_data['parameters']['max_len'],
                        padding_type=config_data['parameters']['padding_type'],
                        truncation=config_data['parameters']['truncation'],
                        add_special_tokens=config_data['parameters']['add_special_tokens'],
                        pad_to_max_length=config_data['parameters']['pad_to_max_length'],
                        return_token_type_ids=config_data['parameters']['return_token_type_ids'],
                        return_attention_mask=config_data['parameters']['return_attention_mask'])
    test_dl=DataLoader(dataset=test_ds,
                        batch_size=config_data['parameters']['batch_size'], 
                        shuffle=False)

    #load the model
    model=torch.load(model_file)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)

    if(config_data['loss']['loss_option']=='cross_entropy_loss'):
        criterion = cross_entropy_loss(device)

    if(config_data['loss']['loss_option']=='cross_entropy_label_smoothing'):
            epsilon=config_data['loss']['epsilon']
            criterion=LabelSmoothingCrossEntropy(epsilon=epsilon)

    test_accuracy,f1_score_test,precision_score_test,recall_score_test,test_loss=gen_eval_score_updated(test_dl,model,device,criterion)
    print('Run:%d' %(i))
    print('Testing accuracy:%f' %(test_accuracy))
    print('Testing f1 score:%f' %(f1_score_test))
    print('Testing recall score:%f' %(recall_score_test))
    print('Testing loss:%f' %(test_loss))
    test_acc.append(test_accuracy)
    test_f1.append(f1_score_test)

print('Average testing accuracy:%f' %(np.mean(test_acc)))
print('Average testing f1 score:%f' %(np.mean(test_f1)))






