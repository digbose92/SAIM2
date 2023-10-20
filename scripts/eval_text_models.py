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
seed_value=123457
np.random.seed(seed_value) # cpu vars
torch.manual_seed(seed_value) # cpu  vars
random.seed(seed_value) # Python
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_config(config_file):

    with open(config_file,'r') as f:
        config_data=yaml.safe_load(f)

    return(config_data)


def gen_eval_score(valid_dl,model,device,criterion):
    model.eval()
    total_correct=0
    pred_list=[]
    true=[]
    loss_list=[]
    log_softmax=nn.LogSoftmax(dim=-1)
    with torch.no_grad():
        for i, (data) in tqdm(enumerate(valid_dl)):
            
            input_ids=data['ids'].to(device)
            input_masks=data['mask'].to(device)
            label=data['label'].to(device)
            token_type_ids=data['token_type_ids'].to(device)
            pred,_=model(input_ids,input_masks,token_type_ids)
            pred=log_softmax(pred)
            y_pred = torch.max(pred, 1)[1]
            loss_val=criterion(pred,label)
            loss_list.append(loss_val.item())
            #total_correct=total_correct+(y_pred==target).sum()
            true=true+label.cpu().numpy().tolist()
            pred_list=pred_list+y_pred.cpu().numpy().tolist()
                # print('True list:', true)
        # print('Predicted list:', pred_list)
        valid_accuracy=accuracy_score(true,pred_list)
        f1_score_val=f1_score(true, pred_list, average='macro')  
        precision_score_val=precision_score(true,pred_list,average='macro')
        recall_score_val=recall_score(true,pred_list,average='macro')
        val_loss_avg=mean(loss_list)

    return(valid_accuracy,f1_score_val,precision_score_val,recall_score_val,val_loss_avg)


def gen_eval_score_updated(valid_dl,model,device,criterion):
    model.eval()
    total_correct=0
    pred_list=[]
    true=[]
    loss_list=[]
    log_softmax=nn.LogSoftmax(dim=-1)
    with torch.no_grad():
        for i, (data) in tqdm(enumerate(valid_dl)):
            
            input_ids=data['ids'].to(device)
            input_masks=data['mask'].to(device)
            label=data['label'].to(device)
            token_type_ids=data['token_type_ids'].to(device)
            output=model(input_ids,input_masks,token_type_ids)
            logits=output[0]
            pred=log_softmax(logits)
            y_pred = torch.max(pred, 1)[1]
            loss_val=criterion(logits,label)
            loss_list.append(loss_val.item())
            #total_correct=total_correct+(y_pred==target).sum()
            true=true+label.cpu().numpy().tolist()
            pred_list=pred_list+y_pred.cpu().numpy().tolist()
                # print('True list:', true)
        # print('Predicted list:', pred_list)
        valid_accuracy=accuracy_score(true,pred_list)
        f1_score_val=f1_score(true, pred_list, average='macro')  
        precision_score_val=precision_score(true,pred_list,average='macro')
        recall_score_val=recall_score(true,pred_list,average='macro')
        val_loss_avg=mean(loss_list)

    return(valid_accuracy,f1_score_val,precision_score_val,recall_score_val,val_loss_avg)
