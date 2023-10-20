import os 
import numpy as np 
import pickle 
import pandas as pd 
import torch
import sys 
import yaml
from tqdm import tqdm
import random 
sys.path.append(os.path.join('..', 'datasets'))
sys.path.append(os.path.join('..', 'models'))
sys.path.append(os.path.join('..', 'configs'))
sys.path.append(os.path.join('..', 'losses'))
sys.path.append(os.path.join('..', 'optimizers'))
sys.path.append(os.path.join('..', 'utils'))

seed_value=123457
np.random.seed(seed_value) # cpu vars
torch.manual_seed(seed_value) # cpu  vars
random.seed(seed_value) # Python
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#forward hook code here 
from text_models import *
from text_dataset import *
import argparse

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output[0].detach()
    return hook

def load_config(config_file):

    with open(config_file,'r') as f:
        config_data=yaml.safe_load(f)

    return(config_data)


#model option config file dest folder path of csv file 
parser = argparse.ArgumentParser()
parser.add_argument('--model_option', help='Location of model configuration', type=str, required=True)
parser.add_argument('--config_file', help='Location of configuration data', type=str, required=True)
parser.add_argument('--dest_folder', help='Location of destination folder', type=str, required=True)
parser.add_argument('--csv_file', help='Location of the csv file to be used', type=str, required=True)

args = vars(parser.parse_args())



#model pt file and config file
model_option=args['model_option']
config_file=args['config_file']
dest_folder=args['dest_folder']
csv_file=args['csv_file']


config_data=load_config(config_file)
model_params={'model_type': config_data['model']['model_type'],
                  'num_classes':config_data['model']['num_classes'],
                  'hidden_dim': config_data['model']['hidden_dim'],
                  'hidden_dim_1': config_data['model']['hidden_dim_1'],
                  'drop_prob': config_data['model']['drop_prob'],
                  'model_freeze_index': config_data['model']['model_freeze_index']}
    

#load the model 
model_decl=torch.load(model_option)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer_model = Bert_Model(**model_params)
#Bert_Model
#(**model_params) #change this to BertClassifier
transformer_model.load_state_dict(model_decl.module.state_dict())
transformer_model.eval()
transformer_model=transformer_model.to(device)

#tokenizer initialize
tokenizer=bert_tokenizer(config_data['model']['model_type'],do_lower_case=config_data['parameters']['do_lower_case'])
#print(transformer_model.module.Bert_Model)
max_length_token=512
#print((transformer_model.bert_model.encoder.layer[-1]))

transformer_model.bert_model.encoder.layer[-1].register_forward_hook(get_activation('last_encoder_layer'))
transformer_model.bert_model.encoder.layer[-2].register_forward_hook(get_activation('penultimate_encoder_layer'))

#csv data file 
target_file_name=csv_file.split("/")[-1].split(".")[0]+".pkl"
csv_data=pd.read_csv(csv_file)
print(csv_data.columns)
#print(csv_data.shape[0])

mean_last_encoder_w_cls_token=np.zeros((csv_data.shape[0],768))
mean_penultimate_encoder_w_cls_token=np.zeros((csv_data.shape[0],768))
mean_last_encoder_wo_cls_token=np.zeros((csv_data.shape[0],768))
mean_penultimate_encoder_wo_cls_token=np.zeros((csv_data.shape[0],768))
cls_token_repr=np.zeros((csv_data.shape[0],768))

for i in tqdm(np.arange(csv_data.shape[0])):

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

    input_ids=dictionary['input_ids'].to(device)
    attention_mask=dictionary['attention_mask'].to(device)
    token_type_ids=dictionary['token_type_ids'].to(device)
    #print(input_ids)
    non_zero_indices=torch.nonzero(input_ids,as_tuple=False).cpu().numpy()[:,1].tolist()
    
    with torch.no_grad():
        outputs = transformer_model(input_ids, attention_mask, token_type_ids)
    
    last_encoder_layer_activation=activation['last_encoder_layer'].squeeze(0).cpu().numpy()    
    penultimate_encoder_layer_activation=activation['penultimate_encoder_layer'].squeeze(0).cpu().numpy()

    last_encoder_layer_activation=np.array([last_encoder_layer_activation[ind,:] for ind in non_zero_indices])
    penultimate_encoder_layer_activation=np.array([penultimate_encoder_layer_activation[ind,:] for ind in non_zero_indices])
    
    last_encoder_layer_mean_w_cls=np.mean(last_encoder_layer_activation,axis=0)
    cls_token_representation=last_encoder_layer_activation[0,:]
    last_encoder_layer_mean_wo_cls=np.mean(last_encoder_layer_activation[1:,:],axis=0)
    penultimate_encoder_layer_mean_w_cls=np.mean(penultimate_encoder_layer_activation,axis=0)
    penultimate_encoder_layer_mean_wo_cls=np.mean(penultimate_encoder_layer_activation[1:,:],axis=0)

    #take the mean upto the length of the token excluding the special padding token
    #print(last_encoder_layer_mean_w_cls==last_encoder_layer_mean_wo_cls)
    
    mean_last_encoder_w_cls_token[i,:]=last_encoder_layer_mean_w_cls
    mean_penultimate_encoder_w_cls_token[i,:]=penultimate_encoder_layer_mean_w_cls
    mean_last_encoder_wo_cls_token[i,:]=last_encoder_layer_mean_wo_cls
    mean_penultimate_encoder_wo_cls_token[i,:]=penultimate_encoder_layer_mean_wo_cls
    cls_token_repr[i,:]=cls_token_representation

print(mean_last_encoder_w_cls_token.shape)

dict_set={'cls_token': cls_token_repr,
        'Last_encoder_w_cls':mean_last_encoder_w_cls_token,
        'Last_encoder_wo_cls': mean_last_encoder_wo_cls_token,
        'Penultimate_encoder_w_cls':mean_penultimate_encoder_w_cls_token,
        'Penultimate_encoder_wo_cls':mean_penultimate_encoder_wo_cls_token }
print(target_file_name)
with open(os.path.join(dest_folder,target_file_name),'wb') as f:
    pickle.dump(dict_set,f)


#output=transformer_model()
