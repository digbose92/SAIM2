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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertTokenizer, BertForSequenceClassification
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


def main(config_data):

    wandb.login()
    wandb.init(project='bert domain adaptation', entity='digbwb', config=config_data)
    print(wandb.config)

    #dataset instantiation
    df=pd.read_csv(config_data['data']['data_file'])

    #preprocess label space to match the dictionary file 
    #print(df.head(5))
    df_train=df[df['split']=='train']
    df_valid=df[df['split']=='val']

    #print(df_valid.shape)
    tokenizer=bert_tokenizer(config_data['model']['model_type'],do_lower_case=config_data['parameters']['do_lower_case'])
    
    train_ds=Bert_dataset(df=df_train,
                    tokenizer=tokenizer,
                    max_len=config_data['parameters']['max_len'],
                    padding_type=config_data['parameters']['padding_type'],
                    truncation=config_data['parameters']['truncation'],
                    add_special_tokens=config_data['parameters']['add_special_tokens'],
                    pad_to_max_length=config_data['parameters']['pad_to_max_length'],
                    return_token_type_ids=config_data['parameters']['return_token_type_ids'],
                    return_attention_mask=config_data['parameters']['return_attention_mask'])
    train_dl=DataLoader(dataset=train_ds,
                    batch_size=config_data['parameters']['batch_size'], 
                    shuffle=config_data['parameters']['train_shuffle'])

    
    valid_ds=Bert_dataset(df=df_valid,
                    tokenizer=tokenizer,
                    max_len=config_data['parameters']['max_len'],
                    padding_type=config_data['parameters']['padding_type'],
                    truncation=config_data['parameters']['truncation'],
                    add_special_tokens=config_data['parameters']['add_special_tokens'],
                    pad_to_max_length=config_data['parameters']['pad_to_max_length'],
                    return_token_type_ids=config_data['parameters']['return_token_type_ids'],
                    return_attention_mask=config_data['parameters']['return_attention_mask'])
    valid_dl=DataLoader(dataset=valid_ds,
                    batch_size=config_data['parameters']['batch_size'], 
                    shuffle=config_data['parameters']['val_shuffle'])

    #model instantiation 
    #model declarations 
    # model_params={'model_type': config_data['model']['model_type'],
    #               'num_classes':config_data['model']['num_classes'],
    #               'hidden_dim': config_data['model']['hidden_dim'],
    #               'hidden_dim_1': config_data['model']['hidden_dim_1'],
    #               'drop_prob': config_data['model']['drop_prob'],
    #               'model_freeze_index': config_data['model']['model_freeze_index']}
    
    model=BertForSequenceClassification.from_pretrained(config_data['model']['model_type'],num_labels=config_data['model']['num_classes'])
    print(model)

    if(config_data['device']['is_cuda']):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_gpus=torch.cuda.device_count()
    if(n_gpus>1):
        model=nn.DataParallel(model)
    #transfer model to device 
    model=model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: %d' %(params))
    print(model)
    #loss functions and optimizer declarations 
    max_epochs=config_data['parameters']['epochs']

    if(config_data['loss']['loss_option']=='cross_entropy_loss'):
        criterion = cross_entropy_loss(device)
    if(config_data['loss']['loss_option']=='cross_entropy_label_smoothing'):
        epsilon=config_data['loss']['epsilon']
        criterion=LabelSmoothingCrossEntropy(epsilon=epsilon)
    
    # #optimizer definition
    if(config_data['optimizer']['choice']=='Adam'):
        optim_example=optimizer_adam(model,float(config_data['optimizer']['lr']),weight_decay=float(config_data['optimizer']['weight_decay']))

    if(config_data['optimizer']['choice']=='AdamW'):
        print(config_data['optimizer']['weight_decay'])
        optim_example=bert_base_AdamW_LLRD(model,float(config_data['optimizer']['lr']))
        #optim_example=optimizer_adamW(model,float(config_data['optimizer']['lr']),weight_decay=config_data['optimizer']['weight_decay'])


        #apply llrd (layer wise learning rate decay here)


    # if(config_data['optimizer']['scheduler']=='linear_scheduler_with_warmup'):
    #         print('Initializing linear scheduler')
    #         num_training_steps=len(train_dl)*max_epochs
    #         scheduler=linear_schedule_with_warmup(optim_example,config_data['optimizer']['num_warmup_steps'],num_training_steps)
    #################### LOGGER + BEST MODEL SAVING INFO HERE ###########################
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename=timestr+'_'+config_data['model']['model_type']+'_log.logs'
    yaml_filename=timestr+'_'+config_data['model']['model_type']+'.yaml'
    log_model_subfolder=os.path.join(config_data['output']['log_dir'],config_data['model']['model_type'])
    log_model_subfolder=os.path.join(log_model_subfolder+"_"+config_data['parameters']['domain_name'])
    if(os.path.exists(log_model_subfolder) is False):
        os.mkdir(log_model_subfolder)
    # #create log folder associated with current model
    sub_folder_log=os.path.join(log_model_subfolder,timestr+'_'+config_data['model']['model_type'])
    if(os.path.exists(sub_folder_log) is False):
        os.mkdir(sub_folder_log)

    # #create model folder associated with current model
    model_loc_subfolder=os.path.join(config_data['output']['model_dir'],config_data['model']['model_type'])
    model_loc_subfolder=os.path.join(model_loc_subfolder+"_"+config_data['parameters']['domain_name'])
    if(os.path.exists(model_loc_subfolder) is False):
        os.mkdir(model_loc_subfolder)

    sub_folder_model=os.path.join(model_loc_subfolder,timestr+'_'+config_data['model']['model_type'])
    if(os.path.exists(sub_folder_model) is False):
        os.mkdir(sub_folder_model)


    # #dump the current config into a yaml file 
    with open (os.path.join(sub_folder_log,yaml_filename),'w') as f:
        yaml.dump(config_data,f)

    # #logger=Logger(os.path.join(config_data['output']['log_dir'],config_data['model']['option']+'_log.txt'))
    logger = log(path=sub_folder_log, file=filename)
    logger.info('Starting training')

    early_stop_counter=config_data['parameters']['early_stop']
    print('Early stop criteria:%d' %(early_stop_counter))
    early_stop_cnt=0

    best_models=config_data['output']['model_dir']
    val_f1_best=0   
    log_softmax=nn.LogSoftmax(dim=-1)

    wandb.watch(model)

    #img_tensor, text_tensor, label=next(iter(train_dl))
    for epoch in range(1, max_epochs+1):
        step=0
        t = time.time()
        tot_pred_list=[]
        tot_GT_list=[]
        target_labels=[]
        train_logits=[]
        train_loss_list=[]
        lr_list=[]
        for id,(dict_data) in enumerate(tqdm(train_dl)):
            
            #pass everything to device 
            input_ids=dict_data['ids'].to(device)
            attention_mask=dict_data['mask'].to(device)
            token_type_ids=dict_data['token_type_ids'].to(device)
            label=dict_data['label'].to(device)

            optim_example.zero_grad()

            output=model(input_ids, attention_mask, token_type_ids)
            logits=output[0]
            train_logits_temp=log_softmax(logits).to('cpu')
            y_pred=torch.max(train_logits_temp, 1)[1]
            train_logits.append(y_pred)
            target_labels.append(label.cpu())


            loss=criterion(logits,label)
            loss.backward()

            # Update parameters and the learning rate
            #print('Loss:',loss.item())
            optim_example.step()

            step=step+1
            lr_step=optim_example.param_groups[0]["lr"]
            train_loss_list.append(loss.item())
            lr_list.append(lr_step)

            if(step%20==0):
                logger_step_dict={'Learning_rate':lr_step,'Running_Train_loss':mean(train_loss_list)}
                #print('Loss:%f' %(mean(train_loss_list)))
                logger.info("Training loss:{:.3f}".format(loss.item()))
                wandb.log(logger_step_dict)
                   #
        mean_train_loss=mean(train_loss_list)


        target_label_np=torch.cat(target_labels).detach().numpy()
        train_predictions = torch.cat(train_logits).detach().numpy()

        #train stats
        train_accuracy=accuracy_score(target_label_np,train_predictions)
        f1_score_train=f1_score(target_label_np, train_predictions, average='macro')  
        precision_score_train=precision_score(target_label_np,train_predictions,average='macro')
        recall_score_train=recall_score(target_label_np,train_predictions,average='macro')

        #print('Epoch:%d, Loss:%f' %(epoch,mean_train_loss))
        logger.info('epoch: {:d}, time:{:.2f}'.format(epoch, time.time()-t))
        logger.info('\ttrain_loss:{:.3f}'.format(mean_train_loss))
        logger.info('\ttrain_loss:{:.3f}, train accuracy:{:.3f}, train f1:{:.3f}'.format(mean(train_loss_list),train_accuracy,f1_score_train))


        logger.info('Evaluating the dataset')
        valid_accuracy,f1_score_val,precision_score_val,recall_score_val,val_loss=gen_eval_score_updated(valid_dl,model,device,criterion)
        logger.info('Validation accuracy:{:.3f},Validation f1:{:.3f},Validation loss:{:.3f}'.format(valid_accuracy,f1_score_val,val_loss))

        #wandb logging
        metrics_dict={'Train_loss':mean(train_loss_list),
            'Train_accuracy':train_accuracy,
            'Train_F1':f1_score_train,
            'Valid_loss':val_loss,
            'Valid_accuracy':valid_accuracy,
            'Valid_F1':f1_score_val,
            'Epoch':epoch}   #add epoch here to later switch the x-axis with epoch rather than actual wandb log
        
        wandb.log(metrics_dict)

        if(f1_score_val>val_f1_best):
            val_f1_best=f1_score_val
            logger.info('Saving the best model')
            #print('Validation accuracy: %f' %(valid_accuracy))
            logger.info('Validation accuracy:{:.3f},Validation f1:{:.3f}'.format(valid_accuracy,val_f1_best))
            #logger.info('Validation f1:{:.3f}'.format(f1_score_val))
            torch.save(model, os.path.join(sub_folder_model,timestr+'_'+config_data['model']['model_type']+'_best_model.pt'))
            early_stop_cnt=0

        else:
            early_stop_cnt=early_stop_cnt+1


        if(early_stop_cnt==early_stop_counter):
            print('Validation performance does not improve for %d iterations' %(early_stop_counter))
            break

        model.train(True)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Location of configuration data', type=str, required=True)
    args = vars(parser.parse_args())
    config_data=load_config(args['config_file'])
    main(config_data)
