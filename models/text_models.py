import os 
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import transformers 

class Bert_Model(nn.Module):
    def __init__(self,model_type,num_classes,hidden_dim,hidden_dim_1,drop_prob,model_freeze_index=None):
        super(Bert_Model,self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained(model_type,output_hidden_states=True)
        self.num_classes=num_classes
        self.hidden_dim=hidden_dim
        self.hidden_dim_1=hidden_dim_1
        self.drop_prob=drop_prob
        #self.freeze_model=freeze_model
        self.drop = torch.nn.Dropout(self.drop_prob)
        self.model_freeze_index=model_freeze_index
        #self.drop_2 = torch.nn.Dropout(self.drop_prob)
        #self.out = torch.nn.Linear(self.hidden_dim,self.hidden_dim_1)
        self.out=torch.nn.Linear(self.hidden_dim,self.num_classes)
        #self.out_1 =  torch.nn.Linear(self.hidden_dim_1,self.num_classes)
        # if(self.freeze_model): #dont want to open up the entire model

        #     #complete freezing of the model
        # for param in self.bert_model.parameters():
        #         param.requires_grad = False
        # if(self.model_freeze_index is not None):
        #     modules = [self.bert_model.embeddings, *self.bert_model.encoder.layer[:self.model_freeze_index]]
        #     for module in modules:
        #         for param in module.parameters():
        #             param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids):

        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hiddn_state=bert_output[-1]
        hidden_state = bert_output[0]  # (bs, seq_len, dim)
        #print(hidden_state.size())
        pooled_output = hidden_state[:, 0, :]  # (bs, dim)
        #pooled_output = self.drop(pooled_output)
        # output = self.fc_1(output_1)
        # output=self.relu_layer(output)
        output=self.out(pooled_output)
        #output=F.relu(output)
        #outptut=self.out_1(output)
        return output, hiddn_state


class Bert_Model_classifier(nn.Module):
    def __init__(self,model_type,num_classes,hidden_dim,hidden_dim_1,drop_prob,model_freeze_index=None):
        super(Bert_Model_classifier,self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained(model_type,output_hidden_states=True)
        self.num_classes=num_classes
        self.hidden_dim=hidden_dim
        self.hidden_dim_1=hidden_dim_1
        self.drop_prob=drop_prob
        #self.freeze_model=freeze_model
        self.drop = torch.nn.Dropout(self.drop_prob)
        self.model_freeze_index=model_freeze_index
        #self.drop_2 = torch.nn.Dropout(self.drop_prob)
        self.fc = torch.nn.Linear(self.hidden_dim,self.hidden_dim_1)
        self.out=torch.nn.Linear(self.hidden_dim_1,self.num_classes)
        
    def forward(self, input_ids, attention_mask, token_type_ids):

        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hiddn_state=bert_output[-1]
        hidden_state = bert_output[0]  # (bs, seq_len, dim)
        #print(hidden_state.size())
        pooled_output = hidden_state[:, 0, :]  # (bs, dim)
        embedding=self.fc(pooled_output)
        output=self.out(embedding)

        return output, hiddn_state
