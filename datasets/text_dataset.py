from transformers import DistilBertTokenizer, RobertaTokenizer, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import os 
import torch 
import pandas as pd 
import pickle

def bert_tokenizer(model_type,do_lower_case):
    tokenizer=BertTokenizer.from_pretrained(model_type,do_lower_case=do_lower_case)
    return(tokenizer)

class Bert_dataset(Dataset):
    def __init__(self,df,tokenizer,
                max_len,
                padding_type='max_length',
                truncation=True,
                add_special_tokens=True,
                pad_to_max_length=True,
                return_token_type_ids=True,
                return_attention_mask=True):

        self.data=df 
        self.tokenizer=tokenizer 
        self.max_len=max_len  
        self.add_special_tokens=add_special_tokens
        self.pad_to_max_length=pad_to_max_length
        self.return_attention_mask=return_attention_mask
        self.return_token_type_ids=return_token_type_ids
        self.padding_type=padding_type
        self.truncation=truncation

    def __getitem__(self,idx):

        text=self.data['Text'].iloc[idx]
        label=int(self.data['Label'].iloc[idx])
        encoded_text=self.tokenizer.encode_plus(
                    text=text,
                    add_special_tokens=self.add_special_tokens, 
                    max_length=self.max_len,
                    padding=self.padding_type,
                    truncation=self.truncation,
                    return_attention_mask=self.return_attention_mask,
                    return_token_type_ids=self.return_token_type_ids
        )

        ids=encoded_text['input_ids']
        mask=encoded_text['attention_mask']
        token_type_ids = encoded_text["token_type_ids"]

        ret_dict={'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'label': torch.tensor(label,dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)}
        return(ret_dict)

    def __len__(self):
        return(len(self.data))


class Bert_json_dataset(Dataset):
    def __init__(self,json_data,
                tokenizer,
                max_len,
                padding_type='max_length',
                truncation=True,
                add_special_tokens=True,
                pad_to_max_length=True,
                return_token_type_ids=True,
                return_attention_mask=True):

        self.data=json_data
        self.tokenizer=tokenizer 
        self.max_len=max_len  
        self.add_special_tokens=add_special_tokens
        self.pad_to_max_length=pad_to_max_length
        self.return_attention_mask=return_attention_mask
        self.return_token_type_ids=return_token_type_ids
        self.padding_type=padding_type
        self.truncation=truncation

    def __getitem__(self,idx):

        text=self.data[idx]['Text'][0]
        label=int(self.data[idx]['Label'][0])
        encoded_text=self.tokenizer.encode_plus(
                    text=text,
                    add_special_tokens=self.add_special_tokens, 
                    max_length=self.max_len,
                    padding=self.padding_type,
                    truncation=self.truncation,
                    return_attention_mask=self.return_attention_mask,
                    return_token_type_ids=self.return_token_type_ids
        )
        ids=encoded_text['input_ids']
        mask=encoded_text['attention_mask']
        token_type_ids = encoded_text["token_type_ids"]

        ret_dict={'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'label': torch.tensor(label,dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)}
        return(ret_dict)

    def __len__(self):
        return(len(self.data))

# if __name__ == "__main__":
#     file="/data/Multi_Domain_Data/parsed_csv_data/dvd/dvd_review_splits_combined.csv"
#     df=pd.read_csv(file)
#     df_train=df[df['split']=='train']
#     tokenizer=bert_tokenizer('bert-base-uncased',do_lower_case=True)
#     ds=Bert_dataset(df=df_train,
#                     tokenizer=tokenizer,
#                     max_len=512,
#                     add_special_tokens=True,
#                     pad_to_max_length=True,
#                     return_token_type_ids=True,
#                     return_attention_mask=True)

#     dl=DataLoader(dataset=ds,batch_size=32,shuffle=False)
#     ret_dict=next(iter(dl))
#     print(ret_dict['ids'])