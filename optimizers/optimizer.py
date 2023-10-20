import torch
from transformers import AdamW, get_linear_schedule_with_warmup

def optimizer_adam(model,lr,weight_decay=0):
    optim_set=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    return(optim_set)

def optimizer_adamW(model,lr,weight_decay):
    #optim_set=AdamW(model.parameters(),lr=lr)
    #default weight decay parameters added
    optim_set=AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    return(optim_set)

def linear_schedule_with_warmup(optimizer,num_warmup_steps,num_training_steps):
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps, # Default value
                                                num_training_steps=num_training_steps)
    return(scheduler)

def reduce_lr_on_plateau(optimizer,mode,patience):
    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode=mode,patience=patience)
    return(lr_scheduler)

def bert_base_AdamW_LLRD(model,init_lr):
    
    opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters()) 
        
    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    #init_lr = 3.5e-6 
    head_lr = init_lr
    lr = init_lr
    
    # === Pooler and regressor ======================================================  
    
    params_0 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n) 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n)
                and not any(nd in n for nd in no_decay)]
    
    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
    opt_parameters.append(head_params)
        
    head_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01}    
    opt_parameters.append(head_params)
                
    # === 12 Hidden layers ==========================================================
    
    for layer in range(11,-1,-1):        
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]
        
        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   
                            
        layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        opt_parameters.append(layer_params)       
        
        lr *= 0.9     
        
    # === Embeddings layer ==========================================================
    
    params_0 = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    
    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)
        
    embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
    opt_parameters.append(embed_params)        
    
    return AdamW(opt_parameters, lr=init_lr)
