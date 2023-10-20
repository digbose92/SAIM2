import torch.nn as nn 
import torch
import torch.nn.functional as F 


def cross_entropy_loss(device):
    criterion=nn.CrossEntropyLoss(reduction='mean').to(device)
    return(criterion)

def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss 

#label smoothing for ground truth in the cross entropy loss function
#reference issue: https://github.com/pytorch/pytorch/issues/7455
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):

        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim
    
    def forward(self, pred, target):
      pred = pred.log_softmax(dim=self.dim)
      with torch.no_grad():
          true_dist = torch.zeros_like(pred)
          true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) #(1-smoothing)*GT
          true_dist += self.smoothing / pred.size(self.dim) #(1-smoothing)*GT + smoothing * predicted 
      return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
      
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)