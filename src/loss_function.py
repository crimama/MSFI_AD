import torch.nn as nn 
import torch.nn.functional as F 
import torch 
class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

    def loss_fn(self,f_s,f_t):
        
        score_map = 1. 
        total_loss = 0 
        #feature_maps = torch.zeros((len(t_features[0]),len(t_features),cfg['imgsize'],cfg['imgsize']))
        for t,s in zip(f_t,f_s):
            t,s = F.normalize(t,dim=1),F.normalize(s,dim=1)
            # for loss 
            total_loss += torch.sum((t.type(torch.float32) - s.type(torch.float32)) ** 2, 1).mean()

        return total_loss

    def forward(self, f_s,f_t):
        total_loss = self.loss_fn(f_s,f_t)
        return total_loss