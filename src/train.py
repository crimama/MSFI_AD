import torch 
import random 
import numpy as np 
import os 
from src.callback import Callbacks 


def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
        
def train_epoch(model,train_loader,criterion,optimizer,cfg):
    model.train()
    train_loss = AverageMeter()
    for batch_imgs,_ in train_loader:
        
        # predict 
        batch_imgs = batch_imgs.to(cfg['device']).type(torch.float32)
        t_features, s_features = model(batch_imgs)
        
        # backward 
        loss = criterion(t_features, s_features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        train_loss.update(loss.item())
    return train_loss.avg 

def valid_epoch(model,test_loader,criterion,cfg):
    model.eval()
    valid_loss = AverageMeter()
    
    for batch_imgs,_ in test_loader:
        # predict 
        batch_imgs = batch_imgs.to(cfg['device']).type(torch.float32)
        t_features, s_features = model(batch_imgs)
        # backward 
        loss = criterion(t_features, s_features)
        
        valid_loss.update(loss.item())
    return valid_loss.avg     


def fit(model,train_loader,test_loader,criterion,optimizer,scheduler,cfg):
    
    callbacks = Callbacks(cfg)
    
    total_loss = {} 
    total_loss['train'] = [] 
    total_loss['valid'] = [] 
    
    for epoch in range(cfg['nepochs']):
        train_loss = train_epoch(model,train_loader,criterion,optimizer,cfg)
        valid_loss = valid_epoch(model,test_loader,criterion,cfg)

        
        total_loss['train'].append(train_loss)
        total_loss['valid'].append(valid_loss)
        scheduler.step()
        
        log = {'Epoch' : epoch,
               'train_loss' : train_loss,
               'valid_loss' : valid_loss,
               'learing_rate' : optimizer.param_groups[0]['lr']}
        
        #check point 
        callbacks.epoch(model,log)
    callbacks.epoch(model,log,'last')