import torch 
import torch.nn.functional as F 
import random 
import numpy as np 
import os 
from src.callback import Callbacks 
from tqdm import tqdm 
from sklearn.metrics import roc_curve, auc 

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
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
    
    anomaly_maps = [] 
    ground_msks = [] 
    
    for batch_imgs,batch_msks in train_loader:
        
        # predict 
        batch_imgs = batch_imgs.to(cfg['device']).type(torch.float32)
        t_features, s_features = model(batch_imgs)
        
        # backward 
        #loss,batch_anomaly_map = criterion(t_features, s_features)
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

def build_anomaly_map(t_features,s_features,cfg):
    score_map = 1. 
    feature_maps = torch.zeros((len(t_features[0]),len(t_features),cfg['imgsize'],cfg['imgsize']))
    for i in range(len(t_features)):
        #feature_maps = torch.zeros((len(t_features[0]),len(t_features),cfg['imgsize'],cfg['imgsize']))
        t_f,s_f = t_features[i],s_features[i]
        t_f,s_f = F.normalize(t_f,dim=1), F.normalize(s_f,dim=1)
        
        layer_maps = torch.sum((t_f - s_f)**2,dim=1,keepdim=True)
        layer_maps = F.interpolate(input = layer_maps,
                                size  = (cfg['imgsize'],cfg['imgsize']),
                                mode  = 'bilinear',
                                align_corners = False)
        #score_map = score_map*layer_maps
        
        #feature_maps[:,i,:,:] = score_map.squeeze(dim=1)
        feature_maps[:,i,:,:] = layer_maps.squeeze(dim=1)
    anomaly_map = torch.mean(feature_maps,dim=1,keepdim=True)  
    return anomaly_map 
    

def inference(model,test_loader,cfg):
    model.eval()
    device = cfg['device']
    
    anomaly_map = []
    ground_msk = [] 
    for batch_imgs,batch_msks in test_loader:
        batch_imgs = batch_imgs.type(torch.float32).to(device)
        t_features,s_features = model(batch_imgs)
        batch_anomaly_map = build_anomaly_map(t_features = t_features,
                                              s_features = s_features,
                                              cfg        = cfg)
        anomaly_map.extend(batch_anomaly_map.detach().cpu().numpy())
        ground_msk.extend(batch_msks.detach().cpu().numpy())
        
    anomaly_map = np.array(anomaly_map).reshape(len(anomaly_map),-1)      
    ground_msk = np.array(ground_msk).reshape(len(ground_msk),-1)
    ground_msk = np.where(ground_msk==0,ground_msk,1).astype(int)
    
    return anomaly_map,ground_msk

def roc(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc    


def cal_metric(anomaly_map,ground_msk):
    pixel_roc_auc = roc(ground_msk.flatten(),anomaly_map.flatten())
    img_roc_auc = roc(ground_msk.max(axis=1),anomaly_map.max(axis=1))
    return img_roc_auc,pixel_roc_auc



def fit(model,train_loader,test_loader,criterion,optimizer,scheduler,cfg):
    
    callbacks = Callbacks(cfg)
    
    total_loss = {} 
    total_loss['train'] = [] 
    total_loss['valid'] = []
     
    
    for epoch in tqdm(range(cfg['nepochs'])):
        train_loss = train_epoch(model,train_loader,criterion,optimizer,cfg)
        valid_loss = valid_epoch(model,test_loader,criterion,cfg)
        anomaly_map,ground_msk = inference(model,test_loader,cfg)
        img_roc_auc,pixel_roc_auc = cal_metric(anomaly_map,ground_msk)
        
        total_loss['train'].append(train_loss)
        total_loss['valid'].append(valid_loss)
        
        
        if cfg['usescheduler']:
            scheduler.step()
        
        log = {'Epoch' : epoch,
               'train_loss'   : train_loss,
               'valid_loss'   : valid_loss,
               'learing_rate' : optimizer.param_groups[0]['lr'],
               'image_auroc'  : img_roc_auc,
               'pixel_auroc'  : pixel_roc_auc
               }
        
        
        
        #check point 
        callbacks.epoch(model,log)
    callbacks.epoch(model,log,'last')
    
