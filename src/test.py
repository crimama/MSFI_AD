import numpy as np 
import pandas as pd 
from PIL import Image 
import matplotlib.pyplot as plt 
import cv2 
import yaml 
from sklearn.metrics import roc_curve,auc 

import torch 
from torchvision import transforms 
import torch.nn.functional as F 
import torch.nn as nn 

from src.data.augmentation import *
from src.data.factory import create_dataset,create_dataloader
from src.train import torch_seed
from src.options import Options

def build_anomaly_map(t_features,s_features):
    score_map = 1. 
    for i in range(len(t_features)):
        feature_maps = torch.zeros((len(t_features[0]),len(t_features),cfg['imgsize'],cfg['imgsize']))
        t_f,s_f = t_features[i],s_features[i]
        t_f,s_f = F.normalize(t_f,dim=1), F.normalize(s_f,dim=1)
        
        layer_maps = torch.sum((t_f - s_f)**2,dim=1,keepdim=True)
        layer_maps = F.interpolate(input = layer_maps,
                                size  = (cfg['imgsize'],cfg['imgsize']),
                                mode  = 'bilinear',
                                align_corners = False)
        score_map = score_map*layer_maps
        
        feature_maps[:,i,:,:] = score_map.squeeze(dim=1)
    anomaly_map = torch.mean(feature_maps,dim=1,keepdim=True)  
    return anomaly_map 

def inference(cfg,model,testset,trainset,test_loader):
    model.eval()
    device = cfg['device']

    anomaly_map = [] 
    ground_msk = [] 
    for batch_imgs,batch_msks in test_loader:
        
        batch_imgs = batch_imgs.type(torch.float32).to(device)
        t_features,s_features = model(batch_imgs)
        
        batch_anomaly_map = build_anomaly_map(t_features = t_features,
                                            s_features = s_features)
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