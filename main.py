import warnings
warnings.filterwarnings('ignore')
from glob import glob 
import os 
import random 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 
import cv2
import argparse
import yaml 


import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms 
import torchvision
import timm 

from src.data.augmentation import *
from src.data.factory import create_dataset,create_dataloader
from src.options import Options
from src.models import Model 
from src.loss_function import LossFunction
from src.train import fit,torch_seed


    
def run(cfg):
    torch_seed(cfg['SEED'])
    
    # build train,test loader 
    trainset,testset = create_dataset(
                            cfg      = cfg,
                            data_dir = cfg['datadir'],
                            img_cls  = cfg['imgcls'],
                            img_size = cfg['imgsize'],
                            mode     = cfg['mode']
                            )
    train_loader = create_dataloader(
                            dataset    = trainset,
                            batch_size = cfg['Batchsize'],
                            shuffle    = True)
    test_loader = create_dataloader(
                            dataset    = testset,
                            batch_size = cfg['Batchsize'],
                            shuffle    = False)

    # build a model, criterion and optimizer 
    model = Model(
                training_type = cfg['modeltype']
                ).to(cfg['device'])
    
    # set training 
    criterion = LossFunction()
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[cfg['optimizer']](model.parameters(), lr=cfg['lr'],betas=(cfg['beta1'],0.999))    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['nepochs'])
    
    print('All loaded, Training start')
    fit(
        model             = model,
        train_loader      = train_loader,
        test_loader       = test_loader,
        criterion         = criterion,
        optimizer         = optimizer,
        scheduler         = scheduler,
        cfg               = cfg
        )
    
def init():
    parser = argparse.ArgumentParser(description='MSFI')
    parser.add_argument('--yaml_config', type=str, default='./configs/default.yaml', help='exp config file')    
    args = parser.parse_args()
    
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)
    cfg['transform'] = __import__('src').data.augmentation.__dict__[f"{cfg['transform']}"]()
    return cfg 
    
if __name__=='__main__':
    cfg = init()
    run(cfg)
    
    