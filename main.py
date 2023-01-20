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
from src.train import fit 

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
    
def run(cfg):
    torch_seed(cfg['seed'])
    
    # build train,test loader 
    trainset,testset = create_dataset(cfg)
    train_loader = create_dataloader(
                                    dataset    = trainset,
                                    batch_size = cfg['Batchsize'],
                                    shuffle    = True)
    test_loader = create_dataloader(
                                    dataset    = testset,
                                    batch_size = cfg['Batchsize'],
                                    shuffle    = False)

    # build a model, criterion and optimizer 
    model = Model(cfg['modeltype']).to(cfg['device'])
    criterion = LossFunction()
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[cfg['optimizer']](model.parameters(), lr=cfg['lr'],betas=(cfg['beta1'],0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['nepochs'])
    
    print('All loaded, Training start')
    fit(model,train_loader,test_loader,criterion,optimizer,scheduler,cfg)

    
if __name__=='__main__':
    
    cfg = Options().parse()
    run(cfg)
    
    