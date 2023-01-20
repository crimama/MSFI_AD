import pandas as pd 
import numpy as np 
import cv2 
import torch 
from torchvision import transforms 
from torch.utils.data import Dataset,DataLoader
import os 
from PIL import Image 


def create_dataset(cfg,data_dir,img_cls,img_size,mode):
    trainset = VisaDataset(
                            root         = data_dir,
                            img_cls       = img_cls,
                            img_size      = img_size,
                            transform     = cfg['transform'],
                            mode          = mode
                            )
    testset = VisaDataset(
                            root         = data_dir,
                            img_cls       = img_cls,
                            img_size      = img_size,
                            transform     = transforms.Compose([transforms.ToTensor()]),
                            mode          = cfg['mode'],
                            train = False
                            )

    return trainset, testset


def create_dataloader(dataset, batch_size: int = 32, shuffle: bool = False):

    return DataLoader(
                      dataset     = dataset,
                      batch_size  = batch_size,
                      shuffle     = shuffle
                     )



class VisaDataset(Dataset):
    def __init__(self,root,img_size,transform,img_cls = 'candle',mode='full',train=True):
        super(VisaDataset,self).__init__()
        
        self.root = root                             # Dataset directory 
        self.img_size = img_size 
        self.mode = mode                             # Training mode : Fullshot, 2cls Fewshot, 2cls Highshot 
        self.img_cls = img_cls                       # Image Class 
        
        self.df = self._read_csv(mode)               # Load df containing information of img and mask 
        self._load_dirs(img_cls,train)               # Following df, load directorys of imgs and masks 
        self.img_transform  = transforms.Compose(transform.transforms + [transforms.Resize((img_size,img_size))])
        self.msk_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((img_size,img_size))])
        
        self.train = train  
        
    def __len__(self):
        return len(self.img_dirs)
            
    def _read_csv(self,mode):
        # Choose a mode of Training : Fullshot, 2cls Fewshot, 2cls Highshot 
        if mode == 'full':
            df = pd.read_csv(os.path.join(self.root,'split_csv','1cls.csv'))
        elif mode == 'fewshot':
            df = pd.read_csv(os.path.join(self.root,'split_csv','2cls_fewshot.csv'))
        elif mode == 'highshot':
            df = pd.read_csv(os.path.join(self.root,'split_csv','2cls_highshot.csv'))
        return df 
    
    def _load_dirs(self,img_cls,train):
        # Choose either Training or Test and additionaly Class of img (ex : Candle)
        if img_cls == 'all': # In case using All type of Image Claases 
            if train:
                self.img_dirs = self.df[self.df['split']=='train']['image'].values
                self.msk_dirs = self.df[self.df['split']=='train']['mask'].values
            else:
                self.img_dirs = self.df[self.df['split']=='test']['image'].values
                self.msk_dirs = self.df[self.df['split']=='test']['mask'].values
        else: # In case only using one class of image 
            if train:
                self.img_dirs = self.df[(self.df['split']=='train') & (self.df['object'] == self.img_cls)]['image'].values
                self.msk_dirs = self.df[(self.df['split']=='train') & (self.df['object'] == self.img_cls)]['mask'].values
            else:
                self.img_dirs = self.df[(self.df['split']=='test') & (self.df['object'] == self.img_cls)]['image'].values
                self.msk_dirs = self.df[(self.df['split']=='test') & (self.df['object'] == self.img_cls)]['mask'].values
            
    def load_img(self,img_dir):
        img = Image.open(os.path.join(self.root,img_dir))
        img = self.img_transform(img)        
        return img 

    def load_msk(self,msk_dir):
        try:
            msk = Image.open(os.path.join(self.root,msk_dir))
            msk = cv2.resize(np.array(msk),dsize=(self.img_size,self.img_size))
            msk = np.expand_dims(msk,axis=-1)
        except:
            msk = np.zeros((self.img_size,self.img_size,1))
        return msk 
        
    
    def __getitem__(self,idx):
        img = self.img_dirs[idx]
        msk = self.msk_dirs[idx]
        
        img = self.load_img(img)
        msk = self.load_msk(msk)
        
        return img,msk 