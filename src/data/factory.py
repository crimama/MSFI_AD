import pandas as pd 
import torch 
from torchvision import transforms 
from torch.utils.data import Dataset,DataLoader
import os 
from PIL import Image 



def create_dataset(cfg : dict):
    trainset = VisaDataset(
                            root          = cfg['root'],
                            img_cls       = cfg['imgcls'],
                            img_size      = cfg['imgsize'],
                            transform     = cfg['transform'],
                            mode          = cfg['mode']
                            )
    testset = VisaDataset(
                            root          = cfg['root'],
                            img_cls       = cfg['imgcls'],
                            img_size      = cfg['imgsize'],
                            transform     = __import__('src').data.augmentation.__dict__["default_augmentation"](),
                            mode          = cfg['mode'],
                            train = False
                            )

    return trainset, testset


def create_dataloader(dataset, batch_size: int = 32, shuffle: bool = False):

    return DataLoader(
                      dataset     = dataset,
                      batch_size  = batch_size,
                      shuffle     = shuffle,
                      num_workers = 0
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
        # Choose btw Training or Test and additionaly Class of img (ex : Candle)
        if img_cls == 'all': # In case using All type of Image Claases 
            if train:
                self.img_dirs = self.df[self.df['split']=='train']['image'].values
                self.msk_dirs = self.df[self.df['split']=='train']['mask'].values
            else:
                self.img_dirs = self.df[self.df['split']=='test']['image'].values
                self.msk_dirs = self.df[self.df['split']=='test']['mask'].values
        else: # In case only using one class of image 
            if train:
                self.img_dirs = self.df[(self.df['split']=='train') & (self.df['object'] == 'candle')]['image'].values
                self.msk_dirs = self.df[(self.df['split']=='train') & (self.df['object'] == 'candle')]['mask'].values
            else:
                self.img_dirs = self.df[(self.df['split']=='test') & (self.df['object'] == 'candle')]['image'].values
                self.msk_dirs = self.df[(self.df['split']=='test') & (self.df['object'] == 'candle')]['mask'].values
            
    def load_img(self,img_dir):
        img = Image.open(os.path.join(self.root,img_dir))
        img = self.img_transform(img)
        return img 

    def load_msk(self,msk_dir):
        try:
            msk = Image.open(os.path.join(self.root,msk_dir))
            msk = self.msk_transform(msk)
        except:
            msk = torch.zeros((1,256,256))
        return msk 
        
    
    def __getitem__(self,idx):
        img = self.img_dirs[idx]
        msk = self.msk_dirs[idx]
        
        img = self.load_img(img)
        msk = self.load_msk(msk)
        
        return img,msk 