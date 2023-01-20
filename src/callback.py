import torch 
import wandb 
import os 
import numpy as np 

class Callbacks:
    def __init__(self,cfg):
        #init 
        self.cfg = cfg 
        self.save_dir = os.path.join(cfg['savedirs'],cfg['savename'])
        self.make_dir()
        #logging 
        self.wandb_init()
        #checkpoint 
        self.best = np.inf
        self.best_epoch = 0 
        
    def make_dir(self):
        if os.path.exists(self.save_dir):
            n = 0 
            while os.path.exists(save_dir + str(n)):
                n +=1 
            self.save_dir = self.save_dir + str(n)
            os.mkdir(self.save_dir)
        else:
            os.mkdir(self.save_dir)
            
        
    def check_point(self,model,name):
        torch.save(model,os.path.join(self.save_dir,name))
    
    def wandb_init(self):
        if self.cfg['usewandb']:
            wandb.init(project="MSFI_AD",name=cfg['savename'])
    
    def logging(self,log):
        print(f" \nEpoch : {log['Epoch']}")
        print(f" Train loss : {log['train_loss']:.3f} | Valid loss : {log['valid_loss']:.3f}")
    
    def epoch(self,model,log,check='best'):
        # logging 
        if check == 'last':
            print(f" Best loss : {self.best} at Epoch : {self.best_epoch}")
        else:
            self.logging(log)
            
        # Wandb 
        if self.cfg['usewandb']:
            wandb.log(log)
        
        # Check point
        if check == 'best':
            if log['valid_loss'] < self.best:
                self.check_point(model,'best.pt')
                self.best = log['valid_loss']
                self.best_epoch = log['Epoch']
                print(f"Best model saved at {log['Epoch']}")
        else:
            self.check_pint(model,'last.pt')