import argparse
import os
import torch
import parser 
import pandas as pd 
class Options():

    def __init__(self):
        ##
        #
        #self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = argparse.ArgumentParser()

        ##
        # Base
        self.parser.add_argument('--root',         type=str,        default = '/Volume/MSFI_AD/datasets/' )
        self.parser.add_argument('--imgcls',       type=str,        default = 'candle')
        self.parser.add_argument('--imgsize',      type=int,        default = 256)
        self.parser.add_argument('--transform',    type=str,        default = 'weak')
        self.parser.add_argument('--mode',         type=str,        default = 'full')
        self.parser.add_argument('--train',        type=bool,       default =  True)
        self.parser.add_argument('--device',       type=str,        default = 'cuda', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids',      type=str,        default = '0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        

        ##
        # Train
        self.parser.add_argument('--Batchsize',    type=int,      default = 32)
        self.parser.add_argument('--nepochs',      type=int,      default = 100)
        self.parser.add_argument('--beta1',        type=float,    default = 0.5)
        self.parser.add_argument('--lr',           type=float,    default = 0.001)
        self.parser.add_argument('--seed',         type = int,    default = 42)
        self.parser.add_argument('--modeltype',    type = str,    default = 'independent')
        self.parser.add_argument('--optimizer',    type = str,    default = 'Adam')
        
        # Save 
        self.parser.add_argument('--savedirs',     type = str,    default = '/Volume/MSFI_AD/save_models')        
        self.parser.add_argument('--savename',     type = str,    default = 'Baseline')
        self.parser.add_argument('--usewandb',     type = bool,   default = False)
        

        ## Test
        self.parser.add_argument('--threshold',    type=float,    default=0.05)

    
    def parse(self):
        self.args = self.parser.parse_args()
        # change argparse to dict type 
        arg_list = pd.Series(dir(self.args))[pd.Series(dir(self.args)).apply(lambda x : '_' not in x )].values
        temp = [] 
        cfg = {} 
        for i in arg_list:
            exec(f'temp.append(self.args.{i})')
        for n,arg in enumerate(arg_list):
            cfg[arg] = temp[n]
            
        # Transform 
        cfg['transform'] = __import__('src').data.augmentation.__dict__[f"{cfg['transform']}_augmentation"]()

        return cfg

    def to_dict(self):
        self.args = self.parser.parse_args(args=[])
        # change argparse to dict type 
        arg_list = pd.Series(dir(self.args))[pd.Series(dir(self.args)).apply(lambda x : '_' not in x )].values
        temp = [] 
        cfg = {} 
        for i in arg_list:
            exec(f'temp.append(self.args.{i})')
        for n,arg in enumerate(arg_list):
            cfg[arg] = temp[n]
            
        # Transform 
        cfg['transform'] = __import__('src').data.augmentation.__dict__[f"{cfg['transform']}_augmentation"]()

        return cfg
        
        