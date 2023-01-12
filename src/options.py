import argparse
import os
import torch
import parser 
import pandas as pd 
class Options():

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--root',         type=str,         default='/Volume/MSFI_AD/datasets/' )
        self.parser.add_argument('--imgcls',      type=str,        default='candle')
        self.parser.add_argument('--imgsize',     type=int,        default=256)
        self.parser.add_argument('--transform',    type=str,        default='weak')
        self.parser.add_argument('--mode',         type=str,        default='full')
        self.parser.add_argument('--train',        type=bool,       default= True)
        

        ##
        # Train
        self.parser.add_argument('--Batchsize',    type=int,      default = 32)
        self.parser.add_argument('--nepochs',      type=int,      default = 100)
        self.parser.add_argument('--beta1',        type=float,    default = 0.5)
        self.parser.add_argument('--lr',           type=float,    default = 0.001)
        self.parser.add_argument('--savedirs',    type = str,    default = '/Volume/MSFI_AD/save_models')
        
        
        
        

        ## Test
        self.parser.add_argument('--threshold', type=float, default=0.05)

        self.args = self.parser.parse_args(args=[])

    def parse(self):
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
        
        