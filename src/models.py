import torch.nn as nn 
import torch 
import timm 


def build_net(pretrained=False):
    #net = timm.create_model('wide_resnet101_2',pretrained=pretrained)
    net = timm.create_model('resnet18',pretrained=pretrained)
    if pretrained:
        model = torch.nn.Sequential(*(list(net.children())[:-2]))
        for param in model.parameters():
            param.requires_grad = False
    else:
        model = torch.nn.Sequential(*(list(net.children())[:-2]))
        
    return model 

class Model(nn.Module):
    def __init__(self,training_type='deafult',device='cuda'):
        super(Model,self).__init__()
        self.teacher = build_net(True)
        self.student = build_net()
        self.training_type = training_type
        
    def train(self):
        self.teacher.eval()
        self.student.train()
        
    def eval(self):
        self.teacher.eval()
        self.student.eval()
    
    def train_independent_student(self,x):
        t_features = []
        s_features = []
        for (t_name,t_module),(s_name,s_module) in zip(self.teacher._modules.items(),self.student._modules.items()):
            if t_name in ['0','1','2','3']:
                x = t_module(x)
                #t_features.append(x)                    
            else:
                s = x.clone()
                x = t_module(x)
                s = s_module(s)
                
                t_features.append(x)
                s_features.append(s)
        return t_features,s_features
    
    def train_default_student(self,x):
        t_features = [] 
        s_features = []
        for (t_name,t_module),(s_name,s_module) in zip(self.teacher._modules.items(),self.student._modules.items()):
            if t_name == '0':
                x_s = s_module(x)
                x_t = t_module(x)
            else:
                x_s = s_module(x)
                x_t = t_module(x)
                if t_name in ['4','5','6','7']:
                    s_features.append(x_s)
                    t_features.append(x_t)
        return t_features,s_features
            
        
    def forward(self,x):
        if self.training_type =='default':
            t_features,s_features = self.train_default_student(x)
        else: 
            t_features,s_features = self.train_independent_student(x)
        return t_features,s_features