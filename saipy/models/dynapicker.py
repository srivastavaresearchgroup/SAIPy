import os,sys
sys.path.insert(0, '..')
import pandas as pd
import h5py
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.functional as F
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        return torch.nn.functional.relu6(x + 3., inplace=self.inplace) / 3.

class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )
    def forward(self, x):
        y = self.fc(x)
        return x * y

class conv_basic_dy(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super(conv_basic_dy, self).__init__()
        
        self.conv = conv3x3(inplanes, planes, stride)
        self.dim = int(math.sqrt(inplanes*2))
        squeeze = max(inplanes*4, self.dim ** 2) // 16
        if squeeze < 4:
            squeeze = 4
        self.q = nn.Conv1d(inplanes, self.dim, 1, stride, 0, bias=False)
        self.p = nn.Conv1d(self.dim, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm1d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(2)  
        self.fc = nn.Sequential(
            nn.Linear(inplanes*2, squeeze, bias=False),
            SEModule_small(squeeze),
        ) 
        self.fc_phi = nn.Linear(squeeze, self.dim**2, bias=False)
        self.fc_scale = nn.Linear(squeeze, planes, bias=False)
        self.hs = Hsigmoid()     
    def forward(self, x):
        r = self.conv(x)
        b, c, _= x.size()
        y = self.avg_pool(x).view(b, c*2)
        y = self.fc(y)
        phi = self.fc_phi(y).view(b, self.dim, self.dim)
        scale = self.hs(self.fc_scale(y)).view(b,-1,1)
        r = scale.expand_as(r)*r        
        out = self.bn1(self.q(x))
        _, _, w = out.size()
        out = out.view(b,self.dim,-1)
        out = self.bn2(torch.matmul(phi, out)) + out
        out = out.view(b,-1,w)        
        out = self.p(out) + r
        return out    
    
class DYCls(nn.Module):
    def __init__(self, inp, oup):
        super(DYCls, self).__init__()
        self.dim = 32
        self.cls = nn.Linear(inp*100, oup)
        self.cls_q = nn.Linear(inp*100, self.dim, bias=False)
        self.cls_p = nn.Linear(self.dim, oup, bias=False)
        mid = 32
        self.fc = nn.Sequential(
            nn.Linear(inp*100, mid, bias=False),
            SEModule_small(mid),
        )
        self.fc_phi = nn.Linear(mid, self.dim**2, bias=False)
        self.fc_scale = nn.Linear(mid, oup, bias=False)
        self.hs = Hsigmoid()
        self.bn1 = nn.BatchNorm1d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)
    def forward(self, x):
        #r = self.cls(x)
        b, c = x.size()
        y = self.fc(x)
        dy_phi = self.fc_phi(y).view(b, self.dim, self.dim)
        dy_scale = self.hs(self.fc_scale(y)).view(b, -1)
        r = dy_scale*self.cls(x)
        x = self.cls_q(x)
        x = self.bn1(x)
        x = self.bn2(torch.matmul(dy_phi, x.view(b, self.dim, 1)).view(b, self.dim)) + x
        x = self.cls_p(x)
        return x + r

class dynapickerv1(nn.Module):
    def __init__(self):
        super().__init__()
        # input: 1,400,3
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 3, out_channels = 32, kernel_size = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size = 2),
            nn.Dropout(p=0.2, inplace = False)
            )
        self.layer1 = nn.Sequential(
            conv_basic_dy(32, 64, stride =1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace = False)
        )
        self.layer2 = nn.Sequential(
            conv_basic_dy(64, 128, stride =1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace = False)
        )
        self.layer3 = nn.Sequential(
            conv_basic_dy(128, 256, stride =1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace = False)
        )
        self.layer4= nn.Sequential(
            conv_basic_dy(256, 512, stride =1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace = False)
        )
        self.layer5= nn.Sequential(
            conv_basic_dy(512, 1024, stride =1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace = False)
        )
        self.layer6= nn.Sequential(
            conv_basic_dy(1024, 2048, stride =1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace = False)
        )
        self.avgpool =nn.AvgPool1d(2)
        self.dropout = nn.Dropout(0.2)
        classifier1 = []
        classifier1.append(DYCls(512, 2))
        self.classifier1 = nn.Sequential(*classifier1)
        classifier2 = []
        classifier2.append(DYCls(2048, 3))
        self.classifier2 = nn.Sequential(*classifier2)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = x2 = self.layer4(x1)
        x1 = self.avgpool(x1)
        x1 = self.dropout(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.classifier1(x1)
        x2 = self.layer5(x2)
        x2 = self.layer6(x2)
        x2 = self.avgpool(x2)
        x2 = self.dropout(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.classifier2(x2)
        return x2

    

def load_model(path):
    if os.path.exists(path):
        model = torch.jit.load(path, map_location='cpu')
    else:
        model = torch.jit.load('../saipy/saved_models/saved_model.pt', map_location='cpu')
    return model

def arguments():
    parser = argparse.ArgumentParser(description='Phase identification')
    parser.add_argument('--batch_size', action='store', type=int, default=3, help='number of data in a batch')
    parser.add_argument('--lr', action='store', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epochs', action='store', type=int, default=20, help='train rounds over training set')
    parser.add_argument('--num_classes', action='store', type=int, default=3, help='class number')
    parser.add_argument('--patience', action='store', type=int, default=5, help='How many epochs to wait after last time validation loss improved')
    parser.add_argument('--verbose', action='store_true', default=True, help='if True, prints a message for each validation loss improvement')
    parser.add_argument('--model_save_path', action='store', type=str, default='./saving_model', help='the path to save trained model')
    return parser.parse_args("")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)