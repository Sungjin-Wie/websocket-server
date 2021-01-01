#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,stack_num,n_freq, bilinear=True):
        super(UNet, self).__init__()
        n_hid = 2048
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.stack_num =stack_num
        self.n_freq=n_freq
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down2 = Down(128, 256// factor)
        
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64, bilinear)
        
        self.outc = OutConv(64, n_classes)
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_freq)
        
    def forward(self, x):
        with torch.cuda.amp.autocast():
            drop_p = 0.2
            x1 = self.inc(x)
            x2 = F.dropout(F.relu(self.down1(x1)), p=drop_p, training=self.training)
            x3 = F.dropout(F.relu(self.down2(x2)), p=drop_p, training=self.training)
            x = self.up1(x3, x2)
            x = self.up2(x, x1)
            x = self.outc(x)
            x = x.view(-1,self.stack_num * self.n_freq)
            x = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
            x = self.fc2(x)
            
        return x
    
    
def Load_model():
         # Model
    n_freq = 257
    stack_num = 7
    model = UNet(1,1, stack_num, n_freq)
    device = torch.device('cpu')
    model.load_state_dict(torch.load('/home/ubuntu/server/Inference/Unet_Model_param_cpu.pt',map_location=device))
    
    return model


# In[ ]:




