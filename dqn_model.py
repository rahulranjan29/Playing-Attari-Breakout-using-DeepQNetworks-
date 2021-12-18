# -*- coding: utf-8 -*-
"""
Created on Mon May 25 23:16:26 2020

@author: Rahul Verma
"""


import torch
import torch.nn as nn
#import math
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self,in_channels,num_actions):
        super(DQN,self).__init__()
        
        self.conv1=nn.Conv2d(in_channels,32,kernel_size=8,stride=4)
        #elf.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,64,kernel_size=4,stride=2)
        #self.bn2=nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(64,64,kernel_size=3,stride=1)
       # self.bn3=nn.BatchNorm2d(64)
        
        self.fc1=nn.Linear(7*7*64,512)
        self.head=nn.Linear(512,num_actions)
        #self.init_weights()
        
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=F.relu(self.fc1(x.contiguous().view(x.size(0),-1)))
        return self.head(x)
    
    def init_weights(m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.uniform(m.weight, -0.01, 0.01)
            m.bias.data.fill_(0.01)

#if __name__=="__main__":
 #   k=torch.rand(1,4,84,84)
  #  model=DQN(4,8)
   # output=model(k)
    #print(output.size())
    
    
    