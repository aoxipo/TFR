import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
from torch.autograd import Variable
from torchsummary import summary
import torch.nn as nn
class GACNN(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(GACNN, self).__init__( )

        self.Flatten = torch.nn.Sequential(  
            torch.nn.Flatten()
        )
        self.attention = Gobal_Attention(100, 100, out_channels)
        # self.BN = torch.nn.Sequential(
        #     torch.nn.BatchNorm1d(100)
        # )
        self.dense = torch.nn.Sequential( 
            torch.nn.Linear(100,64),
            torch.nn.ReLU(),
        )
        self.dense2 = torch.nn.Sequential( 
            torch.nn.Linear(64,32),
            torch.nn.ReLU(),
        )
        self.dense3 = torch.nn.Sequential( 
            torch.nn.Linear(32, out_channels),
            torch.nn.ReLU(),
        )

    def summary(self):
        summary(self, (1, 10, 10))
    def forward(self, x):
        #print(x.size())
        x = self.Flatten(x)
        #print(x.size())
        #x = self.BN(x)
        #print(x.size())
        p = self.attention(x)
        x = self.dense(x)
        
        x = self.dense2(x)
        
        x = self.dense3(x)
        
        x =  torch.mul(x,p)
        return x


class Gobal_Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, out_channels):
        super(Gobal_Attention, self).__init__()
        self.attn = nn.Linear(enc_hid_dim , dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, out_channels, bias = False)
        self.exp = torch.exp
        
    def forward(self, enc_output):
        energy = torch.tanh(self.attn( enc_output ))
        attention = self.v(energy)
        exp_a =  self.exp(attention)
        exp_a = exp_a/torch.sum(exp_a)
        return exp_a

class Attention_average(nn.Module):
    def __init__(self, sequence, img_dim):
        super(Attention_average, self).__init__()
        self.sequence = sequence
        self.img_dim = img_dim

    def forward(self, x):
        output = self.pooling(x).view(-1, self.sequence, self.img_dim)
        return output

    def pooling(self, x):
        output = torch.mean(torch.mean(x, dim=3), dim=2)
        return output

#####
## To Do .
#####
class Local_Attention(nn.Module):
    pass
    
class CNN(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__( )

        middle = 18*18
        self.Flatten = torch.nn.Sequential(  
            torch.nn.Flatten()
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(middle,out_channels),
            torch.nn.ReLU(),
        )
        self.BN = torch.nn.Sequential(
            torch.nn.BatchNorm1d(middle)
        )
        self.dense = torch.nn.Sequential( 
            torch.nn.Linear(middle,128),
            torch.nn.ReLU(),
        )
        self.dense2 = torch.nn.Sequential( 
            torch.nn.Linear(128,64),
            torch.nn.ReLU(),
        )
        self.dense3 = torch.nn.Sequential( 
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
        )
        self.dense4 = torch.nn.Sequential( 
            torch.nn.Linear(32, out_channels),
            torch.nn.ReLU(),
        )

    def summary(self):
        summary(self, (1, 18, 18))
    def forward(self, x):
        x = self.Flatten(x)
        x = self.BN(x)
        p = self.attention(x)
        x = self.dense(x)
        
        x = self.dense2(x)
        
        x = self.dense3(x)
        x = self.dense4(x)
        x =  torch.mul(x,p)
        return x


class SMCNN(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(SMCNN, self).__init__( )
        
        self.Flatten = torch.nn.Sequential(  
            torch.nn.Flatten()
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(100,out_channels),
            torch.nn.ReLU(),
        )
        self.BN = torch.nn.Sequential(
            torch.nn.BatchNorm1d(100)
        )
        self.dense = torch.nn.Sequential( 
            torch.nn.Linear(100,64),
            torch.nn.ReLU(),
        )
        self.dense2 = torch.nn.Sequential( 
            torch.nn.Linear(64,32),
            torch.nn.ReLU(),
        )
        self.dense3 = torch.nn.Sequential( 
            torch.nn.Linear(32, out_channels),
            torch.nn.ReLU(),
        )

    def summary(self):
        summary(self, (1, 10, 10))
    def forward(self, x):
        x = self.Flatten(x)
        x = self.BN(x)
        p = self.attention(x)
        x = self.dense(x)
        
        x = self.dense2(x)
        
        x = self.dense3(x)
        
        x =  torch.mul(x,p)
        return x

if __name__ == '__main__':
    model = CNN(3, 8)
    model.summary() 