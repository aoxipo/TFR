import torch
import torch.nn as nn
import torch.nn.functional as F

# downsample data and warpper data
# crop data to 8x8 or 16x16 
class DaTa_Warpper(nn.Module):

    def __init__(self):
        super(DaTa_Warpper, self).__init__()
        self.downsample = nn.AvgPool2d(2,2)

    def forward(self, x):
        out = self.downsample(x)
        return out