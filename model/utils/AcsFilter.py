from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary 
from utils import cat_tensor, crop_tensor
import matplotlib.pyplot as plt


class MinPool(nn.Module):
    def __init__(self, kernel_size, ndim=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(MinPool, self).__init__()
        self.pool = getattr(nn, f'MaxPool{ndim}d')(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                                  return_indices=return_indices, ceil_mode=ceil_mode)
    def forward(self, x):
        x = self.pool(-x)
        return -x
@torch.no_grad()
class ApprConSolution(nn.Module):
    # Approximate connected solution
    def __init__(self, scale = 2):
        super().__init__()
        self.up = nn.Upsample(scale_factor = scale)
        self.thre = 0.3# [0 - 0.2 ,0.2 - 0.3 , 0.3 - 1] small middle big
        self.magic_number = 169 # 13 * 13
        self.few_dilate_col = nn.MaxPool2d(kernel_size = (2,1), stride = 1, padding = 0)
        self.few_erode_col = MinPool(kernel_size = (2,1), stride = 1, padding = 0)
        self.few_dilate_raw = nn.MaxPool2d(kernel_size = (1,2), stride = 1, padding = 0)
        self.few_erode_raw = MinPool(kernel_size = (1,2), stride = 1, padding = 0)
        self.dense_dilate = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.dense_erode = MinPool(kernel_size = 2, stride = 1, padding = 0)
    
    def forward_dense(self, x):
        # print("call forward dense")
        # pad first
        
        x_pad = torch.nn.functional.pad(x, (0, 1, 0, 1))
        # erode x
        ans = self.dense_erode(x_pad)
        # up scale
        ans = self.up(ans)
        
        # enchance the max 
        ans = self.dense_dilate(ans)
        ans = self.dense_dilate(ans)
        ans = self.up(ans)
        return ans
        
    def forward_few(self,x):
        #print("call forward few")
        # pad in different direction

        a_pad = torch.nn.functional.pad(x, (0, 0, 1, 1))
        ans = self.few_erode_col(a_pad)
        ans_col = self.few_dilate_col(ans)
        
        a_pad = torch.nn.functional.pad(x, (1, 1, 0, 0))
        ans = self.few_erode_raw(a_pad)
        ans_raw = self.few_dilate_raw(ans)
        return ans_col + ans_raw

    def forward(self,x):
        activate_mask = self.forward_dense(x)
        return activate_mask
    
    def forward(self,x):
        # B,C,W,H = x.shape
        density_rate_list = torch.sum(x, dim = [2,3])/self.magic_number # better more then 14 * 14

        out = []
        for index, density_rate in enumerate(density_rate_list):
            if density_rate < self.thre :
        
                activate_mask = self.forward_few(x[index].unsqueeze(0))
            else:
                activate_mask = self.forward_dense(x[index].unsqueeze(0))
               
            if torch.sum(activate_mask) == 0:
                activate_mask = x[index].unsqueeze(0)
                
            out.append(activate_mask)
        out = torch.cat(out)
        return out


if __name__ == "__main__":
    acs_filter = ApprConSolution()

    # 密集
    a = torch.randn((1,1,14,14))
    a[a<0.5] = 0
    a[a>0.5] = 1
    
    ans = acs_filter(a)
    plt.figure(figsize=(8,8))
    plt.subplot(121)
    plt.imshow(a.squeeze().numpy())
    plt.xlabel(torch.sum(a)/169)
    plt.subplot(122)
    plt.imshow(ans.squeeze().numpy())
    plt.show()

    # 稀疏
    a = torch.randn((1,1,14,14))
    a[a<1] = 0
    a[a>1] = 1
    
    ans = acs_filter(a)
    plt.figure(figsize=(8,8))
    plt.subplot(121)
    plt.imshow(a.squeeze().numpy())
    plt.xlabel(torch.sum(a)/169)
    plt.subplot(122)
    plt.imshow(ans.squeeze().numpy())
    plt.show()

    # 密集 多batch
    a = torch.randn((2,1,14,14))
    a[a<0.5] = 0
    a[a>0.5] = 1
    torch.sum(a, dim = [2,3])/169

    ans = acs_filter(a)
    plt.figure(figsize=(8,8))
    plt.subplot(121)
    plt.imshow(a[0].squeeze().numpy())
    plt.subplot(122)
    plt.imshow(ans[0].squeeze().numpy())
    plt.show()

    plt.figure(figsize=(8,8))
    plt.subplot(121)
    plt.imshow(a[1].squeeze().numpy())
    plt.subplot(122)
    plt.imshow(ans[1].squeeze().numpy())
    plt.show()

    # 稀疏 多batch
    a = torch.randn((2,1,14,14))
    a[a<1] = 0
    a[a>1] = 1
    torch.sum(a,dim=[2,3])/169

    ans = acs_filter(a)
    plt.figure(figsize=(8,8))
    plt.subplot(121)
    plt.imshow(a[0].squeeze().numpy())
    plt.subplot(122)
    plt.imshow(ans[0].squeeze().numpy())
    plt.show()

    plt.figure(figsize=(8,8))
    plt.subplot(121)
    plt.imshow(a[1].squeeze().numpy())
    plt.subplot(122)
    plt.imshow(ans[1].squeeze().numpy())
    plt.show()