
import numpy as np
import scipy.signal as s
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
logger = logging.basicConfig(
        level=logging.DEBUG #设置日志输出格式
        ,filename="./log/predict.log" #log日志输出的文件位置和文件名
        ,filemode="w" #文件的写入格式，w为重新写入文件，默认是追加
        ,format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s" #日志输出的格式
            # -8表示占位符，让输出左对齐，输出长度都为8位
        ,datefmt="%Y-%m-%d %H:%M:%S" #时间输出的格式
)

"""
import pandas as pd
import ephem
import cv2
import astropy.io.fits as pyfits
import os
import sys
from decimal import Decimal
import matplotlib
"""

def plot_data(data, mask=None):
    channels = np.arange(data.shape[1])
#     c = Candidate(fp=fil, dm=dm, tcand=2.0288800, width=64, label=-1, snr=16.8128, min_samp=256, device=0)
#     c.data = data
#     c.dedisperse(target='GPU')
#     data = c.dedispersed
    bandpass = data.mean(0)
    ts = s.detrend( data.mean(1))
    
    if mask is not None:
        data[:, mask] = 0

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])

    ax0 = plt.subplot(gs[1, 0])
    ax0.imshow(data.T, aspect="auto", interpolation=None)
    ax0.set_xlabel("Time Samples")
    ax0.set_ylabel("Frequency Channels")

    ax1 = plt.subplot(gs[1, 1], sharey=ax0)
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.plot(bandpass, channels, "k")
    ax1.set_xlabel("Flux (Arb. Units)")
    
    ax2 = plt.subplot(gs[0, 0], sharex=ax0)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.plot(ts, "k")
    ax2.set_ylabel("Flux (Arb. Units)")

    if mask is not None:
        for channel, val in zip(channels[mask], bandpass[mask]):
            ax0.axhline(channel, color="r", xmin=0, xmax=0.03, lw=0.1)
            ax1.scatter(val, channel, color="r", marker="o")

    plt.tight_layout()
    plt.show()
    return  bandpass, ts
    



class waterflow():
    def __init__(self,data_map):
        self.clear(data_map)

    def clear(self,data_map):
        self.data_map = data_map
        self.next_step = np.array([
            [0,1],
            [1,0],
            [1,1],
            [-1,0],
            [0,-1],
            [-1,-1],
            [-1,1],
            [1,-1],
        ],dtype = np.int)
        self.block_dict = {"map":np.zeros((10,10))}
        self.block_dict["data"] = data_map
    
    def findmax(self, crop_image_list, show = True):
        image = np.zeros((10,10))
        max1 = 0
        r_number = 0 
        number = 1
        ans_vector = []
        for i in crop_image_list:
            row_256_s, row_256_e, col_256_s, col_256_e = i["coord"]
            row_256_s = row_256_s%2048
            x = int(row_256_s/256) +1
            y = int(col_256_s/256) +1
            self.findmaxgroup(number, x, y)
            number += 1

        for i in self.block_dict:
            if( i == "map" or i == "data"):
                continue
            total = len(self.block_dict[i])
            if(max1 < total):
                max1 = total
                r_number = i

        for i in crop_image_list:
            row_256_s, row_256_e, col_256_s, col_256_e = i["coord"]
            row_256_s = row_256_s%2048
            x = int(row_256_s/256) +1
            y = int(col_256_s/256) +1
            if((x,y) in self.block_dict[r_number]):
                ans_vector.append(1)
                image[x][y]  = 1
            else:
                ans_vector.append(0)
           
        if(show):
            plt.imshow(image)
            plt.show()
        return ans_vector 

    def findmaxgroup(self, number ,x, y):
        if(self.block_dict["map"][x][y]):
            return 
        self.block_dict["map"][x][y] = 1
        if(number in self.block_dict):
            if((x,y) not in  self.block_dict[number]):
                self.block_dict[number].append((x,y))
        else:
            self.block_dict[number] = []
            self.block_dict[number].append((x,y))
            
        for i in range(len(self.next_step)):
            next_x = x + self.next_step[i][0]
            next_y = y + self.next_step[i][1]
            if( 1 <= next_x and next_x < 10 and 1 <= next_y and next_y < 10 and self.block_dict["data"][next_x][next_y]):
                self.findmaxgroup(number, next_x, next_y) 