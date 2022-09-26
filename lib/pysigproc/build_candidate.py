
from .candidate import Candidate
import numpy as np
#from scipy.signal import detrend
#import bin.h5plotter
import matplotlib.pyplot as plt
#import logging
#import math
import datetime
#import cv2
#import os
import io
#import h5py
from PIL import Image

def convert_to_image(data, show = False):
    data_ft = data
    data_ft /= np.std(data_ft)
    data_ft -= np.median(data_ft)
    dst = data_ft.mean(1)

    dst = dst.reshape((256))
    
    fig=plt.figure()
    plt.plot(dst)
    plt.axis('off') # 关掉坐标轴为 off
          
    fig.set_size_inches(312/100,312/100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # plt.gca()表示获取当前子图"Get Current Axes"。
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    img = np.array(Image.open(buffer))
    
    #
    if(show):
        plt.imshow(img)
        plt.show()
        print(img[:,:,:3].shape)
    buffer.close()
    return img[:,:,:3]

def process_data(fpath, dm, label, show = False):
    cand = Candidate(fp=fpath, dm=dm, label=label, snr=88.694, tcand=0.826, width=2**1 , data_source = "国台天文台" )
    cand.get_chunk()
    cand.dmtime()
    cand.dedisperse()
    if(show):
        plt.imshow(cand.dedispersed.T)
        plt.show()
    img = convert_to_image(cand.dedispersed.T, show)
    ans = {"data_dm_time": cand.dmt, "data_freq_time": cand.dedispersed.T, "data_sum_time": img, "name": fpath.split('/')[-1]}
    del cand
    return ans

def save_as_h5(file_path_list, f = None):
    data_list = []
    count = 0;
    total = len(file_path_list)
    show = True
    for file_path in file_path_list:
        count += 1
        stime = datetime.datetime.now()
        if(count % 50 == 0):
            data_dict = process_data(file_path, show)
        else:
            data_dict = process_data(file_path)
        etime = datetime.datetime.now()
        data_list.append(data_dict)
        if(count % 50 ==0):
            print("index:", count," process: ", count/total * 100,"%","ETA:",  str((etime-stime)*total) )
        break;
        #return data_dict
    print(data_list)
    if(f != None):
        f.create_dataset("data", data=data_list)