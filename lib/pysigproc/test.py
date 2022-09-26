# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 10:37:42 2021

@author: Administrator
"""

from candidate import Candidate
import numpy as np
from scipy.signal import detrend
import bin.h5plotter
import matplotlib.pyplot as plt
import logging
import math
import cv2
fpath = f'F:/fast/data/askap_frb_180417/28.fil'
fpath = "F:\\fast\\data\\askap_frb_180417\\29.fil"
fpath = "C:Users\\Administrator\\Desktop\\fast\\fast\\FRB180417.fil"
fpath = "I:/1159.fits"

logger = logging.getLogger()
logger = logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s -'
                                                        ' %(message)s')
#cand = Candidate(fp=fpath, dm=56.838, label=1, snr=88.694, tcand=0.826, width=2**1)
cand = Candidate(fp=fpath, dm=56.838, label=1, snr=88.694, tcand=0.826, width=2**1 , data_source = "国台天文台" )

cand.get_chunk()
#print(cand.data)
#plt.imshow( cand.data_origin )
#plt.show()
#plt.imshow( cand.data )
#plt.show()
#print(cand.get_snr())
cand.dmtime()
print( cand.dmt )
plt.imshow( cand.dmt,aspect="auto" )
plt.show()

cand.dedisperse()
plt.imshow( cand.dedispersed.T, aspect="auto")
plt.show()

plt.imshow( cand.data.T, aspect="auto" )
plt.show()

#print(cand.get_snr())

# print(cand.data, cand.data.shape,cand.dtype)



#cand = Candidate(fp=fpath, dm=56.838, label=1, snr=88.694, tcand=0.826, width=2**1)

#cand.get_chunk()
#print(cand.data)
#print(cand.get_snr())
#cand.dmtime()

def psnr(img1, img2):
    mse = np.mean( (img1/1.0 - img2/1.0)**2 )
    if(mse < 1.0e-10):
        return 100
    return 10*math.log10(255.0**2/mse)

print(psnr(cand.data_origin, cv2.resize(cand.data, (2048,2048), cv2.INTER_NEAREST)))

