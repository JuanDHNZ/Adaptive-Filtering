# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:19:53 2021

@author: Juan David
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# PATHS ONLY FOR TESTING 
filename = "../data_4C/NEW_22ch_A01.mat"
data = sio.loadmat(filename)
Xdata = data['X']
labels = data['labels'].reshape(-1,)
fs = 250#int(data['fs'].reshape(-1,))
print('Loading',filename,'with sampling frequency of',fs,'Hz.')
Xdata = np.transpose(Xdata,(2,1,0)) #trials x ch x time


# PATHS ONLY FOR TESTING 
filename = "../data_4C/MAC_filtered/macfBCI_s01train.mat"
data = sio.loadmat(filename)
Xfdata = data['X']
flabels = data['labels'].reshape(-1,)
fs = 250#int(data['fs'].reshape(-1,))
print('Loading',filename,'with sampling frequency of',fs,'Hz.')
Xfdata = np.transpose(Xfdata,(2,1,0)) #trials x ch x time

for ta,tb in zip(Xdata,Xfdata):
    for cha,chb in zip(ta,tb):
        plt.title('Raw Vs. Deniosed')
        plt.plot(cha,label='Raw')
        plt.plot(chb,label='Denoised')
        plt.legend()
        plt.show()