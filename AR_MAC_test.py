# -*- coding: utf-8 -*-
"""
Created on Sun May  9 10:36:41 2021

@author: Juan David
"""
import numpy as np
import scipy.io as sio
from sklearn.pipeline import Pipeline

import KAF 
from mac import MAC

ths = [0.144004463,0.15777807,0.148215102,0.08950537,0.147071399,0.172103277,0.134589128,0.168674552,0.077062943]

#%%
 
for n,th in enumerate(ths):
    print(n, th)
# PATHS ONLY FOR TESTING 
    filename = "../BCI/NEW_22ch_A0{}.mat".format(n+1)
    savename = "../BCI/MAC_filtered/macfBCI_s0{}train.mat".format(n+1)
    
    data = sio.loadmat(filename)
    Xdata = data['X']
    labels = data['labels'].reshape(-1,)
    # fs = int(data['fs'].reshape(-1,))
    fs = 250
    print('Loading',filename,'with sampling frequency of',fs,'Hz.')
    Xdata = np.transpose(Xdata,(2,1,0)) #trials x ch x time
    
    #%%
    from OA import artifact_removal_with_MAC as ar_mac
    
    ar = ar_mac(th=th)
    Xclean = ar.fit_transform(Xdata,labels)
    
    # Save clean BCI data
    cleanEEG = np.transpose(Xclean,(2,1,0))
    new_filename = 'filtered' + filename
    new_data = {'X': cleanEEG,
                'labels': data['labels'],
                'fs': fs}
    sio.savemat(savename, new_data)