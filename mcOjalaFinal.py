# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:19:21 2021

@author: dcard
"""

import TimeSeriesGenerator as tsg
import matplotlib.pyplot as plt

from tqdm import tqdm

import numpy as np
import comPlot as cp

from KAF import QKLMS

L = 5
db = 'lorenz'

R = 10 #MC repetions
tsLength = 1000

trainLength = int(tsLength*0.7)
testLength = tsLength - trainLength

samples = R*tsLength

sigma = 1
eps = 1e-4

scale = 5 #For plotting purposes

x, y, z = tsg.chaoticSystem(samples=samples+L,systemType=db)

x -= x.mean()
x /= x.std()

mse = []

for r in range(R):
    #print(L+r*tsLength,L+r*tsLength+trainLength,L+r*tsLength+trainLength,L+(r+1)*tsLength)
    u_train = np.array([x[i-L:i] for i in range(L+r*tsLength,L+r*tsLength+trainLength)])
    d_train = np.array([x[i] for i in range(L+r*tsLength,L+r*tsLength+trainLength)]).reshape(-1,1)
    
    u_test = np.array([x[i-L:i] for i in range(L+r*tsLength+trainLength,L+(r+1)*tsLength)])
    d_test = np.array([x[i] for i in range(L+r*tsLength+trainLength,L+(r+1)*tsLength)]).reshape(-1,1)
 
   
    print('Input',u_train.shape,'Output',d_train.shape)
    
    mse_r = []
    f = QKLMS(sigma=sigma,epsilon=eps) 
    for i in tqdm(range(len(u_train))):
        ui,di=u_train[i],d_train[i]
        f.evaluate(ui,di)
        
        if np.mod(i,scale)==0:            
            y_pred = f.predict(u_test)     
            err = d_test-y_pred
            mse_r.append(np.mean(err**2))
            
    mse.append(np.array(mse_r))
    
    signalPower = x.var()
    
    mse_mean = np.mean(np.array(mse),axis=0)/signalPower
    mse_std  = np.std(np.array(mse),axis=0)/signalPower
       
    plt.figure(figsize=(15,9))
    plt.title("MSE RELATIVO - Lorenz - $\sigma$ = {} ; epsilon = {} ; Reps = {}".format(sigma,eps,r))
    plt.yscale("log")
    plt.ylim( (1e-2,1e1))    
    plt.fill_between(range(0,len(u_train),scale),mse_mean-mse_std,mse_mean+mse_std,alpha=0.5)
    plt.plot(range(0,len(u_train),scale),mse_mean)
    plt.ylabel("Testing MSE")
    plt.xlabel("iterations")
    #plt.savefig("Montecarlo1000/"+ "ER_lorenz" +".png", dpi = 300)
    plt.show()    