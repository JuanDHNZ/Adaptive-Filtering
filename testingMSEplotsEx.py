# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 23:42:38 2021

@author: Juan David
"""

import pandas as pd

folder = 'GridSearchResults3rdRun'
QKLMS = pd.read_csv(folder + '/testingMSE_QKLMS_4000-1000_lorenz.csv')['MSE']
AKB = pd.read_csv(folder + '/testingMSE_QKLMS_AKB_4000-1000_lorenz.csv')['MSE']
AMK = pd.read_csv(folder + '/testingMSE_QKLMS_AMK_4000-1000_lorenz.csv')['MSE']

folder = 'GridSearchEx'
QKLMS_2 = pd.read_csv(folder + '/testingMSE_QKLMS_4000-1000_lorenz.csv')['MSE']
AKB_2 = pd.read_csv(folder + '/testingMSE_QKLMS_AKB_4000-1000_lorenz.csv')['MSE']
AMK_2 = pd.read_csv(folder + '/testingMSE_QKLMS_AMK_4000-1000_lorenz.csv')['MSE']


inputSizeXrun = 5000
trainSplit = 0.8

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

plt.figure(figsize=(8,6))
plt.yscale("log")
plt.ylim((1e-3,1e1))
plt.title("Testing MSE") 
plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),QKLMS, linestyle="--", color="magenta",label="QKLMS")
plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),QKLMS_2, linestyle="--", color="cyan",label="ExQKLMS")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.yscale("log")
plt.ylim((1e-3,1e1))
plt.title("Testing MSE") 
plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),AKB, linestyle="--", color="magenta",label="QKLMS_AKB")
plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),AKB_2, linestyle="--", color="cyan",label="ExQKLMS_AKB")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.yscale("log")
plt.ylim((1e-3,1e1))
plt.title("Testing MSE") 
plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),AMK, linestyle="--", color="magenta",label="QKLMS_AMK")
plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),AMK_2, linestyle="--", color="cyan",label="ExQKLMS_AMK")
plt.legend()
plt.show()
