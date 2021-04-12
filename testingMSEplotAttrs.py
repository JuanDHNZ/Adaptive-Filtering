# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 22:10:17 2021

@author: Juan David
"""


import pandas as pd
folder = 'GridSearchWang2'
attr = 'wang'

QKLMS = pd.read_csv(folder + '/testingMSE_QKLMS_4000-1000_' + attr + '.csv')['MSE']
# AKB = pd.read_csv(folder + '/testingMSE_QKLMS_AKB_4000-1000_' + attr + '.csv')['MSE']
# AMK = pd.read_csv(folder + '/testingMSE_QKLMS_AMK_4000-1000_' + attr + '.csv')['MSE']

QKLMSEX = pd.read_csv(folder + '/testingMSE_QKLMS_4000-1000_' + attr + 'Ex.csv')['MSE']
# AKBEX = pd.read_csv(folder + '/testingMSE_QKLMS_AKB_4000-1000_' + attr + 'Ex.csv')['MSE']
# AMKEX = pd.read_csv(folder + '/testingMSE_QKLMS_AMK_4000-1000_' + attr + 'Ex.csv')['MSE']


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()


inputSizeXrun = 5000
trainSplit = 0.8

plt.figure(figsize=(8,6))
plt.yscale("log")
plt.ylim((1e-3,1e1))
plt.title("Testing MSE") 
plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),QKLMS, linestyle="--", color="magenta",label="QKLMS")
plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),QKLMSEX, linestyle="--", color="cyan",label="ExQKLMS")
plt.legend()
plt.show()

# plt.figure(figsize=(8,6))
# plt.yscale("log")
# plt.ylim((1e-3,1e1))
# plt.title("Testing MSE") 
# plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),AKB, linestyle="--", color="magenta",label="QKLMS_AKB")
# plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),AKBEX, linestyle="--", color="cyan",label="ExQKLMS_AKB")
# plt.legend()
# plt.show()

# plt.figure(figsize=(8,6))
# plt.yscale("log")
# plt.ylim((1e-3,1e1))
# plt.title("Testing MSE") 
# plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),AMK, linestyle="--", color="magenta",label="QKLMS_AMK")
# plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),AMKEX, linestyle="--", color="cyan",label="ExQKLMS_AMK")
# plt.legend()
# plt.show()

# plt.figure(figsize=(8,6))
# plt.yscale("log")
# plt.ylim((1e-3,1e1))
# plt.title("Testing MSE") 
# plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),QKLMS, linestyle="--", color="magenta",label="QKLMS")
# plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),AKB, linestyle="--", color="cyan",label="QKLMS_AKB")
# plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),AMK, linestyle="--", color="purple",label="QKLMS_AMK")
# plt.legend()
# plt.show()





