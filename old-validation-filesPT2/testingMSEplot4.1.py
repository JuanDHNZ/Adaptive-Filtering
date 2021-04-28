# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:02:52 2021

@author: Juan David
"""

import pandas as pd
folder = 'GridSearch4.1'
QKLMS = pd.read_csv(folder + '/testingMSE_QKLMS_4000-1000_41.csv')['MSE']
AKB = pd.read_csv(folder + '/testingMSE_QKLMS_AKB_4000-1000_41.csv')['MSE']
AMK = pd.read_csv(folder + '/testingMSE_QKLMS_AMK_4000-1000_41.csv')['MSE']


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()


inputSizeXrun = 5000
trainSplit = 0.8

plt.figure(figsize=(8,6))

plt.yscale("log")
plt.ylim((1e-5,1e1))
plt.title("Testing MSE on 4.1 system") 
plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),QKLMS, linestyle="--", color="magenta",label="QKLMS")
plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),AKB, linestyle="--", color="cyan",label="QKLMS_AKB")
plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),AMK, linestyle="--", color="purple",label="QKLMS_AMK")
plt.legend()
plt.show()


