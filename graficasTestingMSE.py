# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 16:48:17 2021

@author: Juan David
"""

import pandas as pd
folder = 'GridSearchResults3rdRun'
QKLMS = pd.read_csv(folder + '/testingMSE_QKLMS_4000-1000_42.csv')['MSE']
AKB = pd.read_csv(folder + '/testingMSE_QKLMS_AKB_4000-1000_42.csv')['MSE']
AMK = pd.read_csv(folder + '/testingMSE_QKLMS_AMK_4000-1000_42.csv')['MSE']


inputSizeXrun = 5000
trainSplit = 0.8

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()
plt.figure(figsize=(8,6))
plt.yscale("log")
plt.ylim((1e-30,1e1))
plt.title("Testing MSE") 
plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),QKLMS, linestyle="--", color="magenta",label="QKLMS")
plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),AKB, linestyle="--", color="cyan",label="QKLMS_AKB")
plt.plot(np.linspace(0,int(inputSizeXrun*trainSplit),800),AMK, linestyle="--", color="lightgreen",label="QKLMS_AMK")
plt.legend()
plt.show()



# Predictions
from test_on_KAF import selectBestResultFromKafSearch
folder = 'GridSearchResults3rdRun'
QKLMS_lorenz = 'QKLMS_lorenz_5000.csv'
AKB_lorenz = 'QKLMS_AKB_lorenz_5000.csv'
AMK_lorenz = 'QKLMS_AMK_lorenz_5000.csv'

QKLMS_42 = 'QKLMS_4.2_5000.csv'
AMK_42 = 'QKLMS_AMK_4.2_5000.csv'
AKB_42 = 'QKLMS_AKB_4.2_5000.csv'

params = selectBestResultFromKafSearch(folder + '/' + AMK_42)

import KAF
import TimeSeriesGenerator

n_samples = 5000
n_train = int(n_samples*0.8)

'''lorenz'''
# embedding = 5
# x, y, z = TimeSeriesGenerator.chaoticSystem(samples=n_samples+embedding,systemType='lorenz')
# x -= x.mean()
# x /= x.std()

'''4.2'''
embedding = 2
import testSystems as ts
x = ts.testSystems(samples = n_samples+embedding, systemType = "4.2_AKB")
x -= x.mean()
x /= x.std()

u_train = np.array([x[i-embedding:i] for i in range(embedding,n_train)])
d_train = np.array([x[i] for i in range(embedding,n_train)]).reshape(-1,1)
      
u_test = np.array([x[i-embedding:i] for i in range(n_train,n_samples)])
d_test = np.array([x[i] for i in range(n_train,n_samples)]).reshape(-1,1)

# f = KAF.QKLMS(eta=params['eta'], epsilon=params['epsilon'], sigma=params['sigma'])
# f = KAF.QKLMS_AKB(eta=params['eta'], epsilon=params['epsilon'], sigma_init= params['sigma_init'], mu=params['mu'], K=int(params['K']))
f = KAF.QKLMS_AMK(eta=params['eta'], epsilon=params['epsilon'], mu=params['mu'], Ka=int(params['K']),A_init="pca")
y_train = f.evaluate(u_train, d_train)
y_pred = f.predict(u_test)

plt.figure(figsize=(8,6))
plt.title("QKLMS AMK predict on test split") 
plt.plot(y_pred, linestyle="--", color="red",label="predict")
# plt.plot(d_test, linestyle="--", color="green",label="target")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.title("QKLMS AMK predict on train split")
plt.plot(y_train, linestyle="--", color="red",label="predict")
plt.plot(d_train, linestyle="--", color="green",label="target")
plt.legend()
plt.show()

print("CB = {}".format(len(f.CB)))


err = d_test-y_pred.reshape(-1,1)
mse = np.mean(err**2)
signalPower = x.var()
print(mse/signalPower)

# plt.plot(d_test - y_pred)


"""Codebook vs MSE"""
from test_on_KAF import selectBestResultFromKafSearch
'''lorenz'''
# folder = 'GridSearchResults2ndRun'
# QKLMS_lorenz = 'QKLMS_lorenz_5000.csv'
# AKB_lorenz = 'QKLMS_AKB_lorenz_5000.csv'
# AMK_lorenz = 'QKLMS_AMK_lorenz_5000.csv'


folder = 'GridSearchResults3rdRun'
QKLMS_lorenz = 'QKLMS_4.2_5000.csv'
AKB_lorenz = 'QKLMS_AKB_4.2_5000.csv'
AMK_lorenz = 'QKLMS_AMK_4.2_5000.csv'

plt.figure(figsize=(8,8))
plt.title("Codebook vs MSE")
QKLMS = pd.read_csv(folder + "/" + QKLMS_lorenz)
QKLMS_AKB = pd.read_csv(folder + "/" + AKB_lorenz)
QKLMS_AMK = pd.read_csv(folder + "/" + AMK_lorenz)
plt.scatter(QKLMS['CB_size']/n_train, QKLMS['testing_mse'], marker='*',label='QKLMS')
plt.scatter(QKLMS_AKB['CB_size']/n_train, QKLMS_AKB['testing_mse'], marker='X',label='QKLMS_AKB')
plt.scatter(QKLMS_AMK['CB_size']/n_train, QKLMS_AMK['testing_mse'], marker='>', label='QKLMS_AMK')
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.xlabel('Codebook')
plt.ylabel('Testing MSE')
plt.ylim([1e-30,1e1])
plt.xlim([1e-4,1e1])
plt.show()

d_opt = (QKLMS_AMK['CB_size']/n_train)**2 + QKLMS_AMK['testing_mse']**2
d_min = np.argmin(d_opt)

