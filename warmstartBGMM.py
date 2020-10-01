# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:38:38 2020

@author: USUARIO
"""


import numpy as np
from sklearn.mixture import BayesianGaussianMixture as BGMM 
import TimeSeriesGenerator as tsg
import matplotlib.pyplot as plt

samples = 400
"""ATRACTOR DE LORENZ"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="lorenz")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

"""
Offline test for comparison purpuses

"""
bgm_off = BGMM(n_components=samples, weight_concentration_prior=1e-3)
bgm_off.fit(u)
labels_off = bgm_off.predict(u)
plt.scatter(u[:,0],u[:,1], c=labels_off, cmap="Set1")
print('Prediction unique labels = ',np.unique(labels_off))
print('score = ', bgm_off.score(u))

"""
Variational Bayesian Gaussian Mixture Model using:
    warmstar=True
    max_iter=10
    
"""

scr = []
bgm = BGMM(n_components=samples, weight_concentration_prior=1e-3, warm_start=True, max_iter=10)
for i in range(100):
    bgm.fit(u)
    scr.append(bgm.score(u))
plt.figure(figsize=(16,9))     
plt.plot(scr,'b',label='score',markersize=8)
plt.ylabel('LL')
plt.xlabel('iteraciones')
plt.legend()
plt.grid()
plt.show()



"""
Por batches
"""

scr = [] #score

bgm = BGMM(n_components=samples, weight_concentration_prior=1e-3, warm_start=True, max_iter=10)

u_batch = u
np.random.shuffle(u_batch)
for i in range(100):
    bgm.fit(u)
    scr.append(bgm.score(u))
plt.figure(figsize=(16,9))     
plt.plot(scr,'m',label='score',markersize=8)
plt.ylabel('LL')
plt.xlabel('iteraciones')
plt.legend()
plt.grid()
plt.show()


"""Otras pruebas"""

batch_size = 100 #Tamaño del batch
n_batches = int(samples/batch_size) # No. de batches

bgm = BGMM(n_components=batch_size, weight_concentration_prior=1e-3, warm_start=True, max_iter=10)

scr = []
u_batch = u
for k in range(100):
    np.random.shuffle(u_batch)
    batch_index = 0
    for i in range(n_batches):
        bgm.fit(u_batch[batch_index:batch_index+batch_size])
        batch_index += batch_size
    scr.append(bgm.score(u_batch))

    
    
plt.figure(figsize=(16,9))     
plt.plot(scr,'r',label='score',markersize=8)
plt.ylabel('LL')
plt.xlabel('iteraciones')
plt.legend()
plt.grid()
plt.show()

