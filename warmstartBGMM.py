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


"""
Variational Bayesian Gaussian Mixture Model using warmstar=True

Batch training is needed
"""

batch_size = 100
n_batches = samples/batch_size
bgm = BGMM(n_components=batch_size-1, weight_concentration_prior=1e-3, warm_start=True)

batch_index = 0


while True:
    print(u[batch_index:batch_index+batch_size-1])
    bgm.fit(u[batch_index:batch_index+batch_size-1])
    batch_index += 100
    if batch_index >= samples:
        break
    
labels = bgm.predict(u)

import matplotlib.pyplot as plt
plt.scatter(u[:,0],u[:,1], c=labels, cmap="Set1")   
print(np.unique(labels))


