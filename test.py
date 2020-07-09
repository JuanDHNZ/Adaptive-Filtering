# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:33:28 2020

@author: Juan David
"""
# Attractors tester

# attr = ["chua","lorenz","duffing","nose_hoover","rikitake","rossler","wang"]
# import TimeSeriesGenerator as tsg

# for sys in attr:
#     x, y, z = tsg.chaoticSystem(samples=1000,systemType=sys)
    
    
# ***********************************
    
import numpy as np 
import KAF 
import matplotlib.pyplot as plt
import scipy.io as sp
import search
import TimeSeriesGenerator as tsg
from sklearn.model_selection import train_test_split

x, y, z = tsg.chaoticSystem(samples=1000,systemType="chua")
signal = x.reshape(-1,1)
samples = 400

d = signal[-samples-1:-1]

u = signal[-samples-2:-2].reshape(-1,1)
# ub = signal[-samples-3:-3].reshape(-1,1)
# uc = signal[-samples-4:-4].reshape(-1,1)
# u = np.concatenate((ua,ub,uc), axis=1) 
plt.figure()
plt.plot(u)

u_train, u_test, d_train, d_test = train_test_split(u, d, test_size=1/4, random_state=42)

filtro = KAF.QKLMS2(epsilon=200, sigma = 1000)
filtro.fit(u_train,d_train)
pred = filtro.predict(u_test, d_test)
score = filtro.score(d_test, pred)

plt.figure()
plt.plot(pred, label="predict")
plt.plot(d_test, label="target")
plt.legend()
plt.title("Chua's attractor random test set")

# plt.plot(ep,cb_size)
print("Tamaño de set de entrenamiento = ", u_train.shape)
print("Tamaño de set de prueba = ", u_test.shape)
print("Tamaño del codebook = ", len(filtro.CB))
print("SCORE = ", score)

# GRID SEARCH
epsilon = np.linspace(100,2000,2)
sigma = np.linspace(100,2000,2)
parameters ={'epsilon':epsilon, 'sigma':sigma}

from sklearn.model_selection import GridSearchCV
filtro = KAF.QKLMS2()
mqklms = GridSearchCV(filtro,parameters)
mqklms.fit(u,d)
print(mqklms.best_score_)

import pandas as pd
df = pd.DataFrame(mqklms.cv_results_)
df.head(-1)



