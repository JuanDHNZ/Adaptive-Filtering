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

ua = signal[-samples-2:-2].reshape(-1,1)
ub = signal[-samples-3:-3].reshape(-1,1)
uc = signal[-samples-4:-4].reshape(-1,1)
u = np.concatenate((ua,ub,uc), axis=1) 
plt.figure()
plt.plot(u)

train_part = 3/4
test_part = 1/4

train_set = int(samples * train_part)
test_set = int(samples * test_part)

u_train = u[:train_set]
u_test = u[train_set:]
d_train = d[:train_set]
d_test = d[train_set:]


filtro = KAF.QKLMS2(epsilon=200, sigma = 1000)
filtro.fit(u_train,d_train)
pred = filtro.predict(u_test)
score = filtro.score(u_test, d_test)

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
epsilon = np.linspace(100,2000,10)
sigma = np.linspace(100,2000,10)

parameters ={'epsilon':epsilon, 'sigma':sigma}

from sklearn.model_selection import GridSearchCV
filtro = KAF.QKLMS2()
cv = [(slice(None), slice(None))]
mqklms = GridSearchCV(filtro,parameters,cv=cv)
mqklms.fit(u,d)

a_results = mqklms.cv_results_

print("Mejores parametros : ", mqklms.best_params_)
print("Mejor score : ", mqklms.best_score_)

# *****************************************
signal = z.reshape(-1,1)
samples = 100

d2 = signal[-samples-1:-1]

ua = signal[-samples-2:-2].reshape(-1,1)
ub = signal[-samples-3:-3].reshape(-1,1)
u2 = np.concatenate((ua,ub), axis=1) 

u2_train = u2[0:74]
u2_test = u2[75:99]
d2_train = d2[0:74]
d2_test = d2[75:99]

flt = KAF.QKLMS2(epsilon = 2000, sigma = 1000)
flt.fit(u2_train,d2_train)
y_pred = flt.predict(u2_test)
scr = flt.score(u2_test,d2_test.reshape(-1,))
print("Score = ", scr)

# plt.plot(y_pred, label='predict')
# plt.plot(d_test, label='target')
# plt.legend()




