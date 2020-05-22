# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:56:18 2020

@author: dcard
"""


import numpy as np
import matplotlib.pyplot as plt
import KAF

X1 = np.random.multivariate_normal([1,1], [[5,-8],[-8,20]],size=200)
X2 = np.random.multivariate_normal([-15,-15], [[20,8],[8,5]],size=200)
X = np.concatenate((X1,X2),axis=0)

np.random.shuffle(X)

y = X[1:-1,1].reshape(-1,1)

X = X[0:-2,:]


print(np.linalg.eig(np.cov(X,rowvar=False))[0])

print(np.cov(X,rowvar=False))

#Modo offline
filtro = KAF.QKLMS(epsilon=20)

filtro.evaluate(X,y)

plt.figure(figsize=(10,10))
plt.scatter(X[:,0],X[:,1])
for cent in filtro.CB:
    plt.scatter(cent[0],cent[1],color='red')

plt.xlim([-30,14])
plt.ylim([-30,14])

#Mode online
filtro = KAF.QKLMS(epsilon=20)
for l in range(0,1000,100):
    filtro.evaluate(X[l:l+100,:],y[l:l+100])
    
    plt.figure(figsize=(10,10))
    plt.scatter(X[:,0],X[:,1])
    for cent in filtro.CB:
        plt.scatter(cent[0],cent[1],color='red')
    
    plt.xlim([-30,12])
    plt.ylim([-30,12])