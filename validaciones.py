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
filtro = KAF.QKLMS2(epsilon=20)

filtro.evaluate(X,y)

plt.figure(figsize=(10,10))
plt.scatter(X[:,0],X[:,1])
for cent in filtro.CB:
    plt.scatter(cent[0],cent[1],color='red')

plt.xlim([-30,14])
plt.ylim([-30,14])

#Mode online
filtro = KAF.QKLMS2(epsilon=20)
for l in range(0,1000,100):
    filtro.evaluate(X[l:l+100,:],y[l:l+100])
    
    plt.figure(figsize=(10,10))
    plt.scatter(X[:,0],X[:,1])
    for cent in filtro.CB:
        plt.scatter(cent[0],cent[1],color='red')
    
    plt.xlim([-30,12])
    plt.ylim([-30,12])

#******************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import KAF

X1 = np.random.multivariate_normal([1,1], [[5,-8],[-8,20]],size=200)
X2 = np.random.multivariate_normal([-15,-15], [[20,8],[8,5]],size=200)
X = np.concatenate((X1,X2),axis=0)

np.random.shuffle(X)

X_ant = X[0:-2,:]
y = X[1:-1,1].reshape(-1,1)

X_ant.shape
y.shape
l = 0
filtro = KAF.QKLMS2(epsilon=20)
for l in range(0,200,50):
    out = filtro.evaluate(X_ant[l:l+50,:],y[l:l+50])
    # print(out)
    
    plt.figure(figsize=(10,10))
    plt.scatter(X_ant[:,0],X_ant[:,1])
    for cent in filtro.CB:
        plt.scatter(cent[0],cent[1],color='red')
        
        
print("Codebook size:", len(filtro.CB))
filtro.testCB_means

filtro = KAF.QKLMS2(epsilon=20)
entrada = X_ant[0:50,:]
salida_d = y[0:50]
out = filtro.evaluate(entrada,salida_d)
oit2 = filtro.evaluate(entrada,salida_d)


filtro = KAF.QKLMS2(epsilon=20)
for l in range:
    out = filtro.evaluate(X_ant[l],y[l])
    # print(out) 
    
plt.figure(figsize=(10,10))
plt.scatter(X_ant[:,0],X_ant[:,1])
for cent in filtro.testCB_means:
    plt.scatter(cent[0],cent[1],color='red')

import pandas as pd
import numpy as np
import KAF
import matplotlib.pyplot as plt

sp500 = pd.read_csv("spx.csv")

# Señal deseada
d = sp500.close.iloc[-366:-1].to_numpy().reshape(-1,1)
# Señal de entrada
u1 = sp500.close.iloc[-367:-2].to_numpy().reshape(-1,1)
u2 = sp500.close.iloc[-368:-3].to_numpy().reshape(-1,1)
u3 = sp500.close.iloc[-369:-4].to_numpy().reshape(-1,1)
u4 = sp500.close.iloc[-370:-5].to_numpy().reshape(-1,1)
u5 = sp500.close.iloc[-371:-6].to_numpy().reshape(-1,1)
u = np.concatenate((u1,u2,u3,u4,u5), axis=1)
#filtro = KAF.QKLMS2(epsilon=200,sigma=1000)
filtro = KAF.QKLMS2(epsilon=100000,sigma=1000)
out = filtro.evaluate(u,d)    
plt.plot(out[:-2],label="Predicted")
plt.plot(d,label="Target")
plt.legend()
print("CB:",len(filtro.CB))
print("Sigma:", filtro.sigma)

plt.plot(filtro.CB_growth)

filtro.testDists


import pandas as pd
import numpy as np
import KAF
import matplotlib.pyplot as plt

sp500 = pd.read_csv("spx.csv")

# Señal deseada
d = sp500.close.iloc[-366:-1].to_numpy().reshape(-1,1)
# Señal de entrada
u1 = sp500.close.iloc[-367:-2].to_numpy().reshape(-1,1)
u2 = sp500.close.iloc[-368:-3].to_numpy().reshape(-1,1)
u3 = sp500.close.iloc[-369:-4].to_numpy().reshape(-1,1)
u4 = sp500.close.iloc[-370:-5].to_numpy().reshape(-1,1)
u5 = sp500.close.iloc[-371:-6].to_numpy().reshape(-1,1)
u = np.concatenate((u1,u2,u3,u4,u5), axis=1)
#filtro = KAF.QKLMS2(epsilon=200,sigma=1000)
filtro = KAF.QKLMS(epsilon=100000,sigma=1000)
out = filtro.evaluate(u,d)    
plt.plot(out[:-2],label="Predicted")
plt.plot(d,label="Target")
plt.legend()
print("CB:",len(filtro.CB))
print("Sigma:", filtro.sigma)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(4, 4.2)
max(out)
