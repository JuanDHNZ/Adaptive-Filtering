# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:22:35 2020

@author: Juan David
"""


#Validaciones extra
import KAF
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

X1 = np.random.multivariate_normal([1,1], [[5,-8],[-8,20]],size=200)
X2 = np.random.multivariate_normal([-15,-15], [[20,8],[8,5]],size=200)
X = np.concatenate((X1,X2),axis=0)
y = X[1:-1,1].reshape(-1,1)
X = X[0:-2,:]

f = KAF.QKLMS()
y_ = f.evaluate(X,y)
# print("Tamaño de y = ",y_.shape)
r2 = r2_score(y[1:],y_)

plt.plot(y[1:])
plt.plot(y_)
print("R2 = ", r2)


import pandas as pd
from sklearn.metrics import r2_score
sp500 = pd.read_csv("datasets/spx.csv")

samples = 365

# Señal deseada
d1 = sp500.close.iloc[-samples-1:-1].to_numpy().reshape(-1,1)
# Señal de entrada
u1 = sp500.close.iloc[-samples-2:-2].to_numpy().reshape(-1,1)
u2 = sp500.close.iloc[-samples-3:-3].to_numpy().reshape(-1,1)
u3 = sp500.close.iloc[-samples-4:-4].to_numpy().reshape(-1,1)
u4 = sp500.close.iloc[-samples-5:-5].to_numpy().reshape(-1,1)
u5 = sp500.close.iloc[-samples-6:-6].to_numpy().reshape(-1,1)
u1 = np.concatenate((u1,u2,u3,u4,u5), axis=1) 


sigmas = np.linspace(200,10000,20)

epsln = 200
 
r2_QKLMS = []
mse_QKLMS2 = []
CB_size1 = []
CB_size2 = []

s1 = 1000

for eps in ep:
    #QKLMS normal
    filtro1 = KAF.QKLMS(epsilon=eps,sigma=s1)
    out1 = filtro1.evaluate(u1,d1)
    r2_QKLMS.append(r2_score(d1[1:], out1))
    CB_size1.append(filtro1.CB_growth[-1])
    
    #QKLMS con distancia de Mahalanobis
    # filtro2 = KAF.QKLMS2(epsilon=eps,sigma=s2)
    # out2 = filtro2.evaluate(u1,d1)
    # mse_QKLMS2.append(mean_squared_error(d1, out2))
    # CB_size2.append(filtro2.CB_growth[-1])
    
plt.figure(4)    
plt.title("SP500 - Epsilon logaritmico")
#plt.yscale("log")
plt.ylim([0,1])
plt.xscale("log")
plt.xlabel("Epsilon")
plt.ylabel("R2")
plt.plot(ep,r2_QKLMS, 'b', marker="o", label="QKLMS")
# plt.plot(ep,mse_QKLMS2, 'm', marker="o", label="QKLMS2")
plt.legend()


print("R2 maximo: ",max(r2_QKLMS))
print("R2 minimo: ", min(r2_QKLMS))

