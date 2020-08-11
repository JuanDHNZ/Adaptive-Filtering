# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:33:47 2020

@author: USUARIO

SEARCH TEST 
"""


import numpy as np 
import matplotlib.pyplot as plt
import testSystems as ts
import KAF 
import search

def plot_pair(u = None,d = None,labelu="u",labeld="d", title = ""):
    plt.title(title)
    plt.plot(u,label = labelu)
    plt.plot(d, label = labeld)
    plt.legend()
    plt.show()

"""NUMERO DE MUESTRAS"""
samples = 100
""""
SISTEMA 1 

u -> proceso Gaussiano de media cero y varianza unitaria
d -> resultado de aplicar filtro FIR a u

"""
u, d = ts.testSystems(samples=samples, systemType="1")
plot_pair(u,d,labelu="input", labeld="target", title="Sistema 1")

epsilon = np.logspace(-6, 6, 20)
sigma = np.logspace(-6, 6, 20)

R2_QKLMS, R2_QKLMS2, CB_size_QKLMS, CB_size_QKLMS2 = search.pSearchCurve(u=u.reshape(-1,1), d=d.reshape(-1,1), sigmaList = sigma, epsilonList = epsilon)

print("\n************************************")
print("R2 Maximo en QKLMS = ", max(R2_QKLMS))
print("R2 Maximo en M-QKLMS = ", max(R2_QKLMS2))
print("************************************\n")




""" Señal de prueba 1 -  SP500 """
import pandas as pd
sp500 = pd.read_csv("datasets/spx.csv")

samples = 365

# Señal deseada
d = sp500.close.iloc[-samples-1:-1].to_numpy().reshape(-1,1)
# Señal de entrada
u1 = sp500.close.iloc[-samples-2:-2].to_numpy().reshape(-1,1)
u2 = sp500.close.iloc[-samples-3:-3].to_numpy().reshape(-1,1)
u3 = sp500.close.iloc[-samples-4:-4].to_numpy().reshape(-1,1)
u4 = sp500.close.iloc[-samples-5:-5].to_numpy().reshape(-1,1)
u5 = sp500.close.iloc[-samples-6:-6].to_numpy().reshape(-1,1)
u = np.concatenate((u1,u2,u3,u4,u5), axis=1) 

epsilon = np.logspace(2, 5, 20)
sigma = np.logspace(2, 5, 20)

sig, eps = search.pSearchCurve(u=u, d=d, sigmaList = sigma, epsilonList = epsilon)

print("\n************************************")
print("R2 Maximo en QKLMS = ", max(R2_QKLMS))
print("R2 Maximo en M-QKLMS = ", max(R2_QKLMS2))
print("************************************\n")


sg = sig[8]
ep = eps[8]

filtro1 = KAF.QKLMS(epsilon=ep,sigma=sg)
filtro2 = KAF.QKLMS2(epsilon=ep, sigma=sg)
for i in range(len(d)):
    out1 = filtro1.evaluate(u[i],d[i])                        
    out2 = filtro2.evaluate(u[i],d[i])

plt.plot(filtro1.CB_growth)
plt.plot(filtro2.CB_growth)


















out1 = []
out2 = []
r2_filtro1 = []
r2_filtro2 = []
CB_size1 = []
CB_size2 = []
from sklearn.metrics import r2_score
  
filtro1 = KAF.QKLMS(epsilon=100,sigma=100)
filtro2 = KAF.QKLMS2(epsilon=100, sigma=100)
for i in range(len(d)):
    out1.append(filtro1.evaluate(u[i].reshape(-1,1),d[i].reshape(-1,1)))                        
    out2.append(filtro2.evaluate(u[i].reshape(-1,1),d[i].reshape(-1,1)))
                
#Remove NoneTypes that result from initialization 
out1 = [j.item() for j in out1 if j is not None]
out2 = [j.item() for j in out2 if j is not None]
 
r2_filtro1.append(r2_score(d[1:], out1))
r2_filtro2.append(r2_score(d[1:], out2))
CB_size1.append(len(filtro1.CB))
CB_size2.append(len(filtro2.CB))


