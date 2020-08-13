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

sg1, ep1, sg2, ep2 = search.pSearchCurve(u=u.reshape(-1,1), d=d.reshape(-1,1), sigmaList = sigma, epsilonList = epsilon)



""" Señal de prueba 1 -  SP500 """
import pandas as pd
sp500 = pd.read_csv("datasets/spx.csv")

samples = 50

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

sg1, ep1, sg2, ep2 = search.pSearchCurve(u=u, d=d, sigmaList = sigma, epsilonList = epsilon, r2_threshold=0.2)

from sklearn.metrics import r2_score
""" Para  QKLMS """
pred = []
qklms = KAF.QKLMS(sigma=sg1,epsilon=ep1)
for i in range(len(u)):
   pred.append(qklms.evaluate(u[i],d[i]))
pred = [i.item() for i in pred if i is not None]
#Grafico
plt.title("QKLMS")
plt.plot(pred, label="Predict")
plt.plot(d[1:], label="Target")
plt.legend()
plt.show()
plt.title("QKLMS codebook growth")
plt.plot(qklms.CB_growth)
plt.show()

R2_qklms = r2_score(d[1:], pred)
print("R2 QKLMS = ", R2_qklms)

""" Para  QKLMS 2 """
print("Best Sigma = ", sg1)
print("Best Epsilon = ", ep1)
pred = []
mqklms = KAF.QKLMS(sigma=sg2,epsilon=ep2)
for i in range(len(u)):
   pred.append(mqklms.evaluate(u[i],d[i]))
pred = [i.item() for i in pred if i is not None]
#Grafico
plt.title("M-QKLMS")
plt.plot(pred, label="Predict")
plt.plot(d[1:], label="Target")
plt.legend()
plt.show()
plt.title("M-QKLMS codebook growth")
plt.plot(qklms.CB_growth)
plt.show()

R2_qklms = r2_score(d[1:], pred)
print("R2 QKLMS = ", R2_qklms)




