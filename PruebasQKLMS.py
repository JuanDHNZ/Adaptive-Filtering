# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:29:55 2020

@author: Juan

Pruebas QKLMS vs QKLMS2
"""
import pandas as pd
import numpy as np
import KAF
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


""" Señal de prueba 1 -  SP500 """

sp500 = pd.read_csv("spx.csv")

# Señal deseada
d1 = sp500.close.iloc[-366:-1].to_numpy().reshape(-1,1)
# Señal de entrada
u1 = sp500.close.iloc[-367:-2].to_numpy().reshape(-1,1)
u2 = sp500.close.iloc[-368:-3].to_numpy().reshape(-1,1)
u3 = sp500.close.iloc[-369:-4].to_numpy().reshape(-1,1)
u4 = sp500.close.iloc[-370:-5].to_numpy().reshape(-1,1)
u5 = sp500.close.iloc[-371:-6].to_numpy().reshape(-1,1)
u1 = np.concatenate((u1,u2,u3,u4,u5), axis=1) 


""" Señal de prueba 1 -  SP500 """
""" Señal de prueba 1 -  SP500 """
sigmas = np.linspace(200,10000,20)

mse_QKLMS = []
mse_QKLMS2 = []
CB_size1 = []
CB_size2 = []

for s in sigmas:
    #QKLMS normal
    filtro1 = KAF.QKLMS(epsilon=200,sigma=s)
    out1 = filtro1.evaluate(u1,d1)
    mse_QKLMS.append(mean_squared_error(d1, out1))
    CB_size1.append(filtro1.CB_growth[-1])
    #QKLMS con distancia de Mahalanobis
    filtro2 = KAF.QKLMS2(epsilon=200,sigma=s)
    out2 = filtro2.evaluate(u1,d1)
    mse_QKLMS2.append(mean_squared_error(d1, out2))
    CB_size2.append(filtro2.CB_growth[-1])
    
plt.figure(0)    
plt.title("SP500 - Sigma lineal")
plt.yscale("log")
# plt.xscale("log")
plt.xlabel("Sigma")
plt.ylabel("MSE")
plt.plot(sigmas,mse_QKLMS, 'b', marker="o", label="QKLMS")
plt.plot(sigmas,mse_QKLMS2, 'm', marker="o", label="QKLMS2")
plt.legend()
plt.figure(1)    
plt.title("SP500 - Sigma lineal")
# plt.yscale("log")
# plt.xscale("log")
plt.xlabel("Sigma")
plt.ylabel("Tamaño CB")
plt.plot(sigmas,CB_size1, 'b', marker="o", label="QKLMS")
plt.plot(sigmas,CB_size2, 'm', marker="o", label="QKLMS2")
plt.legend()

print("Codebook Size QKLMS: ", CB_size1[-1])

print("Codebook Size: QKLMS2", CB_size2[-1])

# #sigmas logaritmicos
sigmas = np.logspace(1,4,20)
mse_QKLMS = []
mse_QKLMS2 = []
CB_size1 = []
CB_size2 = []
for s in sigmas:
    #QKLMS normal
    filtro1 = KAF.QKLMS(epsilon=200,sigma=s)
    out1 = filtro1.evaluate(u1,d1)
    mse_QKLMS.append(mean_squared_error(d1, out1))
    CB_size1.append(filtro1.CB_growth[-1])
    #QKLMS con distancia de Mahalanobis
    filtro2 = KAF.QKLMS2(epsilon=200,sigma=s)
    out2 = filtro2.evaluate(u1,d1)
    mse_QKLMS2.append(mean_squared_error(d1, out2))
    CB_size2.append(filtro2.CB_growth[-1])
    
plt.figure(2)    
plt.title("SP500 - Sigma logaritmico")
plt.yscale("log")
# plt.xscale("log")
plt.xlabel("Sigma")
plt.ylabel("MSE")
plt.plot(sigmas,mse_QKLMS, 'b', marker="o", label="QKLMS")
plt.plot(sigmas,mse_QKLMS2, 'm', marker="o", label="QKLMS2")
plt.legend()
plt.figure(3)    
plt.title("SP500 - Sigma logaritmico")
# plt.yscale("log")
# plt.xscale("log")
plt.xlabel("Sigma")
plt.ylabel("Tamaño CB")
plt.plot(sigmas,CB_size1, 'b', marker="o", label="QKLMS")
plt.plot(sigmas,CB_size2, 'm', marker="o", label="QKLMS2")
plt.legend()

print("Codebook Size QKLMS: ", CB_size1[-1])
print("Codebook Size: QKLMS2", CB_size2[-1])



