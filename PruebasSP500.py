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
 
mse_QKLMS = []
mse_QKLMS2 = []
CB_size1 = []
CB_size2 = []

for s in sigmas:
    #QKLMS normal
    filtro1 = KAF.QKLMS(epsilon=epsln,sigma=s)
    out1 = filtro1.evaluate(u1,d1)
    mse_QKLMS.append(mean_squared_error(d1, out1))
    CB_size1.append(filtro1.CB_growth[-1])
    #QKLMS con distancia de Mahalanobis
    filtro2 = KAF.QKLMS2(epsilon=epsln,sigma=s)
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
print("**********************************************")
print("Codebook Size QKLMS: ", CB_size1[-1])
print("Codebook Size: QKLMS2", CB_size2[-1])
print("Sigma de minimo MSE QKLMS = ", sigmas[np.argmin(mse_QKLMS)])
print("Sigma de minimo MSE QKLMS2 = ", sigmas[np.argmin(mse_QKLMS2)])
print("Minimo MSE QKLMS = ", mse_QKLMS[np.argmin(mse_QKLMS)])
print("Minimo MSE QKLMS2 = ", mse_QKLMS2[np.argmin(mse_QKLMS2)])


# #sigmas logaritmicos
sigmas = np.logspace(1,4,20)
mse_QKLMS = []
mse_QKLMS2 = []
CB_size1 = []
CB_size2 = []
for s in sigmas:
    #QKLMS normal
    filtro1 = KAF.QKLMS(epsilon=epsln,sigma=s)
    out1 = filtro1.evaluate(u1,d1)
    mse_QKLMS.append(mean_squared_error(d1, out1))
    CB_size1.append(filtro1.CB_growth[-1])
    #QKLMS con distancia de Mahalanobis
    filtro2 = KAF.QKLMS2(epsilon=epsln,sigma=s)
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
print("**********************************************")
print("Codebook Size QKLMS: ", CB_size1[-1])
print("Codebook Size: QKLMS2", CB_size2[-1])
print("Sigma de minimo MSE QKLMS = ", sigmas[np.argmin(mse_QKLMS)])
print("Sigma de minimo MSE QKLMS2 = ", sigmas[np.argmin(mse_QKLMS2)])
print("Minimo MSE QKLMS = ", mse_QKLMS[np.argmin(mse_QKLMS)])
print("Minimo MSE QKLMS2 = ", mse_QKLMS2[np.argmin(mse_QKLMS2)])

#Prueba variando 



""" Señal de prueba 1 -  SP500  variando epsilon para cada algoritomo """
    
""" Rejilla para QKLMS: """
mse_QKLMS = []
mse_QKLMS2 = []
CB_size1 = []
CB_size2 = []

# ep = np.linspace(100,10000,20)
ep = np.logspace(1,4,30)
s1 = 1000
s2 = 10000
for eps in ep:
    #QKLMS normal
    filtro1 = KAF.QKLMS(epsilon=eps,sigma=s1)
    out1 = filtro1.evaluate(u1,d1)
    mse_QKLMS.append(mean_squared_error(d1, out1))
    CB_size1.append(filtro1.CB_growth[-1])
    #QKLMS con distancia de Mahalanobis
    filtro2 = KAF.QKLMS2(epsilon=eps,sigma=s2)
    out2 = filtro2.evaluate(u1,d1)
    mse_QKLMS2.append(mean_squared_error(d1, out2))
    CB_size2.append(filtro2.CB_growth[-1])
    
plt.figure(4)    
plt.title("SP500 - Epsilon logaritmico")
plt.yscale("log")
# plt.xscale("log")
plt.xlabel("Epsilon")
plt.ylabel("MSE")
plt.plot(ep,mse_QKLMS, 'b', marker="o", label="QKLMS")
plt.plot(ep,mse_QKLMS2, 'm', marker="o", label="QKLMS2")
plt.legend()
plt.figure(5)    
plt.title("SP500 - Epsilon logaritmico")
# plt.yscale("log")
# plt.xscale("log")
plt.xlabel("Epsilon")
plt.ylabel("Tamaño CB")
plt.plot(ep,CB_size1, 'b', marker="o", label="QKLMS")
plt.plot(ep,CB_size2, 'm', marker="o", label="QKLMS2")
plt.legend()
print("**********************************************")
print("Codebook Size QKLMS: ", CB_size1[-1])
print("Codebook Size: QKLMS2", CB_size2[-1])
print("Epsilon de minimo MSE QKLMS = ", ep[np.argmin(mse_QKLMS)])
print("Epsilon de minimo MSE QKLMS2 = ", ep[np.argmin(mse_QKLMS2)])
print("Minimo MSE QKLMS = ", mse_QKLMS[np.argmin(mse_QKLMS)])
print("Minimo MSE QKLMS2 = ", mse_QKLMS2[np.argmin(mse_QKLMS2)])



# test = KAF.QKLMS2(sigma=1000, epsilon=10000)



