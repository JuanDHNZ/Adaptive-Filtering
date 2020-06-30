# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 07:40:11 2020

@author: Juan David
"""

""" Señal de prueba 2 -  EGG Neurotycho """

import numpy as np
import KAF
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import scipy.io as sp

EEG_ch1 = sp.loadmat("datasets/ECoG_ch1.mat")
signal = EEG_ch1['ECoGData_ch1'].T
samples = 500 

d2 = signal[-samples-1:-1]

ua = signal[-samples-2:-2].reshape(-1,1)
ub = signal[-samples-3:-3].reshape(-1,1)
uc = signal[-samples-4:-4].reshape(-1,1)
u2 = np.concatenate((ua,ub,uc), axis=1) 

plt.plot(u2)

sigmas = np.logspace(1,5,30)
mse_QKLMS = []
mse_QKLMS2 = []
CB_size1 = []
CB_size2 = []

for s in sigmas:
    #QKLMS normal
    filtro1 = KAF.QKLMS(epsilon=200,sigma=s)
    out1 = filtro1.evaluate(u2,d2)
    mse_QKLMS.append(r2_score(d2[1:], out1))
    CB_size1.append(filtro1.CB_growth[-1])
    
    #QKLMS con distancia de Mahalanobis
    filtro2 = KAF.QKLMS2(epsilon=200,sigma=s)
    out2 = filtro2.evaluate(u2,d2)
    mse_QKLMS2.append(r2_score(d2[1:], out2))
    CB_size2.append(filtro2.CB_growth[-1])
    plt.plot(out1[:-5], label="predict")
    plt.plot(d2, label="target")
    plt.legend()
    # plt.title("Sigma = " + str(s))   
    # plt.show()
    
plt.figure(2)    
plt.title("EGG Ch1 - Neurotycho - Sigma logaritmico")
# plt.yscale("log")
plt.xscale("log")
plt.xlabel("Sigma")
plt.ylabel("R2")
plt.ylim([-1,1.1])
plt.plot(sigmas,mse_QKLMS, 'b', marker="o", label="QKLMS")
plt.plot(sigmas,mse_QKLMS2, 'm', marker="o", label="QKLMS2")
plt.legend()

plt.figure(3)    
plt.title("EEG Ch1 - Neurotycho - Sigma logaritmico")
# plt.yscale("log")o
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





""" Señal de prueba 2 -  EGG Ch1 - Neurotycho  variando epsilon para cada algoritomo """
    
""" Rejilla para QKLMS: """
r2_QKLMS = []
r2_QKLMS2 = []
CB_size1 = []
CB_size2 = []

#Completar con grilla variable 
# ep = np.logspace(1,4,20) 
ep = np.linspace(100,100000,100)
s1 = 10000
s2 = 5000
for eps in ep:
    #QKLMS normal
    filtro1 = KAF.QKLMS(epsilon=eps,sigma=s1)
    out1 = filtro1.evaluate(u2,d2)
    r2_QKLMS.append(r2_score(d2[1:], out1))
    CB_size1.append(filtro1.CB_growth[-1])
    
    #QKLMS con distancia de Mahalanobis
    filtro2 = KAF.QKLMS2(epsilon=eps,sigma=s2)
    out2 = filtro2.evaluate(u2,d2)
    r2_QKLMS2.append(r2_score(d2[1:], out2))
    CB_size2.append(filtro2.CB_growth[-1])
    
plt.figure(4)    
plt.title("EGG Ch1 - Neurotycho - Epsilon logaritmico")
#plt.yscale("log")
plt.xscale("log")
plt.xlabel("Epsilon")
plt.ylabel("R2")
plt.plot(ep,r2_QKLMS, 'b', marker="o", label="QKLMS")
plt.plot(ep,r2_QKLMS2, 'm', marker="o", label="QKLMS2")
plt.legend()

plt.figure(5)    
plt.title("EGG Ch1 - Neurotycho- Epsilon logaritmico")
# plt.yscale("log")
plt.xscale("log")
plt.xlabel("Epsilon")
plt.ylabel("Tamaño CB")
plt.plot(ep,CB_size1, 'b', marker="o", label="QKLMS")
plt.plot(ep,CB_size2, 'm', marker="o", label="QKLMS2")
plt.legend()
print("**********************************************")
print("\nPRUEBA VARIANDO EPSILON:")
print("Codebook Size QKLMS: ", CB_size1[-1])
print("Codebook Size: QKLMS2", CB_size2[-1])
print("Epsilon de minimo MSE QKLMS = ", ep[np.argmin(r2_QKLMS)])
print("Epsilon de minimo MSE QKLMS2 = ", ep[np.argmin(r2_QKLMS2)])
print("Minimo MSE QKLMS = ", min(r2_QKLMS))
print("Minimo MSE QKLMS2 = ", min(r2_QKLMS2))



#

