# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:08:02 2020

@author: USUARIO
"""

import numpy as np
import TimeSeriesGenerator as tsg
import KAF
from sklearn.metrics import r2_score
"""NUMERO DE MUESTRAS PARA LAS PRUEBAS"""
samples = 1000

"""
    ATRACTOR DE LORENZ

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="lorenz")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

out1 = []
out2 = []
r2_filtro1 = []
r2_filtro2 = []
CB_size1 = []
CB_size2 = []
sigma_track = []
epsilon_track = []

# epsilonList = np.logspace(0, 2, 20)
# sigmaList = np.logspace(0, 2, 20)

epsilonList = np.linspace(1, 25, 20)
sigmaList = np.linspace(10,60,20)

           
for sigma in sigmaList:
    for epsilon in epsilonList:
        filtro1 = KAF.QKLMS(epsilon=epsilon,sigma=sigma)
        filtro2 = KAF.QKLMS2(epsilon=epsilon, sigma=sigma)
        sigma_track.append(sigma)
        epsilon_track.append(epsilon)
        for i in range(len(d)):
            out1.append(filtro1.evaluate(u[i],d[i]))                        
            out2.append(filtro2.evaluate(u[i],d[i]))
            
        #Remove NoneTypes that result from initialization 
        out1 = [j.item() for j in out1 if j is not None]
        out2 = [j.item() for j in out2 if j is not None]
             
        r2_filtro1.append(r2_score(d[1:], out1))
        r2_filtro2.append(r2_score(d[1:], out2))
        CB_size1.append(len(filtro1.CB))
        CB_size2.append(len(filtro2.CB))
        out1.clear()
        out2.clear()
        

Ns = len(sigmaList)
Ne = len(epsilonList)
r2_filtro1_ = np.asarray(r2_filtro1).reshape([Ns,Ne])
CB_size1_ = np.asarray(CB_size1).reshape([Ns,Ne])  
r2_filtro2_ = np.asarray(r2_filtro2).reshape([Ns,Ne])
CB_size2_ = np.asarray(CB_size2).reshape([Ns,Ne])

import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt

n = len(sigmaList)
colors = pl.cm.jet(np.linspace(0,1,n))

fig, ax = plt.subplots()
norm = mpl.colors.Normalize(min(sigmaList),max(sigmaList))

for i in range(Ns):    
    im = ax.plot(CB_size2_[i],r2_filtro2_[i], color=colors[i])
plt.ylim([0,1])
plt.ylabel("R2")
plt.xlabel("Codebook Size")
plt.title("QKLMS 2 ")
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax)
cbar.set_label('sigma')
# plt.savefig("pruebasGMM/MQKLMS_sig_10_100_2.png", dpi = 300)
plt.show()    

#GRAFICOS
n = len(sigmaList)
colors = pl.cm.jet(np.linspace(0,1,n))

fig, ax = plt.subplots()
norm = mpl.colors.Normalize(min(sigmaList),max(sigmaList))
norm.autoscale(sigmaList)

for i in range(Ns):    
    im = ax.plot(CB_size1_[i],r2_filtro1_[i], color=colors[i])
plt.ylim([0,1])
plt.ylabel("R2")
plt.xlabel("Codebook Size")
plt.title("QKLMS")
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax)
cbar.set_label('sigma')
# plt.savefig("pruebasGMM/QKLMS_sig_10_100_2.png", dpi = 300)
plt.show()    

