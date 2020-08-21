# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:35:31 2020

@author: USUARIO
"""

"""TEST GRAFICAS"""

import KAF
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


sp500 = pd.read_csv("datasets/spx.csv")

samples = 400
# Señal deseada
d = sp500.close.iloc[-samples-1:-1].to_numpy().reshape(-1,1)
# Señal de entrada
u1 = sp500.close.iloc[-samples-2:-2].to_numpy().reshape(-1,1)
u2 = sp500.close.iloc[-samples-3:-3].to_numpy().reshape(-1,1)
u3 = sp500.close.iloc[-samples-4:-4].to_numpy().reshape(-1,1)
u4 = sp500.close.iloc[-samples-5:-5].to_numpy().reshape(-1,1)
u5 = sp500.close.iloc[-samples-6:-6].to_numpy().reshape(-1,1)
u = np.concatenate((u1,u2,u3,u4,u5), axis=1)
    
out1 = []
out2 = []
r2_filtro1 = []
r2_filtro2 = []
CB_size1 = []
CB_size2 = []
sigma_track = []
epsilon_track = []

epsilonList = np.logspace(2, 5, 20)
sigmaList = np.logspace(2, 3, 20)
           
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
    
#Para graficar
import numpy as np
Ns = len(sigmaList)
Ne = len(epsilonList)
r2_filtro1_ = np.asarray(r2_filtro1).reshape([Ns,Ne])
CB_size1_ = np.asarray(CB_size1).reshape([Ns,Ne])  
r2_filtro2_ = np.asarray(r2_filtro2).reshape([Ns,Ne])
CB_size2_ = np.asarray(CB_size2).reshape([Ns,Ne])

import matplotlib as mpl
import matplotlib.pylab as pl

n = len(sigmaList)
colors = pl.cm.jet(np.linspace(0,1,n))

fig, ax = plt.subplots()
norm = mpl.colors.Normalize(0,1)
cmap = mpl.colors.Colormap('jet',256)

for i in range(Ns):    
    im = ax.plot(CB_size1_[i],r2_filtro1_[i], color=colors[i])
plt.ylim([0,1])
plt.ylabel("R2")
plt.xlabel("Codebook Size")
plt.title("QKLMS")
# fig.colorbar(pl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
plt.show()    

fig, ax = plt.subplots()
norm = mpl.colors.Normalize(0,1)
norm = norm(sigmaList)
cmap = mpl.colors.Colormap('jet',256)
for i in range(Ns):    
    im = ax.plot(CB_size2_[i],r2_filtro2_[i], color=colors[i])
plt.ylim([0,1])
plt.ylabel("R2")
plt.xlabel("Codebook Size")
plt.title("M-QKLMS")
# fig.colorbar(pl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
plt.show()

c = np.empty((20,4))
c[:,3] = np.linspace(0,1,20)
c[:,0] = c[:,1] = 0.0
c[:,2] = 1.0

fig, ax = plt.subplots()
for i in range(Ns):    
    ax.plot(CB_size1_[i],r2_filtro1_[i], color=c[i])
plt.ylim([0,1])
plt.ylabel("R2")
plt.xlabel("Codebook Size")
plt.title("QKLMS")
# fig.colorbar(, ax=ax)
plt.show()    
for i in range(Ns):    
    plt.plot(CB_size2_[i],r2_filtro2_[i], color=c[i])
    plt.ylim([0,1])
    plt.ylabel("R2")
    plt.xlabel("Codebook Size")
    plt.title("M-QKLMS")
plt.show()

colors = pl.cm.jet(np.linspace(0,1,n))
