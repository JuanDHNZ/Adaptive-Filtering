# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:35:31 2020

@author: USUARIO
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

th = np.linspace(0, 2*np.pi, 100)

plt.plot(np.cos(th))
plt.plot(np.sin(th))

norm = mpl.colors.Normalize(min(th),max(th))
norm(th).shape




"""TEST GRAFICAS"""

import KAF
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
    
out1 = []
out2 = []
r2_filtro1 = []
r2_filtro2 = []
CB_size1 = []
CB_size2 = []
sigma_track = []
epsilon_track = []

epsilonList = np.logspace(2, 6, 20)
sigmaList = np.logspace(2, 6, 20)
           
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
    
for i in range(Ns):    
    plt.plot(CB_size1_[i],r2_filtro1_[i])
    plt.ylim([0,1])
    plt.ylabel("R2")
    plt.xlabel("Codebook Size")
    plt.title("QKLMS")
plt.show()    
for i in range(Ns):    
    plt.plot(CB_size2_[i],r2_filtro2_[i])
    plt.ylim([0,1])
    plt.ylabel("R2")
    plt.xlabel("Codebook Size")
    plt.title("M-QKLMS")
plt.show()

import numpy as np
import matplotlib.pylab as pl

x = np.linspace(0, 2*np.pi, 64)
y = np.cos(x) 

pl.figure()
pl.plot(x,y)

n = 20
colors = pl.cm.jet(np.linspace(0,1))

for i in range(n):
    pl.plot(x, i*y, color=colors[i])
 




"""TEST QKLMS3"""

import KAF
import numpy as np

import pandas as pd
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

filt = KAF.QKLMS3(epsilon=40000)
out, dist = filt.evaluate(u,d)

import matplotlib.pyplot as plt
plt.plot(out)
plt.plot(d)
