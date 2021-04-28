# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 00:28:14 2021

@author: USUARIO
"""

import numpy as np
import testSystems as ts
N = 5000
u,d = ts.testSystems(N,'4.1_AKB')

import TimeSeriesGenerator
x, y, z = TimeSeriesGenerator.chaoticSystem(samples=6000,systemType='wang')

def std(x):
    x = x - x.mean()
    return x/x.std()

# signalEmbedding = 5
# X = np.array([u[i-signalEmbedding:i] for i in range(signalEmbedding,len(u))])
# y = np.array([d[i] for i in range(signalEmbedding,len(u))]).reshape(-1,1)

u = d = std(x)

 
import KAF
f = KAF.QKLMS_AMK(eta=0.01,epsilon=0.5,embedding=10, Ka=10, mu=0.05)
f.evaluate(u[:100],d[:100])

f.fit(u[:4000],d[:4000])
y_pred = f.predict(u[4000:])

print(len(f.CB))

import matplotlib.pyplot as plt
plt.plot(y_pred)
plt.plot(d[4000:])

scr = f.score(d[4000:],y_pred)

