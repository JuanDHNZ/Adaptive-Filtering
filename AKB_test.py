# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 02:29:04 2020

@author: Juan David
"""

import numpy as np
import KAF
import TimeSeriesGenerator as tsg
"""ATRACTOR DE LORENZ"""
samples = 1000
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="lorenz")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

akb = KAF.QKLMS_AKB(sigma_init=10,K=5)
out = akb.evaluate(u, d)

from sklearn.metrics import r2_score
r2 = r2_score(d[1:],out)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.plot(d[1:],label="Target")
plt.plot(out,label="Predict")
plt.legend()
