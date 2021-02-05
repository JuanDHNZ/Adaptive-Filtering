# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:42:00 2021

@author: Juan David
"""
def db(samples=1000,system='lorenz',L=40):
    import numpy as np
    import TimeSeriesGenerator as tsg
    x, y, z = tsg.chaoticSystem(samples=samples,systemType=system)
    ux = np.array([x[i-L:i] for i in range(L,len(x))])
    uy = np.array([y[i-L:i] for i in range(L,len(y))])
    u = np.concatenate((ux,uy), axis=1) # INPUT
    d = np.array([z[i] for i in range(L,len(z))]).reshape(-1,1)
    return u,d

u,d = db(3000,'lorenz')

from KAF import QKLMS
import numpy as np

n=10
epsilon = np.linspace(1e-2,2000,n)
sigma = np.linspace(1e-2,2000,n)

from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

params = [{'epsilon':ep, 'sigma':sg} for ep in epsilon for sg in sigma]

results = []

for p in tqdm(params):
    try:
        f = QKLMS(epsilon=p['epsilon'],sigma=p['sigma'])
        y = f.evaluate(u,d)
        y = np.array(y).reshape(-1,1)
        # y_ = np.sum(y)
        print("d shape {} and y shape {}".format(d.shape,y.shape))   
        p['r2'] = r2_score(d[1:], y[1:])
        p['mse'] = mean_squared_error(d[1:], y[1:])
    except:
        p['r2'] = np.nan
        p['mse'] = np.nan
    results.append(p)