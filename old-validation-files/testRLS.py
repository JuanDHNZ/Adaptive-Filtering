# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 09:54:11 2020

@author: USUARIO
"""

import numpy as np
import seaborn as sns
sns.set()

"""Prueba simple"""
N = 200
# X = np.linspace(-np.pi,np.pi,N)
X = np.linspace(0,N,N)
y = 0.02*X**2 + 2*X - 3.1
noise = np.mean(y)*np.random.randn(N)
y = y + noise

import matplotlib.pyplot as plt
plt.plot(X,y,'c', label='Target')

lam = 0.98 #Forgeting factor
dlt = 1
order = 3


from LinearAdaptiveFilters import RLS
rls = RLS(order,lam = lam, delta = dlt)

y_pred = []
x_pred = []

for i in range(N):
    x = np.matrix(np.zeros((1,3)))
    x[0,0] = 1
    x[0,1] = i
    x[0,2] = i**2
    x_pred.append(i)
    y_pred.append(float(x*rls.w))
    rls.add_obs(x.T,y[i])

plt.title("RLS")  
plt.plot(X,y_pred,'r',label='Predict')
plt.legend()
plt.show()

plt.title("Loss")
plt.plot(X, y-y_pred)
plt.show()

#weights comparison
X = np.linspace(-np.pi,np.pi,N)
# X = np.linspace(0,N,N)
y = 0.2*X**2 + 2*X - 3.1
y_ = rls.w[2].item()*X**2 + rls.w[1].item()*X + rls.w[0].item()

print(rls.w)



"""Prueba con KERNEL"""
import TimeSeriesGenerator as tsg
from sklearn.metrics import r2_score
samples = 50
scr = []
x, y, z = tsg.chaoticSystem(samples = samples + 10, systemType = 'lorenz')
u = np.concatenate((x[-samples-2:-2].reshape(-1,1),y[-samples-3:-3].reshape(-1,1)), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

from sklearn.mixture import BayesianGaussianMixture
bgmm = BayesianGaussianMixture(n_components=samples)

bgmm.fit(u)

Mn = bgmm.means_ #means
Sn = bgmm.precisions_ #presitions
Nc = bgmm.n_components #Number of components

from scipy.spatial.distance import cdist
MK = [cdist(u, Mn[j].reshape(1,-1), 'mahalanobis', VI=Sn[j]) for j in range(Nc)]
MK = [np.exp((-f**2)/2) for f in MK] 
Phi = np.concatenate(MK,axis=1)

reg = RLS(num_vars=samples,lam=0.98,delta=1)
print(Phi.shape,d.shape)
reg.fit(Phi,d.reshape(-1,))

y_pred = reg.predict(Phi)#prediction
scr.append(r2_score(d,y_pred.T)) # R2 performance

