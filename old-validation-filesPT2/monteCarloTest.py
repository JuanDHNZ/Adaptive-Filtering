# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 08:44:32 2021

@author: Juan David
"""

def db(samples=5040,system='lorenz',L=40):
    import numpy as np
    import TimeSeriesGenerator as tsg
    x, y, z = tsg.chaoticSystem(samples=samples,systemType=system)
    ux = np.array([x[i-L:i] for i in range(L,len(x))])
    uy = np.array([y[i-L:i] for i in range(L,len(y))])
    u = np.concatenate((ux,uy), axis=1) # INPUT
    d = np.array([z[i] for i in range(L,len(z))]).reshape(-1,1)
    return u,d

def db2(samples=1000):
    import numpy as np
    import testSystems as ts
    var = 0.01
    noise = np.sqrt(var)*np.random.randn(samples)
    s = ts.testSystems(samples = samples+2, systemType = "4.2_AKB")
    u = np.array([s[-samples-1:-1],s[-samples-2:-2]]).T
    d = np.array(s[-samples:]).reshape(-1,1) + noise.reshape(-1,1)
    return u,d

def db3(samples=1000):
    import testSystems as ts
    return ts.testSystems(samples=samples, systemType="4.1_AKB")

from KAF import QKLMS_AKB
from KAF import QKLMS_AMK
from KAF import QKLMS
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

L = 40

# db3
# sgm = 0.24
# eps = 1e-6
# eta = 0.1
# K = 5
# mu = 0.1

#QKLMS for lorenz
sgm = 10000
eps = 0.01
eta = 0.1

samples = 1240
N = 100


var = 0.1
# u,d = db(samples=samples, system="lorenz",L=L)

# un,dn = db2(samples=samples)

# sgm = 0.666
# eps = 1e-6
# eta = 0.1
from tqdm import tqdm
for _ in tqdm(range(N)):
    start = np.random.randint(0,150)
    un,dn = db3(samples=samples)
    
    un = un[start:start+1000]
    dn = dn[start:start+1000] + 2 
    
    sgm = 0.22
    eps = 1e-6
    eta = 0.9
    
    mse_ = []
    
    # f = QKLMS_AKB(eta=eta,epsilon=eps,sigma_init=sgm,mu=mu,K=K)
    # y = f.evaluate(u, d)
    # mse_.append((y-d[1:])**2)
    
    # noise = np.sqrt(var)*np.random.randn(samples-L+1).reshape(-1,1)
    # un = u + noise
    # dn = d + noise
    
    f = QKLMS(epsilon=eps, sigma=sgm, eta=eta)
    y_pred = f.evaluate(un,dn)
    mse_.append(((dn-np.array(y_pred).reshape(-1,1))**2)/dn**2)

    
mse_av = np.sum(np.array(mse_),axis=0)/N

plt.figure(figsize=(15,9))
plt.title("System 4.2  -   $\sigma$ = {}; epsilon = {}; eta={}".format(sgm,eps,eta))
plt.yscale("log")
plt.plot(mse_av,"r-",linewidth=1)
plt.ylabel("MSE")
plt.xlabel("iterations")
plt.show()

# plt.title(datb)
plt.plot(dn, label="target")
plt.plot(y_pred, label="predict")
plt.legend()
plt.show()

