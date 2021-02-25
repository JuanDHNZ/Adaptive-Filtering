# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:42:00 2021

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


# ## Sistema 4.2 de AKB
# u,d = db2(samples=400)

# plt.plot(d)
# plt.show()

# # 1. QKLMS
# eta = 0.1
# eps = 0.4
# sigma = 0.5

# f = QKLMS(eta=eta,epsilon=eps, sigma=sigma)
# y_pred = f.evaluate(u,d)
# mse = f.mse_ins

# #2 QKLMS AKB
# eta = 0.584285714285714
# mu = 0.005
# K = 1
# eps = 1.43142857142857
# sigma = 0.5

# f = QKLMS_AKB(eta=eta,epsilon=eps,sigma_init=sigma,mu=mu,K=K)
# y = f.evaluate(u, d)
# mse_akb1 = f.mse

# K=2
# f = QKLMS_AKB(eta=eta,epsilon=eps,sigma_init=sigma,mu=mu,K=K)
# y = f.evaluate(u, d)
# mse_akb2 = f.mse

# K=4
# f = QKLMS_AKB(eta=eta,epsilon=eps,sigma_init=sigma,mu=mu,K=K)
# y = f.evaluate(u, d)
# mse_akb4 = f.mse

# K=6
# f = QKLMS_AKB(eta=eta,epsilon=eps,sigma_init=sigma,mu=mu,K=K)
# y = f.evaluate(u, d)
# mse_akb6 = f.mse

# #3 QKLMS AMK
# eta = 0.584285714285714
# mu = 0.005
# K = 1
# eps = 1.43142857142857
# sigma = 0.5

# f = QKLMS_AMK(eta=eta,epsilon=eps,mu=mu,Ka=K,sigma=0.5)
# y = f.evaluate(u,d)
# mse_amk1 = f.mse_ins

# K=2
# f = QKLMS_AMK(eta=eta,epsilon=eps,mu=mu,Ka=K,sigma=0.5)
# y = f.evaluate(u,d)
# mse_amk2 = f.mse_ins

# K=4
# f = QKLMS_AMK(eta=eta,epsilon=eps,mu=mu,Ka=K,sigma=0.5)
# y = f.evaluate(u,d)
# mse_amk4 = f.mse_ins

# K=6
# f = QKLMS_AMK(eta=eta,epsilon=eps,mu=mu,Ka=K,sigma=0.5)
# y = f.evaluate(u,d)
# mse_amk6 = f.mse_ins

# plt.title("Sistema 4.2")
# plt.yscale("log")
# plt.plot(mse,label='QKLMS')
# plt.plot(mse_akb1,label='QKLMS AKB K=1')
# # plt.plot(mse_akb2,label='QKLMS AKB K=2')
# # plt.plot(mse_akb4,label='QKLMS AKB K=4')
# # plt.plot(mse_akb6,label='QKLMS AKB K=6')
# plt.plot(mse_amk1,label='QKLMS AMK K=1')
# # plt.plot(mse_amk2,label='QKLMS AMK K=2')
# # plt.plot(mse_amk4,label='QKLMS AMK K=4')
# # plt.plot(mse_amk6,label='QKLMS AMK K=6')
# plt.legend()
# plt.show()

# ## CHUA
# u,d = db(samples=5000, system='chua')

# # 1. QKLMS
# sgm = 10000
# eps = 0.01


# f = QKLMS(sigma=sgm,epsilon=eps)
# y = f.evaluate(u,d)
# mse = f.mse_ins

# #2 QKLMS AKB
# eta = 0.584285714285714
# mu = 0.005
# K = 1
# eps = 1.43142857142857
# sigma = 0.5

# f = QKLMS_AKB(eta=eta,epsilon=eps,sigma_init=sigma,mu=mu,K=K)
# y = f.evaluate(u, d)

# mse_akb = f.mse


# #3 QKLMS AMK
# eta = 0.584285714285714
# mu = 0.005
# K = 2
# eps = 1.43142857142857
# sigma = 0.5

# f = QKLMS_AMK(eta=eta,epsilon=eps,mu=mu,Ka=K,A_init="pca")
# y = f.evaluate(u,d)
# mse_amk = f.mse_ins


# plt.title("Chua")
# plt.yscale("log")
# plt.plot(mse,label='QKLMS')
# plt.plot(mse_akb,label='QKLMS AKB')
# plt.plot(mse_amk,label='QKLMS AMK')
# plt.legend()
# plt.show()




# ## LORENZ
# u,d = db(samples=5000, system='lorenz')

# # 1. QKLMS
# sgm = 10000
# eps = 0.01

# f = QKLMS(sigma=sgm,epsilon=eps)
# y = f.evaluate(u,d)
# mse = f.mse_ins
 
# #2 QKLMS AKB
# eta = 1
# mu = 0.0217142857142857
# K = 1
# eps = 0.01
# sigma = 0.5

# f = QKLMS_AKB(eta=eta,epsilon=eps,sigma_init=sigma,mu=mu,K=K)
# y = f.evaluate(u, d)
# mse_akb1 = f.mse

# K=2
# f = QKLMS_AKB(eta=eta,epsilon=eps,sigma_init=sigma,mu=mu,K=K)
# y = f.evaluate(u, d)
# mse_akb2 = f.mse

# K=4
# f = QKLMS_AKB(eta=eta,epsilon=eps,sigma_init=sigma,mu=mu,K=K)
# y = f.evaluate(u, d)
# mse_akb4 = f.mse

# K=6
# f = QKLMS_AKB(eta=eta,epsilon=eps,sigma_init=sigma,mu=mu,K=K)
# y = f.evaluate(u, d)
# mse_akb6 = f.mse


# #3 QKLMS AMK
# eta = 1
# mu = 0.0217142857142857
# K = 1
# eps = 0.01
# sigma = 0.5

# f = QKLMS_AMK(eta=eta,epsilon=eps,mu=mu,Ka=K,A_init='pca')
# y = f.evaluate(u,d)
# mse_amk1 = f.mse_ins

# K=2
# f = QKLMS_AMK(eta=eta,epsilon=eps,mu=mu,Ka=K,A_init="pca")
# y = f.evaluate(u,d)
# mse_amk2 = f.mse_ins

# K=4
# f = QKLMS_AMK(eta=eta,epsilon=eps,mu=mu,Ka=K,A_init="pca")
# y = f.evaluate(u,d)
# mse_amk4 = f.mse_ins

# K=6
# f = QKLMS_AMK(eta=eta,epsilon=eps,mu=mu,Ka=K,A_init="pca")
# y = f.evaluate(u,d)
# mse_amk6 = f.mse_ins

# plt.title("Lorenz")
# plt.yscale("log")
# plt.plot(mse,label='QKLMS')
# plt.plot(mse_akb1,label='QKLMS AKB K=1')
# plt.plot(mse_akb2,label='QKLMS AKB K=2')
# plt.plot(mse_akb4,label='QKLMS AKB K=4')
# plt.plot(mse_akb6,label='QKLMS AKB K=6')
# plt.plot(mse_amk1,label='QKLMS AMK K=1')
# plt.plot(mse_amk2,label='QKLMS AMK K=2')
# plt.plot(mse_amk4,label='QKLMS AMK K=4')
# # plt.plot(mse_amk6,label='QKLMS AMK K=6')
# plt.legend()
# plt.show()


# #Montecarlo test 50
# ## Sistema 4.2 de AKB

# import seaborn as sns
# sns.set()

N = 1
datb = "System 4.2"

# #2 QKLMS AKB
eta = 0.584285714285714
mu = 0.05
K = 1
eps = 0.3
sigma = 0.35

# 1. QKLMS
# sgm = 0.2
# eps = 0.1
# eta = 0.2

vec = np.ones((50,))/50

mse_ = []
for _ in range(N):
    u,d = db3(samples=1000)
    f = QKLMS_AKB(eta=eta,epsilon=eps,sigma_init=sigma,mu=mu,K=K)
    y = f.evaluate(u, d)
    mse_.append((y-d[1:])**2)
    
    # f = QKLMS(epsilon=eps, sigma=sgm, eta=eta)
    # y_pred = f.evaluate(u,d)
    # mse_.append((d-np.array(y_pred).reshape(-1,1))**2)
    
mse_av = np.sum(np.array(mse_),axis=0)/N
# mse_av = np.convolve(mse_av.reshape(-1,),vec,mode='same')
    
plt.title(datb)
plt.yscale("log")
plt.plot(mse_av)
plt.ylabel("MSE")
plt.xlabel("iterations")
plt.show()

plt.title(datb)
plt.plot(d, label="target")
plt.plot(y_pred, label="predict")
plt.legend()
plt.show()


    
    
    
## TESTING MSE  
# The other approach is by setting aside a testing data set before the training.
# For each iteration, we have the weight estimate w(i).We compute the mean
# square error on the testing data set by using w(i).
    
    
# ## LORENZ

## Lorenz
# u,d = db(samples=1200, system='lorenz',L=40)
# datb = "lorenz"

## Sistema 4.2 de AKB
# datb = "System 4.2"
# u,d = db2(samples=6000)

## Sistema 4.1 de AKB
u,d = db3(samples=6000)
datb = "System 4.1"
from sklearn.model_selection import train_test_split
u_train, u_test, d_train, d_test = train_test_split(u,d, test_size=1/6, shuffle=False)

# 1. QKLMS
sgm = 0.2
eps = 0.1
eta = 0.2
f = QKLMS(epsilon=eps, sigma=sgm, eta=eta)


# #3 QKLMS AMK
# eta = 1
# mu = 0.02
# K = 1
# eps = 0.01
# sigma = 0.5
# f = QKLMS_AMK(eta=eta,epsilon=eps,mu=mu,Ka=K,A_init='pca')


yp = f.evaluate(u_train,d_train)
mse_ = (d_train - np.array(yp).reshape(-1,1))**2
plt.title(datb)
plt.figure(figsize=(15,9))
plt.yscale("log")
plt.ylabel("Training MSE")
plt.xlabel("iterations")
plt.plot(mse_,"r")
plt.show()


plt.title(datb)
plt.figure(figsize=(15,9))
plt.plot(d_train, label="target")
plt.plot(yp, label="predict")
plt.legend()
plt.show()

y_pred_test = f.predict(u_test)
# y_pred_test = f.predict(u_train)


mse = (d_test - y_pred_test)**2
# mse = (d_train - y_pred_test)**2
plt.title(datb)
plt.yscale("log")
plt.ylabel("Testing MSE")
plt.xlabel("iterations")
plt.plot(mse,"r")
plt.show()

plt.title(datb)
plt.plot(d_test,label="target")
# plt.plot(d_train,label="target")
plt.plot(y_pred_test,label="predict")
plt.legend()
plt.show()


sigma = np.linspace(1e-6,1,10)
epsilon = np.linspace(1e-6,1,10)

from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

params = [{'s':s, 'e':ep} for s in sigma for ep in epsilon]
    
results = []
u,d = db3(samples=1000)

for p in tqdm(params):
    try:
        f = QKLMS(sigma=p["s"],epsilon=p["e"])
        y = f.evaluate(u,d)
        y = np.array(y).reshape(-1,1)
        # y_ = np.sum(y)        
        p['r2'] = r2_score(d,y)
        p['mse'] = mean_squared_error(d,y)
    except:
        p['r2'] = np.nan
        p['mse'] = np.nan
    results.append(p)

import pandas as pd   
Df = pd.DataFrame(data=results)
Df.to_csv('QKLMS_4-1.csv')
