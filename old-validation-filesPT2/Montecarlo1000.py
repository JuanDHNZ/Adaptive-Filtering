# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:55:42 2021

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

samples = 2500
L = 40 
N = 1000

# 1. Lorenz
var = 0.1

#QKLMS for lorenz
sgm = 10000
eps = 1e-1

init_b = 150

    
from tqdm import tqdm
for _ in tqdm(range(N)):
    start = np.random.randint(0,init_b)
    u,d = db(samples=samples, system="lorenz",L=L)
    
    noise = np.sqrt(var)*np.random.randn(samples-L+1).reshape(-1,1)
    u = u + noise
    d = d + noise
    
    u_train = u[start:start+1000]
    d_train = d[start:start+1000]
    
    u_test = u[start+1000:start+2000]
    d_test = d[start+1000:start+2000]
        
    mse = []
    mse_ = []
        
    f = QKLMS(epsilon=eps, sigma=sgm)
    f.evaluate(u_train,d_train)
    y_pred = f.predict(u_test)
    
    mse.append(((d_test-np.array(y_pred).reshape(-1,1))**2))
    mse_.append(((d_test-np.array(y_pred).reshape(-1,1))**2)/d_test**2)

    
mse_av = np.sum(np.array(mse),axis=0)/N
mse_av_ = np.sum(np.array(mse_),axis=0)/N

plt.figure(figsize=(15,9))
plt.title("ERROR INSTANTANEO - Lorenz - $\sigma$ = {} ; epsilon = {} ".format(sgm,eps))
plt.yscale("log")
plt.plot(mse_av,"r-",linewidth=1)
plt.ylabel("MSE")
plt.xlabel("iterations")
plt.savefig("Montecarlo1000/"+ "EI_lorenz" +".png", dpi = 300)
plt.show()

plt.figure(figsize=(15,9))
plt.title("ERROR RELATIVO - Lorenz - $\sigma$ = {} ; epsilon = {} ".format(sgm,eps))
plt.yscale("log")
plt.plot(mse_av_,"r-",linewidth=1)
plt.ylabel("MSE")
plt.xlabel("iterations")
plt.savefig("Montecarlo1000/"+ "ER_lorenz" +".png", dpi = 300)
plt.show()




# 2. Sistema 4.1 
init_b = 300

sgm = 0.24
eps = 1e-6

from tqdm import tqdm
for _ in tqdm(range(N)):
    start = np.random.randint(0,init_b)
    u,d = db3(samples=samples)
     
    u_train = u[start:start+1000]
    d_train = d[start:start+1000] + 2
    
    u_test = u[start+1000:start+2000]
    d_test = d[start+1000:start+2000] + 2
        
    mse = []
    mse_ = []
      
    
    f = QKLMS(epsilon=eps, sigma=sgm)
    f.evaluate(u_train,d_train)
    y_pred = f.predict(u_test)
    mse.append(((d_test-np.array(y_pred).reshape(-1,1))**2))
    mse_.append(((d_test-np.array(y_pred).reshape(-1,1))**2)/d_test**2)


    
mse_av = np.sum(np.array(mse),axis=0)/N
mse_av_ = np.sum(np.array(mse_),axis=0)/N

plt.figure(figsize=(15,9))
plt.title("ERROR INSTANTANEO - 4.1 - $\sigma$ = {} ; epsilon = {} ".format(sgm,eps))
plt.yscale("log")
plt.plot(mse_av,"r-",linewidth=1)
plt.ylabel("MSE")
plt.xlabel("iterations")
plt.savefig("Montecarlo1000/"+ "EI_4-1" +".png", dpi = 300)
plt.show()

plt.figure(figsize=(15,9))
plt.title("ERROR RELATIVO - 4.1 - $\sigma$ = {} ; epsilon = {}".format(sgm,eps))
plt.yscale("log")
plt.plot(mse_av_,"r-",linewidth=1)
plt.ylabel("MSE")
plt.xlabel("iterations")
plt.savefig("Montecarlo1000/"+ "ER_4-1" +".png", dpi = 300)
plt.show()




# 3.Sistema 4.2
sgm = 0.666
eps = 1e-6

from tqdm import tqdm
for _ in tqdm(range(N)):
    start = np.random.randint(0,init_b)
    u,d = db2(samples=samples)
   
    u_train = u[start:start+1000]
    d_train = d[start:start+1000] + 3
    
    u_test = u[start+1000:start+2000]
    d_test = d[start+1000:start+2000] + 3
    
    
    mse = []
    mse_ = []
      
    
    f = QKLMS(epsilon=eps, sigma=sgm)
    f.evaluate(u_train,d_train)
    y_pred = f.predict(u_test)
    
    mse.append(((d_test-np.array(y_pred).reshape(-1,1))**2))
    mse_.append(((d_test-np.array(y_pred).reshape(-1,1))**2)/d_test**2)

    
mse_av = np.sum(np.array(mse),axis=0)/N
mse_av_ = np.sum(np.array(mse_),axis=0)/N

plt.figure(figsize=(15,9))
plt.title("ERROR INSTANTANEO - 4.1 - $\sigma$ = {} ; epsilon = {} ".format(sgm,eps))
plt.yscale("log")
plt.plot(mse_av,"r-",linewidth=1)
plt.ylabel("MSE")
plt.xlabel("iterations")
plt.savefig("Montecarlo1000/"+ "EI_4-2" +".png", dpi = 300)
plt.show()

plt.figure(figsize=(15,9))
plt.title("ERROR RELATIVO - 4.1 - $\sigma$ = {} ; epsilon = {}".format(sgm,eps))
plt.yscale("log")
plt.plot(mse_av_,"r-",linewidth=1)
plt.ylabel("MSE")
plt.xlabel("iterations")
plt.savefig("Montecarlo1000/"+ "ER_4-2" +".png", dpi = 300)
plt.show()