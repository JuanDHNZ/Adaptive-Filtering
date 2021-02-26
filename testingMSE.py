# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:21:48 2021

@author: JUAN
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:41:32 2021

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
from sklearn.model_selection import train_test_split
sns.set()

samples = 2200
L = 40 
N = 1 

# 1. Lorenz
var = 0.1

#QKLMS for lorenz
sgm = 50
eps = 1

 
from tqdm import tqdm

for rep in range(100):

u,d = db(samples=samples+L-1, system="lorenz",L=L)
  
noise = np.sqrt(var)*np.random.randn(samples).reshape(-1,1)
u_train = u + noise
d_train = d + noise

u,d = db(samples=int(samples/2)+L-1, system="lorenz",L=L)

noise = np.sqrt(var)*np.random.randn(samples).reshape(-1,1)
u_test = u + noise
d_test = d + noise


mse = []
#mse_ = []
#y_ = []

f = QKLMS(epsilon=eps, sigma=sgm)
for ui,di in tqdm(zip(u_train,d_train)):
    f.evaluate(ui,di)
    y_pred = f.predict(u_test) 
#    y_.append(y_pred)
    mse.append(np.mean((d_test-np.array(y_pred).reshape(-1,1))**2/d_test**2))
    
plt.figure(figsize=(15,9))
plt.title("MSE RELATIVO - Lorenz - $\sigma$ = {} ; epsilon = {} ".format(sgm,eps))
plt.yscale("log")
# plt.ylim( (10**-4,10**0))
plt.plot(mse,"r-",linewidth=1)
plt.ylabel("MSE")
plt.xlabel("iterations")
plt.savefig("Montecarlo1000/"+ "ER_lorenz" +".png", dpi = 300)
plt.show()

# plt.title("Prediccion")
# plt.plot(d_test)
# plt.plot(y_pred)
# plt.show()

# 2. Sistema 4.1 
sgm = 0.24
eps = 1e-6

u,d = db3(samples=samples)

u_train, u_test, d_train, d_test = train_test_split(u,d, test_size=1/10, shuffle=False)

f = QKLMS(epsilon=eps, sigma=sgm)

mse = []

for ui,di in tqdm(zip(u,d)):
    f.evaluate(ui,di)
    y_pred = f.predict(u_test)
    mse.append(np.mean((d_test-np.array(y_pred).reshape(-1,1))**2/d_test**2))
    

plt.figure(figsize=(15,9))
plt.title("MSE RELATIVO - 4.2 - $\sigma$ = {} ; epsilon = {} ".format(sgm,eps))
plt.yscale("log")
# plt.ylim( (10**-4,10**3) )
plt.plot(mse,"r-",linewidth=1)
plt.ylabel("MSE")
plt.xlabel("iterations")
plt.savefig("Montecarlo1000/"+ "ER_4-2" +".png", dpi = 300)
plt.show()

# plt.title("Prediccion")
# plt.plot(d_test)
# plt.plot(y_pred)
# plt.show()


# 3.Sistema 4.2
sgm = 0.666
eps = 1e-6

u,d = db2(samples=samples)

u_train, u_test, d_train, d_test = train_test_split(u,d, test_size=1/10, shuffle=False)

f = QKLMS(epsilon=eps, sigma=sgm)

mse = []

for ui,di in tqdm(zip(u,d)):
    f.evaluate(ui,di)
    y_pred = f.predict(u_test)
    mse.append(np.mean((d_test-np.array(y_pred).reshape(-1,1))**2/d_test**2))
    
plt.figure(figsize=(15,9))
plt.title("MSE RELATIVO - 4.1 - $\sigma$ = {} ; epsilon = {} ".format(sgm,eps))
plt.yscale("log")
# plt.ylim( (10**-4,10**3))
plt.plot(mse,"r-",linewidth=1)
plt.ylabel("MSE")
plt.xlabel("iterations")
plt.savefig("Montecarlo1000/"+ "ER_4-1" +".png", dpi = 300)
plt.show()

# plt.title("Prediccion")
# plt.plot(d_test)
# plt.plot(y_pred)
# plt.show()


# 1. Chua
var = 0.1

u,d = db(samples=samples+L-1, system="lorenz",L=L)
noise = np.sqrt(var)*np.random.randn(samples).reshape(-1,1)

u_train = u + noise
d_train = d + noise

noise = np.sqrt(var)*np.random.randn(samples).reshape(-1,1)

u_test = u + noise
d_test = d + noise
 
#QKLMS for lorenz
sgm = 50
eps = 1

 
from tqdm import tqdm

mse = []
mse_ = []
y_ = []

f = QKLMS(epsilon=eps, sigma=sgm)
for ui,di in tqdm(zip(u,d)):
    f.evaluate(ui,di)
    y_pred = f.predict(u_test) 
    y_.append(y_pred)
    mse.append(np.mean((d_test-np.array(y_pred).reshape(-1,1))**2/d_test**2))
    
plt.figure(figsize=(15,9))
plt.title("MSE RELATIVO - Chua - $\sigma$ = {} ; epsilon = {} ".format(sgm,eps))
plt.yscale("log")
# plt.ylim( (10**-4,10**0))
plt.plot(mse,"r-",linewidth=1)
plt.ylabel("MSE")
plt.xlabel("iterations")
plt.savefig("Montecarlo1000/"+ "ER_Chua" +".png", dpi = 300)
plt.show()
