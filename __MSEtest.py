# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 09:21:04 2021

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

import padasip as pa 
# u,d = db(samples=5000,system='chua',L=2)
u,d = db2(samples=2000)
# u,d = db3(samples=5000)


# identification
f = pa.filters.FilterLMS(n=2, mu=0.01, w="random")
y, e, w = f.run(d,u)

# show results
plt.figure(figsize=(15,9));plt.title("Adaptation");plt.xlabel("samples - k")
plt.plot(d,"b", label="d - target");plt.plot(y,"g", label="y - output");
plt.legend();plt.show()

plt.figure(figsize=(15,9)),plt.title("Filter error");plt.xlabel("samples - k");
plt.plot(10*np.log10(e**2),"r", label="e - error [dB]");plt.legend();plt.show()



# identification
f = pa.filters.FilterRLS(n=1, mu=0.1, w="random")
y, e, w = f.run(d,u)

# show results
plt.figure(figsize=(15,9));plt.title("Adaptation");plt.xlabel("samples - k")
plt.plot(d,"b", label="d - target");plt.plot(y,"g", label="y - output");
plt.legend();plt.show()

plt.figure(figsize=(15,9)),plt.title("Filter error");plt.xlabel("samples - k");
plt.plot(10*np.log10(e**2),"r", label="e - error [dB]");plt.legend();plt.show()





# EXAMPLE FROM matousc89/Python-Adaptive-Signal-Processing-Handbook
# https://github.com/matousc89/Python-Adaptive-Signal-Processing-Handbook/blob/master/notebooks/padasip_adaptive_filters_basics.ipynb
N = 5000
n = 10
u = np.sin(np.arange(0, N/10., N/50000.))
v = np.random.normal(0, 1, N)
d = u + v

# filtering
x = pa.input_from_history(d, n)[:-1]
d = d[n:]
u = u[n:]
f = pa.filters.FilterRLS(mu=0.9, n=n)
y, e, w = f.run(d, x)

# error estimation
MSE_d = np.dot(u-d, u-d) / float(len(u))
MSE_y = np.dot(u-y, u-y) / float(len(u))

# results
plt.figure(figsize=(12.5,6))
plt.plot(u, "r:", linewidth=4, label="original")
plt.plot(d, "b", label="noisy, MSE: {}".format(MSE_d))
plt.plot(y, "g", label="filtered, MSE: {}".format(MSE_y))
plt.xlim(N-100,N)
plt.legend()
plt.tight_layout()
plt.show()