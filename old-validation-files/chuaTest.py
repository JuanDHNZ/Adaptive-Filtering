# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 12:14:01 2020

@author: Juan David
"""
import numpy as np
import TimeSeriesGenerator as tsg
import matplotlib.pyplot as plt
import KAF

samples = 1000
x, y, z = tsg.chaoticSystem(samples=samples+100,systemType="")

ds = x[-samples-1:-1]

u1 = x[-samples-2:-2].reshape(-1,1)
u2 = x[-samples-3:-3].reshape(-1,1)
u = np.concatenate((u1,u2), axis=1) 


# plt.plot(u)
# plt.title("Fracmento de atractor de chua")

