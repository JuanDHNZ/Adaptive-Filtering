# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:29:55 2020

@author: USUARIO
"""
import numpy as 

def chaoticSystem(samples = 1000,systemType = None):
    if systemType == None:
        raise ValueError('System type is missing')
    X = np.empty(samples + 1)
    Y = np.empty(samples + 1)
    Z = np.empty(samples + 1)
    
    dt = 0.01
    
    # Set initial values
    X[0], Y[0], Z[0] = (0., 1., 1.05)

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(samples):
        x_dot, y_dot, z_dot = lorenz(X[i], Y[i], Z[i])
        X[i + 1] = X[i] + (x_dot * dt)
        Y[i + 1] = Y[i] + (y_dot * dt)
        Z[i + 1] = Z[i] + (z_dot * dt)
