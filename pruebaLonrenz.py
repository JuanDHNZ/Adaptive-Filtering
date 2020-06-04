# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:04:09 2020

@author: Juan David
"""


import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


dt = 0.01
num_steps = 10000

# Need one more for the initial values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Set initial values
xs[0], ys[0], zs[0] = (0., 1., 1.05)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)


# # Plot atractor
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(xs, ys, zs, lw=0.5)
# ax.set_xlabel("X Axis")
# ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
# ax.set_title("Lorenz Attractor")
# plt.show()
    
""" Prueba de QKLMS y MQKLMS en atractor de lorentz """
import KAF
from sklearn.metrics import mean_squared_error

samples = 400

ua = xs[-samples-2:-2].reshape(-1,1)
ub = xs[-samples-3:-3].reshape(-1,1)
uc = xs[-samples-4:-4].reshape(-1,1)
ud = xs[-samples-5:-5].reshape(-1,1)

u3 = np.concatenate((ua,ub,uc,ud), axis=1) 
u3 = -u3

d3 = xs[-samples-1:-1].reshape(-1,1)

sigmas = np.logspace(1,4,20)
mse_QKLMS = []
mse_QKLMS2 = []
CB_size1 = []
CB_size2 = []

for s in sigmas:
    #QKLMS normal
    filtro1 = KAF.QKLMS(epsilon=200,sigma=s)
    out1 = filtro1.evaluate(u3,d3)
    mse_QKLMS.append(mean_squared_error(d3, out1))
    CB_size1.append(filtro1.CB_growth[-1])
    #QKLMS con distancia de Mahalanobis
    filtro2 = KAF.QKLMS2(epsilon=200,sigma=s)
    out2 = filtro2.evaluate(u3,d3)
    mse_QKLMS2.append(mean_squared_error(d3, out2))
    CB_size2.append(filtro2.CB_growth[-1])
    
plt.figure(2)    
plt.title("Segmento atractor de lorenz - Sigma logaritmico")
plt.yscale("log")
# plt.xscale("log")
plt.xlabel("Sigma")
plt.ylabel("MSE")
plt.plot(sigmas,mse_QKLMS, 'b', marker="o", label="QKLMS")
plt.plot(sigmas,mse_QKLMS2, 'm', marker="o", label="QKLMS2")
plt.legend()
plt.figure(3)    
plt.title("Segmento atractor de lorenz - Sigma logaritmico")
# plt.yscale("log")
# plt.xscale("log")
plt.xlabel("Sigma")
plt.ylabel("Tamaño CB")
plt.plot(sigmas,CB_size1, 'b', marker="o", label="QKLMS")
plt.plot(sigmas,CB_size2, 'm', marker="o", label="QKLMS2")
plt.legend()
print("**********************************************")
print("Codebook Size QKLMS: ", CB_size1[-1])
print("Codebook Size: QKLMS2", CB_size2[-1])
print("Sigma de minimo MSE QKLMS = ", sigmas[np.argmin(mse_QKLMS)])
print("Sigma de minimo MSE QKLMS2 = ", sigmas[np.argmin(mse_QKLMS2)])