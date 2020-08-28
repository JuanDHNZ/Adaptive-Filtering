# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:34:48 2020

@author: USUARIO
"""


import numpy as np
import KAF
import TimeSeriesGenerator as tsg
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(cov, mean, ax, n_std=3.0, facecolor='none', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


samples = 400
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="lorenz")


ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1)

d = z[-samples-1:-1].reshape(-1,1)

import matplotlib.pyplot as plt
# plt.title("Segmento de Atractor de lorenz")
# plt.plot(u)
# plt.show()
pred = []
filt = KAF.QKLMS3(epsilon=50)
for i in range(len(d)):
    pred.append(filt.evaluate(u[i],d[i]))    


plt.title("QKLMS3")
plt.plot(d, label="Target")
plt.plot(pred, label="Predict")
plt.ylim([0,60])
plt.legend()
plt.show()

u_in = u.T

#SCATTER
fig, ax = plt.subplots()
ax.scatter(u.T[0],u.T[1], alpha=0.5, label="Input")
CB = np.asarray(filt.CB)
ax.scatter(CB.T[0],CB.T[1],color='red', marker="x", label="Codebook")
plt.legend()

cov = filt.CB_cov
for i in range(len(CB)):
    confidence_ellipse(cov=cov[i], edgecolor='red', mean=CB[i], ax=ax,  n_std=0.5)
plt.show()












           
