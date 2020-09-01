# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 09:25:37 2020

@author: USUARIO

GMM Test
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from matplotlib.patches import Ellipse
       

""" DATOS - ATRACTOR DE LORENZ """
import TimeSeriesGenerator as tsg
samples = 400
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="lorenz")


ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1)
d = z[-samples-1:-1].reshape(-1,1)


""" GAUSSIAN MIXTURE MODEL """

clusters = 7
from sklearn.mixture import GaussianMixture as GMM
gmm = GMM(n_components=clusters).fit(u)
labels = gmm.predict(u)

fig, ax = plt.subplots()
ax.scatter(u[:, 0], u[:, 1], c=labels, s=40, cmap='viridis')
plt.ylim([-30,50])
plt.xlim([-30,30])
plt.title("GMM for lorenz attractor for {} clusters".format(clusters))
# plot_gmm(gmm, u)
for pos, covar in zip(gmm.means_, gmm.covariances_):
    confidence_ellipse(cov=covar, mean=pos, ax=ax, n_std=3, edgecolor='red')
plt.scatter(gmm.means_[:,0], gmm.means_[:,1], color="magenta", marker="x", label="means")
plt.legend()
plt.show()

""" PRUEBA QKLMS CON GMMs para 7 cl"""
import KAF
fil = KAF.GMM_KLMS(clusters=clusters)#clusters=int(cl)
fil.fit(u,d)
y = fil.predict(u)
plt.title("No. de clusters = {}".format(clusters))
plt.plot(y, label="predict")
plt.plot(d, label="target")
plt.show()
        
""" PRUEBA QKLMS CON GMMs para diferentes cl"""
# import KAF
# clus = np.linspace(2,400,40)
# for cl in clus:
#     fil = KAF.GMM_KLMS(clusters=cl)#clusters=int(cl)
#     fil.fit(u,d)
#     y = fil.predict(u)
#     plt.title("No. de clusters = {}".format(cl))
#     plt.plot(y, label="predict")
#     plt.plot(d, label="target")
#     plt.show()

