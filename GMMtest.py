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

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        print("WEEE")
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
        # confidence_ellipse(cov=covar, mean=pos, ax=ax, n_std=3, facecolor="red")

import matplotlib.transforms as transforms
def confidence_ellipse(cov, mean, ax, n_std=3.0, facecolor='none', edgecolor='none', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
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

