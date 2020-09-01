# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:08:02 2020

@author: USUARIO
"""

import numpy as np
import TimeSeriesGenerator as tsg
import KAF
from sklearn.metrics import r2_score


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
from matplotlib.patches import Ellipse
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

"""NUMERO DE MUESTRAS PARA LAS PRUEBAS"""
samples = 1000

# """
#     ATRACTOR DE LORENZ

# """
# x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="lorenz")
# ua = x[-samples-2:-2].reshape(-1,1)
# ub = y[-samples-3:-3].reshape(-1,1)
# u = np.concatenate((ua,ub), axis=1) # INPUT
# d = z[-samples-1:-1].reshape(-1,1) #TARGET


import testSystems
import matplotlib.pyplot as plt
"""Sistema 3 """
samples = 500
offset = 10
s = testSystems.testSystems(samples=samples+offset , systemType="3")
s = s.to_numpy()

ua = s[-samples-2:-2].reshape(-1,1)
ub = s[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1)
d = s[-samples-1:-1].reshape(-1,1 )

out1 = []
out2 = []
r2_filtro1 = []
r2_filtro2 = []
CB_size1 = []
CB_size2 = []
sigma_track = []
epsilon_track = []

# epsilonList = np.logspace(0, 2, 20)
# sigmaList = np.logspace(0, 2, 20)

epsilonList = np.logspace(1, 3, 20)
sigmaList = np.logspace(1,3,20)

           
for sigma in sigmaList:
    for epsilon in epsilonList:
        filtro1 = KAF.QKLMS(epsilon=epsilon,sigma=sigma)
        filtro2 = KAF.QKLMS2(epsilon=epsilon, sigma=sigma)
        sigma_track.append(sigma)
        epsilon_track.append(epsilon)
        for i in range(len(d)):
            out1.append(filtro1.evaluate(u[i],d[i]))                        
            out2.append(filtro2.evaluate(u[i],d[i]))
            
        #Remove NoneTypes that result from initialization 
        out1 = [j.item() for j in out1 if j is not None]
        out2 = [j.item() for j in out2 if j is not None]
             
        r2_filtro1.append(r2_score(d[1:], out1))
        r2_filtro2.append(r2_score(d[1:], out2))
        CB_size1.append(len(filtro1.CB))
        CB_size2.append(len(filtro2.CB))
        out1.clear()
        out2.clear()
        

""" Graficar curvas de parametros"""
Ns = len(sigmaList)
Ne = len(epsilonList)
r2_filtro1_ = np.asarray(r2_filtro1).reshape([Ns,Ne])
CB_size1_ = np.asarray(CB_size1).reshape([Ns,Ne])  
r2_filtro2_ = np.asarray(r2_filtro2).reshape([Ns,Ne])
CB_size2_ = np.asarray(CB_size2).reshape([Ns,Ne])

import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt

""" MAHALANOBIS QKLMS """

n = len(sigmaList)
colors = pl.cm.jet(np.linspace(0,1,n))

fig, ax = plt.subplots()
norm = mpl.colors.Normalize(min(sigmaList),max(sigmaList))


for i in range(Ns):    
    im = ax.plot(CB_size2_[i],r2_filtro2_[i], color=colors[i])
plt.ylim([0,1])
plt.ylabel("R2")
plt.xlabel("Codebook Size")
plt.title("QKLMS 2 ")
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax)
cbar.set_label('sigma')
# plt.savefig("pruebasGMM/MQKLMS_sig_10_100_2.png", dpi = 300)
plt.show()    

""" QKLMS """
n = len(sigmaList)
colors = pl.cm.jet(np.linspace(0,1,n))

fig, ax = plt.subplots()
norm = mpl.colors.Normalize(min(sigmaList),max(sigmaList))
norm.autoscale(sigmaList)

for i in range(Ns):    
    im = ax.plot(CB_size1_[i],r2_filtro1_[i], color=colors[i])
plt.ylim([0,1])
plt.ylabel("R2")
plt.xlabel("Codebook Size")
plt.title("QKLMS")
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax)
cbar.set_label('sigma')
# plt.savefig("pruebasGMM/QKLMS_sig_10_100_2.png", dpi = 300)
plt.show()    

""" BUSQUEDA DE PARAMETROS """
import search
sg1, ep1, sg2, ep2 = search.pSearchCurve(u=u, d=d, sigmaList = sigmaList, epsilonList = epsilonList, r2_threshold=0.0)
search.parameterTest(u,d,sg1, ep1, sg2, ep2)


import KAF
sg1 = 20.69
ep1 = 23.35

sg2 = 33.59
ep2 = 69.51

d_est = []
fil = KAF.QKLMS(sigma=sg1,epsilon=ep1)
for i in range(len(d)):
   d_est.append(fil.evaluate(u[i],d[i]))

fig, ax = plt.subplots()
ax.scatter(u[:,0],u[:,1])
plt.scatter(fil.CB[0][0],fil.CB[0][1],color='red')
plt.ylim([-40,40])
plt.xlim([-40,40])
import matplotlib as mpl
circle = mpl.patches.Circle(tuple(fil.CB[0]), radius=ep1, edgecolor='red',facecolor='none')
ax.add_patch(circle)
plt.savefig("pruebasGMM/QKLMS.png", dpi = 300)
plt.show()

d_est2 = []
fil2 = KAF.QKLMS2(sigma=sg2,epsilon=ep2)
for i in range(len(d)):
   d_est2.append(fil2.evaluate(u[i],d[i]))

fig, ax = plt.subplots()
ax.scatter(u[:,0],u[:,1])
means = np.asarray(fil2.CB)
cov = [fil2.CB_cov[k]/fil2.n_cov[k] for k in range(len(fil2.n_cov))]

for pos, covar in zip(means, cov):
    confidence_ellipse(cov=covar, mean=pos, ax=ax, n_std=3, edgecolor='red')

plt.scatter(np.asarray(fil2.CB)[:,0], np.asarray(fil2.CB)[:,1], color="magenta", marker="x", label="means")
plt.savefig("pruebasGMM/QKLMS.png", dpi = 300)
plt.show()

d_est = [j.item() for j in d_est2 if j is not None]
d_est2 = [j.item() for j in d_est2 if j is not None]

plt.plot(d, label="target")
plt.plot(d_est, label="predict")
plt.title("QKLMS")
plt.show()
plt.plot(d, label="target")
plt.plot(d_est2, label="predict")
plt.title("M-QKLMS")
plt.show()

# from sklearn.metrics import r2_score
# r2 = r2_score(d[1:],np.asarray(d_est2))
print("QKLMS R2 = ",r2_score(d[1:],np.asarray(d_est)))
print("M-QKLMS R2 = ",r2_score(d[1:],np.asarray(d_est2)))


