# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 14:15:13 2020

@author: Juan

Graficas de comparacion

"""

def dbPlot(u, d, sgm, eps, r2_umbral,clusters, testName):
    #Busqueda de parametros en QKLMS y M-QKLMS
    import search
    r2_1, cb_1, r2_2, cb_2 = search.pSearchCurve(u=u,d=d,sigmaList=sgm,epsilonList=eps,r2_threshold=r2_umbral)
    #Entrenamiento de GMM
    gmm = search.searchGMMCurve(u,d,clusters)
    
    #Grafica combinada  
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pl
    import numpy as np
    mpl.rcParams['lines.linewidth'] = 1
    mpl.rcParams['lines.linestyle'] = '-'
    #QKLMS
    n = len(sgm)
    colors = pl.cm.jet(np.linspace(0,1,n))
    fig, ax = plt.subplots()
    norm = mpl.colors.Normalize(min(sgm),max(sgm))   
    for i in range(len(sgm)):    
        ax.plot(cb_1[i],r2_1[i], color=colors[i], alpha=0.7)
    plt.ylim([0,1])
    plt.ylabel("R2")
    plt.xlabel("Codebook Size")
    plt.title("QKLMS vs GMM: " + testName)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax)
    cbar.set_label('sigma')
    # plt.savefig("pruebasGMM/MQKLMS_sig_10_100_2.png", dpi = 300)
    #GMM
    r2gmm = gmm.cv_results_['mean_test_score']
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.linestyle'] = '--'
    # plt.yticks(np.linspace(0,1,11))
    # plt.xticks(np.linspace(0,samples,11))
    plt.plot(clusters.astype(np.int64),r2gmm,'m', label="GMM")
    plt.plot(clusters.astype(np.int64),r2gmm,'ro',alpha=0.3)
    plt.savefig("pruebasGMM/gmmVsqklms/"+ testName +".png", dpi = 300)
    plt.legend()
    plt.show()  
