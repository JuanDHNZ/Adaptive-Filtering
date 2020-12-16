# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 14:15:13 2020

@author: Juan

Graficas de comparacion

"""

def dbPlot(u, d, sgm, eps, r2_umbral,clusters, testName):
    """
    Comparativa entre QKLMS tradicional y QKLMS usando GMMs:           
        - Se hace gridSearch a los parametros de QKLMS
        - Se entrena el GMM-QKLMS
        - Se grafican las curvas           
    """
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
    
def dbPlot2(u,d,clusters_gmm, wcp, testName):
    """Comparativa entre GMM-QKLMS y BGMM-QKLMS"""
    import search
    import numpy as np
    gmm = search.searchGMMCurve(u,d,clusters_gmm)
    
    bgmm = search.searchBGMMCurve(u,d,wcp)
    
    #GRAFICAS :
    # import matplotlib as mpl
    import matplotlib.pyplot as plt
    #GMM
    r2gmm = gmm.cv_results_['mean_test_score']
    # mpl.rcParams['lines.linewidth'] = 2
    # mpl.rcParams['lines.linestyle'] = '--'
    # plt.yticks(np.linspace(0,1,11))
    # plt.xticks(np.linspace(0,samples,11))
    plt.ylim([0,1])
    plt.ylabel("R2")
    plt.xlabel("Codebook Size")
    plt.plot(clusters_gmm.astype(np.int64),r2gmm,'m', label="GMM")
    plt.plot(clusters_gmm.astype(np.int64),r2gmm,'ro',alpha=0.3)
    plt.show()
    r2bgmm = bgmm.cv_results_['mean_test_score']
    Mcl = u.shape[0]
    Nwcp = len(wcp)
    rtest = r2bgmm.reshape(Mcl,Nwcp)
    rtest = rtest.T
    print("result shape: {}".format(rtest.shape))
    
    import matplotlib as mpl
    import matplotlib.pylab as pl
    n = len(wcp)
    colors = pl.cm.jet(np.linspace(0,1,n))
    fig, ax = plt.subplots()
    norm = mpl.colors.Normalize(min(wcp),max(wcp)) 
    for i in range(rtest.shape[0]):
        plt.plot(clusters_bgmm,rtest[i,:], color=colors[i])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax)
    cbar.set_label('wcp')
    plt.ylabel("R2")
    plt.xlabel("Codebook Size")
    plt.title("Bayesian GMM")
    plt.ylim([0,1])
    plt.savefig("pruebasGMM/GMM_Vs_BGMM/"+ testName +".png", dpi = 300)
    plt.show()
    
    # print("r2 shape = ", r2bgmm.shape)
    # print("r2 shape = ", r2bgmm.shape)
    # plt.plot(clusters_bgmm.astype(np.int64),r2bgmm,'c', label="BGMM")
    # plt.plot(clusters_bgmm.astype(np.int64),r2bgmm,'bo',alpha=0.3)
    
    # plt.savefig("pruebasGMM/GMM_Vs_BGMM/"+ testName +".png", dpi = 300)
    # plt.legend()
    # plt.show()
    
       
def dbPlot3(u,d,clusters_gmm, wcp, testName):
    """Comparativa entre GMM-QKLMS y BGMM-QKLMS"""
    import search
    import numpy as np
    gmm = search.searchGMMCurve(u,d,clusters_gmm)  
    cl = u.shape[0]
    n_comps, r2bgmm = search.searchBGMM(u,d,wcp)
    
    #GRAFICAS :
    # import matplotlib as mpl
    import matplotlib.pyplot as plt
    #GMM
    r2gmm = gmm.cv_results_['mean_test_score']
  
    import matplotlib as mpl
    import matplotlib.pylab as pl
    n = len(wcp)
    colors = pl.cm.jet(np.linspace(0,1,n))
    fig, ax = plt.subplots()
    
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    
    norm = mpl.colors.Normalize(min(wcp),max(wcp)) 
    for i in range(n):
        plt.scatter(n_comps[i],r2bgmm[i], color=colors[i], alpha=0.5)

    plt.ylabel("R2")
    plt.xlabel("Codebook Size")
    plt.plot(clusters_gmm.astype(np.int64),r2gmm,'m', label="GMM")
    plt.plot(clusters_gmm.astype(np.int64),r2gmm,'ro',alpha=0.3)
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax)
    cbar.set_label('WCP')
    plt.ylabel("R2")
    plt.xlabel("Codebook SizSe")
    plt.title(testName)
    plt.ylim([0,1])
    plt.ylim([0,1])
    plt.grid()
    plt.legend()
    plt.savefig("pruebasGMM/GMM_Vs_BGMM/t2/"+ testName +".png", dpi = 300)
    plt.show()
    
    
def dbPlot4(u,d,clusters_gmm, wcp, testName):
    """
    Mejor resultado entre GMM-QKLMS y BGMM-QKLMS usando un criterio de distancia.
    
    Se mide la distacia del resultado obtenido respecto al resultado ideal de R2 = 1 y codebookSize = 0
    
    Se retorna el mejor resultado para cada modelo por prueba realizada
    
    """
    import search
    import numpy as np
    gmm = search.searchGMMCurve(u,d,clusters_gmm)  
    n_comps, r2bgmm = search.searchBGMM(u,d,wcp)    
    #GMM
    r2gmm = np.asarray(gmm.cv_results_['mean_test_score']).reshape(-1,1) #Se ajusta tamaño
    clusters_gmm = clusters_gmm.reshape(-1,1) #Se ajusta el tamaño
    norm_clusters_gmm = clusters_gmm/max(clusters_gmm)
    #BGMM
    r2bgmm = np.asarray(r2bgmm).reshape(-1,1)
    n_comps = np.asarray(n_comps).reshape(-1,1) #Se juasta tamño
    norm_comps = n_comps/max(n_comps)
    #Referencia
    ref = np.array((0,1)).reshape(1,-1)
    # Se concatenan los resultados para cada prueba
    gmm_results = np.concatenate((norm_clusters_gmm,r2gmm),axis=1) #gmm
    bgmm_results = np.concatenate((norm_comps,r2bgmm),axis=1) #bgmm
    from scipy.spatial.distance import cdist
    #Se miden las distancias respecto a la referencia
    dist_gmm = cdist(gmm_results, ref)
    dist_bgmm = cdist(bgmm_results,ref)
    #Se toma el idex de la menos distancia para cada prueba
    gmm_index = np.argmin(dist_gmm)#gmm
    bgmm_index = np.argmin(dist_bgmm)#bgmm
    
    best_gmm_cl = clusters_gmm[gmm_index]
    best_gmm_r2 = r2gmm[gmm_index]
    best_bgmm_cl = n_comps[bgmm_index]
    best_bgmm_r2 = r2bgmm[bgmm_index]
    
    return best_gmm_cl.item(), best_gmm_r2.item(), best_bgmm_cl.item(), best_bgmm_r2.item()


def KRLS_ALD_PLOT(u,d,sgm,eps):
    import search
    results = search.gridSearchKRLS(u,d,sgm,eps)

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    
    plt.plot(d,"r*",label="target")
    for r,index in zip(results,range(sgm.shape[0])):
        plt.plot(r,label='s={:.1f} & e={:.1f}'.format(sgm[index],eps[index]))     
    plt.legend()
    plt.show()
    
    
    


    
    
    
    
    
    
    
    
    
    

