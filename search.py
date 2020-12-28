# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:04:41 2020

@author: Juan David

Rejilla para probar diferentes sigma y epsilon

"""

def draw_ellipse(position, covariance, ax=None, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Ellipse
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
    import matplotlib.pyplot as plt
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
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Ellipse
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


def startTimer():
    import time
    return time.time()

def stopTimer(startTime):
    import time
    return time.time() - startTime


def pSearchCurve(u=None,d=None,sigmaList = None, epsilonList = None, r2_threshold = 0.9):
    if u is None or d is None or sigmaList is None or epsilonList is None:
        raise ValueError("Argument is missing")

      
    import KAF
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    
    out1 = []
    out2 = []
    r2_filtro1 = []
    r2_filtro2 = []
    CB_size1 = []
    CB_size2 = []
    sigma_track = []
    epsilon_track = []
           
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
    
    #Para graficar
    import numpy as np
    Ns = len(sigmaList)
    Ne = len(epsilonList)
    r2_filtro1_ = np.asarray(r2_filtro1).reshape([Ns,Ne])
    CB_size1_ = np.asarray(CB_size1).reshape([Ns,Ne])  
    r2_filtro2_ = np.asarray(r2_filtro2).reshape([Ns,Ne])
    CB_size2_ = np.asarray(CB_size2).reshape([Ns,Ne])
    
    return r2_filtro1_, CB_size1_, r2_filtro2_, CB_size2_
    
    
    # import matplotlib.pylab as pl
    # colors = pl.cm.jet(np.linspace(0,1,Ns))
    
    # for i in range(Ns):    
    #     plt.plot(CB_size1_[i],r2_filtro1_[i], color=colors[i])
    #     plt.ylim([0,1])
    #     plt.ylabel("R2")
    #     plt.xlabel("Codebook Size")
    #     plt.title("QKLMS")
    # plt.show()    
    # for i in range(Ns):    
    #     plt.plot(CB_size2_[i],r2_filtro2_[i], color=colors[i])
    #     plt.ylim([0,1])
    #     plt.ylabel("R2")
    #     plt.xlabel("Codebook Size")
    #     plt.title("M-QKLMS")
    # plt.show()
    
    # best_r2_index1 = [i for i in range(len(r2_filtro1)) if r2_filtro1[i] >= r2_threshold]
    # best_r2_index2 = [i for i in range(len(r2_filtro2)) if r2_filtro2[i] >= r2_threshold]
    
    # best_CB_size = u.shape[0]
    # best_CB_index1 = None
    # for i in best_r2_index1:
    #     if CB_size1[i] < best_CB_size: 
    #         best_CB_size = CB_size1[i]
    #         best_CB_index1 = i
            
    # best_CB_size = u.shape[0]
    # best_CB_index2 = None
    # for i in best_r2_index2:
    #     if CB_size2[i] < best_CB_size: 
    #         best_CB_size = CB_size2[i]
    #         best_CB_index2 = i
    
    # if(best_CB_index1 is None):
    #     raise ValueError("R2 QKLMS under the threshold")
    # if(best_CB_index2 is None):
    #     raise ValueError("R2 M-QKLMS under the threshold")
              
    # return sigma_track[best_CB_index1], epsilon_track[best_CB_index1], sigma_track[best_CB_index2], epsilon_track[best_CB_index2]
    

      
def parameterTest(u,d,sg1,sg2,ep1,ep2):
    import KAF
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    """ Para  QKLMS """
    print("QKLMS")
    print("Best Sigma = ", sg1)
    print("Best Epsilon = ", ep1)
    pred = []
    qklms = KAF.QKLMS(sigma=sg1,epsilon=ep1)
    for i in range(len(u)):
       pred.append(qklms.evaluate(u[i],d[i]))
    pred = [i.item() for i in pred if i is not None]
    #Grafico
    plt.title("QKLMS")
    plt.plot(pred, label="Predict")
    plt.plot(d[1:], label="Target")
    plt.legend()
    plt.show()
    # plt.title("QKLMS codebook growth")
    # plt.plot(qklms.CB_growth)
    # plt.show()
    
    R2_qklms = r2_score(d[1:], pred)
    print("R2 QKLMS = ", R2_qklms)
    
    """ Para  QKLMS 2 """
    print("\nM-QKLMS")
    print("Best Sigma = ", sg2)
    print("Best Epsilon = ", ep2)
    pred = []
    mqklms = KAF.QKLMS2(sigma=sg2,epsilon=ep2)
    for i in range(len(u)):
       pred.append(mqklms.evaluate(u[i],d[i]))
    pred = [i.item() for i in pred if i is not None]
    #Grafico
    plt.title("M-QKLMS")
    plt.plot(pred, label="Predict")
    plt.plot(d[1:], label="Target")
    plt.legend()
    plt.show()
    # plt.title("M-QKLMS codebook growth")
    # plt.plot(qklms.CB_growth)
    # plt.show()
    
    R2_qklms = r2_score(d[1:], pred)
    print("R2 QKLMS = ", R2_qklms)
    
    
    print("\nCodebook Sizes:")
    print("QKLMS = ", len(qklms.CB))
    print("M-QKLMS = ", len(mqklms.CB))
    

def searchGMMCurve(u=None,d=None,clusters=None):
    import numpy as np
    cl = clusters.astype(np.int64)
    parameters ={'clusters':cl}  
    import KAF
    from sklearn.model_selection import GridSearchCV
    filtro = KAF.GMM_KLMS()
    cv = [(slice(None), slice(None))]
    gmmklms = GridSearchCV(filtro,parameters,cv=cv)
    gmmklms.fit(u,d)
    return gmmklms


def searchBGMMCurve(u=None,d=None, wpc=None):
    cl = [u.shape[0]]
    parameters ={'wpc':wpc,'clusters': cl}  
    import KAF
    from sklearn.model_selection import GridSearchCV
    filtro = KAF.BGMM_KLMS()
    cv = [(slice(None), slice(None))]
    gmmklms = GridSearchCV(filtro,parameters,cv=cv)
    gmmklms.fit(u,d)
    return gmmklms

def searchBGMM(u=None,d=None, wcp=None):
   cl = u.shape[0]
   n_track = []
   r2 = []
   import KAF
   import numpy as np
   for wcp_ in wcp:
       m = KAF.BGMM_KLMS(clusters=cl, wcp=wcp_)
       m.fit(u,d)  
       n_track.append(np.sum(m.bgmm.weights_ > 0.01))
       r2.append(m.score(u,d))
   return n_track, r2

def gridSearchKRLS(u,d,sgm,eps):
    import KAF
    from sklearn.metrics import r2_score
    result = []
    for i in sgm:
        partial = []
        for j in eps:
            kf = KAF.KRLS_ALD(sigma=i,epsilon=j)
            out = kf.evaluate(u,d)
            partial.append(r2_score(d,out.reshape(-1,1)))
        result.append(partial)
    return result


def gridSearchKRLS_plot_predict(u,d,sgm,eps,testName):
    import KAF
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import r2_score
    

    sns.set()
    result = []
    for i in sgm:
        partial = []
        for j in eps:
            kf = KAF.KRLS_ALD(sigma=i,epsilon=j)
            out = kf.evaluate(u,d)
            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)
            plt.plot(d,"r", label="target")
            plt.plot(out,"g",label="predict")
            plt.legend()
            plt.title('Prueba ' + testName + "    $\sigma$ ="+ "{:.4f}".format(i) + ";  $\epsilon$ = " + "{:.4f}".format(j) + ";  R2 = " + "{:.4f}".format(r2_score(d,out)))
            plt.savefig("pruebasKRLS/predicciones/"+ testName + "sg"+"{:.4f}".format(i)+ "ep" + "{:.4f}".format(i) +".png", dpi = 300)
            plt.show()   
            
def searchQKLMS(u,d,eps,testName):
    import KAF
    from sklearn.metrics import r2_score as r2
    from sklearn.metrics import mean_squared_error as mse
    import pandas as pd
    
    print("QKLMS " + testName + " |Searching...")
    #Inicializacion     
    kf = KAF.QKLMS3(epsilon=eps[0])
    out = kf.evaluate(u,d)
    best_r2 = r2(d[1:],out)
    best_mse = mse(d[1:],out)
    best_r2_cb = len(kf.CB)
    best_r2_ep = eps[0]
    best_mse_cb = len(kf.CB)
    best_mse_ep = eps[0]
    for j in eps[1:]:
        # print("eps = ",j)
        kf = KAF.QKLMS3(epsilon=j)
        out = kf.evaluate(u,d)
        partial_r2 = r2(d[1:],out)
        partial_mse = mse(d[1:],out)
        # print("r2 = ",partial_r2)
        # print("mse = ",partial_mse)
        # print("\n")
        # print(kf.testDists)
        if partial_r2 > best_r2:
            best_r2 = partial_r2
            best_r2_cb = len(kf.CB)
            best_r2_ep = j 
        if partial_mse < best_mse:
            best_mse = partial_mse
            best_mse_cb = len(kf.CB)
            best_mse_ep = j 
        
    results = {"Best_R2":best_r2, 
               "Best_R2_CB_size":best_r2_cb, 
               "Best_R2_epsilon":best_r2_ep, 
               "Best_MSE":best_mse,
               "Best_MSE_CB_size":best_mse_cb,
               "Best_MSE_epsilon":best_mse_ep}
    
    return pd.DataFrame(data=results,index=[testName])
 
def searchKRLS_ALD(u,d,sgm,eps,testName):
    import KAF
    from sklearn.metrics import r2_score as r2
    from sklearn.metrics import mean_squared_error as mse
    import pandas as pd
    
    print("KRLS search...")
    #Inicializacion     
    kf = KAF.QKLMS3(epsilon=eps[0])
    out = kf.evaluate(u,d)
    best_r2 = r2(d[1:],out)
    best_mse = mse(d[1:],out)
    best_r2_cb = len(kf.CB)
    best_r2_ep = eps[0]
    best_r2_sgm = sgm[0]
    best_mse_cb = len(kf.CB)
    best_mse_ep = eps[0]
    best_mse_sgm = sgm[0]
    
    for i in sgm[1:]:
        for j in eps[1:]:
            # print("eps = ",j)
            kf = KAF.KRLS_ALD(epsilon=j,sigma=i)
            out = kf.evaluate(u,d)
            partial_r2 = r2(d,out.reshape(-1,1))
            partial_mse = mse(d,out.reshape(-1,1))
            # print("r2 = ",partial_r2)
            # print("mse = ",partial_mse)
            # print("\n")
            # print(kf.testDists)
            if partial_r2 > best_r2:
                best_r2 = partial_r2
                best_r2_cb = len(kf.CB)
                best_r2_ep = j 
            if partial_mse < best_mse:
                best_mse = partial_mse
                best_mse_cb = len(kf.CB)
                best_mse_ep = j 
        
    results = {"Best_R2":best_r2, 
               "Best_R2_CB_size":best_r2_cb, 
               "Best_R2_epsilon":best_r2_ep,
               "Best_R2_sigma":best_r2_sgm,
               "Best_MSE":best_mse,
               "Best_MSE_CB_size":best_mse_cb,
               "Best_MSE_epsilon":best_mse_ep,
               "Best_MSE_sigma":best_mse_sgm}
    
    return pd.DataFrame(data=results,index=[testName])   


def searchQKLMS_base(u,d,sgm,eps,testName):
    import KAF
    from sklearn.metrics import r2_score as r2
    from sklearn.metrics import mean_squared_error as mse
    import pandas as pd
    
    print("QKLMS search...")
    #Inicializacion     
    kf = KAF.QKLMS(epsilon=eps[0],sigma=sgm[0])
    out = kf.evaluate(u,d)
    best_r2 = r2(d[1:],out)
    best_mse = mse(d[1:],out)
    best_r2_cb = len(kf.CB)
    best_r2_ep = eps[0]
    best_r2_sgm = sgm[0]
    best_mse_cb = len(kf.CB)
    best_mse_ep = eps[0]
    best_mse_sgm = sgm[0]
    for i in sgm[1:]:
        for j in eps[1:]:
            # print("eps = ",j)
            kf = KAF.QKLMS(epsilon=j,sigma=i)
            out = kf.evaluate(u,d)
            partial_r2 = r2(d[1:],out)
            partial_mse = mse(d[1:],out)
            # print("r2 = ",partial_r2)
            # print("mse = ",partial_mse)
            # print("\n")
            # print(kf.testDists)
            if partial_r2 > best_r2:
                best_r2 = partial_r2
                best_r2_cb = len(kf.CB)
                best_r2_sgm = i
                best_r2_ep = j
            if partial_mse < best_mse:
                best_mse = partial_mse
                best_mse_cb = len(kf.CB)
                best_mse_ep = j 
                best_mse_sgm = i 
        
    results = {"Best_R2":best_r2, 
               "Best_R2_CB_size":best_r2_cb, 
               "Best_R2_epsilon":best_r2_ep,
               "Best_R2_sigma":best_r2_sgm,
               "Best_MSE":best_mse,
               "Best_MSE_CB_size":best_mse_cb,
               "Best_MSE_epsilon":best_mse_ep,
               "Best_MSE_sigma":best_mse_sgm}
    
    return pd.DataFrame(data=results,index=[testName])  


def searchKLMS_BGMM(u=None,d=None, wcp=None, testName="Prueba",batchSize=100):
    import KAF
    from sklearn.metrics import r2_score as r2
    from sklearn.metrics import mean_squared_error as mse
    import pandas as pd
    
    cl = batchSize
    print("KLMS BGMM in " + testName + " | Search...")
    
    u_train = u.reshape(-1,batchSize,u.shape[1]) 
    d_train = d.reshape(-1,batchSize,d.shape[1])
    
    #Inicializacion     
    kf = KAF.BGMM_KLMS(clusters=cl,wcp=wcp[0])
    for u_,d_ in zip(u_train,d_train):
        kf.fit(u_,d_)  
    out = kf.predict(u)
    best_r2 = r2(d,out)
    best_mse = mse(d,out)
    best_r2_cb = len(kf.CB)
    best_r2_wcp = wcp[0]
    best_mse_cb = len(kf.CB)
    best_mse_wcp = wcp[0]
    
    for wcp_ in wcp[1:]:
        m = KAF.BGMM_KLMS(clusters=cl, wcp=wcp_)
        for u_,d_ in zip(u_train,d_train):
            m.fit(u_,d_)  
        out = m.predict(u)
        partial_r2 = r2(d,out)
        partial_mse = mse(d,out)
        if partial_r2 > best_r2:
            best_r2 = partial_r2
            best_r2_cb = len(kf.CB)
            best_r2_wcp = wcp_
        if partial_mse < best_mse:
            best_mse = partial_mse
            best_mse_cb = len(kf.CB)
            best_mse_wcp = wcp_
        
    results = {"Best_R2":best_r2, 
               "Best_R2_CB_size":best_r2_cb, 
               "Best_R2_wcp":best_r2_wcp,
               "Best_MSE":best_mse,
               "Best_MSE_CB_size":best_mse_cb,
               "Best_MSE_wcp":best_mse_wcp}
    print("Finished")
    return pd.DataFrame(data=results,index=[testName])

def searchQKLMS_GMM(u,d,sgm,eps,testName):
    import KAF
    from sklearn.metrics import r2_score as r2
    from sklearn.metrics import mean_squared_error as mse
    import pandas as pd
    
    print("QKLMS search...")
    #Inicializacion     
    kf = KAF.QKLMS(epsilon=eps[0],sigma=sgm[0])
    out = kf.evaluate(u,d)
    best_r2 = r2(d[1:],out)
    best_mse = mse(d[1:],out)
    best_r2_cb = len(kf.CB)
    best_r2_ep = eps[0]
    best_r2_sgm = sgm[0]
    best_mse_cb = len(kf.CB)
    best_mse_ep = eps[0]
    best_mse_sgm = sgm[0]
    for i in sgm[1:]:
        for j in eps[1:]:
            # print("eps = ",j)
            kf = KAF.QKLMS3(epsilon=j,sigma=i)
            out = kf.evaluate(u,d)
            partial_r2 = r2(d[1:],out)
            partial_mse = mse(d[1:],out)
            # print("r2 = ",partial_r2)
            # print("mse = ",partial_mse)
            # print("\n")
            # print(kf.testDists)
            if partial_r2 > best_r2:
                best_r2 = partial_r2
                best_r2_cb = len(kf.CB)
                best_r2_sgm = i
                best_r2_ep = j
            if partial_mse < best_mse:
                best_mse = partial_mse
                best_mse_cb = len(kf.CB)
                best_mse_ep = j 
                best_mse_sgm = i 
        
    results = {"Best_R2":best_r2, 
               "Best_R2_CB_size":best_r2_cb, 
               "Best_R2_epsilon":best_r2_ep,
               "Best_R2_sigma":best_r2_sgm,
               "Best_MSE":best_mse,
               "Best_MSE_CB_size":best_mse_cb,
               "Best_MSE_epsilon":best_mse_ep,
               "Best_MSE_sigma":best_mse_sgm}
    
    return pd.DataFrame(data=results,index=[testName])  