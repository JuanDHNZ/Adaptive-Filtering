# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:04:41 2020

@author: Juan David

Rejilla para probar diferentes sigma y epsilon

"""

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
    
    import matplotlib.pylab as pl
    colors = pl.cm.jet(np.linspace(0,1,Ns))
    
    for i in range(Ns):    
        plt.plot(CB_size1_[i],r2_filtro1_[i], color=colors[i])
        plt.ylim([0,1])
        plt.ylabel("R2")
        plt.xlabel("Codebook Size")
        plt.title("QKLMS")
    plt.show()    
    for i in range(Ns):    
        plt.plot(CB_size2_[i],r2_filtro2_[i], color=colors[i])
        plt.ylim([0,1])
        plt.ylabel("R2")
        plt.xlabel("Codebook Size")
        plt.title("M-QKLMS")
    plt.show()
    
    best_r2_index1 = [i for i in range(len(r2_filtro1)) if r2_filtro1[i] >= r2_threshold]
    best_r2_index2 = [i for i in range(len(r2_filtro2)) if r2_filtro2[i] >= r2_threshold]
    
    best_CB_size = u.shape[0]
    best_CB_index1 = None
    for i in best_r2_index1:
        if CB_size1[i] < best_CB_size: 
            best_CB_size = CB_size1[i]
            best_CB_index1 = i
            
    best_CB_size = u.shape[0]
    best_CB_index2 = None
    for i in best_r2_index2:
        if CB_size2[i] < best_CB_size: 
            best_CB_size = CB_size2[i]
            best_CB_index2 = i
    
    if(best_CB_index1 is None):
        raise ValueError("R2 QKLMS under the threshold")
    if(best_CB_index2 is None):
        raise ValueError("R2 M-QKLMS under the threshold")
              
    return sigma_track[best_CB_index1], epsilon_track[best_CB_index1], sigma_track[best_CB_index2], epsilon_track[best_CB_index2]
    

    


