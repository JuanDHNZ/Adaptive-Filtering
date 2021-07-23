# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 20:55:34 2021

@author: Asus
"""

import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis


class KURTOSIS():
    def __init__(self, th = 0.1):
        self.th = th
        return 
    
    def set_params(self, th):
        self.th = th
        
        return

    def fit(self, X,y):
       
        N, C, T = X.shape
        
        X = np.transpose(X,(1,0,2)).reshape(C,-1) #Matriz con dimensiones C x NT
        ica = FastICA(n_components=C)
        S = ica.fit_transform(X.T)  # Reconstruct signals
        
        self.A = ica.mixing_  # Get estimated mixing matrix #Se debe guardar 
        self.W = np.linalg.inv(self.A) #Se debe guardar 
    
        S = S.reshape(N, C, T)
        
        kur = kurtosis(S, axis = 2).reshape(-1)
        
        
        self.U_k = np.mean(kur) #Se debe guardar 
        self.STD_k = np.std(kur)#Se debe guardar 
        
    
    def transform(self, X):      
        #S = self.W @ X
        S = np.array([self.W@Xn for Xn in X])
        N, D, T = S.shape
        
        kur = kurtosis(S, axis = 2)              
        k_norm = (kur - self.U_k) / self.STD_k
        
        S_ = [] 
        A_ = []
        Xrec = []
        
        for trial in range(N):            
            ind = np.where(k_norm[trial] < self.th)[0]

                   
            S_.append(S[trial,ind])
            A_.append(self.A[:,ind])


            Xrec.append(np.dot(A_[trial], S_[trial]))
        
        return np.array(Xrec)
            
            
    def fit_transform(self, X,y):
            self.fit(X,y)
            return self.transform(X)