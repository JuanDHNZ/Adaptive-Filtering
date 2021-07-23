# -*- coding: utf-8 -*-
"""
Created on Tue May  4 20:59:47 2021

@author: Asus
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class FISHER():
    def __init__(self,cov=True):
        self.cov = cov
        return
    
                
    def set_params(self,cov):
        
        self.cov = cov      
        return
        
    def fit(self,X,y):
        
        lda = LDA(store_covariance=self.cov)   
        self.fit_ = lda.fit(X, y)
        return self
    
    def transform(self,X):    
        weights = self.fit_.coef_
        return X@weights.T
    

    def fit_transform(self, X,y):
        self.fit(X,y)
        return self.transform(X)
    
    def predict(self, X):
        return self.transform(X).reshape(-1,)
        