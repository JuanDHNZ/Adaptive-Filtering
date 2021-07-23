# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:23:57 2021

"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from scipy import signal
from scipy.integrate import simps

class TBR(BaseEstimator, ClassifierMixin):
    def __init__(self,fs=1.0, window='hann', nperseg=0.5, 
                 noverlap=0.5, nfft=None, detrend='constant',
                 return_onesided=True, scaling='density', 
                 axis=- 1, average='mean'):#parametros del modelo
        self.fs = fs
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.detrend = detrend
        self.return_onesided = return_onesided
        self.scaling = scaling
        self.axis = axis
        self.average = average
        
        self.low_theta = 4 
        self.high_theta = 8
        self.low_beta = 12.5
        self.high_beta = 25

    def fit(self, X, y):#entrenar modelo
        # Check that X and y have correct shape
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        # Return the classifier
        return self

    def predict(self, X):#prediccion
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]
    
    def transform(self,X):
        # comprobar si el modelo se entrenó
        check_is_fitted(self)
        np = self.nperseg*self.fs
        no = self.noverlap*np
        
        Theta_Beta = []
        
        for trial in X:
            freq, Pow = signal.welch(trial,fs= self.fs, 
                        nperseg= np,  noverlap= no, nfft=self.nfft, 
                        detrend=self.detrend,
                        return_onesided=self.return_onesided, 
                        scaling=self.scaling, axis=self.axis,
                        average=self.average)
            
            idx_theta = self.indices(freq, self.low_theta, self.high_theta)
            idx_beta = self.indices(freq, self.low_beta, self.high_beta)
            
            freq_res = freq[1] - freq[0]
            
            Pow_1 = Pow.T[idx_theta]
            Pow_2 = Pow.T[idx_beta]
            
            theta_power = simps(Pow_1, dx=freq_res, axis = 0)
            beta_power = simps(Pow_2, dx=freq_res, axis = 0)
            Theta_Beta.append(theta_power/beta_power)
        return Theta_Beta  

    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)
    
    def get_params(self, deep=True):
        return {'fs':self.fs, 'window':self.window, 'nperseg':self.nperseg, 
                 'noverlap':self.noverlap, 'nfft':self.nfft, 'detrend':self.detrend,
                 'return_onesided':self.return_onesided, 'scaling':self.scaling, 
                 'axis':self.axis, 'average':self.average}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def indices(self,fvec,flow,fhigh):     
        al = abs(fvec-flow*np.ones([fvec.shape[0]])).tolist()
        indl = al.index(min(al))
        ah = abs(fvec-fhigh*np.ones([fvec.shape[0]])).tolist()
        indh = ah.index(min(ah))
        return np.arange(indl,indh+1,1)