# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 08:56:46 2021

@author: Asus
"""
import pandas as pd 
import mne
from kurtosis import KURTOSIS
from adhd_theta_beta import Ratio_Theta_Beta
import scipy.io as sio 
from mne.decoding import CSP
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import signal
import matplotlib.pyplot as plt
from scipy.integrate import simps

path = r'G:\Mi unidad\Trabajo\2021\Kurtosis\TBR\data\data\adhd\set\SST_DC_AM_new.set'
labels = r'G:\Mi unidad\Trabajo\2021\Kurtosis\TBR\data\data\adhd\csv\SST_DC_AM_new.csv'


x = mne.read_epochs_eeglab(path)

X = x.get_data()

y = pd.read_csv(labels)['Block']


 
X_filt = KURTOSIS(th = 3).fit_transform(X,y)



def indices(fvec,flow,fhigh):
        al = abs(fvec-flow*np.ones([fvec.shape[0]])).tolist()
        indl = al.index(min(al))
        ah = abs(fvec-fhigh*np.ones([fvec.shape[0]])).tolist()
        indh = ah.index(min(ah))
        return np.arange(indl,indh+1,1)

Theta_Beta = []

for i in range(X_filt.shape[0]):
    
    freq, Pow = signal.welch(X_filt[i], fs=250, nperseg=125)
    # plt.plot(freq[:13], Pow[0][:13])
    # plt.xticks(np.arange(0,30,2))
    # plt.grid()
    
    low_theta, high_theta = 4, 8
    low_beta, high_beta = 12.5, 25
    
    
    idx_theta = indices(freq, low_theta, high_theta)
    
    idx_beta = indices(freq, low_beta, high_beta)
    
    freq_res = freq[1] - freq[0] 
    
    Pow_1 = Pow.T[idx_theta]
    Pow_2 = Pow.T[idx_beta] 
    
    theta_power = simps(Pow_1, dx=freq_res, axis = 0)
    beta_power = simps(Pow_2, dx=freq_res, axis = 0)
    ratio = (theta_power/beta_power)
    Theta_Beta.append(ratio)
    


# Theta_Beta = Ratio_Theta_Beta(fs=128,nperseg=0.5,window='hann',
#                                noverlap=0.5).fit_transform(X_filt,y)



Xdata = np.array(Theta_Beta)

lda = LDA(store_covariance=True)



fit_ = lda.fit(Xdata, y)


weights = fit_.coef_
cov = fit_.covariance_
means = fit_.means_

num = weights @ means.T

den = weights @ cov @ weights.T







