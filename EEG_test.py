# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:34:51 2021

@author: Juan David
"""

import numpy as np 
import pickle
from EEG import TBR
from scipy.integrate import simps
from scipy import signal
import scipy.io as sio

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#-----------------------------------------------------
# PATHS ONLY FOR TESTING 
filename = "../Dataset/adhd_Trial_block_allchannels_hmm.pkl"
savename = "../data_4C/BCI_s05trainTEST.csv"

#-----------------------------------------------------
# EEG DATA LOAD PICKLE

# import pickle
# with open(filename,'rb') as f: DB = pickle.load(f)

# index = 0
# Xdata = DB[index]['block1']
# y = DB[index]['label']
# fs = 250
# print('Loading',filename,'with sampling frequency of',fs,'Hz.')
# Xdata = np.transpose(Xdata,(2,0,1)) #trials x ch x time

#-----------------------------------------------------
# EEG DATA LOAD MAT
filename = "../data_4C/BCI_s05train.mat"
savename = "../data_4C/BCI_s05trainTEST.csv"

data = sio.loadmat(filename)
Xdata = data['X']
labels = data['labels'].reshape(-1,)
fs = int(data['fs'].reshape(-1,))
print('Loading',filename,'with sampling frequency of',fs,'Hz.')

Xdata = np.transpose(Xdata,(2,1,0)) #trials x ch x time

def indices(fvec,flow,fhigh):
        al = abs(fvec-flow*np.ones([fvec.shape[0]])).tolist()
        indl = al.index(min(al))
        ah = abs(fvec-fhigh*np.ones([fvec.shape[0]])).tolist()
        indh = ah.index(min(ah))
        return np.arange(indl,indh+1,1)

Theta_Beta = []

for i in range(Xdata.shape[0]):
    
    freq, Pow = signal.welch(Xdata[i], fs=250, nperseg=125)
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
    


#%% THETA BETA CLASS
   
tbr = TBR(fs = fs)
tbr.fit(Xdata, labels)
TBratio = tbr.transform(Xdata)
#%%

# Theta_Beta = Ratio_Theta_Beta(fs=128,nperseg=0.5,window='hann',
#                                noverlap=0.5).fit_transform(X_filt,y)



# Xdata = np.array(Theta_Beta)

# lda = LDA(store_covariance=True)

# fit_ = lda.fit(Xdata, y)


# weights = fit_.coef_
# cov = fit_.covariance_
# means = fit_.means_

# num = weights @ means.T

# den = weights @ cov @ weights.T