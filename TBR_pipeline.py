# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 09:38:12 2021

@author: Juan David
"""

import argparse
import scipy.io as sio
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel,SelectKBest,mutual_info_classif
# from fbcsp import FBCSP
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from scipy.stats import randint,uniform

from sklearn.metrics import cohen_kappa_score,make_scorer 

from adhd_theta_beta import Ratio_Theta_Beta
from OA import artifact_removal

#%%
def labeler(subj):
    if (subj.get('label') == 'Control'):
        label = np.zeros(subj.get('smiley').shape[2])
        
    elif(subj.get('label') == 'tdah'):
        label = np.ones(subj.get('smiley').shape[2])   
    return label

#%%
#-----------------------------------------------------
# 1 . Get both data and save paths

# parser = argparse.ArgumentParser(description='THETA_BETA_RATIO -> MIBIF -> SVC.')
# parser.add_argument('--input',required=True, help='Input filename with path')
# parser.add_argument('--out',required=True, help='Input savename with path')

# args = parser.parse_args()

# filename = args.input
# savename = args.out


#-----------------------------------------------------
# PATHS ONLY FOR TESTING 
filename = '../Dataset/adhd_Trial_reward_allchannels_hmm.pkl'
savename = "../Dataset/test.csv"

#-----------------------------------------------------
# 2. Load data

with open(filename,'rb') as f: 
    DB = pickle.load(f) 

x = [i for i in DB if (i.get('cond') == 'IC')]

X_ = []
labels_ = []

for subj in x:
    X_.append(subj.get('smiley'))
    arr = labeler(subj)
    labels_.append(arr)

X = np.concatenate((X_), axis = 2)
Xdata = np.transpose(X, [2,0,1])
labels = np.concatenate((labels_))
fs = 250

#-----------------------------------------------------
# 3. Pipeline definition
## TBR DEFAULT PARAMETERS
steps = [ ('clean', artifact_removal()), ## ICA -> artifact estimation and removal
          ('extract', Ratio_Theta_Beta(fs, nperseg=0.5,window='hann',noverlap=0.5)), ## TBR feature extraction
          ('select', LDA())
        ]

pipeline = Pipeline(steps = steps)

threshold = 1.5

param_dist = {'clean__th': [threshold],
              'extract__fs':[fs],
              'extract__nperseg':[0.5],
              'extract__window':['hann'],
              'extract__noverlap':[0.5]
              }

kappa_corr = lambda target,output : (cohen_kappa_score(target,output)+1)/2


search = RandomizedSearchCV(pipeline, param_distributions=param_dist,
                            scoring=make_scorer(kappa_corr),
                            n_iter=5,n_jobs=5,verbose=10,cv=10)

search.fit(Xdata,labels)
cv_results = search.cv_results_
cv_results = pd.DataFrame.from_dict(cv_results)
cv_results.to_csv(savename)



for r in range(49):
  search.fit(Xdata,labels)
  r_results = search.cv_results_
  r_results = pd.DataFrame.from_dict(r_results)
  cv_results=cv_results.append(r_results)  
  cv_results.to_csv(savename)

