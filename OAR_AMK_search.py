# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:26:18 2021

@author: USUARIO
"""
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# 1. Load raw BCI data
folder = 'data_4c/'
filename = 'BCI_s06train.mat'
data = sio.loadmat(folder + filename)
Xdata = data['X']
Xdata = np.transpose(Xdata,(2,1,0))
labels = data['labels'].reshape(-1,)
fs = int(data['fs'].reshape(-1,))

# 2. Load kurthosis thresholds for each subject
import pandas as pd
ths = pd.read_csv("data_4c/score_filterbank.csv")['param_preproc__th'].to_list()

# 3. Artifacts removal stage
from OA import artifactRemoval
ar = artifactRemoval(th=float(ths[5]))
ar.fit(Xdata,labels)

R = ar.noiseEstimation(Xdata)

Rsum = np.array([np.sum(r) for r in R])
r_index = np.where(Rsum!=0.0)[0]

rn = R[r_index][0][0]

# rx = rn[:-1].reshape(-1,1)
# ry = rn[1:].reshape(-1,1)

""" 
For Subject 06 there are 57 trials with noise out of 219
"""

from scipy.stats import randint,uniform
param_dist = {'embedding':randint(5,10),
              'eta':uniform(0.1,0.9),
              'epsilon':uniform(1e-1,2),
              'mu': uniform(1e-2,1),
              "Ka": randint(5,15)
              }

from KAF import QKLMS_AMK
f = QKLMS_AMK()
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(f, param_distributions=param_dist,
                            scoring='r2',
                            n_iter=5,verbose=10,cv=5)

search.fit(rn,rn)

cv_results = search.cv_results_

cv_results = pd.DataFrame.from_dict(cv_results)
cv_results.to_csv("random_sarch_t0_c0.csv")

# for r in range(49):
#   X, y = embedderForSearch(rn[], signalEmbedding, channel, singleRunDataSize)
#   search.fit(Xdata,labels)
#   r_results = search.cv_results_
#   r_results = pd.DataFrame.from_dict(r_results)
#   cv_results=cv_results.append(r_results)  
#   cv_results.to_csv(savename)
  




