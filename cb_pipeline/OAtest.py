# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:28:29 2021

@author: Juan David
"""


# Pipeline test for OA artifact removal

import argparse
import pandas as pd
from kurtosis import KURTOSIS
import numpy as np
import pickle 
from tbr import TBR
from fisher import FISHER
from sklearn.pipeline import Pipeline
from sklearn.metrics import cohen_kappa_score,make_scorer
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from OA import artifact_removal
# path = r'G:\Mi unidad\Trabajo\2021\Kurtosis\TBR\data\data\adhd\set\SST_DC_AM_new.set'
# labels = r'G:\Mi unidad\Trabajo\2021\Kurtosis\TBR\data\data\adhd\csv\SST_DC_AM_new.csv'
# path = r'G:\Mi unidad\Trabajo\2021\Kurtosis\TBR\data\pkl\adhd_Trial_reward_allchannels_hmm.pkl'


#-----------------------------------------------------
# 1 . Get both data and save paths

parser = argparse.ArgumentParser(description='AR -> THETA_BETA_RATIO -> LDA')
parser.add_argument('--input',required=True, help='Input filename with path')
parser.add_argument('--out',required=True, help='Input savename with path')

args = parser.parse_args()

filename = args.input
savename = args.out

# filename = '../../Dataset/adhd_Trial_reward_allchannels_hmm.pkl'
# savename = 'randomSearch.csv'
# savename = args.out

def score(fisher,labels):
    import numpy as np
    sb = (np.mean(fisher[labels>0.5])-np.mean(fisher[labels<0.5]))**2
    sw = np.var(fisher)
    return sb/sw

""".pkl load dataset"""
# with open(filename,'rb') as f: 
#     Data = pickle.load(f) 

# x = [i for i in Data if (i.get('cond') == 'IC')]

# X_ = []
# labels_ = []
# for i,j in enumerate(x):
#     X_.append(j.get('smiley'))     
    
#     if (j.get('label') == 'Control'):
#         arr = np.zeros(j.get('smiley').shape[2])
    
#     elif(j.get('label') == 'tdah'):
#         arr = np.ones(j.get('smiley').shape[2])   
#     labels_.append(arr)

    
# X = np.concatenate((X_), axis = 2)
# Xdata = np.transpose(X, [2,0,1])

# labels = np.concatenate((labels_)).reshape(-1,)

""".mat load dataset"""
import scipy.io as sio
data = sio.loadmat(filename)
Xdata = data['X']
labels = data['labels'].reshape(-1,)
fs = int(data['fs'].reshape(-1,))
print('Loading',filename,'with sampling frequency of',fs,'Hz.')
Xdata = np.transpose(Xdata,(2,1,0)) #trials x ch x time

#%% MI TEST PIPELINE

from OA import artifact_removal_with_MAC as AR

steps = [('ar', artifact_removal()),
         ('extract', TBR()),
         ('classify', FISHER())]

pipeline = Pipeline(steps)

grid = {'ar__th':[args.th],
        'extract__fs':[250]}

search = RandomizedSearchCV(pipeline, param_distributions=grid,return_train_score=True,
                            scoring=make_scorer(score),
                            n_iter=5,verbose=10,cv=10)
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


from datetime import datetime
now = datetime.now()
print(now)



