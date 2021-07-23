# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 08:25:26 2021

@author: dcard
"""

import argparse
import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel,SelectKBest,mutual_info_classif
from fbcsp import FBCSP
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from scipy.stats import randint,uniform
from mac import MAC
from sklearn.metrics import cohen_kappa_score,make_scorer

parser = argparse.ArgumentParser(description='MAC -> FBCSP -> LDA')
parser.add_argument('--input',required=True, help='Input filename with path')
parser.add_argument('--out',required=True, help='Input savename with path')

args = parser.parse_args()

filename = args.input
savename = args.out

filename = "../../data_4C/BCI_s01train.mat"

data = sio.loadmat(filename)
Xdata = data['X']
labels = data['labels'].reshape(-1,)
fs = int(data['fs'].reshape(-1,))
print('Loading',filename,'with sampling frequency of',fs,'Hz.')
Xdata = np.transpose(Xdata,(2,1,0)) #trials x ch x time

steps = [ ('preproc', MAC()),
         ('extract', FBCSP(fs,4,40,4,4,n_components=4)),
          #('select', SelectKBest()),          
          ('classify',LDA())
        ]

pipeline = Pipeline(steps = steps)

param_dist = {'extract__n_components':[4],
              'extract__fs':[fs],
              'extract__f_low':[4],
              'extract__f_high':[40],
              'extract__bandwidth':[4],
              'extract__step':[4],
              #'select__score_func':[mutual_info_classif],
              #'select__k':randint(1,145),                            
              'preproc__th':uniform(0,1),
              'preproc__ch_ref' : [0]
              }


kappa_corr = lambda target,output : (cohen_kappa_score(target,output)+1)/2

search = RandomizedSearchCV(pipeline, param_distributions=param_dist,
                            scoring=make_scorer(kappa_corr),
                            n_iter=5,n_jobs=1,verbose=10,cv=10)

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