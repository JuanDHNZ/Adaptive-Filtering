# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 20:21:25 2020

@author: Camila
"""

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import KernelPCA
from adhd_theta_beta import Ratio_Theta_Beta
from sklearn.pipeline import Pipeline
from scipy import stats
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy.io import loadmat
import seaborn as sns  
import pandas as pd
import numpy as np 
import tikzplotlib
import pickle

def num(metric):
    mean = str(np.mean(metric).round(3))
    pm = ' +/- '
    std = str(np.std(metric).round(3))
    return mean+pm+std


with open('adhd_Trial_hmm_resting.pkl','rb') as f:
    Data = pickle.load(f) 
    
data = [stats.zscore(Trial['Trial']).T for Trial in Data]
label = [T['clase'] for T in Data]
y = np.array(label)
cat_encoder = OrdinalEncoder ()
y = cat_encoder.fit_transform(y.reshape(-1,1))
y = y.reshape(y.shape[0],)  

kfd = StratifiedKFold(shuffle=True,random_state=22032020)
Theta_Beta = Ratio_Theta_Beta(fs=128,nperseg=0.5,window='hann',
                              noverlap=0.5).fit_transform(data,y)
steps = [('svc', SVC())]                     
parameters = {'svc__C':np.logspace(-6,4,1000)}
pipeline = Pipeline(steps=steps)
clf = GridSearchCV(pipeline,parameters, cv=kfd.split(Theta_Beta  ,y))
clf.fit(Theta_Beta  ,y) 
clf.best_score_

scoring = {'f1': make_scorer(f1_score),'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(recall_score,pos_label=0)}
metric = cross_validate(clf.best_estimator_,Theta_Beta,y, scoring=scoring,
                        cv=kfd.split(Theta_Beta,y))
print(num(metric["test_f1"]))
print(num(metric["test_sensitivity"]))
print(num(metric["test_specificity"]))
print(clf.best_score_)