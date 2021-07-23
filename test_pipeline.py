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

from OA import artifact_removal_with_MAC as ar, artifact_removal_MAC_AMK as ar_amk

from sklearn.metrics import cohen_kappa_score,make_scorer 

parser = argparse.ArgumentParser(description='FBCSP -> MIBIF -> SVC.')
parser.add_argument('--input',required=True, help='Input filename with path')
parser.add_argument('--params',required=True, help='Parameters search results file')

args = parser.parse_args()

filename = args.input
savename = args.out
params_file = args.params

## PATHS ONLY FOR TESTING 
filetrain = "../data_4C/BCI_s09train.mat"
filetest = "../data_4C/BCI_s09test.mat"
params = "../ResultsKLMS/fBCI_s09train.csv"

# 1. load BCI data train
data = sio.loadmat(filetrain)
X_train = data['X']
y_train = data['labels'].reshape(-1,)
fs = int(data['fs'].reshape(-1,))
print('Loading',filetrain,'with sampling frequency of',fs,'Hz.')
X_train = np.transpose(X_train,(2,1,0)) #trials x ch x time

print(X_train.shape, y_train.shape)

# 2. Parameter selection
parameters = pd.read_csv(params)
bp = parameters[parameters.mean_test_score == parameters.mean_test_score.max()].iloc[0]

# AMK parameters


# 3. Pipeline definition and trainnig
steps = [ ('artifact_removal', ar(th=0.890070294773727)),
          ('extract', FBCSP(fs,4,40,4,4,n_components=4)),
          ('select', SelectKBest(score_func=mutual_info_classif,k=bp.param_select__k)),          
          ('classify',SVC(kernel='linear', C=bp.param_classify__C))
        ]

# ('artifact_removal', ar(th=0.890070294773727))

pipeline = Pipeline(steps = steps)

pipeline.fit(X_train,y_train) 

# 4. Prediction
data = sio.loadmat(filetest)
X_test = data['X']
y_test = data['labels'].reshape(-1,)
fs = int(data['fs'].reshape(-1,))
print('Loading',filetest,'with sampling frequency of',fs,'Hz.')
X_test = np.transpose(X_test,(2,1,0)) #trials x ch x time

print(X_test.shape, y_test.shape)

y_pred = pipeline.predict(X_test)

print(y_pred.shape, y_test.shape)

kappa_corr = lambda target,output : (cohen_kappa_score(target,output)+1)/2

score = kappa_corr(y_test,y_pred)

print('score = ', score)

ar_rem = ar(th=0.9)
ar_rem.fit(X_train,y_train)

ar_rem.transform(X_test)
