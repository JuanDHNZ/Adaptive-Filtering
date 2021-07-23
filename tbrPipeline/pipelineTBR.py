# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:28:29 2021

@author: Juan David
"""
# Pipeline test for OA artifact removal

import argparse
import pandas as pd
import numpy as np
import pickle 
from tbr import TBR
from fisher import FISHER
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from OA import artifact_removal
from scipy.stats import uniform
from OA import artifact_removal_MAC_AMK as AR
from sklearn.model_selection import GridSearchCV
from datetime import datetime

#-----------------------------------------------------
# 1 . Get both data and save paths

parser = argparse.ArgumentParser(description='AR -> THETA_BETA_RATIO -> LDA')
parser.add_argument('--input',required=True, help='Input filename with path')
parser.add_argument('--output',required=True, help='Input savename with path')
parser.add_argument('--th',required=True, type=float, help='MAC threshold')

args = parser.parse_args()

filename = args.input
savename = args.output

#-----------------------------------------------------
# 2 . Custom scorer
def score(fisher,labels):
    import numpy as np
    sb = (np.mean(fisher[labels>0.5])-np.mean(fisher[labels<0.5]))**2
    sw = np.var(fisher)
    return sb/sw

#-----------------------------------------------------
# 3 . Load dataset in Xdata and labels
with open(filename,'rb') as f: 
    Data = pickle.load(f) 

x = [i for i in Data if (i.get('cond') == 'IC')]

X_ = []
labels_ = []
for i,j in enumerate(x):
    X_.append(j.get('smiley'))     
    
    if (j.get('label') == 'Control'):
        arr = np.zeros(j.get('smiley').shape[2])
    
    elif(j.get('label') == 'tdah'):
        arr = np.ones(j.get('smiley').shape[2])   
    labels_.append(arr)
   
X = np.concatenate((X_), axis = 2)
Xdata = np.transpose(X, [2,0,1])
labels = np.concatenate((labels_)).reshape(-1,)


params_file = 'AMK_params_tdah.csv'
params = pd.read_csv(params_file)
params = params[params.r2_mean == params.r2_mean.max()].iloc[0]

#-----------------------------------------------------
# 3 . Define pipeline

steps = [('pp', AR(params,args.th)),
         ('extract', TBR()),
         ('classify', FISHER())
    ]

pipeline = Pipeline(steps = steps)


grid = {'pp__filter_parameters': [params],
        'pp__th':[args.th],
        'extract__fs':[250]
        }

print("Starting CV...")
start = datetime.now()
search = GridSearchCV(pipeline,grid,cv=5,verbose=10,scoring=make_scorer(score),return_train_score=True)
search.fit(Xdata,labels)
cv_results = search.cv_results_

cv_results = pd.DataFrame.from_dict(cv_results)
cv_results.to_csv(savename)
end = datetime.now()
print("Execution time: {}".format(end-start))




