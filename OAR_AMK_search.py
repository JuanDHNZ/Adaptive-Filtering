# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:26:18 2021

@author: USUARIO
"""
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import randint,uniform
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterSampler,KFold
from KAF import QKLMS_AMK
from OA import artifactRemoval
import pandas as pd

import seaborn as sns
sns.set()


# 1. Load raw BCI data
# parser = argparse.ArgumentParser(description='FBCSP -> MIBIF -> LDA.')
# parser.add_argument('--input',required=True, help='Input filename with path')
# parser.add_argument('--out',required=True, help='Input savename with path')
# #parser.add_argument('--fs',required=True, type=float, help='Input filename with path')

# args = parser.parse_args()

# filename = args.input
# savename = args.out

#filename = r'G:\Shared drives\datasets\BCI\Competition IV\dataset 2a\Trials\NEW_22ch_A01.mat'
# filename = r'G:\My Drive\Students\vigomez\Code_A1_Application\data_4C\BCI_s02train.mat'
filename = '..\data_4C\BCI_s01train.mat'

data = sio.loadmat(filename)



#folder = 'data_4c/'
#filename = 'BCI_s06train.mat'
#data = sio.loadmat(folder + filename)
Xdata = data['X']
Xdata = np.transpose(Xdata,(2,1,0))
labels = data['labels'].reshape(-1,)
fs = int(data['fs'].reshape(-1,))
print('Loading',filename,'with sampling frequency of',fs,'Hz.')

# 2. Load kurthosis thresholds for each subject

#ths = pd.read_csv("data_4c/score_filterbank.csv")['param_preproc__th'].to_list()

# 3. Artifacts removal stage

ar = artifactRemoval(th=0.976930899214176)
ar.fit(Xdata,labels)

Noise = ar.noiseEstimation(Xdata)

Rsum = np.array([np.sum(r) for r in Noise])
r_index = np.where(Rsum!=0.0)[0]

Xdata = Xdata[r_index]
Noise = Noise[r_index]
labels = labels[r_index]


""" 
For Subject 06 there are 57 trials with noise out of 219
"""


param_dist = {'embedding':randint(5,10),
              'eta':uniform(0.1,0.9),
              'epsilon':uniform(1e-1,2),
              'mu': uniform(1e-2,1),
              "Ka": randint(5,15)
              }

param_list = list(ParameterSampler(param_dist, n_iter=50,
                                   random_state=np.random.RandomState(0)))

results = []
folds = 10
for param in param_list:
    fold = 0
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(Xdata,labels):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = Xdata[test_index], Xdata[train_index]
        N_train, N_test = Noise[test_index], Noise[train_index]
        y_train, y_test = labels[test_index], labels[train_index]
        
        f = QKLMS_AMK()
        f.set_params(**param)
        
        r2 = []
        for trial in tqdm(N_train):
            for channel in trial:
                try:
                    pred = np.array(f.evaluate(channel,channel)).reshape(-1,1)
                    _,target = f.embedder(channel,channel)
                    r2.append(r2_score(target,pred))
                except:
                    print('Nan')
        r2 = np.array(r2)
        r2m = np.mean(np.where(r2>-1,r2,-1))
        param['r2_'+str(fold)]
        
    
    
#Terminar de organizar




# cv_results = search.cv_results_

# cv_results = pd.DataFrame.from_dict(cv_results)
# cv_results.to_csv("random_sarch_t0_c0.csv")

  




