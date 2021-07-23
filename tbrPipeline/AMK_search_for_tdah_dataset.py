# -*- coding: utf-8 -*-
"""
Created on Mon May 24 12:26:08 2021

@author: Juan David
"""
import argparse
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import randint,uniform,loguniform
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterSampler,KFold
from KAF import QKLMS_AMK
import pandas as pd
import pickle

# 1. Load raw BCI data
parser = argparse.ArgumentParser(description='Random Search for QKLMS AMK filter parameters')
parser.add_argument('--input',required=True, help='Input filename with path')
parser.add_argument('--out',required=True, help='Input savename with path')
parser.add_argument('--th',required=True, type= float, help='MAC threshold')

args = parser.parse_args()

filename = args.input 
savename = args.out

###Paths for testing only
#filename = r'G:\Shared drives\datasets\BCI\Competition IV\dataset 2a\Trials\NEW_22ch_A01.mat'
#filename = r'G:\My Drive\Students\vigomez\Code_A1_Application\data_4C\BCI_s02train.mat'
#filename = '..\data_4C\BCI_s01train.mat'

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


# 2. Artifacts removal stage - Noise Estimation
from mac import MAC
mac_ = MAC(th=args.th)
noise_est = mac_.fit_transform(Xdata,labels)

Noise_sum = np.array([np.sum(noise) for noise in noise_est])
noise_index = np.where(Noise_sum!=0.0)[0]

Xdata = Xdata[noise_index]
Noise = noise_est[noise_index]
labels = labels[noise_index]


#3. Parameter Grid for AMK AF
param_dist = {'embedding':randint(4,10),
              'eta':loguniform(1e-2,0.9),
              'epsilon':uniform(1e-1,10),
              'mu': uniform(1e-2,1),
              "Ka": randint(5,15)
              }

param_list = list(ParameterSampler(param_dist, n_iter=50,
                                   random_state=np.random.RandomState(0)))

# 4. Grid Search for AMK AF
results = []

folds = 10 if len(noise_index) > 10 else len(noise_index)

print("##### FOLDS : ",folds)

for param in tqdm(param_list):
    fold = 0
    kf = KFold(n_splits=folds)
    r2_temporal = []
    params_results = param.copy()
    for train_index, test_index in kf.split(Xdata,labels):
        
        print("\n###########################################")
        print("\nFILE : ", filename)
        print("\nFOLD: ", fold)
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = Xdata[test_index], Xdata[train_index]
        N_train, N_test = Noise[test_index], Noise[train_index]
        y_train, y_test = labels[test_index], labels[train_index]
               
        f = QKLMS_AMK()
        f.set_params(**param)
        
        r2 = []
                
        N_train_t = N_train.reshape(-1,N_train.shape[2])
       
        ind = np.random.permutation(len(N_train_t))[:20]
        
        for channel in tqdm(N_train_t[ind]):                        
            try:
                pred = np.array(f.evaluate(channel,channel)).reshape(-1,1)
                _,target = f.embedder(channel,channel)                
                r2.append(r2_score(target,pred))                
                #plt.scatter(target,pred)
                #plt.show()
            except:
                nada = np.nan
                #print('NAN')
        r2 = np.array(r2)
        if len(r2)>0:
            r2m = np.mean(np.where(r2>-1,r2,-1))
        else:
            r2m = -1                
        r2_temporal.append(r2m)
        params_results['split'+str(fold)+'_r2']  = r2m
        fold += 1
    params_results['r2_mean'] = np.nanmean(np.array(r2_temporal))
    params_results['r2_std'] = np.nanstd(np.array(r2_temporal))   
    results.append(params_results)
    df = pd.DataFrame.from_dict(results)
    df.to_csv(savename)
    
  




