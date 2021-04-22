# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 06:37:02 2021

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
my_file = open(folder + "th.txt", "r")
th = my_file.read().split("\n")

# 3. Artifacts removal stage
from OA import artifactRemoval
ar = artifactRemoval(th=float(th[5]))
ar.fit(Xdata,labels)

R = ar.noiseEstimation(Xdata)

Rsum = np.array([np.sum(r) for r in R])
r_index = np.where(Rsum!=0.0)[0]

""" 
For Subject 06 there are 27 trials with noise out of 219
"""
r2_4_gamma = []
gammas = [0.1, 0.25, 0.5, 0.75, 1]
for gamma in gammas:
    r2_trial = []
    from tqdm import tqdm
    for trial in tqdm(range(len(r_index))):
        rnoise = R[r_index][trial]
        Xnoise = Xdata[r_index][trial]
        
        from scipy.spatial.distance import cdist
        from sklearn.metrics import r2_score
        r2_ch = []
        
        for rn in rnoise:        
            rn = rn.reshape(-1,1)
            sgm = np.median(cdist(rn,rn)) * gamma #Median criterion
            
            rx = rn[:-1]
            ry = rn[1:]
            from KAF import QKLMS
            f = QKLMS(epsilon = 0, sigma = sgm)
            vn = f.evaluate(rx,ry)
            r2_ch.append(r2_score(ry,vn))
            # plt.plot(rn, label='$\hat{r_n}$')
            # plt.plot(vn, label='$\hat{v_n}$')
            # plt.legend()
            # plt.show()
        r2_trial.append(np.mean(np.array(r2_ch)))
    r2_4_gamma.append(r2_trial)
    
for r2,gamma in zip(r2_4_gamma, gammas):
    plt.plot(r2, label='gamma ' + str(gamma))
plt.legend()
    
    


# cleanEEG = ar.transform(Xdata[:10])

# # 4. Save clean BCI data
# new_filename = 'filtered' + filename
# cleanEEG = np.transpose(cleanEEG,(2,1,0))
# new_filename = 'filtered' + filename
# new_data = {'X': cleanEEG,
#             'labels': data['labels'],
#             'fs': data['fs']}
# sio.savemat(folder + new_filename, new_data)