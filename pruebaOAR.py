# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:29:39 2021

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
cleanEEG = ar.transform(Xdata)

# 4. Save clean BCI data
new_filename = 'filtered' + filename
cleanEEG = np.transpose(cleanEEG,(2,1,0))
new_filename = 'filtered' + filename
new_data = {'X': cleanEEG,
            'labels': data['labels'],
            'fs': data['fs']}
sio.savemat(folder + new_filename, new_data)

# dataF = sio.loadmat(folder + new_filename)
# X_dataF = dataF['X']

# plt.plot(Xdata[2,0])
# plt.plot(cleanEEG[2,0])



