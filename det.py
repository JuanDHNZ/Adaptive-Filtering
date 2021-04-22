# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:32:11 2021

@author: USUARIO
"""

folder = 'data_4c/'
filename = 'BCI_s06T.mat'
filenameF = 'filteredBCI_s06T.mat'


import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = sio.loadmat(folder + filename)
Xdata = data['X']
Xdata = np.transpose(Xdata,(2,1,0))

Covs = [np.cov(trial) for trial in Xdata]
Dets = [np.linalg.det(matrix) for matrix in Covs]

sns.histplot(data=Dets, kde=True, bins=10)
plt.ylim([0,5])


dataF = sio.loadmat(folder + filenameF)
XdataF = dataF['X'].astype(np.float64)
XdataF = np.transpose(XdataF,(2,1,0))

CovsF = [np.cov(trial) for trial in XdataF]
DetsF = [np.linalg.det(matrix) for matrix in CovsF]

sns.histplot(data=DetsF, kde=True, bins=10)


