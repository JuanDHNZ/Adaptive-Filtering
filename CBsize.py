# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 09:24:38 2021

@author: Juan David
"""


from KAF import QKLMS
import numpy as np

"""DATA"""

# ATTRACTOR
import TimeSeriesGenerator
x, y, z = TimeSeriesGenerator.chaoticSystem(samples=6000,systemType='wang')
embedding = 5

N = 5000
trainSplit = 0.8
CB = []

# SISTEMA 4.1

# import testSystems as ts
# x,d = ts.testSystems(samples = 200000, systemType = "4.1_AKB")
# embedding = 2

"""CODEBOOK SIZE FOR KAF"""

from test_on_KAF import codebook4KAF

folder = 'GridSearchWang2'

path = folder + '/QKLMS_wang_5000.csv'
filterType = 'QKLMS'
CB.append(codebook4KAF(x, embedding, N, trainSplit, path, filterType))

path = folder + '/QKLMS_wang_5000_2.csv'
filterType = 'QKLMS'
CB.append(codebook4KAF(x, embedding, N, trainSplit, path, filterType, ExTest=True,y=y, z=z))

# path = folder + '/QKLMS_AKB_4.1_5000.csv'
# filterType = 'QKLMS_AKB'
# CB.append(codebook4KAF(x, embedding, N, trainSplit, path, filterType))

# path = folder + '/QKLMS_AMK_4.1_5000.csv'
# filterType = 'QKLMS_AMK'
# CB.append(codebook4KAF(x, embedding, N, trainSplit, path, filterType))

import pandas as pd 
results = pd.DataFrame(CB, columns=['Filter', 'CB size'])
results.to_csv(folder + '/CB_sizes.csv')


