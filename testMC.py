# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:14:32 2021

@author: Juan David
"""

from KAF import QKLMS
from test_on_KAF import MC_BestParameters
from test_on_KAF import quickGridSearch4QKLMS
import numpy as np

# Sistema 4.2
# import testSystems as ts
# s = ts.testSystems(samples = 20000, systemType = "4.2_AKB")

# LORENZ ATTRACTOR

import TimeSeriesGenerator
x, y, z = TimeSeriesGenerator.chaoticSystem(samples=60000,systemType='lorenz')
embedding = 5

# SISTEMA 4.2

# import testSystems as ts
# x = ts.testSystems(samples = 200000, systemType = "4.2_AKB")
# embedding = 2

from test_on_KAF import selectBestResultFromKafSearch

folder = 'GridSearchEx'
""" lorenz """
# QKLMS_lorenz = 'QKLMS_lorenz_5000.csv'
# params = selectBestResultFromKafSearch(folder + '/' + QKLMS_lorenz)

# AKB_lorenz = 'QKLMS_AKB_lorenz_5000.csv'
# params = selectBestResultFromKafSearch(folder + '/' + AKB_lorenz)

AMK_lorenz = 'QKLMS_AMK_lorenz_5000.csv'
params = selectBestResultFromKafSearch(folder + '/' + AMK_lorenz)

# """ 4.2 """
# QKLMS_42 = 'QKLMS_4.2_5000.csv'
# params = selectBestResultFromKafSearch(folder + '/' + QKLMS_42)

# AKB_42 = 'QKLMS_AKB_4.2_5000.csv'
# params = selectBestResultFromKafSearch(folder + '/' + AKB_42)

# AMK_42 = 'QKLMS_AMK_4.2_5000.csv'
# params = selectBestResultFromKafSearch(folder + '/' + AMK_42)


fType='QKLMS_AMK'
inputSizeXrun = 5000
trainSplit = 0.8

MSE = MC_BestParameters(inputSignal=x, 
              monteCarloRuns=10, 
              singleRunDataSize=inputSizeXrun, 
              trainSplitPercentage=trainSplit,
              signalEmbedding=embedding,
              filterType=fType,
              parameters=params,
              lorenzTest=True,
              y=y,
              z=z)

import pandas as pd
df = pd.DataFrame(MSE, columns=['MSE'])
df.to_csv(folder + "/testingMSE_" + fType + "_4000-1000_lorenz.csv")
