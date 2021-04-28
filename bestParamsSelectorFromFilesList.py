# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:57:36 2021

@author: Juan David
"""
from os import walk
import pandas as pd

# 1. Get filenames
folder = 'AMK_RandomSearch_OAR/'
_, _, filenames = next(walk(folder))

df_best_params = pd.DataFrame()
# 2. Get best params from files
for file in filenames:
    subject_grid_search = pd.read_csv(folder + file)
    best_params = subject_grid_search[subject_grid_search.r2_mean == subject_grid_search.r2_mean.max()]
    df_best_params = pd.concat([df_best_params,best_params])

#3. Save best params to Df
df_best_params.to_csv(folder + 'best_params_for_KAF.csv')