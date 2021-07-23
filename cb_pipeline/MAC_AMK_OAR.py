# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:50:32 2021

@author: Juan David
"""

import argparse
import scipy.io as sio
import numpy as np
import pandas as pd

# parser = argparse.ArgumentParser(description='QKLMS_AMK filtering Ocular Artifact Removal')
# parser.add_argument('--input',required=True, help='Subject filename with path')
# parser.add_argument('--output',required=True, help='Filtered subject savename with path')
# parser.add_argument('--th',required=True, type=float, help='Kurtosis threshold')
# parser.add_argument('--n_subject',required=True, type=float, help='BCI dataset subject for filter parameter asignation')

# args = parser.parse_args()

# filename = args.input
# savename = args.output

ths = [0.144004463,0.15777807,0.148215102,0.08950537,0.147071399,0.172103277,0.134589128,0.168674552,0.077062943]

for i, th in enumerate(ths):
    #For testing purpuses
    filename = r'..\..\BCI\NEW_22ch_A0{}.mat'.format(i+1)
    savename = r'..\..\BCI\MAC_AMK_filtered\fNEW_22ch_A0{}.mat'.format(i+1)
    
    # 1. Load raw BCI data
    data = sio.loadmat(filename)
    Xdata = data['X']
    Xdata = np.transpose(Xdata,(2,1,0))
    labels = data['labels'].reshape(-1,)
    fs = 250
    
    
    
    
    # 1.5. Load filter parameters for respective subject
    parameters_path = '..\..\BCI'
    # parametersFile_name = 'customRandomSearch0' + str(args.n_subject) + '.csv'
    parametersFile_name = '\customRandomSearch0' + str(i+1) + '.csv'
    params = pd.read_csv(parameters_path + parametersFile_name)
    best_params = params[params.r2_mean == params.r2_mean.max()].iloc[0]
    
    # 2. Artifacts removal stage
    from OA import artifact_removal_MAC_AMK as AR
    # ar = AR(th=args.th, filter_parameters=best_params)
    ar = AR(th=th, filter_parameters=best_params)
    ar.fit(Xdata,labels)
    cleanEEG = ar.transform(Xdata)
    
    # 3. Save clean BCI data
    cleanEEG = np.transpose(cleanEEG,(2,1,0))
    new_filename = 'filtered' + filename
    new_data = {'X': cleanEEG,
                'labels': data['labels'],
                'fs': fs}
    sio.savemat(savename, new_data)
    print("FILE SAVED")

# cleanEEG = np.transpose(cleanEEG,(2,1,0))
# import matplotlib.pyplot as plt
# for clean,raw in zip(cleanEEG,Xdata):
#     for ch_1, ch_2 in zip(clean,raw):
#         plt.plot(ch_1,label='clean')
#         plt.plot(ch_2,label='raw')
#         plt.legend()
#         plt.show()

