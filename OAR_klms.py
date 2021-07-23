# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:29:39 2021

@author: USUARIO
"""
import argparse
import scipy.io as sio
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='KLMS filtering Ocular Artifact Removal')
parser.add_argument('--input_file',required=True, help='Subject filename with path')
parser.add_argument('--th',required=True, type=float, help='MAC threshold')
parser.add_argument('--output_file',required=True, help='Filtered subject savename with path')


args = parser.parse_args()

filename = args.input_file
savename = args.output_file

# 1. Load raw BCI data
data = sio.loadmat(filename)
Xdata = data['X']
Xdata = np.transpose(Xdata,(2,1,0))
labels = data['labels'].reshape(-1,)
fs = int(data['fs'].reshape(-1,))

# 3. Artifacts removal stage
from OA import artifact_removal_with_MAC as AR
ar = AR(th=args.th)
ar.fit(Xdata,labels)
cleanEEG = ar.transform(Xdata)

# 4. Save clean BCI data
cleanEEG = np.transpose(cleanEEG,(2,1,0))
new_filename = 'filtered' + filename
new_data = {'X': cleanEEG,
            'labels': data['labels'],
            'fs': data['fs']}
sio.savemat(savename, new_data)




