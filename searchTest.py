# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:33:47 2020

@author: USUARIO

SEARCH TEST 
"""


import numpy as np 
import matplotlib.pyplot as plt
import testSystems as ts
import KAF 
import search

def plot_pair(u = None,d = None,labelu="u",labeld="d", title = ""):
    plt.title(title)
    plt.plot(u,label = labelu)
    plt.plot(d, label = labeld)
    plt.legend()
    plt.show()

"""NUMERO DE MUESTRAS"""
samples = 10
""""
SISTEMA 1 

u -> proceso Gaussiano de media cero y varianza unitaria
d -> resultado de aplicar filtro FIR a u

"""
u, d = ts.testSystems(samples=samples, systemType="1")
plot_pair(u,d,labelu="input", labeld="target", title="Sistema 1")

epsilon = np.logspace(-5, 5, 10)
sigma = np.logspace(-5, 5, 10)

R2_QKLMS, R2_QKLMS2, CB_size_QKLMS, CB_size_QKLMS2 = search.pSearchCurve(u=u.reshape[-1,1], d=d.reshape[-1,1], sigmaList = sigma, epsilonList = epsilon)






