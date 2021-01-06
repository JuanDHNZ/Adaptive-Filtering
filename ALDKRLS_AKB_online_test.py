# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:12:50 2021

@author: Juan David

ALDKRLS_AKB Online test
"""

def db(samples=1000,system='lorenz'):
    import numpy as np
    import TimeSeriesGenerator as tsg
    x, y, z = tsg.chaoticSystem(samples=samples+10,systemType=system)
    ua = x[-samples-2:-2].reshape(-1,1)
    ub = y[-samples-3:-3].reshape(-1,1)
    return np.concatenate((ua,ub), axis=1), z[-samples-1:-1].reshape(-1,1)

def online_plot(samples, system, kaf):
    #1. Get data
    u,d = db(samples,system)
    #2. Inicializacion
    y_pred = []
    y_tar = []
    new_add_x = [0]
    new_add_y = [d[0]]
    cb_ant = 1
    i=0
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 300
    dmax = max(d)
    #3. Predictions and plot
    for ui,di in zip(u,d):
        y_pred.append(kaf.evaluate(ui.reshape(1,-1),di.reshape(-1,1)))
        y_tar.append(di)
        if cb_ant != len(kaf.CB):
            new_add_x.append(i)
            new_add_y.append(y_pred[-1])
            cb_ant = len(kaf.CB)
        i+=1
        # plt.ylim([-dmax-dmax*0.1,dmax+dmax*0.1])
        plt.plot(y_tar,'c', label="Target")
        plt.plot(y_pred, 'magenta', label="Predict")
        plt.legend()
        plt.title("Online test on {}    -    $\sigma = {}$".format(system,kaf.sigma))
        plt.scatter(new_add_x,new_add_y,c="magenta",marker="*")
        plt.show()
        
    return

import KAF
klms = KAF.ALDKRLS_AKB(sigma=40,K=5,epsilon=1e-4)
#input y target
online_plot(1000,'chua',klms)


