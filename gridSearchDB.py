# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 14:26:52 2020

@author: Juan David
"""
import numpy as np
import TimeSeriesGenerator as tsg
samples = 400
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="lorenz")

ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1)
d = z[-samples-1:-1].reshape(-1,1)

def gridSearchDB(u = None, d = None, cl = None):
    """
    Grid Search a un conjunto de datos entrada-salida
    
    u = entrada N muestras x D instantes anteriores
    d = salida esperada
    cl = array con grilla de parametro clusters (int64)
    
    """
    cl = cl.astype(np.int64)
    parameters ={'clusters':cl}
    
    import KAF
    from sklearn.model_selection import GridSearchCV
    filtro = KAF.GMM_KLMS()
    cv = [(slice(None), slice(None))]
    gmmklms = GridSearchCV(filtro,parameters,cv=cv)
    gmmklms.fit(u,d)
    return gmmklms
    # print("Mejores parametros : ", mqklms.best_params_)
    # print("Mejor score : ", mqklms.best_score_)
