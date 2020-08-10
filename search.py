# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:04:41 2020

@author: Juan David

Rejilla para probar diferentes sigma y epsilon

"""

def pSearchCurve(u=None,d=None,sigmaList = None, epsilonList = None ):
    if u is None or d is None or sigmaList is None or epsilonList is None:
        raise ValueError("Argument is missing")
    
    if len(u) is not len(d):
        raise ValueError("u and d must be same lenght")
    
    
    import KAF
    from sklearn.metrics import r2_score
    import testSystems as ts
    import numpy as np

    epsilonList = np.logspace(-5, 5, 10)
    sigmaList = np.logspace(-5, 5, 10)
    
    out1 = []
    out2 = []
    r2_filtro1 = []
    r2_filtro2 = []
    CB_size1 = []
    CB_size2 = []
    
    u, d = ts.testSystems(samples=10, systemType="1")
    u = u.reshape(-1,1)
    d = d.reshape(-1,1)
    for sigma in sigmaList:
        for epsilon in epsilonList:
            filtro1 = KAF.QKLMS(epsilon=epsilon,sigma=sigma)
            filtro2 = KAF.QKLMS2(epsilon=epsilon, sigma=sigma)
            for i in range(len(d)):        
                out1.append(filtro1.evaluate(u[i],d[i]))                        
                out2.append(filtro2.evaluate(u[i],d[i]))
                      
            r2_filtro1.append(r2_score(d, out1))#[1:]
            r2_filtro2.append(r2_score(d, out2))
            CB_size1.append(len(filtro1.CB))
            CB_size2.append(len(filtro2.CB))
            out1.clear()
            out2.clear()
            
    return r2_filtro1, r2_filtro2, CB_size1, CB_size2
    

    


