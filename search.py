# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:04:41 2020

@author: Juan David

Rejilla para probar diferentes sigma y epsilon

"""

def grid(u=None,d=None,sigmaList = None, epsilonList = None ):
    if u is None or d is None or sigmaList is None or epsilonList is None:
        raise ValueError("Argument is missing")
    
    import KAF
    from sklearn.metrics import r2_score
    
    bestR2 = 0;
    bestSigma = 0;
    bestEpsilon = 0;
    for sigma in sigmaList:
        for epsilon in epsilonList:
            filtro = KAF.QKLMS(epsilon=epsilon,sigma=sigma)
            out = filtro.evaluate(u,d)
            r2_filtro = r2_score(d[1:], out)
            if(r2_filtro > bestR2):
                bestSigma = sigma
                bestEpsilon = epsilon
                bestR2 = r2_filtro
            
    return bestSigma, bestEpsilon, bestR2
    

    


