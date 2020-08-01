# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:45:19 2020

@author: Juan David

Test systems from state-of.the-art papers
"""

def testSystems(samples = 1000, systemType = None):
    # Dictionaty of available attractors
    
    if systemType == None:
        raise ValueError('System type is missing')
    
    if not systemType in validKeys:
        raise ValueError('Attractor does not exist or is not supported')
    
    import numpy as np
    if systemType == "1":      
        mean = [0]
        cov = np.asarray([1]).reshape(1,1)
        u = np.random.multivariate_normal(mean,cov,size=samples).reshape(-1,)    
        w = np.asarray([0.227, 0.460, 0.688, 0.460, 0.227])
        d = np.convolve(u,w)
        return u,d
    
    if systemType == "2":
        s = np.empty((samples,))
        s[0] = 0.1
        s[1] = 0.1
        i = 2
        while True:
            s[i] = s[i-1]*(0.8 - 0.5*exp(-s[i-1]**2)) - s[i-2]*
            (0.3)
        
    if systemType == "3":
        import csv
        with open('SN_m_tot_V2.csv','r') as file:
            data = csv.reader(file)
            
    

    


