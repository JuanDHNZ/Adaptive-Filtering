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
    
def system2(samples):
    """ x(t) --> Gaussian process with zero mean and unit variance.
    
        d(t)
    """
    import numpy as np 
    mean = [0]
    cov = np.asarray([1]).reshape(1,1)
    x = np.random.multivariate_normal(mean,cov,size=samples)
    
    w = np.asarray([0.227, 0.460, 0.688, 0.460, 0.227])#.reshape(-1,1)
    d = 
    


