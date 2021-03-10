# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:42:19 2021

@author: Juan David
"""
def kafSearch(filterName,systemName,n_samples,n_paramters):
    import KAF
    from tqdm import tqdm
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    import pandas as pd  
    
    embedding = 40
    
    # 1. Generate data for grid search
    if systemName == "lorenz" or systemName == "chua":
        u,d = db(samples=n_samples,system=systemName,L=embedding)
    elif systemName == "4.2":
        u,d = db2(samples=n_samples)
    else:
        raise ValueError("Database does not exist")
      
    # 2. Pick filter    
    if filterName == "QKLMS":
        # 2.1. Generate parameters for QKLMS grid search
        import numpy as np
        if systemName == "lorenz":
            eta = np.linspace(0.1,0.9,n_paramters)
            sigma = np.linspace(10,200,n_paramters)
            epsilon = np.linspace(5,50,n_paramters)
            
        elif systemName == "chua":
            eta = np.linspace(0.1,0.9,n_paramters)
            sigma = np.linspace(100,500,n_paramters)
            epsilon = np.linspace(0.01,20,n_paramters)
            
        elif systemName == "4.2":
            eta = np.linspace(0.1,0.8,n_paramters)
            sigma = np.linspace(20,200,n_paramters)
            epsilon = np.linspace(4,20,n_paramters)
        params = [{'eta':et,'epsilon':ep, 'sigma':s } for et in eta for ep in epsilon for s in sigma]
                       
        # 2.2. Search over QKLMS
        results = []
        for p in tqdm(params):
            try:
                f = KAF.QKLMS(eta=p['eta'],epsilon=p['epsilon'],sigma=p['sigma'])
                y = f.evaluate(u,d)
                y = np.array(y).reshape(-1,1)       
                p['r2'] = r2_score(d[1:], y[1:])
                p['mse'] = mean_squared_error(d[1:], y[1:])
                p['CB_size'] = len(f.CB)
                p['tradeOff_dist'] = tradeOffDistance(p['mse'],p['CB_size'])
            except:
                p['r2'] = np.nan
                p['mse'] = np.nan
                p['CB_size'] = np.nan
                p['tradeOff_dist'] = np.nan
            results.append(p)
            pd.DataFrame(data=results).to_csv('GridSearchResults/' + filterName + '_' + systemName + '_' + str(n_samples) + '.csv')
            
    elif filterName == "QKLMS_AKB": 
        # 2.1. Generate parameters for QKLMS_AKB grid search
        import numpy as np
        if systemName == "lorenz":
            eta = np.linspace(0.02,1,n_paramters)
            sigma = np.linspace(0.01,200,n_paramters)
            epsilon = np.linspace(1e-3,100,n_paramters)
            mu = np.linspace(1e-4,1,n_paramters)
            K = np.linspace(2,20,n_paramters)

        elif systemName == "chua":
            eta = np.linspace(0.1,0.9,n_paramters)
            sigma = np.linspace(0.01,200,n_paramters)
            epsilon = np.linspace(1e-3,100,n_paramters)
            mu = np.linspace(1e-4,1,n_paramters)
            K = np.linspace(2,20,n_paramters)

        elif systemName == "4.2":
            eta = np.linspace(0.1,0.9,n_paramters)
            sigma = np.linspace(0.01,200,n_paramters)
            epsilon = np.linspace(1e-3,100,n_paramters)
            mu = np.linspace(1e-4,1,n_paramters)
            K = np.linspace(2,20,n_paramters)

        params = [{'eta':et,'epsilon':ep, 'sigma_init':s, 'mu':m, 'K':int(k) } for et in eta for ep in epsilon for s in sigma for m in mu for k in K]
                      
        # 2.2. Search over QKLMS
        results = []
        for p in tqdm(params):
            try:
                f = KAF.QKLMS_AKB(eta=p['eta'],epsilon=p['epsilon'],sigma_init=p['sigma_init'], mu=p['mu'], K=p['K'])
                y = f.evaluate(u,d)
                y = np.array(y).reshape(-1,1)       
                p['r2'] = r2_score(d[1:], y[1:])
                p['mse'] = mean_squared_error(d[1:], y[1:])
                p['CB_size'] = len(f.CB)
                p['tradeOff_dist'] = tradeOffDistance(p['mse'],p['CB_size'])
            except:
                p['r2'] = np.nan
                p['mse'] = np.nan
                p['CB_size'] = np.nan
                p['tradeOff_dist'] = np.nan
            results.append(p)
            pd.DataFrame(data=results).to_csv('GridSearchResults/' + filterName + '_' + systemName + '_' + str(n_samples) + '.csv')
            
    elif filterName == "QKLMS_AMK": 
        # 2.1. Generate parameters for QKLMS_AKB grid search
        import numpy as np
        if systemName == "lorenz":
            eta = np.linspace(0.02,1,n_paramters)
            epsilon = np.linspace(1e-3,100,n_paramters)
            mu = np.linspace(1e-4,1,n_paramters)
            K = np.linspace(2,20,n_paramters)     
        elif systemName == "chua":
            eta = np.linspace(0.02,1,n_paramters)
            epsilon = np.linspace(1e-3,100,n_paramters)
            mu = np.linspace(1e-4,1,n_paramters)
            K = np.linspace(2,20,n_paramters)
        elif systemName == "4.2":
            eta = np.linspace(0.02,1,n_paramters)
            epsilon = np.linspace(1e-3,100,n_paramters)
            mu = np.linspace(1e-4,1,n_paramters)
            K = np.linspace(2,20,n_paramters)
        
    
        params = [{'eta':et,'epsilon':ep, 'mu':m, 'K':int(k) } for et in eta for ep in epsilon for m in mu for k in K]
                      
        # 2.2. Search over QKLMS
        results = []
        for p in tqdm(params):
            try:
                f = KAF.QKLMS_AMK(eta=p['eta'],epsilon=p['epsilon'], mu=p['mu'], Ka=p['K'], A_init="pca")
                y = f.evaluate(u,d)
                y = np.array(y).reshape(-1,1)       
                p['r2'] = r2_score(d[1:], y[1:])
                p['mse'] = mean_squared_error(d[1:], y[1:])
                p['CB_size'] = len(f.CB)
                p['tradeOff_dist'] = tradeOffDistance(p['mse'],p['CB_size'])
            except:
                p['r2'] = np.nan
                p['mse'] = np.nan
                p['CB_size'] = np.nan
                p['tradeOff_dist'] = np.nan
            results.append(p)
            pd.DataFrame(data=results).to_csv('GridSearchResults/' + filterName + '_' + systemName + '_' + str(n_samples) + '.csv')
            
    else:
        raise ValueError("Filter does not exist")   
    return       
    

def db(samples=1000,system='lorenz',L=40):
    import numpy as np
    import TimeSeriesGenerator as tsg
    x, y, z = tsg.chaoticSystem(samples=samples,systemType=system)
    ux = np.array([x[i-L:i] for i in range(L,len(x))])
    uy = np.array([y[i-L:i] for i in range(L,len(y))])
    u = np.concatenate((ux,uy), axis=1) # INPUT
    d = np.array([z[i] for i in range(L,len(z))]).reshape(-1,1)
    return u,d

def db2(samples=1000):
    import numpy as np
    import testSystems as ts
    var = 0.01
    noise = np.sqrt(var)*np.random.randn(samples)
    s = ts.testSystems(samples = samples+2, systemType = "4.2_AKB")
    u = np.array([s[-samples-1:-1],s[-samples-2:-2]]).T
    d = np.array(s[-samples:]).reshape(-1,1) + noise.reshape(-1,1)
    return u,d

def tradeOffDistance(MSE,sizeCB):
    from scipy.spatial.distance import cdist
    import numpy as np
    reference = np.array([0,0]).reshape(1,-1)
    result = np.array([MSE,sizeCB]).reshape(1,-1)
    return cdist(reference,result).item()