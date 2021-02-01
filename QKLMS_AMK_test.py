# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 06:13:26 2021

@author: JUAN 


"""
def db(samples=1000,system='lorenz',L=40):
    import numpy as np
    import TimeSeriesGenerator as tsg
    x, y, z = tsg.chaoticSystem(samples=samples,systemType=system)
    ux = np.array([x[i-L:i] for i in range(L,len(x))])
    uy = np.array([y[i-L:i] for i in range(L,len(y))])
    u = np.concatenate((ux,uy), axis=1) # INPUT
    d = np.array([z[i] for i in range(L,len(z))]).reshape(-1,1)
    return u,d

def db_z(samples=1000,system='lorenz',L=40):
    import numpy as np
    import TimeSeriesGenerator as tsg
    x, y, z = tsg.chaoticSystem(samples=samples,systemType=system)
    ux = np.array([x[i-L:i] for i in range(L,len(x))])
    uy = np.array([y[i-L:i] for i in range(L,len(y))])
    uz = np.array([z[i-L:i] for i in range(L,len(z))]).reshape(-1,1)
    u = np.concatenate((ux,uy,uz), axis=1) # INPUT
    d = np.array([z[i] for i in range(L,len(z))]).reshape(-1,1)
    return u,d

def AMK_test(samples=1000,attr=None):   
    from KAF import QKLMS_AMK as amk
    #1. Input data
    if attr in ['lorenz','chua','duffing','nose_hoover','rikitake','rossler','wang']:
        u,d = db(samples=samples, system=attr)
    elif attr in ['test']:
        u,d = db_z(samples=samples, system=attr)
    else:
        raise ValueError("Unknown dataset")
    
    #2. Parameters for Grid Search
    import numpy as np
    n=8
    eta = np.linspace(0.03,1,n)
    mu = np.linspace(5e-4,5e-2,n)
    epsilon = np.linspace(1e-2,2,n)
    Ka = np.linspace(1,40,n)
    
    #3. Results dataframe
    
    
    #4. Search
    from tqdm import tqdm
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    # best_r2 = best_mse = 0
    
    params = [{'eta':et,'mu':m,'K':int(k),'epsilon':ep} for et in eta for m in mu for k in Ka for ep in epsilon]
    
    results = []
    for p in tqdm(params):
        try:
            f = amk(eta=p['eta'],epsilon=p['epsilon'],mu=p['mu'],Ka=p['K'],A_init="pca")
            y = f.evaluate(u,d)
            y = np.array(y).reshape(-1,1)
            # y_ = np.sum(y)        
            p['r2'] = r2_score(d[1:], y[1:])
            p['mse'] = mean_squared_error(d[1:], y[1:])
        except:
            p['r2'] = np.nan
            p['mse'] = np.nan
        results.append(p)
        
    
    # for et in tqdm(eta):
    #     for m in mu:
    #         for ep in epsilon:
    #             for k in Ka:
    #                 f = amk(eta=et,epsilon=ep,mu=m,Ka=int(k),A_init="pca")
    #                 y = f.evaluate(u,d)
    #                 y = np.array(y).reshape(-1,1)
    #                 y_ = np.sum(y)
    #                 if not np.isnan(y_).any():  
    #                     r2 = r2_score(d[1:], y[1:])
    #                     mse = mean_squared_error(d[1:], y[1:])
    #                     if r2 > best_r2:
    #                         best_r2 = r2
    #                         best_mse = mse
    #                         best_cb = len(f.CB)
    #                         best_mu = m
    #                         best_ep = ep
    #                         best_eta = et
    #                         best_K = k
                        

    import pandas as pd   
    # results = {"Best_R2":best_r2,
    #             "Best_MSE":best_mse,
    #             "Best_CB_size":best_cb,  
    #             "Best_eta":best_eta,
    #             "Best_mu":best_mu,
    #             "Best_epsilon":best_ep,
    #             "Best_K":best_K}
    # results = {"Best_R2":best_r2,
    #             "Best_MSE":best_mse,
    #             "Best_CB_size":best_cb,  
    #             "Best_eta":best_eta,
    #             "Best_mu":best_mu,
    #             "Best_epsilon":best_ep,
    #             "Best_K":best_K}
    
    return pd.DataFrame(data=results,index=[attr])
    


if __name__ == "__main__":
    

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Dataset to use')
    args = parser.parse_args()
    data = args.dataset
    
    print('Testing '+data)
    
    #import pandas as pd
    #dbs = ['lorenz','chua','duffing','nose_hoover','rikitake','rossler','wang']
    #dbs = ['lorenz']
    #first = True
    #for data in dbs:
    Df = AMK_test(3000, data)
        # if first:
        #     Df = frame
        #     first = False
        # else:
        #     Df = pd.concat([Df,frame])
    Df.to_csv('QKLMS_AMK_results_'+data+'.csv')