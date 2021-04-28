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

def QKLMS_test(samples=1000,attr=None):   
    from KAF import QKLMS
    #1. Input data
    if attr in ['lorenz','chua','duffing','nose_hoover','rikitake','rossler','wang']:
        u,d = db(samples=samples, system=attr)
    elif attr in ['test']:
        u,d = db_z(samples=samples, system=attr)
    else:
        raise ValueError("Unknown dataset")
    
    #2. Parameters for Grid Search
    import numpy as np
    n=60
    epsilon = np.linspace(1e-2,10000,n)
    sigma = np.linspace(1e-1,10000,n)
       
    #. Search
    from tqdm import tqdm
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    
    params = [{'epsilon':ep, 'sigma':sg} for ep in epsilon for sg in sigma]
    
    results = []
    for p in tqdm(params):
        try:
            f = QKLMS(epsilon=p['epsilon'],sigma=p['sigma'])
            y = f.evaluate(u,d)
            y = np.array(y).reshape(-1,1)
            # y_ = np.sum(y)
            p['r2'] = r2_score(d, y)
            p['mse'] = mean_squared_error(d, y)
        except:
            p['r2'] = np.nan
            p['mse'] = np.nan
        results.append(p)
    import pandas as pd   
    return pd.DataFrame(data=results)

if __name__ == "__main__":
    

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Dataset to use')
    args = parser.parse_args()
    data = args.dataset
    
    print('Testing '+ data)
    Df = QKLMS_test(3000, data)
    Df.to_csv('QKLMS_results_'+data+'.csv')

