# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:21:48 2021

@author: JUAN
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:41:32 2021

@author: Juan David
"""

def db(samples=5040,system='lorenz',L=40):
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

def db3(samples=1000):
    import testSystems as ts
    return ts.testSystems(samples=samples, systemType="4.1_AKB")

def MC_testingMSE_QKLMS(dataset,eta,epsilon,sigma, trainSetSize,testSetSize, MonteCarlo_N):
    Mc_MSE = []
    samples = trainSetSize*2
    
    from tqdm import tqdm
    from KAF import QKLMS
    import numpy as np
    
    for rep in tqdm(range(MonteCarlo_N)):
        #train set
        if dataset == "lorenz" or dataset == "chua":
            var = 0.1
            u,d = db(samples=samples, system=dataset) 
            noise = np.sqrt(var)*np.random.randn(trainSetSize).reshape(-1,1)
            u_train = u[:trainSetSize] + noise
            d_train = d[:trainSetSize] + noise
            
            # test set
            u,d = db(samples=samples, system=dataset)
            noise = np.sqrt(var)*np.random.randn(testSetSize).reshape(-1,1)
            u_test = u[:testSetSize] + noise
            d_test = d[:testSetSize] + noise
        
        elif dataset == "4.2":
            u,d = db2(samples=samples) 
            u_train = u[:trainSetSize]
            d_train = d[:trainSetSize]
                
            # test set
            u,d = db2(samples=samples)
            u_test = u[:testSetSize]
            d_test = d[:testSetSize]
        
        i=0
        mse = []      
        f = QKLMS(eta,epsilon,sigma)
        
        for ui,di in tqdm(zip(u_train,d_train)):
            f.evaluate(ui,di)
            
            #Option 1
            # y_pred = f.predict(u_test) 
            # mse.append(np.mean((d_test-np.array(y_pred).reshape(-1,1))**2/d_test**2))  
            
            #Option 2
            if np.mod(i,5)==0:
                y_pred = f.predict(u_test) 
                mse.append(np.sum((d_test-np.array(y_pred).reshape(-1,1))**2)/np.sum(d_test**2))
            i+=1
        Mc_MSE.append(mse)
    return np.mean(np.array(Mc_MSE),axis=0).reshape(-1,1)

def MC_testingMSE_QKLMS_AKB(dataset,eta,epsilon,sigma,mu,K, trainSetSize,testSetSize, MonteCarlo_N):
    Mc_MSE = []
    samples = trainSetSize*2
    
    from tqdm import tqdm
    from KAF import QKLMS_AKB
    import numpy as np
    
    for rep in tqdm(range(MonteCarlo_N)):
        #train set
        if dataset == 'lorenz' or dataset == "chua":
            var = 0.1
            u,d = db(samples=samples, system=dataset) 
            noise = np.sqrt(var)*np.random.randn(trainSetSize).reshape(-1,1)
            u_train = u[:trainSetSize] + noise
            d_train = d[:trainSetSize] + noise
            
            # test set
            u,d = db(samples=samples, system=dataset)
            noise = np.sqrt(var)*np.random.randn(testSetSize).reshape(-1,1)
            u_test = u[:testSetSize] + noise
            d_test = d[:testSetSize] + noise
        
        elif dataset == "4.2":
            u,d = db2(samples=samples) 
            u_train = u[:trainSetSize]
            d_train = d[:trainSetSize]
                
            # test set
            u,d = db2(samples=samples)
            u_test = u[:testSetSize]
            d_test = d[:testSetSize]
            
        mse = []
        i = 0
        f = QKLMS_AKB(eta,epsilon,sigma,mu,K)
        
        for ui,di in tqdm(zip(u_train,d_train)):
            f.evaluate(ui,di)
            # Option 1
            # y_pred = f.predict(u_test) 
            # mse.append(np.mean((d_test-np.array(y_pred).reshape(-1,1))**2/d_test**2))
            
            #Option 2
            if np.mod(i,5)==0:
                y_pred = f.predict(u_test) 
                mse.append(np.sum((d_test-np.array(y_pred).reshape(-1,1))**2)/np.sum(d_test**2))
            i+=1
        Mc_MSE.append(mse)
    return np.mean(np.array(Mc_MSE),axis=0).reshape(-1,1)

def MC_testingMSE_QKLMS_AMK(dataset,eta,epsilon,mu,K,trainSetSize,testSetSize, MonteCarlo_N):
    Mc_MSE = []
    samples = trainSetSize*2
    
    from tqdm import tqdm
    from KAF import QKLMS_AMK
    import numpy as np
    for rep in tqdm(range(MonteCarlo_N)):
        #train set
        if dataset == 'lorenz' or dataset == "chua":
            var = 0.1
            u,d = db(samples=samples, system=dataset) 
            noise = np.sqrt(var)*np.random.randn(trainSetSize).reshape(-1,1)
            u_train = u[:trainSetSize] + noise
            d_train = d[:trainSetSize] + noise
            
            # test set
            u,d = db(samples=samples, system=dataset)
            noise = np.sqrt(var)*np.random.randn(testSetSize).reshape(-1,1)
            u_test = u[:testSetSize] + noise
            d_test = d[:testSetSize] + noise
        
        elif dataset == "4.2":
            u,d = db2(samples=samples) 
            u_train = u[:trainSetSize]
            d_train = d[:trainSetSize]
                
            # test set
            u,d = db2(samples=samples)
            u_test = u[:testSetSize]
            d_test = d[:testSetSize]
            
        mse = []
        i=0 # for option 2
        f = QKLMS_AMK(eta,epsilon,mu=mu,Ka=K,A_init="pca")
        f.evaluate(u[:100],d[:100])
        for ui,di in tqdm(zip(u_train,d_train)):
            try:
                f.evaluate(ui,di)
                
                # Option 1
                # y_pred = f.predict(u_test) 
                # mse.append(np.mean((d_test-np.array(y_pred).reshape(-1,1))**2/d_test**2))
                
                # Option 2
                if np.mod(i,5)==0:
                    y_pred = f.predict(u_test) 
                    mse.append(np.sum((d_test-np.array(y_pred).reshape(-1,1))**2)/np.sum(d_test**2))
            except:
                mse.append(0)
            i+=1
        Mc_MSE.append(mse)
        return np.mean(np.array(Mc_MSE),axis=0).reshape(-1,1)


"""
    TESTING MSE ON LORENZ SYSTEM USING QKLMS, QKLMS-AKB AND QKLMS-AMK
     
    Parameters were determined by grid search
     
    QKLMS:
         eta = 0.9
         epsilon = 50 
         sigma = 200
         
    Results:
        R2 score = 0.984
        MSE = 0.88        
        
"""
#Simulation parameters
trainSetSize = 2000
testSetSize = 200
N = 50
# dataset = "lorenz"
dataset = "chua"
# dataset = "4.2"

# 1. QKLMS
"""a. Lorenz"""
# eta = 0.9
# epsilon = 50 
# sigma = 200

"""b. chua"""
# eta = 0.9
# epsilon = 50 
# sigma = 200

"""c. 4.2"""
eta = 0.9
epsilon = 0.01
sigma = 500

QKLMS_MSE = MC_testingMSE_QKLMS(dataset,eta,epsilon,sigma, trainSetSize,testSetSize,N)

# 2. QKLMS_AKB
"""a. Lorenz"""
# eta = 1
# epsilon = 100
# sigma = 200
# mu = 1
# K = 2

"""b. chua"""
eta = 0.9
epsilon = 100
sigma = 50.0075
mu = 1
K = 20

"""c. 4.2"""
# eta = 0.1
# epsilon = 25.00075
# sigma = 200
# mu = 1e-4
# K = 2

AKB_MSE = MC_testingMSE_QKLMS_AKB(dataset,eta,epsilon,sigma,mu,K,trainSetSize,testSetSize,N)

# 3. QKLMS_AMK
"""a. Lorenz"""
# eta = 1
# epsilon = 20.0008
# mu = 1e-4
# K = 2

"""b. chua"""
eta = 0.804
epsilon = 1e-3
mu = 1e-4
K = 12

"""c. 4.2"""
# eta = 0.02
# epsilon = 20.0008
# mu = 1e-4
# K = 2

AMK_MSE = MC_testingMSE_QKLMS_AMK(dataset,eta,epsilon,mu,K,trainSetSize,testSetSize,N)


import pandas as pd
import numpy as np
data = np.concatenate((QKLMS_MSE,AKB_MSE,AMK_MSE),axis=1)
columns = ['QKLMS','QKLMS_AKB','QKLMS_AMK']
results = pd.DataFrame(data=data, columns=columns)
results.to_csv("MSE2_comparison_with_" + dataset + ".csv")





