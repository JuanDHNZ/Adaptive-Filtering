# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:09:36 2021

@author: USUARIO
"""

class artifactRemoval():
    def __init__(self, th = 0.1):
        self.th = th
        return 
    
    def set_params(self, th):
        self.th = th
        return

    def fit(self, X,y):
        import numpy as np
        from sklearn.decomposition import FastICA
        from scipy.stats import kurtosis
        
        N, C, T = X.shape
        
        X = np.transpose(X,(1,0,2)).reshape(C,-1) #Matriz con dimensiones C x NT
        ica = FastICA(n_components=C)
        S = ica.fit_transform(X.T)  # Reconstruct signals
        
        self.A = ica.mixing_  # Get estimated mixing matrix #Se debe guardar 
        self.W = np.linalg.inv(self.A) #Se debe guardar 
    
        S = S.reshape(N, C, T)
        
        kur = kurtosis(S, axis = 2).reshape(-1)
        
        self.U_k = np.mean(kur) #Se debe guardar 
        self.STD_k = np.std(kur)#Se debe guardar 
        
    def transform(self, X):
        R = self.noiseEstimation(X)
        return self.noiseRemoval(X, R)
        
    def noiseEstimation(self, X):
        import numpy as np
        # 1. Calcular S en el espacio de ICA
        S = np.array([self.W@Xn for Xn in X])
        
        # X -> (N,C,T)       
        # W -> (IC,C)
        # A -> (C,IC)
        
        # 2. Calcular kurtosis normalizada
        from scipy.stats import kurtosis
        K = kurtosis(S, axis=2)
        Knormalizado = (K - self.U_k)/self.STD_k # K -> (N,IC)
        
        # 3. Umbralizar S y calcular R
        S_ = []
        R_ = []
        
        for n in range(X.shape[0]):
            C_index = np.where(Knormalizado[n] > self.th)[0]
            An = self.A[:,C_index]
            R_.append(An@S[n,C_index])
        return np.array(R_) 
    
    def noiseRemoval(self, X, r):
        # X -> r -> (N,C,T)
            # Xn -> rn -> (C,T)
                # Xnc -> rnc -> (T,)
        import numpy as np
        from tqdm import tqdm
        return np.array([[self.singleChanelNoiseRemoval(Xnc, rnc) for Xnc, rnc in zip(Xn,rn)] for Xn,rn in tqdm(zip(X,r))])
        
    def singleChanelNoiseRemoval(self, X, r):
        import numpy as np
        signalEmbedding = 5
        r_em = np.array([r[i-signalEmbedding:i] for i in range(signalEmbedding,len(r))])
        X_em = np.array([X[i] for i in range(signalEmbedding,len(r))]).reshape(-1,1)
        if np.sum(r_em) != 0.0:
            from KAF import QKLMS
            f = QKLMS(epsilon = 0)
            v = f.evaluate(r_em,X_em) # r -> X -> (T,)
            # print(v.shape,v.ravel().shape)
            return X_em.ravel() - v.ravel()  #en -> (T,)
        else:
            return X_em.ravel()
        
        
class artifact_removal_with_AMK:
    def __init__(self, th=0.1, embedding=5, eta=0.9, epsilon=0.1, mu=0.1, K=1):
        self.th = th
        self.embedding = embedding
        self.eta = eta
        self.epsilon = epsilon
        self.mu = mu
        self.K = K
        return 
    
    def set_params(self, th):
        self.th = th
        return

    def fit(self, X,y):
        import numpy as np
        from sklearn.decomposition import FastICA
        from scipy.stats import kurtosis
        
        N, C, T = X.shape
        
        X = np.transpose(X,(1,0,2)).reshape(C,-1) #Matriz con dimensiones C x NT
        ica = FastICA(n_components=C)
        S = ica.fit_transform(X.T)  # Reconstruct signals
        
        self.A = ica.mixing_  # Get estimated mixing matrix #Se debe guardar 
        self.W = np.linalg.inv(self.A) #Se debe guardar 
    
        S = S.reshape(N, C, T)
        
        kur = kurtosis(S, axis = 2).reshape(-1)
        
        self.U_k = np.mean(kur) #Se debe guardar 
        self.STD_k = np.std(kur)#Se debe guardar 
        
    def transform(self, X):
        R = self.noiseEstimation(X)
        return self.noiseRemoval(X, R)
        
    def noiseEstimation(self, X):
        import numpy as np
        # 1. Calcular S en el espacio de ICA
        S = np.array([self.W@Xn for Xn in X])
        
        # 2. Calcular kurtosis normalizada
        from scipy.stats import kurtosis
        K = kurtosis(S, axis=2)
        Knormalizado = (K - self.U_k)/self.STD_k # K -> (N,IC)
        
        # 3. Umbralizar S y calcular R
        S_ = []
        R_ = []
        
        for n in range(X.shape[0]):
            C_index = np.where(Knormalizado[n] > self.th)[0]
            An = self.A[:,C_index]
            R_.append(An@S[n,C_index])
        return np.array(R_) 
    
    def noiseRemoval(self, X, r):
        import numpy as np
        from tqdm import tqdm
        return np.array([[self.singleChannelNoiseRemoval(Xnc, rnc) for Xnc, rnc in zip(Xn,rn)] for Xn,rn in tqdm(zip(X,r))])
        
    def singleChannelNoiseRemoval(self, X, r):
        import numpy as np
        signalEmbedding = 5
        r_em = np.array([r[i-signalEmbedding:i] for i in range(signalEmbedding,len(r))])
        X_em = np.array([X[i] for i in range(signalEmbedding,len(r))]).reshape(-1,1)
        if np.sum(r_em) != 0.0:
            from KAF import QKLMS_AMK
            f = QKLMS_AMK(embedding=self.embedding, eta=self.eta, epsilon=self.epsilon, mu=self.mu, K=self.K)
            f.evaluate(r_em[:100],X_em[:100]) # Evaluation for PCA initialization
            v = f.evaluate(r_em,X_em) 
            return X_em.ravel() - v.ravel()  
        else:
            return X_em.ravel()
        

    