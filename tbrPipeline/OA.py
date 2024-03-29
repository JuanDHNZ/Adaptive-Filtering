# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:09:36 2021

@author: USUARIO
"""

class artifact_removal():
    """
    ARTIFACT REMOVAL METHODOLOGY
    
    ICA -> KURTHOSIS THRESHOLDING -> KLMS AF
    
    """
    def __init__(self, th = 0.1):
        self.th = th
        return 

    def fit(self, X,y):
        import numpy as np
        from sklearn.decomposition import FastICA
        from scipy.stats import kurtosis
        
        N, C, T = X.shape
        
        X = np.transpose(X,(1,0,2)).reshape(C,-1) 
        ica = FastICA(n_components=C)
        S = ica.fit_transform(X.T)  
        self.A = ica.mixing_
        self.W = np.linalg.inv(self.A) 
    
        S = S.reshape(N, C, T)
        kur = kurtosis(S, axis = 2).reshape(-1)
        self.U_k = np.mean(kur) 
        self.STD_k = np.std(kur)
        
    def transform(self, X):
        import numpy as np
        R = self.noiseEstimation(X)
        return self.noiseRemoval(X, R)
    
    def predict(self, X):
        return self.transform(X)

    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)
    
    def get_params(self, deep=True):
        return {"th": self.th}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
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
    def __init__(self, filter_parameters, th = 0.1):
        self.embedding = filter_parameters['embedding']
        self.eta = filter_parameters['eta']
        self.epsilon = filter_parameters['epsilon']
        self.mu = filter_parameters['mu']
        self.Ka = filter_parameters['Ka']
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
        from KAF import QKLMS_AMK
        f = QKLMS_AMK(embedding=int(self.embedding), eta=self.eta, epsilon=self.epsilon, mu=self.mu, Ka=int(self.Ka))
        _,X_ = f.embedder(r,X)
        
        if np.sum(r) != 0.0:
            f.evaluate(r[:100],X[:100]) # Evaluation for PCA initialization
            v = np.array(f.evaluate(r,X))
            return X_.ravel() - v.ravel()  
        else:
            return X_.ravel()
        

class artifact_removal_with_MAC:
    def __init__(self, th = 0.1):
        self.th = th
        return 
    
    def get_params(self, deep=True):
        return {"th": self.th}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self,X,y):
        from mac import MAC
        mac_ = MAC(th=self.th)
        self.noise_est = mac_.fit_transform(X,y)
        return
    
    def transform(self, X):
        return self.__noise_removal(X)    
    
    def predict(self, X):
        return self.transform(X)
    
    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)
    
    def __noise_removal(self, X_true):
        import numpy as np
        from tqdm import tqdm
        return np.array([[self.__ch_noise_removal(rnc, Xnc) for rnc, Xnc in zip(rn,Xn)] for rn, Xn in tqdm(zip(self.noise_est ,X_true))])
        
    def __ch_noise_removal(self, noise_true, X_true):
        import numpy as np
        signalEmbedding = 5
        
        noise_em = np.array([noise_true[i-signalEmbedding:i] for i in range(signalEmbedding,len(noise_true))])
        X_em = np.array([X_true[i] for i in range(signalEmbedding,len(noise_true))]).reshape(-1,1)
        
        if np.sum(noise_em) != 0.0:
            from KAF import QKLMS
            f = QKLMS(epsilon = 0)
            noise_pred = f.evaluate(noise_em,X_em)
            
            return X_em.ravel() - noise_pred.ravel()
        else:
            return X_em.ravel()
        
        
class artifact_removal_MAC_AMK:
    def __init__(self, filter_parameters, th = 0.1):
        self.filter_params = filter_parameters
        self.embedding =  self.filter_params['embedding']
        self.eta =  self.filter_params['eta']
        self.epsilon =  self.filter_params['epsilon']
        self.mu =  self.filter_params['mu']
        self.Ka =  self.filter_params['Ka']
        self.th = th
        return 
       
    def get_params(self, deep=True):
        return {"th": self.th,"filter_parameters":self.filter_params}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self,X,y):
        from mac import MAC
        mac_ = MAC(th=self.th)
        self.noise_est = mac_.fit_transform(X,y)
        return
    
    def transform(self, X):
        return self.__noise_removal(X)    
    
    def predict(self, X):
        return self.transform(X)
    
    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)
    
    def __noise_removal(self, X_true):
        import numpy as np
        from tqdm import tqdm
        return np.array([[self.__ch_noise_removal(rnc, Xnc) for rnc, Xnc in zip(rn,Xn)] for rn, Xn in tqdm(zip(self.noise_est ,X_true))])
        
    def __ch_noise_removal(self, noise_true, X_true):
        import numpy as np
        from KAF import QKLMS_AMK
        
        f = QKLMS_AMK(embedding=int(self.embedding), eta=self.eta, epsilon=self.epsilon, mu=self.mu, Ka=int(self.Ka))
        _,X_em = f.embedder(noise_true,X_true)
        
        if np.sum(noise_true) != 0.0:
            noise_pred = f.evaluate(noise_true,X_true)
            import matplotlib.pyplot as plt
            # # Noise
            # plt.figure(figsize=(20,16),dpi=300)
            # plt.title("Noise")
            # plt.plot(noise_true,label='noise true')
            # plt.plot(noise_pred,label='noise predict')
            # plt.legend();plt.show()
            # # X
            # plt.figure(figsize=(20,16),dpi=300)
            # plt.title("X")
            # plt.plot(X_em,label='x raw')
            # plt.plot(X_em.ravel() - noise_pred.ravel(),label='x cleaned')
            # plt.legend();plt.show()
            # # X 
            # plt.figure(figsize=(20,16),dpi=300)
            # plt.title("X raw adn noise pred")
            # plt.plot(X_em,label='x raw')
            # plt.plot(noise_pred.ravel(),label='noise')
            # plt.legend();plt.show()
            # # VS X
            # plt.figure(figsize=(20,16),dpi=300)
            # plt.title("X vs")
            # plt.plot(X_em.ravel(),label='x true')
            # plt.plot(noise_pred,label='noise pred')
            # plt.plot(X_em.ravel() - noise_pred.ravel(),label='x cleaned')
            # plt.legend();plt.show()
            
            return X_em.ravel() - noise_pred.ravel()
        else:
            return X_em.ravel()      
        
        
        