
import numpy as np
from sklearn.decomposition import FastICA


#%%
# import scipy.io as sio 

# path = r'G:\Mi unidad\Trabajo\2021\database\BCI\NEW_22ch_A01.mat'
# data = sio.loadmat(path)

# X = data['X']
# X = np.transpose(X,(2,1,0)) #trials x ch x time


#%%
class MAC():
    def __init__(self, th = 0.1, ch_ref = [0]):
        self.th = th
        self.ch_ref = ch_ref
        return 
    
    def set_params(self, th, ch_ref):
        self.th = th
        self.ch_ref = ch_ref
        return

    def fit(self, X,y):
        
        ref_ = X[:,self.ch_ref,:]
        N, C, T = X.shape
        
        X = np.transpose(X,(1,0,2)).reshape(C,-1) #Matriz con dimensiones C x NT
        ica = FastICA(n_components=C)
        S = ica.fit_transform(X.T)  # Reconstruct signals      
        self.A = ica.mixing_  # Get estimated mixing matrix #Se debe guardar 
        self.W = np.linalg.inv(self.A) #Se debe guardar 
        S = S.reshape(N, C, T)
        
        CORR = np.array([np.corrcoef(S[i].T, ref_[i].T, rowvar=False) for i in range(N)])
        CORR = CORR[:,:C,C:]
        self.mac = np.abs(CORR).mean(axis=2)
        
        return
        
    
    def transform(self, X):
        S = np.array([self.W@Xn for Xn in X])
        N, D, T = S.shape
        S_ = [] 
        A_ = []
        Nrec = [] #Reconstructed noise
        
        for trial in range(N):            
            ind = np.where(self.mac[trial] > self.th)[0]                  
            S_.append(S[trial,ind])
            A_.append(self.A[:,ind])
            Nrec.append(np.dot(A_[trial], S_[trial]))
            
        
        return np.array(Nrec)
            
            
    def fit_transform(self, X,y):
        self.fit(X,y)
        return self.transform(X)