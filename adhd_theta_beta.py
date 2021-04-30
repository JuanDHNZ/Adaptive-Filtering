import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from scipy import signal
from scipy.integrate import simps
    #%%
class Ratio_Theta_Beta(BaseEstimator, ClassifierMixin):
    
    def __init__(self,fs=1.0, window='hann', nperseg=0.5, 
                 noverlap=0.5, nfft=None, detrend='constant',
                 return_onesided=True, scaling='density', 
                 axis=- 1, average='mean'):#parametros del modelo
        self.fs = fs
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.detrend = detrend
        self.return_onesided = return_onesided
        self.scaling = scaling
        self.axis = axis
        self.average = average

    def fit(self, X, y):#entrenar modelo
        # Check that X and y have correct shape
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        # Return the classifier
        return self

    def predict(self, X):#prediccion
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]
    
    def transform(self,X):
        # comprobar si el modelo se entrenó
        check_is_fitted(self)
        TB = []
        np = self.nperseg*self.fs
        no = self.noverlap*np
        for subj in X:
            tbs = []
            for ch in range(subj.shape[0]):
                freqs, psd = signal.welch(subj[ch,:], fs= self.fs, 
                        nperseg= np,  noverlap= no, nfft=self.nfft, 
                        detrend=self.detrend,
                        return_onesided=self.return_onesided, 
                        scaling=self.scaling, axis=self.axis,
                        average=self.average)
                low_theta, high_theta = 4, 8
                low_beta, high_beta = 12.5, 25
                # Find intersecting values in frequency vector
                idx_theta = self.indices(freqs,low_theta,high_theta)
                idx_beta = self.indices(freqs,low_beta,high_beta)
                freq_res = freqs[1] - freqs[0]  
                # Compute the absolute power by approximating 
                #the area under the curve
                theta_power = simps(psd[idx_theta], dx=freq_res)
                beta_power = simps(psd[idx_beta], dx=freq_res)
                tbs.append(theta_power/beta_power)
            TB.append(tbs)
        return TB   
    
    def indices(self,fvec,flow,fhigh):     
        al = abs(fvec-flow*np.ones([fvec.shape[0]])).tolist()
        indl = al.index(min(al))
        ah = abs(fvec-fhigh*np.ones([fvec.shape[0]])).tolist()
        indh = ah.index(min(ah))
        return np.arange(indl,indh+1,1)

    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)
    
    def get_params(self, deep=True):
        return {'fs':self.fs, 'window':self.window, 'nperseg':self.nperseg, 
                 'noverlap':self.noverlap, 'nfft':self.nfft, 'detrend':self.detrend,
                 'return_onesided':self.return_onesided, 'scaling':self.scaling, 
                 'axis':self.axis, 'average':self.average}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    
