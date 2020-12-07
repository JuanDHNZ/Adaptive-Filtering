# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 09:21:48 2020

@author: Juan

"""
def pruningGMM(gmm):
    """
    Ignores GMMs components with weights under an arbitrary threshold
    
    Returns means and covariaces
    """
    return gmm.means_[np.where(gmm.weights_>0.01)], gmm.covariances_[np.where(gmm.weights_>0.01)]

def pruningGMM2(gmm):
    """
    Ignores GMMs components with weights under an arbitrary threshold
    
    Returns means, covariaces and number of components
    """
    return gmm.means_[np.where(gmm.weights_>0.01)], gmm.precisions_[np.where(gmm.weights_>0.01)], np.where(model.weights_>0.01)[0].shape[0]


"""PRUEBA CON LINEAR REGRESSOR SIMPLE"""

    
    
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture as bgm
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
import TimeSeriesGenerator as tsg
import numpy as np

samples = 10000
batch_size = 50 #Batch size

attr_list = ["chua", "duffing", "nose_hoover", "rikitake","rossler","wang", "lorenz"]

for att_type in attr_list:
    x, y, z = tsg.chaoticSystem(samples = samples + 10, systemType = att_type)
    u = np.concatenate((x[-samples-2:-2].reshape(-1,1),y[-samples-3:-3].reshape(-1,1)), axis=1) # INPUT
    d = z[-samples-1:-1].reshape(-1,1) #TARGET
    
    # Training data dimensions
    Nu,Du = u.shape 
    Nd,Dd = d.shape
    
    #inicializacion
    scr = [] #Score

    """PRETRAIN STAGE"""
    #split data in batches 
    u_train = u.reshape(-1,batch_size,Du) 
    d_train = d.reshape(-1,batch_size,Dd)
    
    
    #Bayesian GMM with Warm Start
    model = bgm(n_components=batch_size,weight_concentration_prior=1e-3,warm_start=True,max_iter=20)
    #Bayesian GMM pre-training stage 
    model.fit(u_train[0]) 
    
    #Regressor
    reg = LinearRegression()
    
    #Save first-observed trainning date
    input_train_history = u_train[0]
    target_train_history = d_train[0]
    
    
    # print(input_train_history.shape,target_train_history.shape) #Test
    
    #Exclude pre-training instances
    u_train = u_train[1:]
    d_train = d_train[1:]
    
    # Option 1:
    for u_,d_ in zip(u_train,d_train):
        model.fit(u_)
        #Bayesian GMM model parameters
        Mn = model.means_ #means
        Sn = model.precisions_ #presitions
        Nc = model.n_components #Number of components
        
        # mns, prs, ncomp = pruningGMM2(model)
        
        # Mahalanobis kernel with BGMM parameters
        MK = [cdist(input_train_history, Mn[j].reshape(1,-1), 'mahalanobis', VI=Sn[j]) for j in range(Nc)]
        MK = [np.exp((-f**2)/2) for f in MK] 
        Phi = np.concatenate(MK,axis=1)
        
        # print(Phi.shape)#Test
        
        MK_ = [cdist(u_, Mn[j].reshape(1,-1), 'mahalanobis', VI=Sn[j]) for j in range(Nc)]
        MK_ = [np.exp((-f**2)/2) for f in MK_] 
        Phi_current = np.concatenate(MK_,axis=1)
        
        # print(Phi_current.shape)#Test
        
        #Bayesian Regressor 
        reg.fit(Phi,target_train_history)#training 
        y_pred = reg.predict(Phi_current)#prediction
        # y_pred = reg.partial_fit(Phi,d_).predict(Phi) #training and prediction
        scr.append(r2_score(d_,y_pred)) # R2 performance
        

        # print(att_type + " r2 = " , r2_score(d_,y_pred)) #Test
        
        #Save current batch in history
        input_train_history = np.concatenate((input_train_history,u_))
        target_train_history = np.concatenate((target_train_history,d_))
        
        # print(input_train_history.shape,target_train_history.shape) #Test
        
    plt.title("Test BGMM & BRR - " + att_type)
    plt.ylabel("R2")
    plt.ylim([-1,1])
    plt.xlabel("iterations")
    plt.plot(scr,'m')
    plt.show()
    



"""Mean & Covariance changes  - Tested with lorenz"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="lorenz")
u = np.concatenate((x[-samples-2:-2].reshape(-1,1),y[-samples-3:-3].reshape(-1,1)), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

#split data in batches 
u_train = u.reshape(-1,batch_size,Du) 
d_train = d.reshape(-1,batch_size,Dd)

#Bayesian GMM with Warm Start
model = bgm(n_components=batch_size,warm_start=True,max_iter=20)
#Bayesian GMM pre-training stage 
model.fit(u_train[0]) 

#Regressor
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

#Save first-observed trainning date
input_train_history = u_train[0]
target_train_history = d_train[0]
    
#Exclude pre-training instances
u_train = u_train[1:]
d_train = d_train[1:]

import search

for u_,d_ in zip(u_train,d_train):
    model.fit(u_)
    #Bayesian GMM model parameters
    Mn = model.means_ #means
    Sn = model.precisions_ #presitions
    Nc = model.n_components #Number of components
    
    
    #Mahalanobis kernel with BGMM parameters
    MK = [cdist(u_, Mn[j].reshape(1,-1), 'mahalanobis', VI=Sn[j]) for j in range(Nc)]
    MK = [np.exp((-f**2)/2) for f in MK] 
    Phi = np.concatenate(MK,axis=1)
    
    #Bayesian Regressor 
    y_pred = reg.fit(Phi,d_).predict(Phi) #training and prediction
    scr.append(r2_score(d_,y_pred)) # R2 performance
    
    plt.title("Means & Covs Behavior ")
    # plt.ylim([0,1])
    plt.xlabel("input data")
    plt.scatter(u_[:,0],u_[:,1],linewidths=1,marker="*",alpha=0.7)
    ax = plt.gca()
    
    mns, covs = pruningGMM(model)
    
    print(mns.shape, covs.shape)
    for mean, cov in zip(mns,covs):
        search.confidence_ellipse(cov=cov, mean=mean,ax=ax,edgecolor='red')
        plt.scatter(mean[0],mean[1], marker="x", color="red")
    plt.show()















 

