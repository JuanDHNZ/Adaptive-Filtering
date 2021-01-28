# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 09:55:14 2021

@author: JUAN

Validation QKLMS_AMK

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
    uz = np.array([z[i-1] for i in range(L,len(z))]).reshape(-1,1)
    u = np.concatenate((ux,uy,uz), axis=1) # INPUT
    d = np.array([z[i] for i in range(L,len(z))]).reshape(-1,1)
    return u,d


# 1. Entrada y salida esperada
S = 1000
sys = 'lorenz'
u,d = db(samples=S,system=sys)
# u,d = db_z(samples=S,system=sys)

# 2. Parametros e Inicializacion
eta = 0.9 #Learning rate
epsilon = 1 #Umbral de cuantizacion
sigma = 0.1 #Ancho de banda

CB = [] #Codebook
a_coef = [] #Coeficientes
CB_growth = [] #Crecimiento del codebook por iteracion
initialize = True #Bandera de inicializacion
init_eval = True
evals = 0  #
        
testDists = []
d_m = []
        
N,D = u.shape #Tamaño de u
Nd,Dd = d.shape #Tamaño de d

from scipy.spatial import distance
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

rbf = lambda x,y : np.exp(-0.5*cdist(x, y,'mahalanobis', VI=np.dot(A.T,A))**2)

CB.append(u[0]) #Codebook
a_coef.append(eta*d[0]) #Coeficientes
A = np.eye(D)/sigma #Matriz de proyeccion

start = 1

y_tar = []
y_pred = []
e_t = []


# Filtro
for n in tqdm(range(start,N)):
    ui = u[n]
    di = d[n]
    dis = cdist(CB, ui.reshape(1,-1),'mahalanobis', VI=np.dot(A.T,A)) # Distancia de mahalanobis
    K = np.exp(-0.5*(dis)**2) # Kernel
    yi = K.T.dot(np.array(a_coef)) # Prediccion   
    e = (di - yi).item() # Error
    
    dist_m = [distance.mahalanobis(CB[k],ui,A.T.dot(A)) for k in range(len(CB))]
    d_m.append(dist_m)
    
    e_t.append(e/di.item())
    y_tar.append(di.item())
    y_pred.append(yi.item())
    
    if n > 700:
        # Grafica de validacion 
        plt.scatter(n,yi,marker="X",label="predict")
        plt.scatter(n,di,marker="x",label="target")
        plt.ylim([0,60])
        plt.legend()
        plt.title("Error = {}".format(e))
        plt.show()
    
    plt.imshow(A)
    plt.show()
    
    min_dis = np.argmin(dis) #Index de distancia minima 
    testDists.append(dis[min_dis])         
    if dis[min_dis] <= epsilon:
        a_coef[min_dis] = (a_coef[min_dis] + eta*e)
    else: 
        da = [a_coef[j]*rbf(CB[j].reshape(1,-1),ui.reshape(1,-1))*(CB[j] - ui.reshape(1,-1)).T.dot((CB[j] - ui.reshape(1,-1))) for j in range(len(CB))]
        da  = np.sum(da,axis=0)                       
        A = e*A@da
        CB.append(ui)
        a_coef.append(eta*e)


# Grafica de validacion 
plt.plot(y_pred,label="predict")
plt.plot(y_tar,label="target")
plt.legend()
plt.show()

print(" CB size = {}".format(len(CB)))
# while True:
#     ui = u[i,:].reshape(-1,D) 
#     yi,disti = self.__output(ui) #Salida  
#     # yi = np.sum(self.a_coef)
#     # self.__newEta(yi,err) #Nuevo eta
#     err = d[i] - yi # Error
    
    
#     #Cuantizacion
#     min_index = np.argmin(disti)           
#     if disti[min_index] <= self.epsilon:
#       self.a_coef[min_index] = (self.a_coef[min_index] + self.eta*err).item()
#     else: 
#     #self.A = err*self.A self.a_coef.T.@K *disti*disti
#         da = [self.a_coef[j]*rbf(self.CB[j].reshape(1,-1),ui)*(self.CB[j] - ui).T.dot((self.CB[j] - ui)) for j in range(len(self.CB))]
#         da  = np.sum(da,axis=0)                       
#         self.A = err*self.A@da
#         self.CB.append(u[i,:])
#         self.a_coef.append((self.eta*err).item())
    
#     self.CB_growth.append(len(self.CB)) #Crecimiento del diccionario 
    
#     if self.init_eval: 
#         y[i-1] = yi
#     else:
#         y[i] = yi
    
#     if(i == N-1):
#        if self.init_eval:
#            self.init_eval = False           
#            return y
#     i+=1 

# def __output(self,ui):
#     from scipy.spatial.distance import cdist
#     import numpy as np
#     dist = cdist(self.CB, ui,'mahalanobis', VI=np.dot(self.A.T,self.A))
#     # dist = [cdist(self.CB[k].reshape(1,-1), ui,'mahalanobis', VI=np.dot(self.A.T,self.A)) for k in range(len(self.CB))]
#     K = np.exp(-0.5*(dist**2))
#     y = K.T.dot(np.asarray(self.a_coef))
#     return [y,dist]

# def predict(self, ui):
#     from scipy.spatial.distance import cdist
#     import numpy as np
#     dist = cdist(self.CB, ui,'mahalanobis', VI=np.dot(self.A.T,self.A))
#     K = np.exp(-0.5*(dist**2))
#     y = K.T.dot(np.asarray(self.a_coef))
#     return y, K

