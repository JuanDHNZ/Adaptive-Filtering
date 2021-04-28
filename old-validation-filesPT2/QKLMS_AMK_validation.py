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
S = 4000
sys = 'lorenz'
u,d = db(samples=S,system=sys)
# u,d = db_z(samples=S,system=sys)

# 2. Parametros e Inicializacion
epsilon = 0.5 #Umbral de cuantizacion
sigma = 0.1 #Ancho de banda
eta = 0.5#Learning rate QKLMS

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
A0 = np.eye(D)/sigma #Matriz de proyeccion

start = 1

y_tar = []
y_pred = []
e_t = []

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=5).fit(u[:100,:])
A0 = pca.components_/100

tmp = np.dot(u[:100,:],A0.T)
# tmp = pca.transform(u[:100,:])
dd = np.exp(-0.5*cdist(tmp,tmp)**2)

# plt.imshow(dd)
# plt.colorbar()
# plt.show()


#Inicializacion AMK
A_k = [A0]
K_A = 10 #Memory
mu = 0.001#Step size A update


nda_t = []
na_t = []


# Filtro
for n in tqdm(range(start,N)):
    A = A_k[-1]
    ui = u[n]
    di = d[n]
    dis = cdist(CB, ui.reshape(1,-1),'mahalanobis', VI=np.dot(A.T,A)) # Distancia de mahalanobis
    K = np.exp(-0.5*(dis)**2) # Kernel
    yi = K.T.dot(np.array(a_coef)) # Prediccion   
    e = (di - yi).item() # Error
    
    d_m.append(dis)
    
    e_t.append(e/di.item())
    y_tar.append(di.item())
    y_pred.append(yi.item())
    
    # if n >= 2924:
    #     # Grafica de validacion 
    #     plt.scatter(n,yi,marker="X",label="predict")
    #     plt.scatter(n,di,marker="x",label="target")
    #     plt.legend()
    #     plt.title("Error = {}, i = {}".format(e/di.item(),n))
    #     plt.show()
    
    # plt.imshow(A)
    # plt.colorbar()
    # plt.show()
    
    # plt.imshow(A.T.dot(A))
    # plt.colorbar()
    # plt.show()
    
    min_dis = np.argmin(dis) #Index de distancia minima 
    testDists.append(dis[min_dis])
    """1. Primer metodo de actualizacion de A """  
    # if dis[min_dis] <= epsilon:
    #     a_coef[min_dis] = (a_coef[min_dis] + eta*e)
    # else: 
    #     da = [a_coef[j]*rbf(CB[j].reshape(1,-1),ui.reshape(1,-1))*(CB[j].reshape(1,-1) - ui.reshape(1,-1)).T.dot((CB[j].reshape(1,-1) - ui.reshape(1,-1))) for j in range(len(CB))]
    #     da  = np.sum(da,axis=0)                       
    #     A = e*A@da
    #     CB.append(ui)
    #     a_coef.append(eta*e)
    
    """2. Segundo metodo de actualizacion de A """
    # print("\n*********************************************")
    # print("\n\t\t n = {}\n".format(n))
    # print("CB size -> {}".format(len(CB)))  
    # # print("\ndis -> \n{}".format(dis))
    # print("\nKernel -> \n{}".format(K[-1]))
    # print("\nyi -> {}  Vs. di -> {}".format(yi,di))
    # print("\nerror -> {}".format(e/di.item()))
    # print("\ndis[min_dis] -> {}  Vs. epsilon -> {}".format(dis[min_dis],epsilon))
    # print("\n*********************************************")
    if n == 2780:
        wer = 1
    if dis[min_dis] <= epsilon:
        a_coef[min_dis] = (a_coef[min_dis] + eta*e)
    else:
        if len(CB) >= K_A:                  
            #Actualizar A_k y dejar A como el ultimo A_k
            S = len(CB)
            for i in range(K_A):
                da = 0
                for j in range(S-K_A,S):
                      da += a_coef[j]*K[j]*(CB[j].reshape(1,-1) - ui.reshape(1,-1)).T.dot((CB[j].reshape(1,-1) - ui.reshape(1,-1)))
                da = e*A_k[i]@da
                nda = np.linalg.norm(da,"fro")
                na = np.linalg.norm(A_k[i],"fro")
                nda_t.append(nda)
                na_t.append(na)
                # print("nda = \n{} \n na = \n{}".format(nda,na))
                A_k[i] -= mu*(da/nda)*na
                
            # print("\nSE ACTUALIZA A")
            # A_k = [A_k[i] - mu*e*A_k[i]@np.sum([a_coef[j]*rbf(CB[j].reshape(1,-1),ui.reshape(1,-1))*(CB[j].reshape(1,-1) - ui.reshape(1,-1)).T.dot((CB[j].reshape(1,-1) - ui.reshape(1,-1))) for j in range(len(CB))],axis=0) for i in range(K_A)]                     
        else:
            A_k.append(A0)
            
        CB.append(ui)
        a_coef.append(eta*e)
         
        
        
        
        
        
# Grafica de validacion 
plt.plot(y_pred,label="predict")
plt.plot(y_tar,label="target")
plt.legend()
plt.show()


# plt.imshow(A)
# plt.colorbar()
# plt.show()

print(" CB size = {}".format(len(CB)))



# KC = np.exp(-0.5*cdist(CB,CB,metric='mahalanobis',VI=A.T.dot(A))**2)
# plt.imshow(KC)
# plt.colorbar()
# plt.show()


# plt.plot(nda_t)
# plt.show()
# plt.plot(na_t)
# plt.show()