# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:33:47 2020

@author: USUARIO

SEARCH TEST 
"""


import numpy as np 
import matplotlib.pyplot as plt
import testSystems as ts
import KAF 
import search

def plot_pair(u = None,d = None,labelu="u",labeld="d", title = ""):
    plt.title(title)
    plt.plot(u,label = labelu)
    plt.plot(d, label = labeld)
    plt.legend()
    plt.show()
    
def parameterTest(u,d,sg1,sg2,ep1,ep2):    
    from sklearn.metrics import r2_score
    """ Para  QKLMS """
    print("QKLMS")
    print("Best Sigma = ", sg1)
    print("Best Epsilon = ", ep1)
    pred = []
    qklms = KAF.QKLMS(sigma=sg1,epsilon=ep1)
    for i in range(len(u)):
       pred.append(qklms.evaluate(u[i],d[i]))
    pred = [i.item() for i in pred if i is not None]
    #Grafico
    plt.title("QKLMS")
    plt.plot(pred, label="Predict")
    plt.plot(d[1:], label="Target")
    plt.legend()
    plt.show()
    # plt.title("QKLMS codebook growth")
    # plt.plot(qklms.CB_growth)
    # plt.show()
    
    R2_qklms = r2_score(d[1:], pred)
    print("R2 QKLMS = ", R2_qklms)
    
    """ Para  QKLMS 2 """
    print("\nM-QKLMS")
    print("Best Sigma = ", sg2)
    print("Best Epsilon = ", ep2)
    pred = []
    mqklms = KAF.QKLMS2(sigma=sg2,epsilon=ep2)
    for i in range(len(u)):
       pred.append(mqklms.evaluate(u[i],d[i]))
    pred = [i.item() for i in pred if i is not None]
    #Grafico
    plt.title("M-QKLMS")
    plt.plot(pred, label="Predict")
    plt.plot(d[1:], label="Target")
    plt.legend()
    plt.show()
    # plt.title("M-QKLMS codebook growth")
    # plt.plot(qklms.CB_growth)
    # plt.show()
    
    R2_qklms = r2_score(d[1:], pred)
    print("R2 QKLMS = ", R2_qklms)
    
    
    print("\nCodebook Sizes:")
    print("QKLMS = ", len(qklms.CB))
    print("M-QKLMS = ", len(mqklms.CB))



"""************************************************************************************************************"""

""" Señal de prueba 1 -  SP500 """
import pandas as pd
sp500 = pd.read_csv("datasets/spx.csv")

samples = 400

# Señal deseada
d = sp500.close.iloc[-samples-1:-1].to_numpy().reshape(-1,1)
# Señal de entrada
u1 = sp500.close.iloc[-samples-2:-2].to_numpy().reshape(-1,1)
u2 = sp500.close.iloc[-samples-3:-3].to_numpy().reshape(-1,1)
u3 = sp500.close.iloc[-samples-4:-4].to_numpy().reshape(-1,1)
u4 = sp500.close.iloc[-samples-5:-5].to_numpy().reshape(-1,1)
u5 = sp500.close.iloc[-samples-6:-6].to_numpy().reshape(-1,1)
u = np.concatenate((u1,u2,u3,u4,u5), axis=1) 

epsilon = np.logspace(2, 4, 20)
sigma = np.logspace(2, 4, 20)

# epsilon = np.logspace(3, 8, 20)
# sigma = [1]

print("\n\nSP500 - Economic timeseries")

sg1, ep1, sg2, ep2 = search.pSearchCurve(u=u, d=d, sigmaList = sigma, epsilonList = epsilon, r2_threshold=0.9)

parameterTest(u,d,sg1, ep1, sg2, ep2)

"""************************************************************************************************************"""

""""
SISTEMA 1 

u -> proceso Gaussiano de media cero y varianza unitaria
d -> resultado de aplicar filtro FIR a u

"""
"""NUMERO DE MUESTRAS"""
# samples = 400

u, d = ts.testSystems(samples=samples, systemType="1")
plot_pair(u,d,labelu="input", labeld="target", title="Sistema 1")

epsilon = np.logspace(-3, 6, 20)
sigma = np.logspace(-3, 6, 20)

print("\n\nSistema 1")
sg1, ep1, sg2, ep2 = search.pSearchCurve(u=u.reshape(-1,1), d=d.reshape(-1,1), sigmaList = sigma, epsilonList = epsilon, r2_threshold=0.7)

parameterTest(u.reshape(-1,1),d.reshape(-1,1),sg1, ep1, sg2, ep2)



"""************************************************************************************************************"""

""" Sistema 2 

Sistema Altamente no lineal

s = Sistema
u = Concatenacion de instantes anteriores
d = Instante actual

"""
"""NUMERO DE MUESTRAS"""
# samples = 400
offset = 10

s = ts.testSystems(samples=samples+offset, systemType="2")
plt.plot(s)
plt.title("Highly Non-lineal System - Sistema 3")
plt.show()


ua = s[-samples-2:-2].reshape(-1,1)
ub = s[-samples-3:-3].reshape(-1,1)
uc = s[-samples-4:-4].reshape(-1,1)
ud = s[-samples-5:-5].reshape(-1,1)
ue = s[-samples-6:-6].reshape(-1,1)
u = np.concatenate((ua,ub,uc,ud,ue), axis=1) 

d = s[-samples-1:-1].reshape(-1,1)


epsilon = np.logspace(-3, 6, 20)
sigma = np.logspace(-3, 6, 20)

print("\n\nSistema 2")
sg1, ep1, sg2, ep2 = search.pSearchCurve(u=u, d=d, sigmaList = sigma, epsilonList = epsilon, r2_threshold=0.9)

parameterTest(u,d,sg1, ep1, sg2, ep2)


"""************************************************************************************************************"""

"""Sistema 3 """
samples = 2000
offset = 10
s = ts.testSystems(samples=samples+offset , systemType="3")
s = s.to_numpy()
plt.plot(s)
plt.title("Sunspot dataset - Sistema 3")
plt.show()


ua = s[-samples-2:-2].reshape(-1,1)
ub = s[-samples-3:-3].reshape(-1,1)
uc = s[-samples-4:-4].reshape(-1,1)
ud = s[-samples-5:-5].reshape(-1,1)
ue = s[-samples-6:-6].reshape(-1,1)
u = np.concatenate((ua,ub,uc,ud,ue), axis=1)

d = s[-samples-1:-1].reshape(-1,1 )

epsilon = np.logspace(-3, 6, 20)
sigma = np.logspace(-3, 6, 20)
print("\n\nSistema 3")
sg1, ep1, sg2, ep2 = search.pSearchCurve(u=u, d=d, sigmaList = sigma, epsilonList = epsilon, r2_threshold=0.8)

parameterTest(u,d,sg1, ep1, sg2, ep2)


"""************************************************************************************************************"""

"""************************************************************************************************************"""

"""************************************************************************************************************"""


""" PRUEBAS CON ATRACTORES """

"""ATRACTOR DE CHUA"""

import TimeSeriesGenerator as tsg
samples = 2000
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="chua")
signal = x.reshape(-1,1)


d = signal[-samples-1:-1]

ua = signal[-samples-2:-2].reshape(-1,1)
ub = signal[-samples-3:-3].reshape(-1,1)
uc = signal[-samples-4:-4].reshape(-1,1)
u = np.concatenate((ua,ub,uc), axis=1) 
plt.title("Segmento de Atractor de Chua")
plt.plot(u)
plt.show()

epsilon = np.logspace(-3, 6, 20)
sigma = np.logspace(-3, 6, 20)

print("\n\nAtractor de Chua")

sg1, ep1, sg2, ep2 = search.pSearchCurve(u=u, d=d, sigmaList = sigma, epsilonList = epsilon, r2_threshold=0.8)

parameterTest(u,d,sg1, ep1, sg2, ep2)


"""ATRACTOR DE LORENZ"""
import TimeSeriesGenerator as tsg
samples = 1000
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="lorenz")
signal = x.reshape(-1,1)


d = signal[-samples-1:-1]

ua = signal[-samples-2:-2].reshape(-1,1)
ub = signal[-samples-3:-3].reshape(-1,1)
uc = signal[-samples-4:-4].reshape(-1,1)
u = np.concatenate((ua,ub,uc), axis=1) 
plt.title("Segmento de Atractor de lorenz")
plt.plot(u)
plt.show()

epsilon = np.linspace(1, 100, 40)
sigma = np.linspace(1,100,40)

print("\n\nAtractor de Lorenz")
sg1, ep1, sg2, ep2 = search.pSearchCurve(u=u, d=d, sigmaList = sigma, epsilonList = epsilon, r2_threshold=0.9)

parameterTest(u,d,sg1, ep1, sg2, ep2)







