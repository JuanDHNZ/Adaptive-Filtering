# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:47:27 2020

@author: Juan David

SISTEMAS DEL ESTADO DEL ARTE - PRUEBAS EN QKLMS

"""
import numpy as np
import testSystems
import matplotlib.pyplot as plt
import KAF

def plot_pair(u = None,d = None,labelu="u",labeld="d", title = ""):
    plt.title(title)
    plt.plot(u,label = labelu)
    plt.plot(d, label = labeld)
    plt.legend()
    plt.show()

def plot_codebookGrowth(estimator = None):
    plt.plot(estimator.CB_growth)
    plt.title("Crecimiento Codebook")
    plt.show()
    

"""Sistema 1 
 x = Proceso Gaussiano con media 0 y varianza 1+
 t = Filtro FIR en x
 
 Se prueban 2000 muestras
 
"""
samples = 20

u, d = testSystems.testSystems(samples=samples, systemType="1")
plot_pair(u,d,labelu="input", labeld="target", title="Sistema 1")

# GRID SEARCH
epsilon = np.logspace(-5, 5, 20)
sigma = np.logspace(-5, 5, 20)

parameters ={'epsilon':epsilon, 'sigma':sigma}

from sklearn.model_selection import GridSearchCV
filtro = KAF.QKLMS2()
cv = [(slice(None), slice(None))]
mqklms = GridSearchCV(filtro,parameters,cv=cv)
mqklms.fit(u.reshape(-1,1),d.reshape(-1,1))

a_results = mqklms.cv_results_

print("Mejores parametros : ", mqklms.best_params_)
print("Mejor score : ", mqklms.best_score_)

# CALCULO CON MEJOR ESTIMADOR

out = mqklms.best_estimator_.evaluate(u.reshape(-1,1),d.reshape(-1,1))

plot_pair(d,out,"target","predict", "Salida sistema 1")
plot_codebookGrowth(estimator = mqklms.best_estimator_)


#**********************
""" Sistema 2 

Sistema Altamente no lineal

s = Sistema
u = Concatenacion de instantes anteriores
d = Instante actual

"""
samples = 40
offset = 10

s = testSystems.testSystems(samples=samples+offset, systemType="2")
plt.plot(s)
plt.title("highly non-lineal system")
plt.show()


ua = s[-samples-2:-2].reshape(-1,1)
ub = s[-samples-3:-3].reshape(-1,1)
uc = s[-samples-4:-4].reshape(-1,1)
ud = s[-samples-5:-5].reshape(-1,1)
ue = s[-samples-6:-6].reshape(-1,1)
u = np.concatenate((ua,ub,uc,ud,ue), axis=1) 

d = s[-samples-1:-1].reshape(-1,1)

plot_pair(u,d,labelu="input", labeld="target", title="Sistema 2")

""" GRID SEARCH """
epsilon = np.logspace(-5, 5, 20)
sigma = np.logspace(-5, 5, 20)
parameters ={'epsilon':epsilon, 'sigma':sigma}

from sklearn.model_selection import GridSearchCV
filtro = KAF.QKLMS2()
cv = [(slice(None), slice(None))]
mqklms = GridSearchCV(filtro,parameters,cv=cv)#,verbose=10
mqklms.fit(u,d)

print("Mejores parametros : ", mqklms.best_params_)
print("Mejor score : ", mqklms.best_score_)

# CALCULO CON MEJOR ESTIMADOR
out = mqklms.best_estimator_.evaluate(u,d)
plot_pair(d,out,labelu="target", labeld="predict", title="Sistema 2")


sigma = mqklms.best_params_["sigma"]
epsilon = mqklms.best_params_["epsilon"]


plot_codebookGrowth(estimator = mqklms.best_estimator_)
#**********************

"""Sistema 3 """
samples = 20
offset = 10
s = testSystems.testSystems(samples=samples+offset , systemType="3")
s = s.to_numpy()
plt.plot(s)
plt.title("Sunspot dataset")
plt.show()


ua = s[-samples-2:-2].reshape(-1,1)
ub = s[-samples-3:-3].reshape(-1,1)
uc = s[-samples-4:-4].reshape(-1,1)
ud = s[-samples-5:-5].reshape(-1,1)
ue = s[-samples-6:-6].reshape(-1,1)
u = np.concatenate((ua,ub,uc,ud,ue), axis=1)

d = s[-samples-1:-1].reshape(-1,1 )

plot_pair(u,d,labelu="input", labeld="target", title="Sistema 3")

# GRID SEARCH
epsilon = np.logspace(-5, 5, 20)
sigma = np.logspace(-5, 5, 20)

parameters ={'epsilon':epsilon, 'sigma':sigma}
from sklearn.model_selection import GridSearchCV
filtro = KAF.QKLMS2()
cv = [(slice(None), slice(None))]
mqklms = GridSearchCV(filtro,parameters,cv=cv)
mqklms.fit(u,d)

a_results = mqklms.cv_results_

print("Mejores parametros : ", mqklms.best_params_)
print("Mejor score : ", mqklms.best_score_)

# CALCULO CON MEJOR ESTIMADOR

filter_ = KAF.QKLMS2(epsilon=mqklms.best_params_["epsilon"])
out = filter_.evaluate(u,d)

plot_pair(d,out,labelu="target", labeld="predict", title="Sistema 3")
plot_codebookGrowth(estimator = mqklms.best_estimator_)



    
    