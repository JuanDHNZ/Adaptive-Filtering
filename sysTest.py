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

"""Sistema 1 
 x = Proceso Gaussiano con media 0 y varianza 1+
 t = Filtro FIR en x
 
 Se prueban 2000 muestras
 
"""
samples = 200

u, d = testSystems.testSystems(samples=samples, systemType="1")
plt.plot(u, label="input")
plt.plot(d, label="target")
plt.legend()
plt.show()

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

plt.title("Sistema 1")
plt.plot(d, label="Target")
plt.plot(out, label="Predict")
plt.legend()
plt.show()


#**********************
""" Sistema 2 

Sistema Altamente no lineal

s = Sistema
u = Concatenacion de instantes anteriores
d = Instante actual

"""
samples = 200
offset = 100

s = testSystems.testSystems(samples=samples+offset, systemType="2")
plt.plot(s, label="highly non-lineal system")
plt.legend()
plt.show()


u = s[-samples-2:-2].reshape(-1,1)
# ub = s[-samples-3:-3].reshape(-1,1)
# uc = s[-samples-4:-4].reshape(-1,1)
# ud = s[-samples-5:-5].reshape(-1,1)
# ue = s[-samples-6:-6].reshape(-1,1)
# u = np.concatenate((ua,ub,uc,ud,ue), axis=1) 

d = s[-samples-1:-1].reshape(-1,1)

plt.plot(u, label="input")
plt.plot(d, label="target")
plt.legend()
plt.show()

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

plt.title("Sistema 2")
plt.plot(d, label="Target")
plt.plot(out, label="Predict")
plt.legend()


#**********************
samples = 200
offset = 100
s = testSystems.testSystems(samples=samples+offset , systemType="3")
plt.plot(s, label="sunspot dataset")
plt.legend()


ua = s[-samples-2:-2].reshape(-1,1)
ub = s[-samples-3:-3].reshape(-1,1)
uc = s[-samples-4:-4].reshape(-1,1)
ud = s[-samples-5:-5].reshape(-1,1)
ue = s[-samples-6:-6].reshape(-1,1)
u = np.concatenate((ua,ub,uc,ud,ue), axis=1) 

d = s[-samples-1:-1].reshape(-1,1 )

plt.plot(u, label="input")
plt.plot(d, label="target")
plt.legend()
plt.show()

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

plt.title("Sistema 2")
plt.plot(d, label="Target")
plt.plot(out, label="Predict")
plt.legend()



