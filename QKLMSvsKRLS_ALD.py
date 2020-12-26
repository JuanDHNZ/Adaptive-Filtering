# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 23:17:34 2020

@author: USUARIO
"""

import TimeSeriesGenerator as tsg
import numpy as np
import search
import pandas as pd

samples = 500
# epList = np.logspace(0, 2, 20)
# sgmList = np.logspace(0,2,20)



"""ATRACTOR DE LORENZ"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="lorenz")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

epList = np.linspace(1e-1,300, 300)
sR_QKLMS = search.searchQKLMS(u,d,epList,"lorenz")

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-6,1e-1, 20)
sR_KRLS = search.searchKRLS_ALD(u,d,sgmList, epList,"lorenz")

QKLMS_RESULTS = sR_QKLMS
KRLS_RESULTS = sR_KRLS

"""ATRACTOR DE CHUA"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="chua")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

epList = np.linspace(1e-1,300, 300)
sR_QKLMS = search.searchQKLMS(u,d,epList,"chua")

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-6,1e-1, 20)
sR_KRLS = search.searchKRLS_ALD(u,d,sgmList, epList,"chua")

QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])



"""
    ATRACTOR DE DUFFING

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="duffing")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET


epList = np.linspace(1e-1,300, 300)
sR_QKLMS = search.searchQKLMS(u,d,epList,"duffing")

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-6,1e-1, 20)
sR_KRLS = search.searchKRLS_ALD(u,d,sgmList, epList,"duffing")

QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])


"""
    ATRACTOR DE NOSE HOOVER

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="nose_hoover")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

epList = np.linspace(1e-1,300, 300)
sR_QKLMS = search.searchQKLMS(u,d,epList,"nose_hoover")

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-6,1e-1, 20)
sR_KRLS = search.searchKRLS_ALD(u,d,sgmList, epList,"nose_hoover")

QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])



"""
    ATRACTOR DE RIKITAKE

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="rikitake")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

epList = np.linspace(1e-1,300, 300)
sR_QKLMS = search.searchQKLMS(u,d,epList,"rikitake")

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-6,1e-1, 20)
sR_KRLS = search.searchKRLS_ALD(u,d,sgmList, epList,"rikitake")

QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])




"""
    ATRACTOR DE ROSSLER

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="rossler")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

epList = np.linspace(1e-1,300, 300)
sR_QKLMS = search.searchQKLMS(u,d,epList,"rossler")

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-6,1e-1, 20)
sR_KRLS = search.searchKRLS_ALD(u,d,sgmList, epList,"rossler")

QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])



"""
    ATRACTOR DE WANG

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="wang")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

epList = np.linspace(1e-1,300, 300)
sR_QKLMS = search.searchQKLMS(u,d,epList,"wang")

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-6,1e-1, 20)
sR_KRLS = search.searchKRLS_ALD(u,d,sgmList, epList,"wang")

QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])



"""
    SISTEMA 1 : 
    
    x = Proceso Gaussiano con media 0 y varianza 1+
    t = Filtro FIR en x   
    
"""
import testSystems

u, d = testSystems.testSystems(samples=samples, systemType="1")
u = u.reshape(-1,1)
d = d.reshape(-1,1)

epList = np.linspace(1e-1,300, 300)
sR_QKLMS = search.searchQKLMS(u,d,epList,"Sistema 2")

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-6,1e-1, 20)
sR_KRLS = search.searchKRLS_ALD(u,d,sgmList, epList,"Sistema 1")

QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])



""" 
    SISTEMA 2 

    Sistema Altamente no lineal

    s = Sistema
    u = Concatenacion de instantes anteriores
    d = Instante actual

"""

s = testSystems.testSystems(samples=samples+10, systemType="2")

ua = s[-samples-2:-2].reshape(-1,1)
ub = s[-samples-3:-3].reshape(-1,1)
uc = s[-samples-4:-4].reshape(-1,1)
ud = s[-samples-5:-5].reshape(-1,1)
ue = s[-samples-6:-6].reshape(-1,1)
u = np.concatenate((ua,ub,uc,ud,ue), axis=1) 
d = s[-samples-1:-1].reshape(-1,1)

epList = np.linspace(1e-1,300, 300)
sR_QKLMS = search.searchQKLMS(u,d,epList,"Sistema 2")

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-6,1e-1, 20)
sR_KRLS = search.searchKRLS_ALD(u,d,sgmList, epList,"Sistema 2")

QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])

  

""" 
    Sistema 3 
    
    Sunspot dataset

    s = Sistema
    u = Concatenacion de instantes anteriores
    d = Instante actual
    
"""
import testSystems
s = testSystems.testSystems(samples=samples+10 , systemType="3")
s = s.to_numpy()

ua = s[-samples-2:-2].reshape(-1,1)
ub = s[-samples-3:-3].reshape(-1,1)
uc = s[-samples-4:-4].reshape(-1,1)
ud = s[-samples-5:-5].reshape(-1,1)
ue = s[-samples-6:-6].reshape(-1,1)
u = np.concatenate((ua,ub,uc,ud,ue), axis=1)
d = s[-samples-1:-1].reshape(-1,1 )

epList = np.linspace(1e-1,300, 300)
sR_QKLMS = search.searchQKLMS(u,d,epList,"Sistema 3")

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-6,1e-1, 20)
sR_KRLS = search.searchKRLS_ALD(u,d,sgmList, epList,"Sistema 3")

QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])



"""RESULADOS A CSV"""
QKLMS_RESULTS.to_csv("QKLMS_RESULTS.csv")
KRLS_RESULTS.to_csv("KRLS_RESULTS.csv")
