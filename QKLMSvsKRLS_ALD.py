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
batchSize = 10
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

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-1,300, 300)
sR_QKLMS_base = search.searchQKLMS_base(u,d,sgmList,epList,"lorenz")

wcp = np.linspace(1e-6, 1e-2,50)
sR_KLMS_BGMM = search.searchKLMS_BGMM(u,d,wcp,"lorenz",batchSize)


QKLMS_RESULTS = sR_QKLMS
KRLS_RESULTS = sR_KRLS
QKLMS_base_RESULTS = sR_QKLMS_base
KLMS_BGMM_RESULTS = sR_KLMS_BGMM


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

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-1,300, 300)
sR_QKLMS_base = search.searchQKLMS_base(u,d,sgmList,epList,"chua")

wcp = np.linspace(1e-6, 1e-2,50)
sR_KLMS_BGMM = search.searchKLMS_BGMM(u,d,wcp,"chua",batchSize)

QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])
QKLMS_base_RESULTS = pd.concat([QKLMS_base_RESULTS, sR_QKLMS_base])
KLMS_BGMM_RESULTS = pd.concat([KLMS_BGMM_RESULTS, sR_KLMS_BGMM])


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

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-1,300, 300)
sR_QKLMS_base = search.searchQKLMS_base(u,d,sgmList,epList,"duffing")

wcp = np.linspace(1e-6, 1e-2,50)
sR_KLMS_BGMM = search.searchKLMS_BGMM(u,d,wcp,"duffing",batchSize)


QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])
QKLMS_base_RESULTS = pd.concat([QKLMS_base_RESULTS, sR_QKLMS_base])
KLMS_BGMM_RESULTS = pd.concat([KLMS_BGMM_RESULTS, sR_KLMS_BGMM])


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

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-1,300, 300)
sR_QKLMS_base = search.searchQKLMS_base(u,d,sgmList,epList,"nose_hoover")

wcp = np.linspace(1e-6, 1e-2,50)
sR_KLMS_BGMM = search.searchKLMS_BGMM(u,d,wcp,"nose_hoover",batchSize)

QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])
QKLMS_base_RESULTS = pd.concat([QKLMS_base_RESULTS, sR_QKLMS_base])
KLMS_BGMM_RESULTS = pd.concat([KLMS_BGMM_RESULTS, sR_KLMS_BGMM])



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

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-1,300, 300)
sR_QKLMS_base = search.searchQKLMS_base(u,d,sgmList,epList,"rikitake")

wcp = np.linspace(1e-6, 1e-2,50)
sR_KLMS_BGMM = search.searchKLMS_BGMM(u,d,wcp,"rikitake",batchSize)

QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])
QKLMS_base_RESULTS = pd.concat([QKLMS_base_RESULTS, sR_QKLMS_base])
KLMS_BGMM_RESULTS = pd.concat([KLMS_BGMM_RESULTS, sR_KLMS_BGMM])




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

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-1,300, 300)
sR_QKLMS_base = search.searchQKLMS_base(u,d,sgmList,epList,"rossler")

wcp = np.linspace(1e-6, 1e-2,50)
sR_KLMS_BGMM = search.searchKLMS_BGMM(u,d,wcp,"rossler",batchSize)

QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])
QKLMS_base_RESULTS = pd.concat([QKLMS_base_RESULTS, sR_QKLMS_base])
KLMS_BGMM_RESULTS = pd.concat([KLMS_BGMM_RESULTS, sR_KLMS_BGMM])



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

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-1,300, 300)
sR_QKLMS_base = search.searchQKLMS_base(u,d,sgmList,epList,"wang")

wcp = np.linspace(1e-6, 1e-2,50)
sR_KLMS_BGMM = search.searchKLMS_BGMM(u,d,wcp,"wang",batchSize)


QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])
QKLMS_base_RESULTS = pd.concat([QKLMS_base_RESULTS, sR_QKLMS_base])
KLMS_BGMM_RESULTS = pd.concat([KLMS_BGMM_RESULTS, sR_KLMS_BGMM])



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

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-1,300, 300)
sR_QKLMS_base = search.searchQKLMS_base(u,d,sgmList,epList,"Sistema 1")

wcp = np.linspace(1e-6, 1e-2,50)
sR_KLMS_BGMM = search.searchKLMS_BGMM(u,d,wcp,"Sistema1",batchSize)


QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])
QKLMS_base_RESULTS = pd.concat([QKLMS_base_RESULTS, sR_QKLMS_base])
KLMS_BGMM_RESULTS = pd.concat([KLMS_BGMM_RESULTS, sR_KLMS_BGMM])


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

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-1,300, 300)
sR_QKLMS_base = search.searchQKLMS_base(u,d,sgmList,epList,"Sistema 2")

wcp = np.linspace(1e-6, 1e-2,50)
sR_KLMS_BGMM = search.searchKLMS_BGMM(u,d,wcp,"Sistema 2",batchSize)


QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])
QKLMS_base_RESULTS = pd.concat([QKLMS_base_RESULTS, sR_QKLMS_base])
KLMS_BGMM_RESULTS = pd.concat([KLMS_BGMM_RESULTS, sR_KLMS_BGMM])
  

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

sgmList = np.linspace(0.1,np.max(u),20)
epList = np.linspace(1e-1,300, 300)
sR_QKLMS_base = search.searchQKLMS_base(u,d,sgmList,epList,"Sistema 3")

wcp = np.linspace(1e-6, 1e-2,50)
sR_KLMS_BGMM = search.searchKLMS_BGMM(u,d,wcp,"Sistema 3",batchSize)


QKLMS_RESULTS = pd.concat([QKLMS_RESULTS,sR_QKLMS])
KRLS_RESULTS = pd.concat([KRLS_RESULTS,sR_KRLS])
QKLMS_base_RESULTS = pd.concat([QKLMS_base_RESULTS, sR_QKLMS_base])
KLMS_BGMM_RESULTS = pd.concat([KLMS_BGMM_RESULTS, sR_KLMS_BGMM])


"""RESULADOS A CSV"""     

QKLMS_RESULTS.to_csv("QKLMS_RESULTS.csv")
KRLS_RESULTS.to_csv("KRLS_RESULTS.csv")
QKLMS_base_RESULTS.to_csv("QKLMS_base_RESULTS.csv")
KLMS_BGMM_RESULTS.to_csv("KLMS_BGMM_RESULTS.csv")

