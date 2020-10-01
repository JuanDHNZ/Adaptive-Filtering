# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:14:00 2020

@author: Juan

Prueba QKLMS with GMM vs. QKLMS with BayesianGMM 

Offline learning

"""

"""
-------------------------------------------------------------------------------
----------------------------- PRUEBAS GRAFICAS --------------------------------
-------------------------------------------------------------------------------

"""

import TimeSeriesGenerator as tsg
import numpy as np
import comPlot as cp
import search as s
samples = 400

# executionTime = []

# """ATRACTOR DE LORENZ"""
# x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="lorenz")
# ua = x[-samples-2:-2].reshape(-1,1)
# ub = y[-samples-3:-3].reshape(-1,1)
# u = np.concatenate((ua,ub), axis=1) # INPUT
# d = z[-samples-1:-1].reshape(-1,1) #TARGET

# #parametros
# cl = np.linspace(1,400,40)
# wcp = np.linspace(1e-6,1e3,10)

# st = s.startTimer()
# cp.dbPlot3(u,d,cl,wcp,"Lorenz t2")
# t  = s.stopTimer(st)
# executionTime.append(t)
# print("PRUEBA ATRACTOR DE LORENZ:")
# print("Tiempo de ejecucion:   {:.4f}".format(t))

    


# """ATRACTOR DE CHUA"""
# x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="chua")
# ua = x[-samples-2:-2].reshape(-1,1)
# ub = y[-samples-3:-3].reshape(-1,1)
# u = np.concatenate((ua,ub), axis=1) # INPUT
# d = z[-samples-1:-1].reshape(-1,1) #TARGET

# cp.dbPlot3(u,d,cl,wcp,"Chua")


# """
#     ATRACTOR DE DUFFING

# """
# x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="duffing")
# ua = x[-samples-2:-2].reshape(-1,1)
# ub = y[-samples-3:-3].reshape(-1,1)
# u = np.concatenate((ua,ub), axis=1) # INPUT
# d = z[-samples-1:-1].reshape(-1,1) #TARGET

# cp.dbPlot3(u,d,cl,wcp,"Duffing")


# """
#     ATRACTOR DE NOSE HOOVER

# """
# x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="nose_hoover")
# ua = x[-samples-2:-2].reshape(-1,1)
# ub = y[-samples-3:-3].reshape(-1,1)
# u = np.concatenate((ua,ub), axis=1) # INPUT
# d = z[-samples-1:-1].reshape(-1,1) #TARGET

# cp.dbPlot3(u,d,cl,wcp,"Nose Hoover")



# """
#     ATRACTOR DE RIKITAKE

# """
# x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="rikitake")
# ua = x[-samples-2:-2].reshape(-1,1)
# ub = y[-samples-3:-3].reshape(-1,1)
# u = np.concatenate((ua,ub), axis=1) # INPUT
# d = z[-samples-1:-1].reshape(-1,1) #TARGET

# cp.dbPlot3(u,d,cl,wcp,"Rikitake")



# """
#     ATRACTOR DE ROSSLER

# """
# x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="rossler")
# ua = x[-samples-2:-2].reshape(-1,1)
# ub = y[-samples-3:-3].reshape(-1,1)
# u = np.concatenate((ua,ub), axis=1) # INPUT
# d = z[-samples-1:-1].reshape(-1,1) #TARGET

# cp.dbPlot3(u,d,cl,wcp,"Rossler")


# """
#     ATRACTOR DE WANG

# """
# x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="wang")
# ua = x[-samples-2:-2].reshape(-1,1)
# ub = y[-samples-3:-3].reshape(-1,1)
# u = np.concatenate((ua,ub), axis=1) # INPUT
# d = z[-samples-1:-1].reshape(-1,1) #TARGET

# cp.dbPlot3(u,d,cl,wcp,"Wang")


# """
#     SISTEMA 1 : 
    
#     x = Proceso Gaussiano con media 0 y varianza 1+
#     t = Filtro FIR en x   
    
# """
# import testSystems

# u, d = testSystems.testSystems(samples=samples, systemType="1")
# u = u.reshape(-1,1)
# d = d.reshape(-1,1)

# cp.dbPlot3(u,d,cl,wcp,"Sistema 1")


# """ 
#     SISTEMA 2 

#     Sistema Altamente no lineal

#     s = Sistema
#     u = Concatenacion de instantes anteriores
#     d = Instante actual

# """

# s = testSystems.testSystems(samples=samples+10, systemType="2")

# ua = s[-samples-2:-2].reshape(-1,1)
# ub = s[-samples-3:-3].reshape(-1,1)
# uc = s[-samples-4:-4].reshape(-1,1)
# ud = s[-samples-5:-5].reshape(-1,1)
# ue = s[-samples-6:-6].reshape(-1,1)
# u = np.concatenate((ua,ub,uc,ud,ue), axis=1) 
# d = s[-samples-1:-1].reshape(-1,1)

# cp.dbPlot3(u,d,cl,wcp,"Sistema 2")
  

# """ 
#     Sistema 3 
    
#     Sunspot dataset

#     s = Sistema
#     u = Concatenacion de instantes anteriores
#     d = Instante actual
    
# """
# import testSystems
# s = testSystems.testSystems(samples=samples+10 , systemType="3")
# s = s.to_numpy()

# ua = s[-samples-2:-2].reshape(-1,1)
# ub = s[-samples-3:-3].reshape(-1,1)
# uc = s[-samples-4:-4].reshape(-1,1)
# ud = s[-samples-5:-5].reshape(-1,1)
# ue = s[-samples-6:-6].reshape(-1,1)
# u = np.concatenate((ua,ub,uc,ud,ue), axis=1)
# d = s[-samples-1:-1].reshape(-1,1 )

# cp.dbPlot3(u,d,cl,wcp,"Sistema 3")






"""
-------------------------------------------------------------------------------
------------------------------ PRUEBAS TABLAS ---------------------------------
-------------------------------------------------------------------------------

"""

#Resultados
results = []
#parametros
cl = np.linspace(1,400,40)
wcp = np.linspace(1e-6,1e3,10)

"""ATRACTOR DE LORENZ"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="lorenz")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

cl_gmm,r2_gmm,cl_bgmm,r2_bgmm = cp.dbPlot4(u,d,cl,wcp,"Lorenz t2")

results.append(['Lorenz',cl_gmm,r2_gmm,cl_bgmm,r2_bgmm])


"""ATRACTOR DE CHUA"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="chua")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

cl_gmm,r2_gmm,cl_bgmm,r2_bgmm = cp.dbPlot4(u,d,cl,wcp,"Chua")
results.append(['Chua',cl_gmm,r2_gmm,cl_bgmm,r2_bgmm])

"""
    ATRACTOR DE DUFFING

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="duffing")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

cl_gmm,r2_gmm,cl_bgmm,r2_bgmm = cp.dbPlot4(u,d,cl,wcp,"Duffing")
results.append(['Duffing',cl_gmm,r2_gmm,cl_bgmm,r2_bgmm])

"""
    ATRACTOR DE NOSE HOOVER

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="nose_hoover")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

cl_gmm,r2_gmm,cl_bgmm,r2_bgmm = cp.dbPlot4(u,d,cl,wcp,"Nose Hoover")
results.append(['Nose Hoover',cl_gmm,r2_gmm,cl_bgmm,r2_bgmm])


"""
    ATRACTOR DE RIKITAKE

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="rikitake")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

cl_gmm,r2_gmm,cl_bgmm,r2_bgmm = cp.dbPlot4(u,d,cl,wcp,"Rikitake")
results.append(['Rikitake',cl_gmm,r2_gmm,cl_bgmm,r2_bgmm])


"""
    ATRACTOR DE ROSSLER

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="rossler")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

cl_gmm,r2_gmm,cl_bgmm,r2_bgmm = cp.dbPlot4(u,d,cl,wcp,"Rossler")
results.append(['Rossler',cl_gmm,r2_gmm,cl_bgmm,r2_bgmm])

"""
    ATRACTOR DE WANG

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="wang")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

cl_gmm,r2_gmm,cl_bgmm,r2_bgmm = cp.dbPlot4(u,d,cl,wcp,"Wang")
results.append(['Wang',cl_gmm,r2_gmm,cl_bgmm,r2_bgmm])

"""
    SISTEMA 1 : 
    
    x = Proceso Gaussiano con media 0 y varianza 1+
    t = Filtro FIR en x   
    
"""
import testSystems

u, d = testSystems.testSystems(samples=samples, systemType="1")
u = u.reshape(-1,1)
d = d.reshape(-1,1)

cl_gmm,r2_gmm,cl_bgmm,r2_bgmm = cp.dbPlot4(u,d,cl,wcp,"Sistema 1")
results.append(['Sistema',cl_gmm,r2_gmm,cl_bgmm,r2_bgmm])

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

cl_gmm,r2_gmm,cl_bgmm,r2_bgmm = cp.dbPlot4(u,d,cl,wcp,"Sistema 2")
results.append(['Sistema 2',cl_gmm,r2_gmm,cl_bgmm,r2_bgmm])

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

cl_gmm,r2_gmm,cl_bgmm,r2_bgmm = cp.dbPlot4(u,d,cl,wcp,"Sistema 3")
results.append(['Sistema 3',cl_gmm,r2_gmm,cl_bgmm,r2_bgmm])



#Dataframe with results
import pandas as pd 
df = pd.DataFrame(results, columns = ['prueba', 'cb_gmm','r2_gmm','cb_bgmm','r2_bgmm'])  
#Export results in CSV
df.to_csv('pruebasGMM/GMM_Vs_BGMM/mejor_resultado_x_prueba.csv', index=False)