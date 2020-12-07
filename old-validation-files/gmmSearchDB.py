# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 19:57:28 2020

@author: Juan David
"""

import matplotlib.pyplot as plt
def save_plot(search,clusters,titulo,imagen):
    r2scores = search.cv_results_['mean_test_score']
    import matplotlib as mpl
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.linestyle'] = '--'
    plt.yticks(np.linspace(0,1,11))
    plt.xticks(np.linspace(0,samples,11))
    plt.plot(clusters.astype(np.int64),r2scores,'b')
    plt.plot(clusters.astype(np.int64),r2scores,'ro',alpha=0.3 )
    plt.ylabel("R2")
    plt.xlabel("Codebook size")
    plt.title(titulo)
    plt.grid()
    plt.savefig("pruebasGMM/" + imagen + '.png', dpi = 300)
    plt.show()
    

import numpy as np
import TimeSeriesGenerator as tsg
import testSystems
"""NUMERO DE MUESTRAS PARA LAS PRUEBAS"""
samples = 1000
clusters = np.logspace(0,3,int(samples/20))

"""
    ATRACTOR DE LORENZ

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="lorenz")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

import gridSearchDB as gs
search = gs.gridSearchDB(u=u,d=d,cl=clusters)
save_plot(search,clusters,"Atractor de Lorenz", "pruebaLorenz")



"""
    ATRACTOR DE CHUA

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="chua")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

import gridSearchDB as gs
search = gs.gridSearchDB(u=u,d=d,cl=clusters)
save_plot(search,clusters,"Atractor de Chua", "pruebachua")


"""
    ATRACTOR DE DUFFING

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="duffing")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

import gridSearchDB as gs
search = gs.gridSearchDB(u=u,d=d,cl=clusters)
save_plot(search,clusters,"Atractor de Duffing", "pruebaduffing")


"""
    ATRACTOR DE NOSE HOOVER

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="nose_hoover")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

import gridSearchDB as gs
search = gs.gridSearchDB(u=u,d=d,cl=clusters)
save_plot(search,clusters,"Atractor de Nose Hoover", "pruebanosehoover")


"""
    ATRACTOR DE RIKITAKE

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="rikitake")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

import gridSearchDB as gs
search = gs.gridSearchDB(u=u,d=d,cl=clusters)
save_plot(search,clusters,"Atractor de Rikitake", "pruebarikitake")


"""
    ATRACTOR DE ROSSLER

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="rossler")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

import gridSearchDB as gs
search = gs.gridSearchDB(u=u,d=d,cl=clusters)
save_plot(search,clusters,"Atractor de Rossler", "pruebarossler")


"""
    ATRACTOR DE WANG

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="wang")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

import gridSearchDB as gs
search = gs.gridSearchDB(u=u,d=d,cl=clusters)
save_plot(search,clusters,"Atractor de Wang", "pruebawang")


"""
    SISTEMA 1 : 
    
    x = Proceso Gaussiano con media 0 y varianza 1+
    t = Filtro FIR en x   
    
"""

u, d = testSystems.testSystems(samples=samples, systemType="1")
import gridSearchDB as gs
search = gs.gridSearchDB(u=u.reshape(-1,1),d=d.reshape(-1,1),cl=clusters)
save_plot(search,clusters,"Sistema 1", "sistema1")


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

import gridSearchDB as gs
search = gs.gridSearchDB(u=u,d=d,cl=clusters)
save_plot(search,clusters,"Sistema 2", "sistema2")



""" 
    Sistema 3 
    
    Sunspot dataset

    s = Sistema
    u = Concatenacion de instantes anteriores
    d = Instante actual
    
"""

s = testSystems.testSystems(samples=samples+10 , systemType="3")
s = s.to_numpy()

ua = s[-samples-2:-2].reshape(-1,1)
ub = s[-samples-3:-3].reshape(-1,1)
uc = s[-samples-4:-4].reshape(-1,1)
ud = s[-samples-5:-5].reshape(-1,1)
ue = s[-samples-6:-6].reshape(-1,1)
u = np.concatenate((ua,ub,uc,ud,ue), axis=1)
d = s[-samples-1:-1].reshape(-1,1 )

import gridSearchDB as gs
search = gs.gridSearchDB(u=u,d=d,cl=clusters)
save_plot(search,clusters,"Sistema 3", "sistema3")