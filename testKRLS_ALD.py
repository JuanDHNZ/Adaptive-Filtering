# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:27:40 2020

@author: USUARIO

PRUEBA A KRLS_ALD
"""
<<<<<<< HEAD
   
=======

def trainAndPlot(u,d):
    import KAF
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    
    krls = KAF.KRLS_ALD()
    pred = krls.evaluate(u,d)
    plt.plot(pred,"c",label="predict")
    plt.plot(d,"r",label="target")
    plt.legend()
    
>>>>>>> origin/master
import TimeSeriesGenerator as tsg
import numpy as np
import comPlot as cp

<<<<<<< HEAD
samples = 500
# epList = np.logspace(0, 2, 20)
# sgmList = np.logspace(0,2,20)

epList = np.linspace(1e-6,1e-1, 20)
=======
samples = 100
# epList = np.logspace(0, 2, 20)
# sgmList = np.logspace(0,2,20)

epList = np.linspace(0.01, 50, 20)
sgmList = np.linspace(0.01,50,20)
>>>>>>> origin/master

"""ATRACTOR DE LORENZ"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="lorenz")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

<<<<<<< HEAD
sgmList = np.linspace(0.1,np.max(u),10)

cp.KRLS_ALD_PLOT(u,d,sgmList,epList, "lorenz")



"""ATRACTOR DE CHUA"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="chua")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

sgmList = np.linspace(0.1,np.max(u),10)

cp.KRLS_ALD_PLOT(u,d,sgmList,epList,testName="chua")


"""
    ATRACTOR DE DUFFING

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="duffing")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET


sgmList = np.linspace(0.1,np.max(u),10)
cp.KRLS_ALD_PLOT(u,d,sgmList,epList,testName="duffing")


"""
    ATRACTOR DE NOSE HOOVER

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="nose_hoover")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

sgmList = np.linspace(0.1,np.max(u),10)
cp.KRLS_ALD_PLOT(u,d,sgmList,epList,testName="nose_hoover")



"""
    ATRACTOR DE RIKITAKE

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="rikitake")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

sgmList = np.linspace(0.1,np.max(u),10)
cp.KRLS_ALD_PLOT(u,d,sgmList,epList,testName="rikitake")



"""
    ATRACTOR DE ROSSLER

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="rossler")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

sgmList = np.linspace(0.1,np.max(u),10)
cp.KRLS_ALD_PLOT(u,d,sgmList,epList,testName="rossler")


"""
    ATRACTOR DE WANG

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="wang")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

sgmList = np.linspace(0.1,np.max(u),10)
cp.KRLS_ALD_PLOT(u,d,sgmList,epList,testName="wang")


"""
    SISTEMA 1 : 
    
    x = Proceso Gaussiano con media 0 y varianza 1+
    t = Filtro FIR en x   
    
"""
import testSystems

u, d = testSystems.testSystems(samples=samples, systemType="1")
u = u.reshape(-1,1)
d = d.reshape(-1,1)

sgmList = np.linspace(0.1,np.max(u),10)
cp.KRLS_ALD_PLOT(u,d,sgmList,epList,testName="sistema 1")


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

sgmList = np.linspace(0.1,np.max(u),10)
cp.KRLS_ALD_PLOT(u,d,sgmList,epList,testName="sistema 2")
  

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

sgmList = np.linspace(0.1,np.max(u),10)
cp.KRLS_ALD_PLOT(u,d,sgmList,epList,testName="sistema 3")



"""
***********************************************************

                        PREDICCIONES
                        
***********************************************************
"""
import search


"""
    SISTEMA 1 : 
    
    x = Proceso Gaussiano con media 0 y varianza 1+
    t = Filtro FIR en x   
    
"""
import testSystems

u, d = testSystems.testSystems(samples=samples, systemType="1")
u = u.reshape(-1,1)
d = d.reshape(-1,1)

sgmList = np.linspace(0.1,np.max(u),10)
search.gridSearchKRLS_plot_predict(u,d,sgmList,epList,testName="sistema 1")


"""
    ATRACTOR DE RIKITAKE

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="rikitake")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

sgmList = np.linspace(0.1,np.max(u),10)
search.gridSearchKRLS_plot_predict(u,d,sgmList,epList,testName="rikitake")


"""
    ATRACTOR DE ROSSLER

"""
x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="rossler")
ua = x[-samples-2:-2].reshape(-1,1)
ub = y[-samples-3:-3].reshape(-1,1)
u = np.concatenate((ua,ub), axis=1) # INPUT
d = z[-samples-1:-1].reshape(-1,1) #TARGET

sgmList = np.linspace(0.1,np.max(u),10)
search.gridSearchKRLS_plot_predict(u,d,sgmList,epList,testName="rossler")


=======
cp.KRLS_ALD_PLOT(u,d,sgmList,epList)



# """ATRACTOR DE CHUA"""
# x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="chua")
# ua = x[-samples-2:-2].reshape(-1,1)
# ub = y[-samples-3:-3].reshape(-1,1)
# u = np.concatenate((ua,ub), axis=1) # INPUT
# d = z[-samples-1:-1].reshape(-1,1) #TARGET

# epList = np.logspace(0, 2, 20)
# sgmList = np.logspace(0,2,20)
# cl = np.linspace(1,500,30)

# cp.dbPlot(u,d,sgmList,epList,r2_umbral=0.8,clusters=cl,testName="chua")


# """
#     ATRACTOR DE DUFFING

# """
# x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="duffing")
# ua = x[-samples-2:-2].reshape(-1,1)
# ub = y[-samples-3:-3].reshape(-1,1)
# u = np.concatenate((ua,ub), axis=1) # INPUT
# d = z[-samples-1:-1].reshape(-1,1) #TARGET

# epList = np.logspace(0, 2, 20)
# sgmList = np.logspace(0,2,20)
# cl = np.linspace(1,60,30)

# cp.dbPlot(u,d,sgmList,epList,r2_umbral=0.8,clusters=cl,testName="duffing")


# """
#     ATRACTOR DE NOSE HOOVER

# """
# x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="nose_hoover")
# ua = x[-samples-2:-2].reshape(-1,1)
# ub = y[-samples-3:-3].reshape(-1,1)
# u = np.concatenate((ua,ub), axis=1) # INPUT
# d = z[-samples-1:-1].reshape(-1,1) #TARGET

# epList = np.logspace(0, 2, 20)
# sgmList = np.logspace(0,2,20)
# cl = np.linspace(1,20,20)

# cp.dbPlot(u,d,sgmList,epList,r2_umbral=0.8,clusters=cl,testName="nose_hoover")



# """
#     ATRACTOR DE RIKITAKE

# """
# x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="rikitake")
# ua = x[-samples-2:-2].reshape(-1,1)
# ub = y[-samples-3:-3].reshape(-1,1)
# u = np.concatenate((ua,ub), axis=1) # INPUT
# d = z[-samples-1:-1].reshape(-1,1) #TARGET

# epList = np.logspace(0, 2, 20)
# sgmList = np.logspace(0,2,20)
# cl = np.linspace(1,20,20)

# cp.dbPlot(u,d,sgmList,epList,r2_umbral=0.8,clusters=cl,testName="rikitake")



# """
#     ATRACTOR DE ROSSLER

# """
# x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="rossler")
# ua = x[-samples-2:-2].reshape(-1,1)
# ub = y[-samples-3:-3].reshape(-1,1)
# u = np.concatenate((ua,ub), axis=1) # INPUT
# d = z[-samples-1:-1].reshape(-1,1) #TARGET

# epList = np.logspace(0, 2, 20)
# sgmList = np.logspace(0,2,20)
# cl = np.linspace(1,20,20)

# cp.dbPlot(u,d,sgmList,epList,r2_umbral=0.8,clusters=cl,testName="rossler")


# """
#     ATRACTOR DE WANG

# """
# x, y, z = tsg.chaoticSystem(samples=samples+10,systemType="wang")
# ua = x[-samples-2:-2].reshape(-1,1)
# ub = y[-samples-3:-3].reshape(-1,1)
# u = np.concatenate((ua,ub), axis=1) # INPUT
# d = z[-samples-1:-1].reshape(-1,1) #TARGET

# epList = np.logspace(0, 2, 20)
# sgmList = np.logspace(0,2,20)
# cl = np.linspace(1,50,30)

# cp.dbPlot(u,d,sgmList,epList,r2_umbral=0.8,clusters=cl,testName="wang")


# """
#     SISTEMA 1 : 
    
#     x = Proceso Gaussiano con media 0 y varianza 1+
#     t = Filtro FIR en x   
    
# """
# import testSystems

# u, d = testSystems.testSystems(samples=samples, systemType="1")
# u = u.reshape(-1,1)
# d = d.reshape(-1,1)
# epList = np.logspace(0, 2, 20)
# sgmList = np.logspace(0,2,20)
# cl = np.linspace(1,10,10)

# cp.dbPlot(u,d,sgmList,epList,r2_umbral=0.8,clusters=cl,testName="sistema 1")


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

# epList = np.logspace(0, 2, 20)
# sgmList = np.logspace(0,2,20)
# cl = np.linspace(1,15,15)

# cp.dbPlot(u,d,sgmList,epList,r2_umbral=0.8,clusters=cl,testName="sistema 2")
  

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

# # epList = np.logspace(0, 3, 40)
# # sgmList = np.logspace(0,3,20)
# cl = np.linspace(1,1000,5)

# clb = np.linspace(1,1000,5)
# wcp = np.linspace(0.01,1,5)

# # cp.dbPlot(u,d,sgmList,epList,r2_umbral=0.8,clusters=cl,testName="sistema 3")
# cp.dbPlot2(u,d,clusters_gmm=cl,clusters_bgmm=clb, wcp=wcp ,testName="sistema 3")

# """ PRETESTING"""


# # import KAF
# # fil = KAF.QKLMS(sigma=100, epsilon=1000)
# # y_pred = []
# # for i in range(len(d)):
# #     y_pred.append(fil.evaluate(u[i],d[i]))
# # y_pred = [j.item() for j in y_pred if j is not None]
# # print("CB size = ",len(fil.CB))
>>>>>>> origin/master
