# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:12:50 2021

@author: Juan David

ALDKRLS_AKB Online test
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

def online_plot(samples, system, kaf, L=40):
    #1. Get data
    u,d = db(samples,system,L)
    # u,d = db_z(samples,system,L)
    #2. Inicializacion
    print("Online test on {} system:".format(system))
    y_pred = []
    y_tar = []
    new_add_x = [0]
    new_add_y = [d[0]]
    cb_ant = 1
    i=0
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    import matplotlib as mpl
    # mpl.rcParams['figure.dpi'] = 100
    #3. Predictions and plot
    for ui,di in zip(u,d):
        y_pred.append(kaf.evaluate(ui.reshape(1,-1),di.reshape(-1,1)))
        y_tar.append(di)
        print("Error {} en iteracion {}".format(kaf.error,i))
        if cb_ant != len(kaf.CB):
            new_add_x.append(i)
            new_add_y.append(y_pred[-1])
            cb_ant = len(kaf.CB)
        i+=1
        # plt.ylim([-dmax-dmax*0.1,dmax+dmax*0.1])        
       
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.plot(y_tar,'c', label="Target")
    plt.plot(y_pred, 'magenta', label="Predict")
    plt.legend()
    plt.scatter(new_add_x,new_add_y,c="magenta",marker="*")
    plt.show()
    fig.savefig('KRLS_ALD_AKB/{}_test_{}_samples_.png'.format(system,samples), dpi=300)
    print("Final Codebook Size -> {}".format(len(kaf.CB)))   
    return u,d,kaf




# KRLS ALD with AKB
import KAF
samples = 1000
L = 10

f = KAF.QKLMS_AMK(epsilon=1, sigma=1)
u,d,kaf = online_plot(samples,'lorenz',f)

f1 = KAF.QKLMS_AMK(epsilon=1, sigma=1)
y_pred = f1.evaluate(u,d)

import numpy as np
f2 = KAF.QKLMS_M(epsilon=1, sigma=1, A = np.eye(80)/50)
y_pred = f2.evaluate(u,d)
# Prediccion
u,d,kaf2 = online_plot(samples,'chua',f2)

import matplotlib.pyplot as plt

plt.plot(d, label="Target")
plt.plot(y_pred, label="predict")
plt.legend()
plt.show()

plt.imshow(kaf.A)
plt.colorbar()
plt.show()

# klms = KAF.ALDKRLS_AKB(sigma=50,K_akb=5,epsilon=1e-4)
# online_plot(samples,'lorenz',klms)

# klms = KAF.ALDKRLS_AKB(sigma=50,K_akb=5,epsilon=1e-4)
# online_plot(samples,'chua',klms)

# klms = KAF.ALDKRLS_AKB(sigma=10,K_akb=5,epsilon=1e-4)
# online_plot(samples,'duffing',klms)

# klms = KAF.ALDKRLS_AKB(sigma=10,K_akb=5,epsilon=1e-4)
# online_plot(samples,'nose_hoover',klms)

# klms = KAF.ALDKRLS_AKB(sigma=10,K_akb=5,epsilon=1e-4)
# online_plot(samples,'rossler',klms)

# klms = KAF.ALDKRLS_AKB(sigma=10,K_akb=10,epsilon=1e-4)
# online_plot(samples,'wang',klms)


# KRLS ALD with AKB
# klms = KAF.ALDKRLS_AKB_2(sigma=50,K_akb=5,epsilon=1e-4)
# online_plot(samples,'lorenz',klms)

# klms = KAF.ALDKRLS_AKB_2(sigma=50,K_akb=10,epsilon=1e-4)
# online_plot(samples,'chua',klms)

# klms = KAF.ALDKRLS_AKB_2(sigma=10,K_akb=5,epsilon=1e-4)
# online_plot(samples,'duffing',klms)

# klms = KAF.ALDKRLS_AKB_2(sigma=10,K_akb=5,epsilon=1e-4)
# online_plot(samples,'nose_hoover',klms)

# klms = KAF.ALDKRLS_AKB_2(sigma=10,K_akb=5,epsilon=1e-4)
# online_plot(samples,'rossler',klms)

# klms = KAF.ALDKRLS_AKB_2(sigma=10,K_akb=10,epsilon=1e-4)
# online_plot(samples,'wang',klms)






# # KRLS ALD
# klms = KAF.KRLS_ALD(sigma=50,epsilon=1e-4)
# online_plot(samples,'lorenz',klms)

# klms = KAF.KRLS_ALD(sigma=50,epsilon=1e-4)
# online_plot(samples,'chua',klms)

# klms = KAF.KRLS_ALD(sigma=10,epsilon=1e-4)
# online_plot(samples,'duffing',klms)

# klms = KAF.KRLS_ALD(sigma=10,epsilon=1e-4)
# online_plot(samples,'nose_hoover',klms)

# klms = KAF.KRLS_ALD(sigma=10,epsilon=1e-4)
# online_plot(samples,'rossler',klms)

# klms = KAF.KRLS_ALD(sigma=10,epsilon=1e-4)
# online_plot(samples,'wang',klms)

# # Pruebas en atractores

# u,d = db_z(samples,'lorenz',40)


#KLMS ALD 2
# klms = KAF.KRLS_ALD_2(sigma=50,epsilon=1e-4)
# online_plot(samples,'lorenz',klms)

# klms = KAF.KRLS_ALD(sigma=50,epsilon=1e-4)
# online_plot(samples,'chua',klms)

# klms = KAF.KRLS_ALD(sigma=10,epsilon=1e-4)
# online_plot(samples,'duffing',klms)

# klms = KAF.KRLS_ALD(sigma=10,epsilon=1e-4)
# online_plot(samples,'nose_hoover',klms)

# klms = KAF.KRLS_ALD(sigma=10,epsilon=1e-4)
# online_plot(samples,'rossler',klms)

# klms = KAF.KRLS_ALD(sigma=10,epsilon=1e-4)
# online_plot(samples,'wang',klms)