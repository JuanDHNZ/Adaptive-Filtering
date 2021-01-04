# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:27:40 2020

@author: USUARIO

PRUEBA A KRLS_ALD
"""

import argparse
import TimeSeriesGenerator as tsg
import matplotlib.pyplot as plt

import numpy as np
import comPlot as cp

parser = argparse.ArgumentParser()
parser.add_argument('--kaf', help='Filter to train')
parser.add_argument('--dataset', help='Dataset to use')
parser.add_argument('-N', help='Dataset length (if available)',default=100,type=int)
parser.add_argument('-L', help='Embedding size',default=1,type=int)
parser.add_argument('--epsilon', help='Novelty threshold',default=1e-2,type=float)
parser.add_argument('--sigma', help='Gaussian kernel bandwidth',default=1,type=float)
args = parser.parse_args()
kaf = args.kaf
db = args.dataset
samples = args.N
L = args.L
eps = args.epsilon
sigma = args.sigma

plt.style.use('ggplot')

def live_plotter(x_vec,y1_data,ylim,line1,identifier='',pause_time=0.1):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)        
        #update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    plt.ylim(ylim)
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1

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
    
def main():

    if db in ['lorenz','chua','duffing','nose_hoover','rikitake','rossler','wang']:
        x, y, z = tsg.chaoticSystem(samples=samples,systemType=db)
        ux = np.array([x[i-L:i] for i in range(L,len(x))])
        uy = np.array([y[i-L:i] for i in range(L,len(y))])
        u = np.concatenate((ux,uy), axis=1) # INPUT
        d = np.array([z[i] for i in range(L,len(z))]).reshape(-1,1)
        dmin = d.min() - (d.max()-d.min())*0.1
        dmax = d.max() + (d.max()-d.min())*0.1
        print('Input',u.shape,'Output',d.shape)

    else:
        print('Unknown dataset')
        return
        
    if kaf == 'KRLS_ALD':
        from KAF import KRLS_ALD
        f = KRLS_ALD(sigma=sigma,epsilon=eps)
    elif kaf == 'QKLMS_AKB':
        from KAF import QKLMS_AKB
        f = QKLMS_AKB(sigma_init=sigma,epsilon=eps,K=5)
    
    else:
        print('Unknown KAF')
        return

    fig = plt.figure()
    ax = fig.add_subplot(111)

    y = np.zeros(d.shape)
    CB_ant = 0
    for n in range(len(u)):
        y[n] = f.evaluate(u[n].reshape(1,-1),d[n].reshape(1,-1))[0]
        if n>3:
            ax.plot(np.array([n-1,n]),d[n-2:n],color='blue',label='Target')
            ax.plot(np.array([n-1,n]),y[n-2:n],color='red',label='Predicted')
            plt.ylim([dmin,dmax])
            plt.pause(1e-3)
        if CB_ant != len(f.CB):
            ax.scatter(n,y[n],color='black',label='CB update',marker='+')
            CB_ant = len(f.CB)
    plt.show()
    return


if __name__ == "__main__":
    main()



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
