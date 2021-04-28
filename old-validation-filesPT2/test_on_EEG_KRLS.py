# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:17:25 2021

@author: USUARIO

Test for BCI_s06train on KRLS

"""

# folder = 'data_4c/'
# filename = 'BCI_s06train.mat'

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# data = sio.loadmat(folder + filename)
# Xdata = data['X']
# Xdata = np.transpose(Xdata,(2,1,0))
# labels = data['labels'].reshape(-1,)
# fs = int(data['fs'].reshape(-1,))

# from OA import artifactRemoval

# ar = artifactRemoval(th=2)
# ar.fit(Xdata,labels)
# cleanEEG = ar.transform(Xdata)



from KAF import QKLMS_v2

f = QKLMS_v2(epsilon=0.0,sigma=0.1)

import testSystems as ts
import numpy as np

N = 200
L = 5
u,d = ts.testSystems(N, "4.1_AKB")

X = np.array([u[i-L:i] for i in range(L,len(u))])
y = np.array([d[i] for i in range(L,len(d))])
 
f.fit(X, y)
y_pred = f.predict(X)

plt.plot(y)
plt.plot(y_pred)
plt.show()

# from KAF import QKLMS
# f = QKLMS(epsilon=0.0)

# import testSystems as ts
# X,y = ts.testSystems(1000, "4.1_AKB")

# f.evaluate(X.reshape(-1,1),y.reshape(-1,1))






