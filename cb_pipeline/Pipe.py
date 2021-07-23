# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 08:56:46 2021

@author: Asus
"""
import pandas as pd
from kurtosis import KURTOSIS
import numpy as np
import pickle 
from tbr import TBR
from fisher import FISHER
from sklearn.pipeline import Pipeline
from sklearn.metrics import cohen_kappa_score,make_scorer
from sklearn.model_selection import GridSearchCV
# path = r'G:\Mi unidad\Trabajo\2021\Kurtosis\TBR\data\data\adhd\set\SST_DC_AM_new.set'
# labels = r'G:\Mi unidad\Trabajo\2021\Kurtosis\TBR\data\data\adhd\csv\SST_DC_AM_new.csv'
# path = r'G:\Mi unidad\Trabajo\2021\Kurtosis\TBR\data\pkl\adhd_Trial_reward_allchannels_hmm.pkl'

path = '../../Dataset/adhd_Trial_reward_allchannels_hmm.pkl'

def score(fisher,labels):
    import numpy as np
    sb = (np.mean(fisher[labels>0.5])-np.mean(fisher[labels<0.5]))**2
    sw = np.var(fisher)
    return sb/sw


with open(path,'rb') as f: 
    Data = pickle.load(f) 

x = [i for i in Data if (i.get('cond') == 'IC')]

X_ = []
labels_ = []
for i,j in enumerate(x):
    X_.append(j.get('smiley'))     
    
    if (j.get('label') == 'Control'):
        arr = np.zeros(j.get('smiley').shape[2])
    
    elif(j.get('label') == 'tdah'):
        arr = np.ones(j.get('smiley').shape[2])   
    labels_.append(arr)

    
X = np.concatenate((X_), axis = 2)
Xdata = np.transpose(X, [2,0,1])

labels = np.concatenate((labels_)).reshape(-1,)


steps = [('preproc', KURTOSIS()),
         ('extract', TBR()),
         ('classify', FISHER())
    ]

pipeline = Pipeline(steps = steps)


grid = {'preproc__th':[10],
        'extract__fs':[250]
        }

search = GridSearchCV(pipeline,grid,cv=5,verbose=10,scoring=make_scorer(score),return_train_score=True)
search.fit(Xdata,labels)
cv_results = search.cv_results_

cv_results = pd.DataFrame.from_dict(cv_results)
cv_results.to_csv('gridsearch.csv')


from datetime import datetime
now = datetime.now()
print(now)


# f = TBR(fs=250).fit_transform(Xdata,labels)
# fisher = FISHER().fit_transform(f, labels)

# def score(fisher,labels):
#     import numpy as np
#     sb = (np.mean(fisher[labels==1])-np.mean(fisher[labels==0]))**2
#     sw = np.var(fisher)
#     return sb/sw


# scr = score(fisher,labels)

# import matplotlib.pyplot as plt

# plt.scatter(labels,fisher)















# thold = np.linspace(0,3,10)



 

# Fisher = []
# for a,j in enumerate(thold):
      
#     X_filt = KURTOSIS(th = j).fit_transform(X,labels)
    
#     Theta_Beta = TBR(fs = 250).fit_transform(X_filt,labels)
       
    
#     fisher = FISHER(cov=True).fit_transform(Theta_Beta, labels)
    
    
#     Fisher.append(fisher)

# dic = {'Threshold' : thold,
#        'Fisher' : Fisher}
# df = pd.DataFrame(dic)

# df.to_csv('fisher_scores.csv')

