# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:24:40 2021

@author: Juan David
"""

import pandas as pd
lorenz = pd.read_csv("MSE2_comparison_with_lorenz.csv")
lorenz["QKLMS_AKB"] = pd.to_numeric(lorenz["QKLMS_AKB"], downcast="float")

def clean_akb(df):
    import re
    for i in range(len(df["QKLMS_AKB"])):
        if df["QKLMS_AKB"].iloc[i].startswith('['):
            df["QKLMS_AKB"].iloc[i] = df["QKLMS_AKB"].iloc[i][1:-1]
    return df
    
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.lineplot(data=lorenz["QKLMS"], palette="tab10", linewidth=2.5, label="QKLMS").set_title("Testing MSE - LORENZ -50 MonteCarlo runs ")
sns.lineplot(data=lorenz["QKLMS_AKB"], palette="tab10", linewidth=2.5, label="QKLMS_AKB").set_yscale("log")
sns.lineplot(data=lorenz["QKLMS_AMK"], palette="tab10", linewidth=2.5, label="QKLMS_AMK")
plt.ylim([1e-4,1e2])
plt.yscale("log")
plt.savefig('LORENZ.png', dpi=500)
plt.show()

sys42 = pd.read_csv("MSE2_comparison_with_4.2.csv")

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.lineplot(data=sys42["QKLMS"], palette="tab10", linewidth=2.5, label="QKLMS").set_title("Testing MSE - 4.2 - 50 MonteCarlo runs")
sns.lineplot(data=sys42["QKLMS_AKB"], palette="tab10", linewidth=2.5, label="QKLMS_AKB").set_yscale("log")
sns.lineplot(data=sys42["QKLMS_AMK"], palette="tab10", linewidth=2.5, label="QKLMS_AMK")
plt.ylim([1e-1,1e1])
plt.yscale("log")
plt.savefig('42.png', dpi=500)
plt.show() 

chua = pd.read_csv("MSE2_comparison_with_chua.csv")

chua = clean_akb(chua)
chua["QKLMS_AKB"] = pd.to_numeric(chua["QKLMS_AKB"], downcast="float")

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.lineplot(data=chua["QKLMS"], palette="tab10", linewidth=2.5, label="QKLMS").set_title("Testing MSE - CHUA - 50 MonteCarlo runs")
sns.lineplot(data=chua["QKLMS_AKB"], palette="tab10", linewidth=2.5, label="QKLMS_AKB").set_yscale("log")
sns.lineplot(data=chua["QKLMS_AMK"], palette="tab10", linewidth=2.5, label="QKLMS_AMK")
plt.savefig('CHUA.png', dpi=500)
plt.show()