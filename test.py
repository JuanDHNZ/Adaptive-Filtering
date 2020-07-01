# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:33:28 2020

@author: Juan David
"""
attr = ["chua","lorenz","duffing","nose_hoover","rikitake","rossler","wang"]
import TimeSeriesGenerator as tsg

for sys in attr:
    x, y, z = tsg.chaoticSystem(samples=10000,systemType=sys)
    
