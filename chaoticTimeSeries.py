# -*- coding: utf-8 -*-   
"""
Created on Tue Jun 30 11:51:26 2020

@author: Juan David

Se usan los algorimos desarrollados por
Obtenidos del repositorio https://github.com/capitanov/chaospy
"""
import numpy as np

def chua(x=0, y=0, z=1, **kwargs):
    """
    Calculate the next coordinate X, Y, Z for Chua system.

    Parameters
    ----------
    x, y, z : float
        Input coordinates Z, Y, Z respectively
    kwargs : float
        alpha, beta, mu0, mu1 - are Chua system parameters
    """
    # Default parameters:
    alpha = kwargs.get('alpha', 15.6)
    beta = kwargs.get('beta', 28)
    mu0 = kwargs.get('mu0', -1.143)
    mu1 = kwargs.get('mu1', -0.714)

    ht = mu1*x + 0.5*(mu0 - mu1)*(np.abs(x + 1) - np.abs(x - 1))
    # Next step coordinates:
    # Eq. 1:
    x_out = alpha*(y - x - ht)
    y_out = x - y + z
    z_out = -beta*y
    # Eq. 2:
    # x_out = 0.3*y + x - x**3
    # y_out = x + z
    # z_out = y

    return x_out, y_out, z_out
