# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:25:36 2023

@author: echon
linear regression
y = b0 + b1*x
given data pairs, y,x find values for b0 and b1 that best fit the data
demonstrate use of pandas and probability density functions
"""
import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x, y):
    """
    Parameters
    ----------
    x : TYPE float array
    DESCRIPTION. independent variable
    y : TYPE float array
    DESCRIPTION. dependent variable
    Returns
    -------
    coefficients, b0, b1 where y = b0 + b1*x is the linear
    regression equation
    """

    n = len(x)
    # get means of x and y
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(x*y)- n*m_x*m_y
    SS_xx = np.sum(x*x) - n*m_x*m_x
    b1 = SS_xy/SS_xx
    b0 = m_y - b1*m_x
    return (b0, b1)

if __name__ == '__main__':

    y = np.array([4,-1,2])
    x = np.array([1,2,3])
    b0, b1 = estimate_coef(x, y)
    plt.plot(x,y, '.',label='data')
    xline = np.linspace(0,4)
    yline = b0 + b1*xline
