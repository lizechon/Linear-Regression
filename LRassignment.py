# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 20:54:03 2023

@author: echon
"""

import numpy as np
import regression_equation as lr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def estimate_coef(x, y):
    '''

    Parameters
    ----------
    x : TYPEfloat array
        DESCRIPTION.independent variable
    y : TYPE float array
        DESCRIPTION. dependent variable
    Returns
    -------
    coefficients, b0, b1 where y = b0+b1*x is the linear
    regression equation

    '''
    import numpy as np# get means of x and y
    n = len(x)
            
    m_x = np.mean(x)
    m_y = np.mean(y)
    
    SS_xy = np.sum(x*y)- n*m_x*m_y
    SS_xx = np.sum(x*x) - n*m_x*m_x
    
    b1 = SS_xy/SS_xx
    b0 = m_y - b1*m_x
    
    return (b0, b1)

if __name__ == "__main__":
    
    
    y = np.array([4,-1,2])
    x = np.array([1,2,3])
    
    b0, b1 = estimate_coef(x, y)
    
    plt.plot(x,y, '.')
    
    xline = np.linspace(0,4)
    yline = b0 + b1*xline
    
    plt.plot(xline,yline,color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('regression')
    plt.grid()
    plt.show()

# read in data
#look at column names, and I want to plot bettin line versus what actually
#happened

NFLdata = pd.read_csv('NFL2022.csv')
n = len(NFLdata)

type(NFLdata)
names = list(NFLdata)
print(names)

# SET UP Y WHICH IS THE SCORE =function of x, x=betting line
# delta = home team score - visit team score

x = NFLdata["overUnder"]
y = NFLdata["hScore"] + NFLdata['vScore']

m_x = np.mean(x)
std_x = np.std(x)  

b0, b1 = lr.estimate_coef(x, y)
print(b0, b1)

histAx = np.linspace(10, 90, 90)
plt.hist(x, alpha = 0.35, label='line', bins=20,density=True)
plt.hist(y, alpha = 0.45, label='actual spread',bins=20, density=True)
plt.xlabel('points')
plt.ylabel('Probability')
plt.plot(histAx, norm.pdf(histAx, m_x ,std_x), 'r-')
plt.legend()
plt.grid()
plt.show()

plt.plot(x, y, 'o')
xAxis = np.linspace(30, 60)

yPred = b0 + b1*xAxis
plt.plot(xAxis, yPred, color='red')
plt.xlabel('overUnder')
plt.ylabel('total')
#plt.hlines(y = m_x , xmin= 30, xmax=60, color="orange")
plt.plot(xAxis, xAxis, color='orange')
plt.grid()

bank=0
winCount = 0
loseCount = 0
for i in range (n):
    if x[i] >53.5:
        if y[i] > x[i]: 
            bank += 300
            winCount += 1
        else:
            bank += 330
            loseCount += 1
    if x[i] <32.5:
        if y[i] < x[i]:
            bank += 300
            winCount += 1
        else:
            bank -= 330
            loseCount += 1
            
print(f"bank = {bank}")
print(f"wins: {winCount}, loses {loseCount}")