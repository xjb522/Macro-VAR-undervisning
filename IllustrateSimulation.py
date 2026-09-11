#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:36:02 2023

Simulate VAR(p) model

Load monthly data for 1973.2-2007.12 ordered as:
1. Growth rate of world oil production 
2. Global real activity (index based on dry cargo shipping rates)
3. Real price of oil 
The data sources are described in the KL.

@author: Michael Bergman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from statsmodels.tsa.api import *

# Own functions
from Ownfunctions import *

# Load data
data = np.loadtxt('data.txt')
y = data
t, q = y.shape                      # t=obs, q=K
print(t, q)
K = q
time = pd.date_range(start='1973-02', end='2008-01', freq='ME')  # Time line
h = 15                               # Impulse response horizon
p = 4                                # VAR lag order
con = 1                              # Constant in VAR
tr = 0                               # Linear Trend in VAR
boot = 1                             # =1 Bootstrap initial condition, 0 otherwise                                    
BS_type = 1                          # Bootstrap type: = 0 for nonparametric = 1 for Wild

# Define indicator for trend
if con==1:
    trend = 'c'
    
    if tr==1:
        trend = 'ct'



# Estimate VAR using LS
model = VAR(y)

results = model.fit(p,trend=trend)
print(results.summary())
Beta = results.params


# Define deterministic components and AR parameter matrix
Beta = Beta.T
a = Beta[:,con+tr:len(Beta.T)]
V = Beta[:,0:con+tr]

# Simulate the VAR(p) model

# Transpose residuals
u = results.resid.T

# Prepare initial conditions
if boot == 0:
    y0p = y[:p-1, :K-1]  # in Python, indexing starts from 0
else:
    pos = random.randint(p+1, t)  # in Python, randint's upper limit is inclusive
    y0p = y[(pos-p+1):(pos+1), :]  # add 1 to pos for Python's exclusive upper limit in slicing


Tbig = t-p  # time periods to simulate

# Draw with replacement from uHat
indexur = np.random.randint(0, Tbig, size=Tbig)  # Using numpy.random.randint! Generate Tbig
ur = u[:, indexur]
ur = np.hstack((np.zeros((K, p)), ur))  # Add residuals for initial conditions



# Handle deterministic components: Only constant and linear trend allowed
if con == 1:
      determ = np.ones((t, 1))
else:
      determ = np.array([]).reshape(0, 1)  # define an empty column vector


if tr == 1:
      added_array = np.arange(-p+1, t-p+1).reshape(-1, 1)
      determ = np.column_stack((determ, added_array))


i = p
j = 0
yr = np.zeros((K, Tbig))  # simulated y for t=1:Tbig
yr = np.hstack((y0p.T, yr))  # Add initial values

while i < Tbig+p  :   # Python uses 0-based indexing
     index = np.flip(np.arange(j, j+p))
    # print('This is index after flip\n',index)
     ylags = vec(yr[:, index])
     ylags = np.reshape(ylags, (K,p))
     ylags = vec(ylags.T)
     if BS_type == 0:
      # Non-parametric BS  
         yr[:, i] = (V @ determ[i, :].reshape(-1, 1) + a @ ylags + ur[:, i].reshape(-1,1)).ravel()
     else:
      # If Wild Gaussian BS, multiply residuals with random number
         yr[:, i] = (V @ determ[i, :].reshape(-1, 1) + a @ ylags + np.random.randn() * ur[:, i].reshape(-1,1)).ravel()   # np.random.randn() generates a random number from a standard normal distribution

     i += 1
     j += 1
     
     
yr = yr.T

fig, axs = plt.subplots(3, 1, figsize=(8,10), layout='constrained')  # create 3 rows and 1 columns of subplots

for i in range(3):
      axs[i].plot(time, y[:, i], label='Original')
      axs[i].plot(time, yr[:,i], label='Simulated')
      axs[i].legend(loc='upper right')
      

plt.show()
 
# Then we can estimate the VAR and compare to original estimates
  
model = VAR(yr)
resultsr = model.fit(p,trend=trend)
print(resultsr.summary())
Betar = resultsr.params.T
# Compute differences
diffbeta = Beta-Betar
# Vectorize
vecdiffbeta = vec(diffbeta)
print('Average difference\n',vecdiffbeta.mean(axis=0))
 
 
 
 
 
 