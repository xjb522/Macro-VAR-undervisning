#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script illustrates the Top-Down Sequence.

@author: Michael Bergman
"""

import numpy as np
import pandas
import statsmodels.api as sm
import scipy.stats as stats
import math


# Note: You need to use your own master file here!
from Ownfunctions import *


import matplotlib.pyplot as plt

data = pandas.read_excel("BlanchardQuah.xlsx")
data = data[['dy','u']]

# Add dates to DataFrame
dates = pandas.date_range(start='1948-04-01', end='1988-01-01', freq='QE')
data.index = pandas.DatetimeIndex(dates)


# Set pmax
pmax = 12

# Create dependent and independent variables using lagmat

# Need to convert DataFrame to array
x = data[['dy', 'u']].to_numpy()
[T,K]=x.shape
nlags = np.arange(0,pmax+1)
y = lagmat(x, lags=nlags)

# Remove missing values
y = y[~np.isnan(y).any(axis=1)]

# Create dependent and independent variables

dep = y[:,0:K]
[nobs,K] = dep.shape

# Add constant to indep and lagged endogenous variables
indep = np.concatenate((np.ones([nobs,1]),y[:,K:K*pmax+2]), axis=1)



# Now we can estimate the VAR(m) and the VAR(m+1) models
# Compute all Log Likelihoods
# =============================================================================
LL = np.zeros([1, 2]);
ii = pmax;
 
yl = indep[:,0:ii*K+1]
dep = dep
Beta = np.dot(np.linalg.inv(np.dot(yl.T,yl)),np.dot(yl.T,dep))
# # =============================================================================
# =============================================================================
while ii>-1:
     yl = indep[:,0:ii*K+1]
     Beta = np.dot(np.linalg.inv(np.dot(yl.T,yl)),np.dot(yl.T,dep))
     res = dep-np.dot(yl,Beta)
     sopmax = np.dot(res.T,res)/nobs
     # Log-Likelihood: Hamilton (1994);
     LL1 = [[ii,-(nobs/2)*(np.log(np.linalg.det(sopmax)) + K*np.log(2*math.pi) + K)]]
     LL = np.concatenate((LL1,LL), axis=0)
     ii=ii-1
 
 
# =============================================================================
# Set up table with results
# Note LR-test: LR(p) = 2*(LL(p)-LL(p-1))

LL = np.array(LL)
teststat = np.array([[ 0, LL[0,1], 0, 0]])
 
ii=0;
while ii<pmax:
    help = np.array([[np.round(LL[ii+1,0], 3), np.round(LL[ii+1,1], 3), np.round(2*( LL[ii+1,1] - LL[ii,1] ), 3), np.round(1-chi2.cdf(2*( LL[ii+1,1] - LL[ii,1] ),K*K), 3)]])
    teststat = np.concatenate((teststat,help), axis=0) 
    ii=ii+1

# Table with results

from tabulate import tabulate

print('')
print('Top-Down Testing Sequence')
print('pmax =',pmax)
print("==============================================")
print (tabulate(teststat, headers=["Lag", "Log. Likelihood", "LR test", "p-value"], floatfmt=".3f", numalign="right"))
print("==============================================")

