"""

This script replicates the analysis in Sims (1972) paper.
It's an application of Granger non-causality tests.

@author: Michael Bergman
"""

import numpy as np
import pandas
import statsmodels.api as sm
import scipy.stats as stats
import math

# Note: You need to replace Ownfunctions with your masterfile and
# the masterfile includes the function nans
from Ownfunctions import *
from Functionnans import *

import matplotlib.pyplot as plt

data = pandas.read_excel("Sims1972D.xlsx")
data = data[['fm1','gdph']]

# Add dates to DataFrame
dates = pandas.date_range(start='1959-01-01', end='2000-12-31', freq='QE')
data.index = pandas.DatetimeIndex(dates)

simssample = 1      # Sample 1959:1-1968:4; = 0 for full sample

# Lag length according to Sims

p = 8

seas = 1     #=1 add seasonal dummies, =0 otherwise

# Data is already in logs

fm1 = pandas.DataFrame.to_numpy(data[['fm1']])
gdph = pandas.DataFrame.to_numpy(data[['gdph']])

# First we apply the filters suggested by Sims

lagmatfm1 = lagmat(fm1, [0, 1, 2])
filtfm1 = lagmatfm1[2:len(lagmatfm1),0]-1.50*lagmatfm1[2:len(lagmatfm1),1]+0.5625*lagmatfm1[2:len(lagmatfm1),2]

lagmatgdph = lagmat(gdph, [0, 1, 2])
filtgdph = lagmatgdph[2:len(lagmatgdph),0]-1.50*lagmatgdph[2:len(lagmatgdph),1]+0.5625*lagmatgdph[2:len(lagmatfm1),2]

if simssample==1:
    filtfm1 = filtfm1[0:40]
    filtgdph = filtgdph[0:40]
    

# =============================================================================
# Sims Granger non-causality tests: Table 1
# OLS regressions with linear trend and seasonal dummies
# see footnote to Table 4
# =============================================================================

# First test: 8 lags

# Seasonal dummies

# use own function sdummy
from sdummy import *

freq = 4
seasons = sdummy(len(filtfm1),freq)

# Normalize seasonal dummies

# %%
sd = seasons-np.dot((1/freq),np.ones((40, 4)))


sd =sd[:,0:freq-1]

# Then we run regressions. First step is to define dependent and independent variables

nlags = np.arange(1,p+1)
indep = lagmat(filtfm1, lags=nlags)
determ = np.concatenate((np.ones((len(filtfm1),1)),np.arange(1,40+1).reshape(40,1)), axis=1)


if seas==1:
     determ = np.concatenate((determ,sd), axis=1)

indep = np.concatenate((determ,indep), axis=1)
# Remove missing observations

indep = indep[~np.isnan(indep).any(axis=1)]
dep = filtgdph[p:len(gdph)]


# Run regression

model = sm.OLS(dep,indep).fit()
print(model.summary())


# Then we need to test whether filtfm1 affects filtgdph
# We have 8 lags and 13 coefficients

# We need to define R to single out parameters to test
R = np.concatenate( (np.zeros([p,len(indep.T)-8]) ,np.identity(8)), axis=1    )

# Then we use built-in function to compute an F-test
print('Testing first hypothesis\n',model.f_test(R))

# Second test: 4 future and 8 past lags

newind1= np.hstack(( filtfm1[4:len(filtfm1)], nans([4]) ))
newind2= np.hstack(( filtfm1[3:len(filtfm1)], nans([3])  ))
newind3= np.hstack(( filtfm1[2:len(filtfm1)], nans([2])  ))
newind4= np.hstack(( filtfm1[1:len(filtfm1)], nans([1])  ))
newind5= np.hstack(( filtfm1[0:len(filtfm1)], nans([0]) ))
newind = np.vstack((newind1,newind2,newind3,newind4,newind5)).T

ind2 = np.hstack((determ,newind,lagmat(filtfm1, lags=nlags)))

# Remove missing observations
indep = ind2[~np.isnan(ind2).any(axis=1)]
dep = filtgdph[8:36]

model = sm.OLS(dep,indep).fit()
print(model.summary())
# We need to define R to single out parameters to test
R = np.concatenate( (np.zeros([13,5]) ,np.identity(13)), axis=1    )
print('Testing second hypothesis\n',model.f_test(R))

# Third test: 8 past lags: GDP on M1

p = 8
nlags = np.arange(1,p+1)
indep = lagmat(filtgdph, lags=nlags)
determ = np.concatenate((np.ones((len(filtgdph),1)),np.arange(1,40+1).reshape(40,1)), axis=1)


if seas==1:
     determ = np.concatenate((determ,sd), axis=1)

indep = np.concatenate((determ,indep), axis=1)
# Remove missing observations

indep = indep[~np.isnan(indep).any(axis=1)]
dep = filtfm1[p:len(gdph)]

model = sm.OLS(dep,indep).fit()
print(model.summary())

R = np.concatenate( (np.zeros([8,5]) ,np.identity(8)), axis=1    )
print('Testing third hypothesis\n',model.f_test(R))

# Fourth test: 4 future and 8 past lags: GDP on M1

newind1= np.hstack(( filtgdph[4:len(filtgdph)], nans([4]) ))
newind2= np.hstack(( filtgdph[3:len(filtgdph)], nans([3])  ))
newind3= np.hstack(( filtgdph[2:len(filtgdph)], nans([2])  ))
newind4= np.hstack(( filtgdph[1:len(filtgdph)], nans([1])  ))
newind5= np.hstack(( filtgdph[0:len(filtgdph)], nans([0]) ))
newind = np.vstack((newind1,newind2,newind3,newind4,newind5)).T

ind2 = np.hstack((determ,newind,lagmat(filtgdph, lags=nlags)))

# Remove missing observations
indep = ind2[~np.isnan(ind2).any(axis=1)]
dep = filtfm1[8:36]

model = sm.OLS(dep,indep).fit()
print(model.summary())

R = np.concatenate( (np.zeros([13,5]) ,np.identity(13)), axis=1    )
print('Testing fourth hypothesis\n',model.f_test(R))


# Sims Table 3
# OLS four future quarters
#  
# GDP on M1

p = 8
nlags = np.arange(1,p+1) 
newind = np.vstack((newind1,newind2,newind3,newind4)).T

ind2 = np.hstack((determ,newind))

# Remove missing observations
indep = ind2[~np.isnan(ind2).any(axis=1)]
dep = filtfm1[0:36]

model = sm.OLS(dep,indep).fit()
print(model.summary())

R = np.concatenate( (np.zeros([4,5]) ,np.identity(4)), axis=1    )
print('Testing first hypothesis Table 3 in Sims\n',model.f_test(R))

# M1 on GDP

newind1= np.hstack(( filtfm1[4:len(filtfm1)], nans([4]) ))
newind2= np.hstack(( filtfm1[3:len(filtfm1)], nans([3])  ))
newind3= np.hstack(( filtfm1[2:len(filtfm1)], nans([2])  ))
newind4= np.hstack(( filtfm1[1:len(filtfm1)], nans([1])  ))

newind = np.vstack((newind1,newind2,newind3,newind4)).T

ind2 = np.hstack((determ,newind))

# Remove missing observations
indep = ind2[~np.isnan(ind2).any(axis=1)]
dep = filtgdph[0:36]

model = sm.OLS(dep,indep).fit()
print(model.summary())

R = np.concatenate( (np.zeros([4,5]) ,np.identity(4)), axis=1    )
print('Testing second hypothesis Table 3 in Sims\n',model.f_test(R))

# We can then do the same for unfiltered data, but results are virtually the same.