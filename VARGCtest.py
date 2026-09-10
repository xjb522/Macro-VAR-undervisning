"""

This script illustrates the use of built-in tests
for normality, autocorrelation and Granger non-causality tests.

@author: Michael Bergman
"""

import numpy as np
import pandas
import statsmodels.api as sm
import scipy.stats as stats
import math

import statsmodels.tsa.vector_ar.var_model
from statsmodels.tsa.vector_ar.var_model import *
from scipy.stats import chi2
from scipy.stats import f
import matplotlib.pyplot as plt

data = pandas.read_excel("BlanchardQuah.xlsx")
data = data[['dy','u']]

# Add dates to DataFrame
dates = pandas.date_range(start='1948-04-01', end='1988-01-01', freq='QE')
data.index = pandas.DatetimeIndex(dates)

# Determine lag order using information criteria
model = VAR(data)

# Lag order and # of equations
p = 2
K = 2

# Define VAR model, estimate model and show estimates

results = model.fit(p)
print(results.summary())
# Plot the data
results.plot()

# Use statsmodels tests for normality and autocorrelation
print(results.test_normality().summary())
print(results.test_whiteness().summary())

# Granger non-causality tests in bivariate models

from statsmodels.tsa.stattools import grangercausalitytests

print(results.test_causality('dy', 'u').summary())

grangercausalitytests(data[['dy', 'u']],maxlag=[p])
grangercausalitytests(data[['u', 'dy']],maxlag=[p])

# Note difference in degrees of freedom!
# What is the correct degrees of freedom?

# Correct degrees of freedom should be chi2(p) and F(p,T-K*p-1)
# so in this case
dgf = model.nobs-K*p-1
print('Correct dgf is',dgf)
print('Do not use test_causality!')
print('You should use grangercausalitytests in bivariate models')

# Then we compare to doing this on our own.
# This could be done using the estimates we already have.
# The covariance matrix of parameters cna be obtained using the
# instruction: pd.DataFrame.to_numpy(results.cov_params())
# which produces an array containing the covariance of vec(B)
# where B is the matrix [intercept, A_1, ..., A_p] (K x (Kp + 1))

# First we need varbeta, the covariance matrix of coefficients

varbeta = pd.DataFrame.to_numpy(results.cov_params())

# Then we need the paramater estimates

Beta = pd.DataFrame.to_numpy(results.params)

# and we need to vectorize Beta using the function in ownfunctions.py

bhat = vec(Beta)

# Design restriction matrices for both cases

# GC from variable 2 on variable 1

R1 = np.zeros([p,K*K*p+K])

j=1
for i in range(p):
   R1[i,i+j+3] = 1
   j = j+3


# GC from variable 1 on variable 2

R2 = np.zeros([p,K*K*p+K])

j=1
for i in range(p):
   R2[i,i+j+2] = 1
   j = j+3
   
# Then we can compute the test statistics.
# Note that the LR chi2 test reported above is not the same as
# the Wald test.   

Q1 = np.dot(np.dot(np.dot(-R1,bhat).T,np.linalg.inv(np.dot(np.dot(R1,varbeta),R1.T))),np.dot(-R1,bhat))
pvalQ1 = 1 - chi2.cdf(Q1,p)
Q1F = Q1/p
dgf = results.nobs-K*p-1
pvalQ1F = 1 - f.cdf(Q1F,p,dgf)

Q2 = np.dot(np.dot(np.dot(-R2,bhat).T,np.linalg.inv(np.dot(np.dot(R2,varbeta),R2.T))),np.dot(-R2,bhat))
pvalQ2 = 1 - chi2.cdf(Q2,p)
Q2F = Q2/p
dgf = results.nobs-K*p-1
pvalQ2F = 1 - f.cdf(Q2F,p,dgf)


print('Results own GC tests')
print('Testing GC from variable 2 on variable 1')
print('Wald Chi2 test',Q1,'with p-value',pvalQ1,'with',p,'degrees of freedom')
print('Wald F test',Q1F,'with p-value',pvalQ1F,'with df_denom',dgf,', df_num',p,'\n')
print('Testing GC from variable 1 on variable 2')
print('Wald Chi2 test',Q2,'with p-value',pvalQ2,'with',p,'degrees of freedom')
print('Wald F test',Q2F,'with p-value',pvalQ2F,'with df_denom',dgf,', df_num',p)







