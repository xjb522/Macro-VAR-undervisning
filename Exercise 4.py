import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2, f
from statsmodels.tsa.vector_ar.var_model import *


"""
import the following functions from your master file.

You will need your OWN companion and stability functions here!
"""

from master import jcitestexog, VARlsExog, sdummy, stabVAR, vec, companion, stabVAR

np.random.seed(2)

# Parameters
N = 1000  # Number of observations
rho1 = np.array([[0.8, 0],  # Autoregressive coefficient for lag 1
                 [0.8, 0]])

rho2 = np.array([[0.2, 0],  # Autoregressive coefficient for lag 1
                 [0.2, 0]])

alpha = np.array([0, 0])  # Mean of the VAR process

sigma = np.array([[1, 0],  # Covariance matrix of white noise
                  [0, 1]])
                  

# Pre-allocate the array for the VAR(2) process
y = np.zeros((2, N))

# Generate white noise
epsilon = np.random.multivariate_normal([0, 0], sigma, N).T

# Generate the VAR(2) process with time trend
for t in range(2, N):
    y[:, t] = alpha + rho1 @ y[:, t-1] + rho2 @ y[:, t-2] + epsilon[:, t]

y = y.T