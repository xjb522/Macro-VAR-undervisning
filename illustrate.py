#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:36:02 2023

Illustration: Standard residual-based recursive design bootstrap


@author: Michael Bergman
"""

import numpy as np
import matplotlib.pyplot as plt
import random


# We have estimated the following residuals
# uhat is a 5x3 matrix, K=3 and T=5

uhat = np.asarray([[0.54, -1.31, -1.35],[1.83, -0.43, 3.03],[-2.26, 0.34, 0.72],[0.86, 3.58, -0.06],[0.32, 2.77, 0.71]])
index = np.reshape(np.asarray(range(5)), (1,-1)).T
print(np.hstack((index,uhat)))


# For each bootstrap replication r we generate a new index
# draw random numbers from uniform distribution, a 1 by 5 vector
# Since we have T=5.

# Note that these are already integers, so no need to transform
# to random numbers between 1 and 5.

Tbig = 5
indexstar = np.random.randint(0, Tbig, size=Tbig)  # Using numpy.random.randint! Generate Tbig
print('New index',indexstar)

# We now have a new index
# Draw from uhat using this index

ur = uhat[indexstar,:]
indexstar = np.reshape(indexstar, (1,-1)).T

print('Draw from uhat with index\n',np.hstack((indexstar,ur)))


