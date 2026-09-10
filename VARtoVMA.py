#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Illustrate VAR to VMA

"""

# Import some functions needed below
import numpy as np

# Estimated bi-variate VAR with 2 lags

A1 = np.array([[0.5, 0.1], [0.4, 0.5]])
A2 = np.array([[0, 0], [0.25, 0]])


# Using expressions in the lecture note

Phi0 = np.identity(2)
Phi1 = np.dot(Phi0,A1)
Phi2 = np.dot(Phi1,A1)+np.dot(Phi0,A2)
Phi3 = np.dot(Phi2,A1)+np.dot(Phi1,A2)

print('Phi0\n',np.reshape(Phi0, 4))
print('Phi1\n',np.reshape(Phi1, 4))
print('Phi2\n',np.reshape(Phi2, 4))
print('Phi3\n',np.reshape(Phi3, 4))







