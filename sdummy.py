#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:23:22 2023

@author: Work
"""

def sdummy(nobs,freq):
    """
    PURPOSE: creates a matrix of seasonal dummy variables
    ---------------------------------------------------
    USAGE:      y = sdummy(nobs,freq)
    where:   freq = 4 for quarterly, 12 for monthly
    ---------------------------------------------------
    RETURNS: 
         y = an (nobs x freq) matrix with 0's and 1's
          e.g.,   1 0 0 0  (for freq=4)
                  0 1 0 0
                  0 0 1 0
                  0 0 0 1
                  1 0 0 0 
    ---------------------------------------------------

    Coded by: Michael Bergman, 2023
    """
    import numpy as np
    seas = np.zeros([nobs,freq])

    for i in range(1, nobs, freq):
        seas[i-1:i+freq-1,0:freq] = np.identity(freq)

    return seas