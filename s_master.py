#masterfile
#revised by eske on sep 23


#packages
import math
import sys
import numpy as np
import pandas as pd
import matlab.engine

from numpy.linalg import *
from scipy.linalg import sqrtm
from scipy.stats import chi2, f

from tabulate import tabulate

from statsmodels.base import wrapper as wrap
from statsmodels.iolib.table import SimpleTable
from statsmodels.regression.linear_model import (
    GLS,
    OLS,
    WLS,
    burg,
    yule_walker,
)
from statsmodels.stats.diagnostic import (
    acorr_lm,
    het_arch,
)
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tools.decorators import cache_readonly, deprecated_alias
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.validation import array_like

import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.tsatools import (
    vech,
    duplication_matrix,
    unvec,
    vec,
)
from statsmodels.tsa.base.tsa_model import (
    TimeSeriesModel,
    TimeSeriesResultsWrapper,
)
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
    CausalityTestResults,
    NormalityTestResults,
    WhitenessTestResults,
)
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary

def lagmatrix(data, lags):
    T, K = data.shape
    lagged_data = np.zeros((T, K * lags))
    for lag in range(1, lags + 1):
        lagged_data[lag:, (lag - 1) * K:lag * K] = data[:T - lag, :]
    return lagged_data

def sdummy(nobs, freq):
    seas = np.zeros((nobs, freq))
    for i in range(nobs):
        season_index= i % freq
        seas[i, season_index] = 1
    return seas

def Jmatrix(K,p):
    """
    Compute J matrix to use when computing IRF's

    Input: K=number of variables; p=number of lags
    Michael Bergman 2023
    """
    if p == 1:
        Jmat = np.identity(K, dtype=int)
    else:   
        size = (K, (p-1)*K)
        Jmat = np.concatenate((np.identity(K, dtype=int),np.zeros(size, dtype=int)), axis=1)
    
    return Jmat

def vec(y):
  """
  Based on Lutz Kilian's Matlab code
  This function vectorizes an (a x b) matrix y.  The resulting ktor vecy
  has dimension (a*b x 1).
  Michael Bergman 2023
  """
  [row,column]=y.shape
  vecy=y.reshape(row*column,1);
  return vecy

def IRF(A,K,p,horizon,B0inv):
    """
    Process to invert VAR to VMA
    
    Note: A is the companion matrix
    K = number of variables
    p = number of lags
    horizon = horizon to compute MA representation
    B0inv = Identifying matrix

    Output: IRF's are organized such that the first K columns contain
    the effects of all shocks on the first variable, next K columns contain the
    effects of all shocks on the second variable and so on
    """    
    jmat = Jmatrix(K,p)
    irf = np.dot(np.dot(np.dot(jmat,np.linalg.matrix_power(A,0)),jmat.T),B0inv)
    irf = np.reshape(irf, [ 1, K*K  ])
    i=1
    while i <= horizon:
        help = np.dot(np.dot(np.dot(jmat,np.linalg.matrix_power(A,i)),jmat.T),B0inv)
        irf = np.concatenate((irf,np.reshape(help, [ 1, K*K  ])), axis=0)
        i=i+1
    
    return irf

def MArep(A,K,p,horizon):
    """
    Process to invert VAR to VMA
    
    Note: A is the companion matrix
    K = number of variables
    p = number of lags
    horizon = horizon to compute MA representation

    Output: MA representation organized in the same way as IRF
    """    
    jmat = Jmatrix(K,p)
    ma = np.dot(np.dot(jmat,np.linalg.matrix_power(A,0)),jmat.T)
    ma = np.reshape(ma, [ 1, K*K  ])
    i=1
    while i <= horizon:
        marep = np.dot(np.dot(jmat,np.linalg.matrix_power(A,i)),jmat.T)
        ma = np.concatenate((ma,np.reshape(marep, [ 1, K*K  ])), axis=0)
        i=i+1
    
    return ma

def FEVD(A,B0inv,K,p,h):
   """
   Structural forecast error variance Decomposition
   using B0inv for a K dimensional VAR and horizon h 
   Input:
   A: companion matrix
   B0inv: Identifying matrix
   K: number of variables
   p: number of lags
   h: horizon 

   Output: fevd organized as
   Each row is for horizon 1, 2, ..., h
   First K columns contain effects of shock 1,
   next K columns contain effects of shock 2,
   ....
   last K columns contain effect of shock K
   Michael Bergman December 2018
   """
   # First compute IRF's first horizon
   h = h-2
   jmat = Jmatrix(K,p)
   Theta = np.dot(np.dot(np.dot(jmat,np.identity(K*p, dtype=int)),jmat.T),B0inv) 
   # Need to take transpose such that rows=variables and cols=shocks   
   Theta = Theta.T
   # compute mse(1)
   Theta2 = np.multiply(Theta,Theta)
   Theta_sum = np.sum(Theta2, axis=0)
   VC = np.zeros([K,K])
   i = 0
   while i<=K-1:
     VC[i,:]=Theta2[i,:] / Theta_sum
     i = i+1

   VC = np.reshape(VC,[1, K*K] )
   # Then add the remaining horizons
   Thetah = Theta2
   j = 0
   while j<=h:
       j = j+1 
       Theta1 = np.dot(np.dot(   np.dot(jmat,np.linalg.matrix_power(A,j)) ,jmat.T),B0inv) 
       Theta1 = Theta1.T
       Theta2 = np.multiply(Theta1,Theta1)
       Thetah = np.array(Thetah)+np.array(Theta2)
       Thetah_sum = np.sum(Thetah, axis=0)
       VC1 = np.zeros([K,K])
       i = 0
       while i<=K-1:
           VC1[i,:]=Thetah[i,:] / Thetah_sum
           i = i+1

       VC1 = np.reshape(VC1,[1, K*K] )
       VC = np.concatenate((VC,np.reshape(VC1, [ 1, K*K  ])), axis=0)


   return VC

def hdecomp(A,mu,What,B0inv,K,p,indep):
    """
    # Input
    # A = Companion matrix A
    # mu = constant
    # What = structural shocks
    # B0inv = inv(B0)
    # K = # variables
    # p = # lags
    # indep = dependent variables excluding constant

    # Output
    # HDinit = initial conditions
    # HDconst = constant
    # HDshocks(obs,shock,variable) = (nobs x shock & variable, i.e.,
    # HDshock(:,1,1) contains the effect of shock 1 on variable 1;
    # HDshock(:,2,1) contains the effect of shock 2 on variable 1 and so on.

    # Written by Michael Bergman 2020 inspired by code written by Ambrogio Cesa Bianchi
    """
    nobs,KK = indep.shape 
    jmat = Jmatrix(K,p)
    init_big = np.zeros([p*K,nobs])
    init = np.zeros([K,nobs])
    init_big[:,0] = indep[0,0:p*K].T
    init[:,0] = np.dot(jmat,init_big[:,0])
    # =============================================================================
    i = 1
    while i<=nobs-1:
         init_big[:,i] = np.dot(A,init_big[:,i-1])
         init[:,i] = np.dot(jmat,init_big[:,i])
         i = i+1
     

    # Constant
    const_big = np.zeros([p*K,nobs])
    const = np.zeros([K,nobs]) 
    CC = np.zeros([p*K,1])

    if np.any(mu) == True:
        CC[0:K,:] = mu
        i = 1
        while i<=nobs-1:
           const_big[:,i] = np.array(CC).T + np.array(np.dot(A,const_big[:,i-1])).T
           const[:,i] = np.dot(jmat,const_big[:,i])
           i = i+1

        B_big = np.zeros([p*K,K])
        B_big[0:K,:] = B0inv
        shock_big = np.zeros((p*K,nobs,K))
        shock = np.zeros((K,nobs,K))
        j = 0
        while j<=K-1:
            What_big = np.zeros([K,nobs])
            What_big[j,1:nobs] = What[j,0:nobs-1]
            i = 0
            while i<=nobs-1:
                shock_big[:,i,j] = np.dot(B_big,What_big[:,i]).T + np.dot(A,shock_big[:,i-1,j]).T
                shock[:,i,j] = np.dot(jmat,shock_big[:,i,j])
                i = i + 1

            j = j + 1 
     
    HDendo = init + const + np.sum(shock.T, axis=0).T 
    nan1 = np.zeros([1,K])
    nan1[:,:] = np.nan
    HDendo[:,0] = nan1
    HDshock = np.zeros((nobs,K,K))
     
    i = 0 
    while i <= K-1:
         j = 0
         while j <= K-1:
            HDshock[0:nobs,j,i] = shock[i,0:nobs,j].T
            j = j + 1 
     
         i = i + 1 
     

    HDinit = init.T                       
    HDconst = const.T
    HDendo = HDendo.T 

    
    
    return HDinit, HDconst, HDshock, HDendo

def owndiag(resid,K,ddof,bglag,archlag):

    """
    This function produces and prints the followingn univariate diagnostic tests.
    Breusch-Godfrey test for autocorrelation
    Engel ARCH test
    Jarque-Behra normality test

    Input: 
    resid is a nobs x K array of residuals
    K is the number of variables (equations) in the VAR model
    ddof is the degrees of freedom (number of parameters of auxiliary regression)
    bglag is the number of lags used when computing the Breusch-Godfrey test
    archlag is the number of lags used when computing the ARCH test    
    """
    
    diagres = np.zeros([1,5])
    for j in range(0, K):
        res = resid[:,j]
        (lm,lmpval,fval,fpval) = acorr_lm(res,nlags=bglag,store=False,period=None,ddof=ddof)
        data = [[ j+1 , np.round(lm, 3), np.round(lmpval, 3), np.round(fval, 3), np.round(fpval, 3)]]
        diagres = np.concatenate((diagres,data), axis=0)

    diagres = diagres[1:K+1,:]
    print('Univariate residual analysis')
    print('============================')
    print('')
    print('Breusch-Godfrey test for autocorrelation with',bglag,'lags')
    print('========================================================')
    print (tabulate(diagres, headers=["Equation", "Chi^2 stat", "p-value", "F stat", "p-value"], numalign="right"))
    print('========================================================')
    print('')

    diagres = np.zeros([1,5])
    for j in range(0, K):
        (lm,lmpval,fval,fpval) = het_arch(resid[:,j], nlags=archlag, store=False, ddof=ddof)
        #diagres = [[JB, JBpv]]
        data = [[ j+1 , np.round(lm, 3), np.round(lmpval, 3), np.round(fval, 3), np.round(fpval, 3)]]
        diagres = np.concatenate((diagres,data), axis=0)

    diagres = diagres[1:K+1,:]
    print('Engle ARCH LM test with',archlag,'lags')
    print('========================================================')
    print (tabulate(diagres, headers=["Equation", "Chi^2 stat", "p-value", "F stat", "p-value"], numalign="right"))
    print('========================================================')
    print('')


    diagres = np.zeros([1,3])
    for j in range(0, K):
        (JB,JBpv,skew,kurtosis) = jarque_bera(resid[:,j], axis=0)
        data = [[ j+1 , np.round(JB, 3), np.round(JBpv, 3)]]
        diagres = np.concatenate((diagres,data), axis=0)

    diagres = diagres[1:K+1,:]
    print('Jarque-Bera test for non-normality')
    print('==================================')
    print (tabulate(diagres, headers=["Equation", "Test stat", "p-value"], numalign="right"))
    print('==================================')
    print('')


    return

def trimr(x, front, end):

    if end > 0:

        return x[front:-end]

    else:

        return x[front:]

def JohansenTrace(data,p,exog,model):
    """
    Function starts here
    Input: x is a T x K data array
    p = # lags in underlying VAR
    exog = exogenous variables T x n data array
    Model: Here we use the same as in Matlab
    H2  (no deterministic terms):                      model = 1
    H1* (constant outside the cointegration relation): model = 2
    H1  (Constant within the cointegration relation):  model = 3
    H*  (constant and linear trend in cointegration relation, no quadratic trend):  model = 4
    H:  (constant and linear trend in cointegration relation, quadratic trend) model = 5
     
    or using Python convention
    
    H2:  deterministic="n"
    H1*: deterministic="co"
    H1:  deterministic="ci"
    H*:  deterministic="cili"
    H:   deterministic="colo"
    
    Output:
    [trace statistics, Critical value 5%, Critical value 10%, p-value, eigenvalue,
     beta, alpha, c0, c1, d0, d1]
    11 arrays in total
    """
    print("Using the Johansen Trace with changes by Jens")

    # First we need to load critical values
    JCV = np.load('JCV.npy')
    PSSCV = np.load('PSSCV.npy')    
    x = pd.DataFrame(data).to_numpy()
     
    [T, K] = x.shape
    
    # Setting up first difference of x
     
    dx = x - lagmat(x, [1], 'col')
    dx = dx[1:len(dx)]
    Z0t = dx[p-1:len(dx)]
     
    lagsdx = np.arange(1,p)
    lagsdx = lagsdx.tolist()
     
    if p<=1:
         dxlags = []
         Z1t = []
         if model==3:
             Z1t = np.ones([len(Z0t),1])
         elif model==4:
             Z1t = np.ones([len(Z0t),1])
         elif model==5:
             Z1t = np.concatenate((np.ones([len(Z0t),1]),np.array([np.arange(1, len(Z0t)+1, 1)]).T), axis=1)
             
    elif p>1:
         dxlags = lagmat(dx,lagsdx, 'col')
         dxlags = dxlags[p-1:len(x)]
         Z1t = dxlags
         [nrows, ncols] = dxlags.shape
         # Now we need to add deterministic components depending on model
         if model==3:
               Z1t = np.concatenate( (dxlags,np.ones([nrows,1])), axis=1)
         elif model==4:
               Z1t = np.concatenate( (dxlags,np.ones([nrows,1])), axis=1)
         elif model==5:
               Z1t = np.concatenate((dxlags,np.concatenate((np.ones([nrows,1]),np.array([np.arange(1, nrows+1, 1)]).T), axis=1)  ), axis=1)
        
       
    # Add exogenous variables
    if exog is not None and not np.isscalar(exog):
         if p>1:
            Z1t = np.concatenate([Z1t, exog[p:len(exog)]], axis=1)
         elif p<=1:
             Z1t = exog[p:len(exog)]

     
    # Setting up lagged level
    Zkt = lagmat(x, [1], 'col')
    Zkt = Zkt[p:len(Zkt)]
                          
     
     # Add deterministic components to lagged level depending on model
    if model==2:
        Zkt = np.concatenate((Zkt,np.ones([len(Zkt),1])), axis=1)
    elif model==4:
        Zkt = np.concatenate((Zkt,np.array([np.arange(1, len(Zkt)+1, 1)]).T), axis=1)
    elif model==5:
        Zkt = Zkt
     
    # Ready to run the regressions and to compute the residuals
    # This is done in two steps using OLS
     
    if p>1:
         Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Z0t))
         R0t = Z0t-np.dot(Z1t,Beta)
         Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Zkt))
         R1t = Zkt-np.dot(Z1t,Beta)
    elif p<=1:
         R0t = Z0t
         R1t = Zkt
         if model>2:
             Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Z0t))
             R0t = Z0t-np.dot(Z1t,Beta)
             Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Zkt))
             R1t = Zkt-np.dot(Z1t,Beta)
             
         
    # Compute sum of squares
    S01 = np.dot(R0t.T,R1t)/len(Zkt)
    S10 = S01.T
    S00 = np.dot(R0t.T,R0t)/len(Zkt)
    S00I = np.linalg.inv(S00)
    S11 = np.dot(R1t.T,R1t)/len(Zkt)
    S11I = np.linalg.inv(S11)
    G = np.linalg.inv(sqrtm(S11))
    A = np.dot(np.dot(np.dot(np.dot(G,S10),S00I),S01),G.T)  
    
    # Compute eigenvalues and eigenvectors

    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # ordering eigenvalues and eigenvectors
    index = np.argsort(eigenvalues)
    index = np.flipud(index)
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:,index]

    # Compute cointegration vector beta and adjustment coefficients alpha
    beta = np.dot(G,eigenvectors[:,0:K-1])
    alpha = np.dot(np.dot(S01,beta),np.linalg.inv(np.dot(np.dot(beta.T,S11),beta)))

    if model==1:
        Pi = np.dot(alpha,beta.T)
        c0 = []
        c1 = []
        d0 = []
        d1 = []
        c = []
        d = []
    elif model==2:
        c0 = beta[K,:]
        c1 = []
        d0 = []
        d1 = []
        d = []
        beta = beta[0:K,:]
        Pi = np.dot(alpha,beta.T)
    if model==3:
        W = Z0t - np.dot(np.dot(Zkt,beta),alpha.T)
        P,h1,h2,h3 = np.linalg.lstsq(Z1t, W, rcond=None)
        P = P.T
        c = P[:,len(P.T)-1]
        c0,h1,h2,h3 = np.linalg.lstsq(alpha, c, rcond=None)
        c1 = c - np.dot(alpha,c0)
        d0 = []
        d1 = []
        Pi = np.dot(alpha,beta.T)
    elif model==4:
        d0 = beta[K,:].T
        W = Z0t - np.dot(np.dot(Zkt,beta),alpha.T)
        P,h1,h2,h3 = np.linalg.lstsq(Z1t, W, rcond=None)
        P = P.T
        c = P[:,len(P.T)-1]
        c0,h1,h2,h3 = np.linalg.lstsq(alpha, c, rcond=None)
        c1 = c - np.dot(alpha,c0)
        d1 = []
        beta = beta[0:K,:]
        Pi = np.dot(alpha,beta.T)
    elif model==5:
        W = Z0t - np.dot(np.dot(Zkt,beta),alpha.T)
        P,h1,h2,h3 = np.linalg.lstsq(Z1t, W, rcond=None)
        P = P.T
        c = P[:,len(P.T)-2]
        d = P[:,len(P.T)-1]
        c0,h1,h2,h3 = np.linalg.lstsq(alpha, c, rcond=None)
        c1 = c - np.dot(alpha,c0)
        d0,h1,h2,h3 = np.linalg.lstsq(alpha, d, rcond=None)
        d1 = d - np.dot(alpha,d0)
        beta = beta[0:K,:]
        Pi = np.dot(alpha,beta.T)

    
    # Formulate eigenvalue problem
    
    l = np.linalg.eigvals(A).T
    l = np.sort(l)[::-1]
    l = l[0:K]
     
    # Compute Trace test lr1
    
    lr1 = np.zeros([len(l),1])
    iota = np.ones(len(l))
    
    for i in range(0, len(l)):
          tmp = trimr(np.log(iota - l), i , 0)
          lr1[i] = -len(Zkt) * np.sum(tmp, 0)     
     
     
    # Now we need to add critical values
     
    testStat = np.flip(lr1)
     
    cval5 = np.flip(JCV[0:K,10,model-1,0])
    cval10 = np.flip(JCV[0:K,20,model-1,0])
     
    # Compute p-value using linear interpolation
     
    # Define significance levels
    
    siglevels = pd.read_excel("SigLevels.xlsx")
    siglevels = siglevels[['siglevels']]
    siglevels = pd.DataFrame.to_numpy(siglevels).T
     
    # Then extract relevant critical values
     
    CVTable = JCV[:,:,model-1,0]
    xp = np.flip(CVTable[0:K,:])
   
    # Finally compute p-values using linear interpolation
    pval = np.zeros([K,1])
     
    for j in range(0, K):
         if lr1[j,0] >= xp[j,len(xp.T)-1]:
             pval[j,0] = siglevels[0,0]
         elif lr1[j,0] <= xp[j,0]:
             pval[j,0] = siglevels[0,len(siglevels)]
         else:          
             idx = bisection(xp[j,:],lr1[j,0])
             x1 = xp[j,idx]
             x2 = xp[j,idx+1]
             y1 = siglevels[0,idx]
             y2 = siglevels[0,idx+1]
             pval[j,0] = 1-(y1 + (lr1[j,0]-x1)*(y2-y1)/(x2-x1))
     
    # Print table with trace test results
     
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
     
    print('\n***************************')
    print('Johansen Cointegration test')
    print('Sample size:',len(Z1t))
    print('Lags in VAR:',p)
    print('Lags in Vec:',p-1)
    print('Statistic:','Trace')
    if model==1:
         print('Model: H2 ("n") [no deterministic terms]')
    elif model==2:
         print('Model: H1* ("co") [constant in coint vec, no linear trend in levels]')
    elif model==3:
         print('Model: H1 ("ci") [constant in coint vec and linear trend in levels]')
    elif model==4:
         print('Model: H* ("cili") [constant and linear trend in coint vec, linear trend in levels]')
    elif model==5:
         print('Model: H ("colo") [constant and linear trend in coint vec, quadratic trend in levels]')
     
    if np.isscalar(exog)==False:
         print('Model includes exogenous I(0) variables')
         print('Note: Critical values may not be correct')
     
    diagres = np.zeros([1,6])
    # =============================================================================
    for j in range(0, K):
              data = [[ j , np.round(float(lr1[j]), 3), np.round(cval5[j], 3), np.round(cval10[j], 4), np.round(pval[j,0], 4), np.round(l[j], 4)]]
              #data = np.ones([1,5])
              diagres = np.concatenate((diagres,data), axis=0)
     
    diagres = diagres[1:K+1,:]
    print('========================================================')
    print (tabulate(diagres, headers=["r", "stat", "cVal5%", "cVal10%", "p-value", "EigVal"], numalign="right"))
    print('========================================================')
    print('')
    
    # Then we print ML estimates of betam alpha and the deterministic terms
    print('\n***************************')
    print('ML estimates of beta.T, the cointegration vector')
    print(tabulate(beta.T, numalign="right"))
    print('ML estimates of alpha, the adjustment coefficients')
    print(tabulate(alpha, numalign="right"))
    print('ML estimates of Pi = alpha x beta.T')
    print(tabulate(Pi, numalign="right"))
    print('\n***************************')
    print('Deterministic terms')
    print('ML estimates of c0\n',c0)
    print('ML estimates of c1\n',c1)
    print('ML estimates of d0\n',d0)
    print('ML estimates of d1\n',d1)
    
    
    
    return lr1, cval5, cval10, pval, l, beta, alpha, c0, c1, d0, d1

def FEVD_irf(irf,K,h):
   """
   Structural Forecast Error Variance Decomposition
   computed using Impulse Responses for dimension K and horizon h 
   Input:
   irf: Impulse Response Function
   K: number of variables
   h: horizon 

   Output: fevd organized as
   Each row is for horizon 1, 2, ..., h
   First K columns contain effects of shock 1,
   next K columns contain effects of shock 2,
   ....
   last K columns contain effect of shock K
   Michael Bergman December 2018
   """
   Theta = np.reshape(irf[0,:], [K,K])
   Theta = Theta.T
   # compute mse(1)
   Theta2 = np.multiply(Theta,Theta)
   Theta_sum = np.sum(Theta2, axis=0)
   VC = np.zeros([K,K])
   i = 0
   while i<=K-1:
       VC[i,:]=Theta2[i,:] / Theta_sum
       i = i+1

   VC = np.reshape(VC,[1, K*K] )
   VChelp = np.zeros([K,K])
   # Then add the remaining horizons
   Thetah = Theta2
   j = 0
   while j<=h-1:
         j = j+1 
         Theta1 = np.reshape(irf[j,:], [K,K]) 
         Theta1 = Theta1.T
         Theta2 = np.multiply(Theta1,Theta1)
         Thetah = np.array(Thetah)+np.array(Theta2)
         Thetah_sum = np.sum(Thetah, axis=0)
         i = 0
         while i<=K-1:
             VChelp[i,:]=Thetah[i,:] / Thetah_sum
             i = i+1

         VC1 = np.reshape(VChelp,[1, K*K] )
         VC = np.concatenate((VC,np.reshape(VC1, [ 1, K*K  ])), axis=0)


   return VC

def vectovar(Gamma,Pi):
    # This function converts VEC estimates into companion matrix for VAR in
    # levels.
    #
    # Michael Bergman
    # Checked October 2023
    #
    # Input: Gamma = KxKp coefficient matrix
    #        Pi = alpha*beta' a KxK matrix
    # Output: KpxKp Companion matrix B
    
    if len(Gamma.T) == 0:
        (K,K) = Pi.shape
        B = np.identity(K) + Pi
    else:
        (K,Kp)=Gamma.shape
        p = Kp/K
        B = np.identity(K) + Pi + Gamma[:,0:K]
        Gamma = np.hstack([Gamma,np.zeros([K,K],dtype=np.float64)])
        i=0
        j=K
        while j<=Kp:
           B = np.hstack([B,(Gamma[:,i+K:j+K]-Gamma[:,i:j])])
           i = i+K
           j=j+K
        
        B = np.vstack([B,np.hstack([np.identity(K*int(p)),np.zeros([K*int(p),K])])])
    
      
    

    return B

def LSKnownBeta(y,p,beta,model):
    """
    % y = Data in levels must be an array T x K
    % p = #lags in underlying VAR
    % beta = cointegration vector, i.e., beta' but remember that it must include
             any deterministic components
    %
    % Note: function requires specification of the deterministic components. We use
    % the Matlab conventions to define models.
    %
    % model = 1 corresponds to model H2 ("n" in Python)
    % model = 2 corresponds to model H1* ("co" in Python)
    % model = 3 corresponds to model H1 ("ci" in Python)
    % model = 4 corresponds to model H* ("cili" in Python)
    % model = 5 corresponds to model H ("colo" in Python)
    %
    %
    % Output:
    % Beta: K x (p-1) parameters associated with first differences;
    %       K x r error correction terms
    %       K x 1 constant terms (if included in the model)
    %       K x 1 linear trend terms and (if included in the model)
    % 
    % Betavec: vectorized Beta (vector)      
    % SEBeta: Standard errors associated with each parameter (vector)
    % tratioBeta: t-ratios for all parameters (vector)
    % res: Residuals from VEC estimates
    % so: variance-covariance matrix
    % so_ml: ML estimate of variance-covariance matrix
    % 
    % Note: The cointegration vector beta used as input must include constant and/or trend
    %
    % Michael Bergman
    % 
    % Revised September 2024
    %
    % Verified using Stata
    """
    y = pd.DataFrame(y).to_numpy()
    
    [T, K]=y.shape
    
    dy = y[1:len(y)]-y[0:len(y)-1]
    dep = dy[p-1:len(dy),0:K]
    
    if p>1:
        nlags = np.arange(0,p-1)
        indep = lagmat(dy, lags=nlags)
        indep = indep[p-2:len(indep)-1]
        
        if model==1:
            cointvec = np.dot(y[p-1:len(y)-1,:],beta.T)
            indep = np.concatenate((indep,cointvec), axis=1)
        elif model==2:
            cointvec = np.dot( np.concatenate((y[p-1:len(y)-1,:],np.ones((len(indep),1))), axis=1),beta.T)
            indep = np.concatenate((indep,cointvec), axis=1)
        elif model==3:
            cointvec = np.dot( np.concatenate((y[p-1:len(y)-1,:],np.ones((len(indep),1))), axis=1),beta.T)
            indep =  np.concatenate((np.concatenate((indep,cointvec), axis=1),np.ones((len(indep),1))), axis=1)
        elif model==4:
            help1 = np.concatenate((np.concatenate( (y[p-1:len(y)-1,:],np.ones((len(indep),1)) ), axis=1),np.arange(1,len(indep)+1).reshape(len(indep),1)), axis=1)
            cointvec = np.dot(help1,beta.T)
            indep =  np.concatenate((np.concatenate((indep,cointvec), axis=1),np.ones((len(indep),1))), axis=1)    
        elif model==5:
            help1 = np.concatenate((np.concatenate( (y[p-1:len(y)-1,:],np.ones((len(indep),1))), axis=1),np.arange(1,len(indep)+1).reshape(len(indep),1)), axis=1)
            cointvec = np.dot(help1,beta.T)
            indep = np.concatenate((np.concatenate((np.concatenate( (indep,cointvec), axis=1 ),np.ones((len(indep),1))), axis=1),np.arange(1,len(indep)+1).reshape(len(indep),1)), axis=1)
    elif p==1:
        if model==1:
            cointvec = np.dot(y[p-1:len(y)-1,:],beta.T)
            indep = cointvec
        elif model==2:
            cointvec = np.dot( np.concatenate((y[p-1:len(y)-1,:],np.ones((len(dep),1))), axis=1),beta.T)
            indep = cointvec
        elif model==3:
            cointvec = np.dot( np.concatenate((y[p-1:len(y)-1,:],np.ones((len(dep),1))), axis=1),beta.T)
            indep =  np.concatenate( (cointvec,np.ones((len(dep),1))), axis=1)    
        elif model==4:
            help1 = np.concatenate((np.concatenate( (y[p-1:len(y)-1,:],np.ones((len(dep),1)) ), axis=1),np.arange(1,len(dep)+1).reshape(len(dep),1)), axis=1)
            cointvec = np.dot(help1,beta.T)
            indep =  np.concatenate((cointvec,np.ones((len(dep),1))), axis=1)    
        elif model==5:
            help1 = np.concatenate((np.concatenate( (y[p-1:len(y)-1,:],np.ones((len(dep),1)) ), axis=1),np.arange(1,len(dep)+1).reshape(len(dep),1)), axis=1)
            cointvec = np.dot(help1,beta.T)
            help1 = np.concatenate((np.ones((len(dep),1)),np.arange(1,len(dep)+1).reshape(len(dep),1)), axis=1)
            indep =  np.concatenate((cointvec,help1), axis=1)    

    
    # =============================================================================
    [T2,Kp2]=indep.shape
    Beta = np.dot(np.linalg.inv(np.dot(indep.T,indep)),np.dot(indep.T,dep))
    res = dep-np.dot(indep,Beta)
    so = np.dot(res.T,res)/(T-Kp2)
    so_ml = np.dot(res.T,res)/(T-K*p-1)
    # Compute t-ratios. Remember to sort Beta to match order of the diagonal of SEBeta
    SEBeta = np.sqrt(np.diag(np.kron(so,np.linalg.inv(np.dot(indep.T,indep))))).reshape(Kp2*K,1)
    Betavec = np.reshape(np.ravel(Beta, order='F'), [1,Kp2*K]).T
    tratioBeta = np.divide( Betavec, SEBeta)

    return Beta, Betavec, SEBeta, tratioBeta, res, so, so_ml

def jcitestexog(data,p,exog,model):
    """
    Function starts here
    Input: x is a T x K data array
    p = # lags in VEC model
    exog = Exogenous variables T x M data array
    Model: Here we use the same as in Matlab
    H2  (no deterministic terms):                      model = 1
    H1* (constant outside the cointegration relation): model = 2
    H1  (Constant within the cointegration relation):  model = 3
    H*  (constant and linear trend in cointegration relation, no quadratic trend):  model = 4
    H:  (constant and linear trend in cointegration relation, quadratic trend) model = 5
    
    or using Python convention
   
    H2:  deterministic="n"
    H1*: deterministic="co"
    H1:  deterministic="ci"
    H*:  deterministic="cili"
    H:   deterministic="colo"
   
    Output:
    [trace statistics, Critical value 5%, Critical value 10%, p-value, eigenvalue,
     beta, alpha, c0, c1, d0, d1]
    11 arrays in total
    """
    import pandas
    from tabulate import tabulate
    from scipy.linalg import sqrtm
   
    # Number of lags in underlying VAR
    p = p+1
   
    # First we need to load critical values
    JCV = np.load('JCV.npy')
    PSSCV = np.load('PSSCV.npy')   
    x = pandas.DataFrame(data).to_numpy()
    
    [T, K] = x.shape
   
    # Setting up first difference of x
    
    dx = x - lagmat(x, [1], 'col')
    dx = dx[1:len(dx)]
    Z0t = dx[p-1:len(dx)]
    
    lagsdx = np.arange(1,p)
    lagsdx = lagsdx.tolist()
    
    if p<=1:
         dxlags = []
         Z1t = []
         if model==3:
             Z1t = np.ones([len(Z0t),1])
         elif model==4:
             Z1t = np.ones([len(Z0t),1])
         elif model==5:
             Z1t = np.concatenate((np.ones([len(Z0t),1]),np.array([np.arange(1, len(Z0t)+1, 1)]).T), axis=1)
        
         
    elif p>1:
         dxlags = lagmat(dx,lagsdx, 'col')
         dxlags = dxlags[p-1:len(x)]
         Z1t = dxlags
         [nrows, ncols] = dxlags.shape
         # Now we need to add deterministic components depending on model
         if model==3:
               Z1t = np.concatenate( (dxlags,np.ones([nrows,1])), axis=1)
         elif model==4:
               Z1t = np.concatenate( (dxlags,np.ones([nrows,1])), axis=1)
         elif model==5:
               Z1t = np.concatenate((dxlags,np.concatenate((np.ones([nrows,1]),np.array([np.arange(1, nrows+1, 1)]).T), axis=1)  ), axis=1)
       
     
    # Add exogenous variables
    
    if np.isscalar(exog)==False:
         if p>1:
            Z1t = np.hstack((Z1t,exog[p:len(exog),:]))
         elif p<=1:
             Z1t = exog[p:len(exog),:]
 
       
    # Setting up lagged level
    Zkt = lagmat(x, [1], 'col')
    Zkt = Zkt[p:len(Zkt)]
                         
     
     # Add deterministic components to lagged level depending on model
    if model==2:
        Zkt = np.concatenate((Zkt,np.ones([len(Zkt),1])), axis=1)
    elif model==4:
        Zkt = np.concatenate((Zkt,np.array([np.arange(1, len(Zkt)+1, 1)]).T), axis=1)
    elif model==5:
        Zkt = Zkt
    
    # Ready to run the regressions and to compute the residuals
    # This is done in two steps using OLS
    
    if p>1:
         Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Z0t))
         R0t = Z0t-np.dot(Z1t,Beta)
         Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Zkt))
         R1t = Zkt-np.dot(Z1t,Beta)
    elif p<=1:
         R0t = Z0t
         R1t = Zkt
         if model>2:
             Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Z0t))
             R0t = Z0t-np.dot(Z1t,Beta)
             Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Zkt))
             R1t = Zkt-np.dot(Z1t,Beta)
            
         
    # Compute sum of squares
    S01 = np.dot(R0t.T,R1t)/len(Zkt)
    S10 = S01.T
    S00 = np.dot(R0t.T,R0t)/len(Zkt)
    S00I = np.linalg.inv(S00)
    S11 = np.dot(R1t.T,R1t)/len(Zkt)
    S11I = np.linalg.inv(S11)
    G = np.linalg.inv(sqrtm(S11))
    A = np.dot(np.dot(np.dot(np.dot(G,S10),S00I),S01),G.T) 
    
    # Compute eigenvalues and eigenvectors
 
    eigenvalues, eigenvectors = np.linalg.eig(A)
   
    # ordering eigenvalues and eigenvectors
    index = np.argsort(eigenvalues)
    index = np.flipud(index)
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:,index]
 
    # Compute cointegration vector beta and adjustment coefficients alpha
    beta = np.dot(G,eigenvectors[:,0:K-1])
    alpha = np.dot(np.dot(S01,beta),np.linalg.inv(np.dot(np.dot(beta.T,S11),beta)))
 
    if model==1:
        Pi = np.dot(alpha,beta.T)
        c0 = np.empty(1, dtype=object)
        c1 = np.empty(1, dtype=object)
        d0 = np.empty(1, dtype=object)
        d1 = np.empty(1, dtype=object)
        c = np.empty(1, dtype=object)
        d = np.empty(1, dtype=object)
    elif model==2:
        c0 = beta[K,:]
        c1 = np.empty(1, dtype=object)
        d0 = np.empty(1, dtype=object)
        d1 = np.empty(1, dtype=object)
        d = np.empty(1, dtype=object)
        beta = beta[0:K,:]
        Pi = np.dot(alpha,beta.T)
    if model==3:
        W = Z0t - np.dot(np.dot(Zkt,beta),alpha.T)
        P,h1,h2,h3 = np.linalg.lstsq(Z1t, W, rcond=None)
        P = P.T
        c = P[:,len(P.T)-1]
        c0,h1,h2,h3 = np.linalg.lstsq(alpha, c, rcond=None)
        c1 = c - np.dot(alpha,c0)
        d0 = np.empty(1, dtype=object)
        d1 = np.empty(1, dtype=object)
        Pi = np.dot(alpha,beta.T)
    elif model==4:
        d0 = beta[K,:].T
        W = Z0t - np.dot(np.dot(Zkt,beta),alpha.T)
        P,h1,h2,h3 = np.linalg.lstsq(Z1t, W, rcond=None)
        P = P.T
        c = P[:,len(P.T)-1]
        c0,h1,h2,h3 = np.linalg.lstsq(alpha, c, rcond=None)
        c1 = c - np.dot(alpha,c0)
        d1 = np.empty(1, dtype=object)
        beta = beta[0:K,:]
        Pi = np.dot(alpha,beta.T)
    elif model==5:
        W = Z0t - np.dot(np.dot(Zkt,beta),alpha.T)
        P,h1,h2,h3 = np.linalg.lstsq(Z1t, W, rcond=None)
        P = P.T
        c = P[:,len(P.T)-2]
        d = P[:,len(P.T)-1]
        c0,h1,h2,h3 = np.linalg.lstsq(alpha, c, rcond=None)
        c1 = c - np.dot(alpha,c0)
        d0,h1,h2,h3 = np.linalg.lstsq(alpha, d, rcond=None)
        d1 = d - np.dot(alpha,d0)
        beta = beta[0:K,:]
        Pi = np.dot(alpha,beta.T)
 
   
    # Formulate eigenvalue problem
   
    l = np.linalg.eigvals(A).T
    l = np.sort(l)[::-1]
    l = l[0:K]
    
    # Compute Trace test lr1
   
    lr1 = np.zeros([len(l),1])
    iota = np.ones(len(l))
   
    for i in range(0, len(l)):
          tmp = trimr(np.log(iota - l), i , 0)
          lr1[i] = -len(Zkt) * np.sum(tmp, 0)    
     
     
    # Now we need to add critical values
    
    testStat = np.flip(lr1)
    
    cval5 = np.flip(JCV[0:K,10,model-1,0])
    cval10 = np.flip(JCV[0:K,20,model-1,0])
    
    # Compute p-value using linear interpolation
    
    # Define significance levels
   
    siglevels = pandas.read_excel("siglevels.xlsx")
    siglevels = siglevels[['siglevels']]
    siglevels = pandas.DataFrame.to_numpy(siglevels).T
    
    # Then extract relevant critical values
    
    CVTable = JCV[:,:,model-1,0]
    xp = np.flip(CVTable[0:K,:])
  
    # Finally compute p-values using linear interpolation
    pval = np.zeros([K,1])
    
    for j in range(0, K):
         if lr1[j,0] >= xp[j,len(xp.T)-1]:
             pval[j,0] = siglevels[0,0]
         elif lr1[j,0] <= xp[j,0]:
             pval[j,0] = siglevels[0,len(siglevels)]
         else:         
             idx = bisection(xp[j,:],lr1[j,0])
             x1 = xp[j,idx]
             x2 = xp[j,idx+1]
             y1 = siglevels[0,idx]
             y2 = siglevels[0,idx+1]
             pval[j,0] = 1-(y1 + (lr1[j,0]-x1)*(y2-y1)/(x2-x1))
    
    # Print table with trace test results
    
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    
    print('\n***************************')
    print('Johansen Cointegration test')
    print('Sample size:',len(Z1t))
    print('Lags in VAR:',p)
    print('Lags in Vec:',p-1)
    print('Statistic:','Trace')
    if model==1:
         print('Model: H2 ("n") [no deterministic terms]')
    elif model==2:
         print('Model: H1* ("co") [constant in coint vec, no linear trend in levels]')
    elif model==3:
         print('Model: H1 ("ci") [constant in coint vec and linear trend in levels]')
    elif model==4:
         print('Model: H* ("cili") [constant and linear trend in coint vec, linear trend in levels]')
    elif model==5:
         print('Model: H ("colo") [constant and linear trend in coint vec, quadratic trend in levels]')
   
     
    diagres = np.zeros([1,6])
    # =============================================================================
    for j in range(0, K):
              data = [[ j , np.round(float(lr1[j]), 3), np.round(cval5[j], 3), np.round(cval10[j], 4), np.round(pval[j,0], 4), np.round(l[j], 4)]]
              #data = np.ones([1,5])
              diagres = np.concatenate((diagres,data), axis=0)
   
        
    if np.isscalar(exog)==False:
         print('Model includes exogenous I(0) variables')
         print('Note: Critical values may not be correct')
  
    
   
    diagres = diagres[1:K+1,:]
    print('========================================================')
    print (tabulate(diagres, headers=["r", "stat", "cVal5%", "cVal10%", "p-value", "EigVal"], numalign="right"))
    print('========================================================')
    print('')
   
    # Then we print ML estimates of betam alpha and the deterministic terms
    print('\n***************************')
    print('ML estimates of beta.T, the cointegration vector')
    print(tabulate(beta.T, numalign="right"))
    print('ML estimates of alpha, the adjustment coefficients')
    print(tabulate(alpha, numalign="right"))
    print('ML estimates of Pi = alpha x beta.T')
    print(tabulate(Pi, numalign="right"))
    print('\n***************************')
    print('Deterministic terms')
    print('ML estimates of c0\n',c0)
    print('ML estimates of c1\n',c1)
    print('ML estimates of d0\n',d0)
    print('ML estimates of d1\n',d1)
   
    
    
    return lr1, cval5, cval10, pval, l, beta, alpha, c0, c1, d0, d1

def lagmat(A: np.array, lags: list, orient: str = 'col') -> np.array:
    """Create array with time-lagged copies of the features/variables

    Args:
        A (np.ndarray): Dataset. One column for each features/variables,
            and one row for each example/observation at a certain time step.
        lags (ist, tuple): Definition what time lags the copies of A should
            have.
        orient (str, Default: 'col'): Information if time series in A are in
            column-oriented or row-oriented

    Assumptions:
        - It's a time-homogenous time series (that's why there is
            no time index)
        - Features/Variables are ordered from the oldest example/observation
            (1st row) to the latest example (last row)
        - Any Missing Value treatment have been done previously.
        
    Copyright 2021 Ulf Hamster         
        
    """
    # detect negative lags
    if min(lags) < 0:
        raise Exception((
            "Negative lags are not allowed. Only provided integers "
            "greater equal 0 as list/tuple elements"))
    # None result for missing lags
    if len(lags) == 0:
        return None
    # enforce floating subtype
    if not np.issubdtype(A.dtype, np.floating):
        A = np.array(A, np.float32)

    if orient in ('row', 'rows'):
        # row-oriented time series
        if len(A.shape) == 1:
            A = A.reshape(1, -1)
        A = np.array(A, order='C')
        return lagmat_rows(A, lags)

    elif orient in ('col', 'cols', 'columns'):
        # column-oriented time series
        if len(A.shape) == 1:
            A = A.reshape(-1, 1)
        A = np.array(A, order='F')
        return lagmat_cols(A, lags)
    else:
        return None

def lagmat_rows(A: np.array, lags: list):
    # number of colums and lags
    n_rows, n_cols = A.shape
    n_lags = len(lags)
    # allocate memory
    B = np.empty(shape=(n_rows * n_lags, n_cols), order='C', dtype=A.dtype)
    B[:] = np.nan
    # Copy lagged columns of A into B
    for i, l in enumerate(lags):
        # target rows of B
        j = i * n_rows
        k = j + n_rows  # (i+1) * n_rows
        # number cols of A
        nc = n_cols - l
        # Copy
        B[j:k, l:] = A[:, :nc]
    return B

def lagmat_cols(A: np.array, lags: list):
    # number of colums and lags
    n_rows, n_cols = A.shape
    n_lags = len(lags)
    # allocate memory
    B = np.empty(shape=(n_rows, n_cols * n_lags), order='F', dtype=A.dtype)
    B[:] = np.nan
    # Copy lagged columns of A into B
    for i, l in enumerate(lags):
        # target columns of B
        j = i * n_cols
        k = j + n_cols  # (i+1) * ncols
        # number rows of A
        nl = n_rows - l
        # Copy
        B[l:, j:k] = A[:nl, :]
    return B

def bisection(array,value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0# Initialize lower
    ju = n-1# and upper limits.
    while (ju-jl > 1):# If we are not yet done,
        jm=(ju+jl) >> 1# compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl=jm# and replace either the lower limit
        else:
            ju=jm# or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]):# and top
        return n-1
    else:
        return jl
    
def johcontest(Y,r,test,cons,model,lags,alpha):
    # Now run jcontest! Note that I have added nargout=5 as input to function.
    # This ensures that result now contains all output that we normally obtain
    # from Matlab.
    # Result will be a tuple containing [h,pValue,stat,cValue,mles] where
    # mles is a structure
    # Start MATLAB engine
    # Convert Python types to MATLAB types

    eng = matlab.engine.start_matlab()
    
    if test[0] == 'acon':
        constrtype = '\u03B1'
    elif test[0] == 'bcon':
        constrtype = '\u03B2'
    elif test[0] == 'bvec':
        constrtype = '\u03B2'
    elif test[0] == 'avec':
        constrtype = '\u03B1'
    
    data = matlab.double(Y.tolist())
     
    
    
    result = eng.jcontest(
        data, 
        r, 
        test, 
        cons, 
        'Model', 
        model, 
        'Lags', 
        lags, 
        'Alpha', 
        alpha,
        nargout=5
    )
    
    # We now have all the results from jcontest in the tuple result
    
    # First we extract the LR test statistic and the p-value directly from
    # result
    
    LRtest = result[2]
    pval = result[1]
    
    # Degrees of freedom is hidden in the dictionary in result (it's the
    # 4 element in result).
    # Extract the mles structure from result
    
    res = result[4]
    dof = res['dof']
    
    # Now we print the results
    
    
    
    print('=======================================================================')
    print('Testing restrictions on',constrtype,'in VEC model with cointegration rank = ',int(r),'\n')
    print('The constraint on',constrtype,'tested is\n',cons.T,'\n')
    print('\nLR test of restriction with',int(dof),'degrees of freedom')
    print('Lr test statistic =',np.round(LRtest, 4))
    print('with p-value = ',np.round(pval, 4),'\n')
    
    # Then we need to print the restricted estimates of alpha and beta.
    # First we need to extract the appropriate dictionary from results,
    # the dictionary we extracted above to obtain degrees of freedom.
    # From this dictionary we extract the appropriate lists, define
    # the arrays we are interested in and then we print these arrays.
    
    mylist = list(res.values())[1]
    Arest = np.array(mylist['A'][:])
    Brest = np.array(mylist['B'][:])
    
    diagres = np.round(np.concatenate((Arest,Brest), axis=1), 4)
    tt1 = ["\u03B1"+str(i+1) for i in range(int(r))]
    tt2 = ["\u03B2"+str(i+1) for i in range(int(r))]
    header = tt1+tt2
    print('Restricted estimates of alpha and beta')
    print(tabulate(diagres, headers=header, numalign="right"))
    
    # Restricted estimates of deterministic components
    
    if model=='H2':
        c0 = [[np.nan]]
        c1 = [[np.nan]]
        d0 = [[np.nan]]
        d1 = [[np.nan]]
    elif model=='H1*':
        c1 = [[np.nan]]
        d0 = [[np.nan]]
        d1 = [[np.nan]]
        if r == 1:
            c0 = np.reshape(np.asarray(mylist['c0']), (-1,1))
        else:
            c0 = np.asarray(mylist['c0'][:])
    
    elif model=='H1':
        if r == 1:
            c0 = np.reshape(np.asarray(mylist['c0']), (-1,1))
        else:
            c0 = np.asarray(mylist['c0'][:]) 
             
        c1 = np.array(mylist['c1'][:])
        d0 = [[np.nan]]
        d1 = [[np.nan]]
    elif model=='H*':
        d1 = [[np.nan]]
        if r == 1:
            c0 = np.reshape(np.asarray(mylist['c0']), (-1,1))
            d0 = np.reshape(np.asarray(mylist['d0']), (-1,1))
        else:        
            c0 = np.asarray(mylist['c0'][:])
            d0 = np.asarray(mylist['d0'][:])
            
        c1 = np.array(mylist['c1'][:])
    elif model=='H':
        if r == 1:
            c0 = np.reshape(np.asarray(mylist['c0']), (-1,1))
            d0 = np.reshape(np.asarray(mylist['d0']), (-1.1))
        else:        
            c0 = np.asarray(mylist['c0'][:])
            d0 = np.asarray(mylist['d0'][:])
            
        c1 = np.array(mylist['c1'][:])
        d1 = np.array(mylist['d1'][:])
    
    
    
    print('\n Restricted estimates of deterministic terms \n')
    print(tabulate(c0, headers=["c0"],numalign="right"))
    print("")
    print(tabulate(c1, headers=["c1"],numalign="right"))
    print("")
    print(tabulate(d0, headers=["d0"],numalign="right"))
    print("")
    print(tabulate(d1, headers=["d1"],numalign="right"))
    print('=======================================================================')
    
    # This is all we need so we can now close the Matlab engine
    #eng.quit()
    
    return LRtest, pval, dof, Arest, Brest, c0, c1, d0, d1

def VARlsExog(y, p, con, tr, exog):
    # Define dependent and independent variables for VAR estimation
    T, K = y.shape
    #T er antal tid. K er antal variable.
    Y = y[p:T, :]
    #for alle 2 kolonner, gå fra række p til T.
    #p er noget vi definerer selv i funktionen.
    
    # Independent variable
    X = lagmatrix(y, p)
    #vi lagger data med p lags!
    X = X[p:, :]

    #con=1 hvis der er en konstant i modellen.
    if con == 1:
        X = np.hstack([X, np.ones((len(X), 1))])
    #tr=1 hvis der er en tidstrend i modellen.
    if tr == 1:
        X = np.hstack([X, np.arange(1, len(X) + 1).reshape(-1, 1)])
    
    if isinstance(exog, np.ndarray):
        X = np.hstack([X, exog])
    
    T, Kp = X.shape
    
    # Beta estimation
    Beta = np.linalg.inv(X.T @ X) @ X.T @ Y  
    
    # Residuals
    residuals = Y - X @ Beta
    
    # Covariance matrix of residuals
    SIGMA = (residuals.T @ residuals) / T
    
    # Covariance matrix of Beta
    CovBeta = np.kron(SIGMA, np.linalg.inv(X.T @ X))
    
    tratioBeta = Beta.reshape(-1, order='F') / np.sqrt(np.diag(CovBeta))
    tratioBeta = tratioBeta.reshape(-1, K, order='F')
    
    return Beta, CovBeta, tratioBeta, residuals, X, SIGMA

def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

def march(res, lags, K):
    """
    Test for Multivariate ARCH

    Input:
    res : TxK residual matrix
    lags : number of lagged ARCH effects
    K : number of variables in the underlying VAR/VEC model

    Output:
    Function produces a table with results
    """
    T = res.shape[0]

    # Subtract the mean from each column
    uhat = res - res.mean(axis=0)

    # Initialize list to store UUT columns
    UUT_list = []

    # Construct UUT by stacking 'vech' of uhat[i,:]' * uhat[i,:]
    for i in range(T):
        utut = np.outer(uhat[i, :], uhat[i, :])  # K x K matrix
        tmp = []
        for ii in range(K):
            tmp.extend(utut[ii:, ii])  # Collect lower triangular elements including diagonal
        UUT_list.append(tmp)

    # Convert list to numpy array and transpose
    UUT = np.array(UUT_list).T  # Shape: (K*(K+1)//2, T)

    # Create matrices of regressors
    Y = UUT[:, lags:]  # Shape: (K*(K+1)//2, T - lags)
    T_Y = Y.shape[1]

    Z_list = []
    for i in range(T_Y):
        temp = UUT[:, i:i + lags]  # Shape: (K*(K+1)//2, lags)
        temp_vec = temp.flatten(order='F')  # Vectorize in column-major order
        col = np.vstack(([1], temp_vec.reshape(-1, 1)))  # Prepend a 1
        Z_list.append(col)

    # Stack columns horizontally to form Z
    Z = np.hstack(Z_list)  # Shape: (1 + K*(K+1)//2 * lags, T_Y)

    # Compute omega
    A = Y @ Z.T @ np.linalg.inv(Z @ Z.T)
    residuals = Y - A @ Z
    omega = residuals @ residuals.T / T_Y

    # Compute omega0
    Y_mean = Y.mean(axis=1, keepdims=True)
    omega0 = (Y - Y_mean) @ (Y - Y_mean).T / T_Y

    # Compute R-squared value
    R2 = 1 - (2 / (K * (K + 1)) * np.trace(omega @ np.linalg.inv(omega0)))

    # Compute test statistic
    VARCHLM = 0.5 * T_Y * K * (K + 1) * R2

    # Degrees of freedom
    df = lags * K**2 * (K + 1)**2 / 4

    # Compute p-value
    pvalue = 1 - chi2.cdf(VARCHLM, df)

    # Collect test results
    test = np.array([VARCHLM, pvalue, df])

    # Create result table
    march_table = pd.DataFrame({
        'Test': ['Test statistic', 'P-value', 'Degrees of freedom'],
        'Doornik_Hendry': test
    })

    return test, march_table

def multnorm(umat):
    """
    Purpose: Multivariate test on normality.

    H0: Gaussian data generation process
    Usage: norm, multnorm_table = multnorm(umat)

    Input: umat is a TxK matrix of residuals

    Output:
        - norm: A 7x2 NumPy array containing test statistics and p-values.
            Rows correspond to:
            [Joint test statistic, P-value, Degrees of freedom,
             Skewness only, P-value, Kurtosis only, P-value]
            Columns correspond to:
            [Doornik_Hansen, Lutkepohl]

        - multnorm_table: A pandas DataFrame summarizing the results.

    References:
    - Lütkepohl (1993), Introduction to Multiple Time Series Analysis, 2nd ed., p. 150.
    - Doornik & Hansen (1994)

    Based on JMulTi Gauss procedure by Michael Bergman.
    """
    n, k = umat.shape
    umat = umat - np.mean(umat, axis=0)
    Su = (1 / n) * (umat.T @ umat)
    lambda_, Pmat = np.linalg.eig(Su)
    lambda_diag = np.diag(lambda_)

    x = np.sqrt(np.diag(Pmat.T @ Pmat))
    rP, cP = Pmat.shape

    for i in range(cP):
        Pmat[:, i] = Pmat[:, i] / x[i]

    sqrt_lambda = np.sqrt(lambda_diag)
    Q = Pmat @ sqrt_lambda @ Pmat.T

    v1 = np.linalg.inv(Q) @ umat.T
    L = np.linalg.cholesky(Su).T
    v2 = np.linalg.inv(L.T) @ umat.T

    b21 = (np.sum(v1.T ** 4, axis=0) / n).T
    b11 = (np.sum(v1.T ** 3, axis=0) / n).T
    b22 = (np.sum(v2.T ** 4, axis=0) / n).T
    b12 = (np.sum(v2.T ** 3, axis=0) / n).T

    l11 = n * b11.T @ b11 / 6
    pskew1 = 1 - chi2.cdf(l11, df=k)
    l12 = n * b12.T @ b12 / 6
    pskew2 = 1 - chi2.cdf(l12, df=k)

    l21 = n * (b21 - 3).T @ (b21 - 3) / 24
    pkurt1 = 1 - chi2.cdf(l21, df=k)
    l22 = n * (b22 - 3).T @ (b22 - 3) / 24
    pkurt2 = 1 - chi2.cdf(l22, df=k)

    NormDf = 2 * k
    l31 = l11 + l21
    Normpv1 = 1 - chi2.cdf(l31, df=NormDf)
    l32 = l12 + l22
    Normpv2 = 1 - chi2.cdf(l32, df=NormDf)

    norm = np.array([
        [l31, l32],
        [Normpv1, Normpv2],
        [NormDf, NormDf],
        [l11, l12],
        [pskew1, pskew2],
        [l21, l22],
        [pkurt1, pkurt2]
    ])

    tests = [
        'Joint test statistic:', 'P-value', 'Degrees of freedom',
        'Skewness only', 'P-value', 'Kurtosis only', 'P-value'
    ]
    multnorm_table = pd.DataFrame({
        'Test': tests,
        'Doornik_Hansen': np.round(norm[:, 0], 3),
        'Lutkepohl': np.round(norm[:, 1], 3)
    })

    return norm, multnorm_table

#Companion matrix
def comp(Beta, p):
    K = int(Beta.shape[1])
    A = np.zeros((K*p, K*p))
    A[:K, :] = Beta.T[:, :K*p]
    for i in range(1, p):
        A[i*K:(i+1)*K, (i-1)*K:i*K] = np.eye(K)
    return A

#Stability condition (eigenvalues<1)
def stabVAR(A):
    e = np.sort(np.abs(np.linalg.eigvals(A)))
    return e


def PhillipsNorm(K,p,T,model,r,beta,S00,S10,S01,S11,Gammasum):
    """
    This function normalizes estimated cointegration vector using Phillips
    normalization and computes standard errors of the normalized
    cointegration vector as well as for the implied estimate of alpha.
    In addition, the function also provides an estimate of Pi=alpha*beta'
    including standard errors.
    
    Input: 
    K = number of variables in the VAR model
    p = number of lags in underlying VAR model
    model = type of model. We follow Matlab convention.
    model = 1 corresponds to model H2
    model = 2 corresponds to model H1*
    model = 3 corresponds to model H1
    model = 4 corresponds to model H*
    model = 5 corresponds to model H
    r = number of cointegration vectors
    S01 = Sum of squares from ML estimates of VEC model for unknown coint
    vector computed in VECMLHelp function.
    S11 = Sum of squares from ML estimates of VEC model for unknown coint
    vector computed in VECMLHelp function.
    Gammasum = Sum of autoregressive parameters in estimated VEC model
    
    Output:
        beta_norm = PHillips normalized cointegration vector
        alpha_hat = the implied estimate of alpha
        V_beta    = standard error of beta_nrom
        V_alpha   = standard error of alpha_hat
        Pi        = alpha_hat*beta_norm.T
        V_Pi      = standard error of Pi
    
    Author: Michael Bergman
    
    Verified using STATA September 2024 
    """
    # Normalize coint vector using Phillips normalization

    if r==1:
        beta_hat = beta/beta[0,0]
    elif r>1:
        beta1 = beta[0:r,0:r]
        beta2 = beta[r:len(beta),0:r]
        beta_hat =  np.vstack((np.identity(r),np.dot(beta2,np.linalg.inv(beta1))))
        
        
    beta = beta.T  # Need to take transpose of coint vec


    # Compute number of parameters depending on the model 

    if model == 1:
        m1 = K
        m2 = K*(p-1)
        nparams = K*r+K*r+K*(K*(p-1))
    elif model == 2:
        m1 = K+1
        m2 = K*(p-1)
        nparams = K*r+(K+1)*r+K*(K*(p-1))
    elif model == 3:
        m1 = K
        m2 = K*(p-1)+1
        nparams = K*r+K*r+K*(K*(p-1)+1)
    elif model == 4:
        m1 = K+1;
        m2 = K*(p-1)+1
        nparams = K*r+(K+1)*r+K*(K*(p-1)+1)
    elif model == 5:
        m1 = K
        m2 = K*(p-1)+2
        nparams = K*r+K*r+K*(K*(p-1)+2)


    # We need some additional estimates from MLÖ estimation of VEC model for unknown coint vector

    S00,S01,S10,S11,c0,c1,d0,d1,beta,alpha = VECMLHelp(data,p-1,model)

    # Compute alpha_hat given normalized coint vector

    if model==1:
          alpha_hat = S01 @ beta_hat @ np.linalg.inv(beta_hat.T @ S11 @ beta_hat)
    elif model==2:
          alpha_hat = S01 @ beta_hat @ np.linalg.inv(beta_hat.T @ S11 @ beta_hat)
    elif model==3:
          beta_hat = beta_hat[0:K,0:r]
          alpha_hat = S01 @ beta_hat @ np.linalg.inv(beta_hat.T @ S11 @ beta_hat)
    elif model==4:
          beta_hat = np.vstack((np.vstack((beta_hat[0:K,0:r],beta_hat[K+1,0:r])),beta_hat[K,0:r] ))
          alpha_hat = S01 @ beta_hat[0:K+1,0:r] @ np.linalg.inv(beta_hat[0:K+1,0:r].T @ S11 @ beta_hat[0:K+1,0:r])
    elif model==5:
          alpha_hat = S01 @ beta_hat[0:K,0:r] @ np.linalg.inv(beta_hat[0:K,0:r].T @ S11 @ beta_hat[0:K,0:r])
         
         
         

    beta_norm = beta_hat[0:K,0:r]       # This is the normalized coint vector

    # Compute some matrices used to construct standard errors of alpha and beta

    if model==1:
        SigmaB = np.linalg.inv(beta_hat[0:K,0:r].T @ S11 @ beta_hat[0:K,0:r])
        Omega_hat = S00 - alpha_hat @ beta_hat[0:K,0:r].T @ S10



    if model==2:
        SigmaB = np.linalg.inv(beta_hat[0:K+1,0:r].T @ S11 @ beta_hat[0:K+1,0:r])
        Omega_hat = S00 - alpha_hat @ beta_hat[0:K+1,0:r].T @ S10



    if model==3:
         SigmaB = np.linalg.inv(beta_hat[0:K,0:r].T @ S11 @ beta_hat[0:K,0:r])
         Omega_hat = S00 - alpha_hat @ beta_hat[0:K,0:r].T @ S10


    if model==4:
         SigmaB = np.linalg.inv(beta_hat[0:K+1,0:r].T @ S11 @ beta_hat[0:K+1,0:r])
         Omega_hat = S00 - alpha_hat @ beta_hat[0:K+1,0:r].T @ S10


    if model==5:
         SigmaB = np.linalg.inv(beta_hat[0:K,0:r].T @ S11 @ beta_hat[0:K,0:r])
         Omega_hat = S00 - alpha_hat @ beta_hat[0:K,0:r].T @ S10




    Hj = np.hstack((np.zeros((r,m1-r)).T,np.identity(m1-r))).T
    d = int(nparams/K)

    # Compute standard erorr of coint vec
    a1 = np.array((1/(T-d+(K-1))))
    a2 = np.kron(np.identity(r),Hj)
    a3 = np.linalg.inv(np.kron( alpha_hat.T @ np.linalg.inv(Omega_hat) @ alpha_hat, Hj.T @ S11 @ Hj))
    a4 = np.kron(np.identity(r),Hj).T
    # Note that we take the standard error here so V_beta is the estimated standard error
    V_beta =  np.sqrt(np.diagonal(a1 * (a2 @ a3 @ a4)))
    # If model=2, 3, 4 and 5, replace last row with zeros
    if model==2:
           V_beta = np.reshape(V_beta,(r,K+1)).T
           V_beta = np.vstack((V_beta[0:K,:],np.zeros((1,r))))
    elif model==3 or model==5:
           V_beta = np.vstack((np.reshape(V_beta,(r,K)).T,np.zeros((2,r))))
    elif model==4:
           V_beta = np.vstack((np.reshape(V_beta, (r,K+1)).T,np.zeros((1,r))))


    # Compute standard error of alpha given by normalized coint vector 
    V_alpha = np.kron((1/(T-d+K-1))*Omega_hat,SigmaB)

    # Compute standard error of Pi=alpha*beta'

    V_Pi = (1/(T-d+K-1))*np.kron(Omega_hat,(beta_norm @ SigmaB @ beta_norm.T))

    print('\n')
    print("Normalized cointegration vector")
    print(beta_hat)
    print("Standard errors beta_normalized")
    if model==1:
        print(np.reshape(V_beta,(r,K)).T)
    elif model==2:
        print(V_beta)
    elif model==3:
        print(V_beta[0:K,0:r])
    elif model==4:
        print(np.vstack((V_beta[0:K,0:r],np.zeros((1,r)))))
    elif model==5:
        print(V_beta)


    print("Alpha based on normalized beta")
    print(alpha_hat)
    print("standard errors alpha");
    print(np.reshape(np.sqrt(np.diag(V_alpha)),(K,r)))
    print("Estimate of Pi and standard errors")
    print( np.hstack((  np.reshape((alpha_hat @ beta_norm.T), (K*K,1)), np.reshape(( np.sqrt(np.diag(V_Pi))),(K*K,1))  ))   )   

    # Prepare output
    V_alpha_hat = np.reshape(np.sqrt(np.diag(V_alpha)),(K,r))
    Pi = np.reshape((alpha_hat @ beta_norm.T), (K*K,1))
    V_Pi = np.reshape(( np.sqrt(np.diag(V_Pi))),(K*K,1))
    
    
    return beta_norm,alpha_hat,V_beta,V_alpha,Pi,V_Pi

def VECMLHelp(data,p,model):
    """
    This function computes additional output when estimating VEC for unknown coint vector(s).
 
    Input: x is a T x K data array
    p = # lags in VEC model
    Model: Here we use the same as in Matlab
    H2  (no deterministic terms):                      model = 1
    H1* (constant outside the cointegration relation): model = 2
    H1  (Constant within the cointegration relation):  model = 3
    H*  (constant and linear trend in cointegration relation, no quadratic trend):  model = 4
    H:  (constant and linear trend in cointegration relation, quadratic trend) model = 5
     
    or using Python convention
    
    H2:  deterministic="n"
    H1*: deterministic="co"
    H1:  deterministic="ci"
    H*:  deterministic="cili"
    H:   deterministic="colo"
    
    Output:
    Various matrices used when computing normalized coint vector(s)
    """
    import pandas
    from tabulate import tabulate
    from scipy.linalg import sqrtm
    
    # Number of lags in underlying VAR
    p = p+1
    
    # First we need to load critical values
    JCV = np.load('JCV.npy')
    PSSCV = np.load('PSSCV.npy')    
    x = pandas.DataFrame(data).to_numpy()
     
    [T, K] = x.shape
    
    # Setting up first difference of x
     
    dx = x - lagmat(x, [1], 'col')
    dx = dx[1:len(dx)]
    Z0t = dx[p-1:len(dx)]
     
    lagsdx = np.arange(1,p)
    lagsdx = lagsdx.tolist()
     
    if p<=1:
         dxlags = []
         Z1t = []
         if model==3:
             Z1t = np.ones([len(Z0t),1])
         elif model==4:
             Z1t = np.ones([len(Z0t),1])
         elif model==5:
             Z1t = np.concatenate((np.ones([len(Z0t),1]),np.array([np.arange(1, len(Z0t)+1, 1)]).T), axis=1)
         
            
         
    elif p>1:
         dxlags = lagmat(dx,lagsdx, 'col')
         dxlags = dxlags[p-1:len(x)]
         Z1t = dxlags
         [nrows, ncols] = dxlags.shape
         # Now we need to add deterministic components depending on model
         if model==3:
               Z1t = np.concatenate( (dxlags,np.ones([nrows,1])), axis=1)
         elif model==4:
               Z1t = np.concatenate( (dxlags,np.ones([nrows,1])), axis=1)
         elif model==5:
               Z1t = np.concatenate((dxlags,np.concatenate((np.ones([nrows,1]),np.array([np.arange(1, nrows+1, 1)]).T), axis=1)  ), axis=1)
        
     
     
    # Setting up lagged level
    Zkt = lagmat(x, [1], 'col')
    Zkt = Zkt[p:len(Zkt)]
                          
     
     # Add deterministic components to lagged level depending on model
    if model==2:
        Zkt = np.concatenate((Zkt,np.ones([len(Zkt),1])), axis=1)
    elif model==4:
        Zkt = np.concatenate((Zkt,np.array([np.arange(1, len(Zkt)+1, 1)]).T), axis=1)
    elif model==5:
        Zkt = Zkt
     
    # Ready to run the regressions and to compute the residuals
    # This is done in two steps using OLS
     
    if p>1:
         Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Z0t))
         R0t = Z0t-np.dot(Z1t,Beta)
         Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Zkt))
         R1t = Zkt-np.dot(Z1t,Beta)
    elif p<=1:
         R0t = Z0t
         R1t = Zkt
         if model>2:
             Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Z0t))
             R0t = Z0t-np.dot(Z1t,Beta)
             Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Zkt))
             R1t = Zkt-np.dot(Z1t,Beta)
             
         
    # Compute sum of squares
    S01 = np.dot(R0t.T,R1t)/len(Zkt)
    S10 = S01.T
    S00 = np.dot(R0t.T,R0t)/len(Zkt)
    S00I = np.linalg.inv(S00)
    S11 = np.dot(R1t.T,R1t)/len(Zkt)
    S11I = np.linalg.inv(S11)
    G = np.linalg.inv(sqrtm(S11))
    A = np.dot(np.dot(np.dot(np.dot(G,S10),S00I),S01),G.T)  
    

    
    
    
    return S00,S01,S10,S11


def PhillipsNorm(data,K,p,T,model,r,beta,S00,S10,S01,S11,Gammasum):
    """
    This function normalizes estimated cointegration vector using Phillips
    normalization and computes standard errors of the normalized
    cointegration vector as well as for the implied estimate of alpha.
    In addition, the function also provides an estimate of Pi=alpha*beta'
    including standard errors.
    
    Input:
    data = DataFrame
    K = number of variables in the VAR model
    p = number of lags in underlying VAR model
    model = type of model. We follow Matlab convention.
    model = 1 corresponds to model H2
    model = 2 corresponds to model H1*
    model = 3 corresponds to model H1
    model = 4 corresponds to model H*
    model = 5 corresponds to model H
    r = number of cointegration vectors
    S01 = Sum of squares from ML estimates of VEC model for unknown coint
    vector computed in VECMLHelp function.
    S11 = Sum of squares from ML estimates of VEC model for unknown coint
    vector computed in VECMLHelp function.
    Gammasum = Sum of autoregressive parameters in estimated VEC model
    
    Output:
        beta_norm = PHillips normalized cointegration vector
        alpha_hat = the implied estimate of alpha
        V_beta    = standard error of beta_nrom
        V_alpha   = standard error of alpha_hat
        Pi        = alpha_hat*beta_norm.T
        V_Pi      = standard error of Pi
    
    Author: Michael Bergman
    
    Verified using STATA September 2024 
    """
    # Normalize coint vector using Phillips normalization
    import pandas
    import pandas
    data = pandas.DataFrame(data).to_numpy()
    
    
    # First we normalize beta
    
    if r==1:
        beta_hat = beta/beta[0,0]
    elif r>1:
        beta1 = beta[0:r,0:r]
        beta2 = beta[r:len(beta),0:r]
        beta_hat =  np.vstack((np.identity(r),np.dot(beta2,np.linalg.inv(beta1))))
        
    
        
    beta = beta.T  # Need to take transpose of coint vec


    # Compute number of parameters depending on the model 

    if model == 1:
        m1 = K
        m2 = K*(p-1)
        nparams = K*r+K*r+K*(K*(p-1))
    elif model == 2:
        m1 = K+1
        m2 = K*(p-1)
        nparams = K*r+(K+1)*r+K*(K*(p-1))
    elif model == 3:
        m1 = K
        m2 = K*(p-1)+1
        nparams = K*r+K*r+K*(K*(p-1)+1)
    elif model == 4:
        m1 = K+1;
        m2 = K*(p-1)+1
        nparams = K*r+(K+1)*r+K*(K*(p-1)+1)
    elif model == 5:
        m1 = K
        m2 = K*(p-1)+2
        nparams = K*r+K*r+K*(K*(p-1)+2)


    # We now use some output from VECMLHelp function

    # Compute alpha_hat given normalized coint vector

    if model==1:
          alpha_hat = S01 @ beta_hat @ np.linalg.inv(beta_hat.T @ S11 @ beta_hat)
    elif model==2:
          alpha_hat = S01 @ beta_hat @ np.linalg.inv(beta_hat.T @ S11 @ beta_hat)
    elif model==3:
          beta_hat = beta_hat[0:K,0:r]
          alpha_hat = S01 @ beta_hat @ np.linalg.inv(beta_hat.T @ S11 @ beta_hat)
    elif model==4:
          beta_hat = np.vstack((np.vstack((beta_hat[0:K,0:r],beta_hat[K+1,0:r])),beta_hat[K,0:r] ))
          alpha_hat = S01 @ beta_hat[0:K+1,0:r] @ np.linalg.inv(beta_hat[0:K+1,0:r].T @ S11 @ beta_hat[0:K+1,0:r])
    elif model==5:
          alpha_hat = S01 @ beta_hat[0:K,0:r] @ np.linalg.inv(beta_hat[0:K,0:r].T @ S11 @ beta_hat[0:K,0:r])
         
         
         

    beta_norm = beta_hat[0:K,0:r]       # This is the normalized coint vector

    # Compute some matrices used to construct standard errors of alpha and beta

    if model==1:
        SigmaB = np.linalg.inv(beta_hat[0:K,0:r].T @ S11 @ beta_hat[0:K,0:r])
        Omega_hat = S00 - alpha_hat @ beta_hat[0:K,0:r].T @ S10



    if model==2:
        SigmaB = np.linalg.inv(beta_hat[0:K+1,0:r].T @ S11 @ beta_hat[0:K+1,0:r])
        Omega_hat = S00 - alpha_hat @ beta_hat[0:K+1,0:r].T @ S10



    if model==3:
         SigmaB = np.linalg.inv(beta_hat[0:K,0:r].T @ S11 @ beta_hat[0:K,0:r])
         Omega_hat = S00 - alpha_hat @ beta_hat[0:K,0:r].T @ S10


    if model==4:
         SigmaB = np.linalg.inv(beta_hat[0:K+1,0:r].T @ S11 @ beta_hat[0:K+1,0:r])
         Omega_hat = S00 - alpha_hat @ beta_hat[0:K+1,0:r].T @ S10


    if model==5:
         SigmaB = np.linalg.inv(beta_hat[0:K,0:r].T @ S11 @ beta_hat[0:K,0:r])
         Omega_hat = S00 - alpha_hat @ beta_hat[0:K,0:r].T @ S10




    Hj = np.hstack((np.zeros((r,m1-r)).T,np.identity(m1-r))).T
    d = int(nparams/K)

    # Compute standard erorr of coint vec
    a1 = np.array((1/(T-d+(K-1))))
    a2 = np.kron(np.identity(r),Hj)
    a3 = np.linalg.inv(np.kron( alpha_hat.T @ np.linalg.inv(Omega_hat) @ alpha_hat, Hj.T @ S11 @ Hj))
    a4 = np.kron(np.identity(r),Hj).T
    # Note that we take the standard error here so V_beta is the estimated standard error
    V_beta =  np.sqrt(np.diagonal(a1 * (a2 @ a3 @ a4)))
    # If model=2, 3, 4 and 5, replace last row with zeros
    if model==2:
           V_beta = np.reshape(V_beta,(r,K+1)).T
           V_beta = np.vstack((V_beta[0:K,:],np.zeros((1,r))))
    elif model==3 or model==5:
           V_beta = np.vstack((np.reshape(V_beta,(r,K)).T,np.zeros((2,r))))
    elif model==4:
           V_beta = np.vstack((np.reshape(V_beta, (r,K+1)).T,np.zeros((1,r))))


    # Compute standard error of alpha given by normalized coint vector 
    V_alpha = np.kron((1/(T-d+K-1))*Omega_hat,SigmaB)

    # Compute standard error of Pi=alpha*beta'

    V_Pi = (1/(T-d+K-1))*np.kron(Omega_hat,(beta_norm @ SigmaB @ beta_norm.T))

    print('\n')
    print('Phillips normalized cointegration vector(s)\n')
    print("Normalized cointegration vector")
    print(beta_hat)
    print("Standard errors beta_normalized")
    if model==1:
        print(np.reshape(V_beta,(r,K)).T)
    elif model==2:
        print(V_beta)
    elif model==3:
        print(V_beta[0:K,0:r])
    elif model==4:
        print(np.vstack((V_beta[0:K,0:r],np.zeros((1,r)))))
    elif model==5:
        print(V_beta)


    print("Alpha based on normalized beta")
    print(alpha_hat)
    print("standard errors alpha");
    print(np.reshape(np.sqrt(np.diag(V_alpha)),(K,r)))
    print("Estimate of Pi and standard errors")
    print( np.hstack((  np.reshape((alpha_hat @ beta_norm.T), (K*K,1)), np.reshape(( np.sqrt(np.diag(V_Pi))),(K*K,1))  ))   )   

    # Prepare output
    V_alpha_hat = np.reshape(np.sqrt(np.diag(V_alpha)),(K,r))
    Pi = np.reshape((alpha_hat @ beta_norm.T), (K*K,1))
    V_Pi = np.reshape(( np.sqrt(np.diag(V_Pi))),(K*K,1))
    
    return beta_hat,alpha_hat,V_beta,V_alpha,Pi,V_Pi


def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()
