## Second exercise class

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


A = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
B = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

C = np.array([[5, 6, 7], 
            [8, 3, 1], 
            [6, 3, 8]])

#Transpose vektor

B_transpose = B.T
B_transpose


AB_transpose = A @ B_transpose
AB_transpose


C_inv = np.linalg.inv(C)
C_inv


C_det = np.linalg.det(C)
C_det

## VAR Process

#set amount of decimals
pd.set_option('display.float_format', '{:.4f}'.format)

np.random.seed(3)

# Parameters
N = 100  # Number of observations
rho1 = np.array([[0.8, 0],  # Autoregressive coefficient for lag 1
                 [0, 0.8]])

alpha = np.array([0, 0])  # Mean of the VAR process

sigma = np.array([[1, 0],  # Covariance matrix of white noise
                  [0, 1]])

y = np.zeros((2, N))

# Generate white noise
epsilon = np.random.multivariate_normal([0, 0], sigma, N).T

# Generate the VAR(1) process
for t in range(1, N):
    y[:, t] = alpha + rho1 @ y[:, t-1]  + epsilon[:, t]

y = y.T

py = pd.DataFrame(y)
py.describe()

plt.close()
plt.clf()

plt.plot(py)
plt.title("y1")
plt.xlabel("tid")
plt.ylabel("værdi")
plt.grid()
plt.show()


def pfind(y, pmax):
    t, K = y.shape

    # Construct regressor matrix and dependent variable
    XMAX = np.ones((1, t - pmax))
    
    for i in range(1, pmax + 1):
        XMAX = np.vstack([XMAX, y[pmax - i:t - i, :].T])

    Y = y[pmax:t, :].T

    aiccrit = np.zeros((pmax + 1, 1))
    hqccrit = np.zeros((pmax + 1, 1))
    siccrit = np.zeros((pmax + 1, 1))

    # Evaluate criterion for p = 0,...,pmax
    for j in range(0, pmax + 1):
        m = j
        T = t - pmax
        X = XMAX[:j * K + 1, :]

        B = (Y @ X.T) @ np.linalg.inv(X @ X.T)

        SIGMA = ((Y - B @ X) @ (Y - B @ X).T )/T

        aiccrit[j] = np.log(np.linalg.det(SIGMA)) + (2/T) * (m * K**2 + K)
        hqccrit[j] = np.log(np.linalg.det(SIGMA)) + ((2*np.log(np.log(T)))/T) * (m * K**2 + K)
        siccrit[j] = np.log(np.linalg.det(SIGMA)) + ((np.log(T))/T) * (m * K**2 + K)

    # Retrieve the lag with the lowest information criteria
    aichat = np.min(aiccrit)
    hqchat = np.min(hqccrit)
    sichat = np.min(siccrit)

    infomat = np.hstack([siccrit, hqccrit, aiccrit])
    m = np.arange(0, pmax + 1).reshape(-1, 1)
    imat = np.hstack([m, infomat])

    LagInformationValue = pd.DataFrame(imat, columns=['Lag', 'SIC', 'HQ', 'AIC'])
    OptimalLag = pd.DataFrame([[sichat, hqchat, aichat]], columns=['SIC', 'HQ', 'AIC'])

    return LagInformationValue, OptimalLag


LagInformationValue, OptimalLag = pfind(y,12)

print(LagInformationValue)
print(OptimalLag)



np.random.seed(1)

# Parameters
N = 100  # Number of observations
rho2 = np.array([[0.15, 0],  # Autoregressive coefficient for lag 1
                 [0, 0.15]])

alpha = np.array([0, 0])  # Mean of the VAR process

sigma = np.array([[1, 0],  # Covariance matrix of white noise
                  [0, 1]])

y2 = np.zeros((2, N))

# Generate white noise
epsilon = np.random.multivariate_normal([0, 0], sigma, N).T

# Generate the VAR(2) process
for t in range(1, N):
    y2[:, t] = alpha + rho1 @ y2[:, t-1] + rho2 @ y2[:, t-1]  + epsilon[:, t]

y2 = y2.T
DR ,DE = pfind(y=y2,pmax=12)

print(DR)
print(DE)


plt.clf()
plt.plot(y2[:,1])
plt.grid()
plt.title("y2")


## 

