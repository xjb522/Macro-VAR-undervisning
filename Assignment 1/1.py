from matplotlib.pyplot import figure
from pandas.core.ops.docstrings import key
from pandas.core.interchange import column
from email import header
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from scipy.stats import chi2

# Spørgsmål 1

print ("Succ")

Data = pd.read_excel("Assignment 1/VARSWE.xls",index_col=0)

print(Data.dtypes)

for i in Data.columns:
    pd.to_numeric(Data[i])

Data_punkter = {"GDP":Data["GDP"], "Annual inflation rate":Data["Annual inflation rate"], "CPI seasonal adjusted":Data["CPI seasonal adjusted"],"Domestic interest rate":Data["Domestic interest rate"], "Oil price":Data["Oil price"],"Real effective exchange rate":Data["Real effective exchange rate"],"Foreign interest rate": Data["Foreign interest rate"],"Federal Funds rate": Data["Federal Funds rate"]}

data_navne = Data_punkter.keys()

print(Data)

Data_punkter["GDP_log"] = np.log(Data_punkter["GDP"])

Data_punkter["REER_log_diff"] = np.log(Data_punkter["Real effective exchange rate"]).diff(1)

plt.close()
plt.clf()

fig, ax = plt.subplots(3, 2, figsize=(12, 8))

ax = ax.flatten()

y_labels = {
    "GDP": "GDP (millions)",
    "Annual inflation rate": "Inflation rate",
    "CPI seasonal adjusted": "Index",
    "Domestic interest rate": "Percent (%)",
    "Oil price": "Price",
    "Real effective exchange rate": "Exchange rate",
    "Foreign interest rate": "Percent (%)",
    "Federal Funds rate": "Percent (%)",
    "GDP_log": "log values",
    "REER_log_diff": "log Diff"
}

data_navne_Question_1 = Data_punkter.copy()


for i in ["GDP","CPI seasonal adjusted","Oil price", "Real effective exchange rate", "Federal Funds rate"]:
    data_navne_Question_1.pop(i, None)


for i, navn in enumerate(data_navne_Question_1):
    ax[i].plot(
        Data_punkter[navn],
        label=navn
        
    )

    ax[i].set_ylabel(y_labels[navn])
    ax[i].set_title(navn)
    ax[i].legend(loc="upper right")
    ax[i].grid(True)

plt.tight_layout()
plt.show()

data_navne_question_1 = data_navne_Question_1.keys()
for i in data_navne_question_1:
    print(Data_punkter[i].describe())


# Spørgsmål 2

y = {}

for i in data_navne_question_1:
    y[i] = data_navne_Question_1[i]

y = pd.DataFrame(y).to_numpy()

T, K = y.shape

print("Observations:", T)
print("Variables:", K)
print("Shape:", y.shape)
print(np.isnan(y).sum(axis=0))
print(np.isfinite(y).all())

# Model bygning

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

        B = Y @ X.T @ np.linalg.inv(X @ X.T)

        U = (Y - B @ X)

        SIGMA = U @ U.T / T

        aiccrit[j] = np.log(np.linalg.det(SIGMA)) + 2 / T * (m * K ** 2 + K)       # AIC value
        hqccrit[j] = np.log(np.linalg.det(SIGMA)) + 2 * np.log(np.log(T)) / T * (m * K ** 2 + K)  # HQC value
        siccrit[j] = np.log(np.linalg.det(SIGMA)) + np.log(T) / T * (m * K ** 2 + K)  # SIC value

    # Rank models for p = 0,1,2,...,pmax
    aichat = np.nanargmin(aiccrit)
    hqchat = np.nanargmin(hqccrit)
    sichat = np.nanargmin(siccrit)

    infomat = np.hstack([siccrit, hqccrit, aiccrit])
    m = np.arange(0, pmax + 1).reshape(-1, 1)
    imat = np.hstack([m, infomat])

    LagInformationValue = pd.DataFrame(imat, columns=['Lag', 'SIC', 'HQ', 'AIC'])
    OptimalLag = pd.DataFrame([[sichat, hqchat, aichat]], columns=['SIC', 'HQ', 'AIC'])

    return LagInformationValue, OptimalLag

def top_down(y, pmax):
    t, K = y.shape
    
    # Construct regressor matrix and dependent variable
    XMAX = np.ones((1, t - pmax))
    for i in range(1, pmax + 1):
        XMAX = np.vstack([XMAX, y[pmax - i:t - i, :].T])

    Y = y[pmax:t, :].T

    # Initialize Log Likelihood storage
    LL = []

    # Evaluate models for lag lengths from pmax down to 0
    for j in range(pmax, -1, -1):
        T = t - pmax
        X = XMAX[:j * K + 1, :]

        B = Y @ X.T @ np.linalg.inv(X @ X.T)
        SIGMA = (Y - B @ X) @ (Y - B @ X).T / T

        # Log-Likelihood
        LL.insert(0, [j, T * np.log(np.linalg.det(SIGMA))])

    LL = np.array(LL)

    # Initialize results table
    results = np.array([[0, LL[0, 1], 0, 0]])

    # Calculate LR test statistics and p-values
    for ii in range(pmax):
        lr_stat = LL[ii, 1] - LL[ii + 1, 1]
        p_value = 1 - chi2.cdf(lr_stat, K ** 2)
        results = np.vstack([results, [LL[ii + 1, 0], LL[ii + 1, 1], lr_stat, p_value]])

    # Convert results to a table
    table_lr = pd.DataFrame(results, columns=['Lag', 'Log. Likelihood', 'LR test', 'p-value'])

    return table_lr

Top_down = top_down(y, pmax=6)

LagInformationValue, OptimalLag = pfind(y,6)   

top_down(y, pmax=12)

print(OptimalLag)
print("----------------------")
print(LagInformationValue)
print(Top_down)


