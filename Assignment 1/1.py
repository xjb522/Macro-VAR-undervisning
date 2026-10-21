from pandas._testing import loc
from matplotlib.pyplot import figure
from pandas.core.ops.docstrings import key
from pandas.core.interchange import column
from email import header
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from scipy.stats import chi2, f, jarque_bera

from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)

from s_master import march, multnorm, jcitestexog


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



y = pd.DataFrame(y)

y = y.dropna()

y = np.array(y)

print(y.dtype)

top_down(y, pmax=12)

Top_down = top_down(y, pmax=12)

LagInformationValue, OptimalLag = pfind(y,12)   

print(OptimalLag)
print("----------------------")
print(LagInformationValue)
print(Top_down)

# Opgave 4

def Var_OLS(y_m, lags):
    t, K = y_m.shape

    # Construct regressor matrix with constant
    XMAX = np.ones((1, t - lags))

    # Add lagged endogenous variables
    for i in range(1, lags + 1):
        XMAX = np.vstack([
            XMAX,
            y_m[lags - i:t - i, :].T
        ])

    # Dependent variables
    Y = y_m[lags:t, :].T

    # Number of effective observations
    T = t - lags

    # OLS estimation
    B = Y @ XMAX.T @ np.linalg.inv(XMAX @ XMAX.T)

    # Residual matrix
    U = Y - B @ XMAX

    # Residual covariance matrix
    SIGMA = U @ U.T / T

    return B, U, SIGMA, XMAX, Y

B, U, SIGMA, X, Y = Var_OLS(y, lags=3)

print("B:", B.shape)
print("U:", U.shape)
print("SIGMA:", SIGMA.shape)
print("X:", X.shape)
print("Y:", Y.shape)


def VAR_LM_test(y, p, h):

    # 1. Estimate original VAR(p)
    B, U, SIGMA_u, X, Y = Var_OLS(y, lags=p)

    K, T = U.shape

    # 2. Construct lagged residuals
    U_lags = np.zeros((K * h, T))

    for i in range(1, h + 1):
        U_lags[(i-1)*K:i*K, i:] = U[:, :-i]

    # 3. Auxiliary regression:
    # Original regressors + lagged residuals
    X_aux = np.vstack([X, U_lags])

    # 4. Estimate auxiliary regression by OLS
    B_aux = (
        Y @ X_aux.T
        @ np.linalg.pinv(X_aux @ X_aux.T)
    )

    # 5. Calculate auxiliary residual covariance
    E = Y - B_aux @ X_aux

    SIGMA_e = E @ E.T / T

    # 6. Breusch-Godfrey LM test
    LM = T * (
        K - np.trace(
            np.linalg.solve(SIGMA_u, SIGMA_e)
        )
    )

    df = h * K**2
    LM_pvalue = chi2.sf(LM, df)

    # 7. Edgerton-Shukur correction
    m = K * h
    q = K * m / 2 - 1

    s = np.sqrt(
        (K**2 * m**2 - 4)
        / (K**2 + m**2 - 5)
    )

    N = T - K*p - m - 0.5*(K - m + 1)

    df1 = K * m
    df2 = N*s - q

    if df2 <= 0:
        raise ValueError("Insufficient degrees of freedom")

    sign_u, logdet_u = np.linalg.slogdet(SIGMA_u)
    sign_e, logdet_e = np.linalg.slogdet(SIGMA_e)

    if sign_u <= 0 or sign_e <= 0:
        raise ValueError("Invalid residual covariance matrix")

    FLM = np.expm1(
        (logdet_u - logdet_e) / s
    ) * df2 / df1

    FLM_pvalue = f.sf(FLM, df1, df2)

    # 8. Results
    results = pd.DataFrame({
        "Test": ["Breusch-Godfrey", "Edgerton-Shukur"],
        "Statistic": [LM, FLM],
        "p-value": [LM_pvalue, FLM_pvalue],
        "Residual lags": [h, h]
    })

    return results

results = VAR_LM_test(y, p=3, h=4)

print(results.round(4))

for h in range(1, 5):
    print(f"\nAutocorrelation test, h = {h}")
    print(VAR_LM_test(y, p=3, h=h).round(4))

# ------------------------------------------
# Question 5: Residual diagnostics
# ------------------------------------------

p = 3
h = 4
arch_lags = 2

# Estimate selected VAR model
B, U, SIGMA, X, Y = Var_OLS(y, lags=p)

# Your function returns residuals as K x T.
# Diagnostic functions expect T x K.
residuals = U.T

T, K = residuals.shape

print("Observations:", T)
print("Variables:", K)


# ------------------------------------------
# 1. Multivariate ARCH test
# ------------------------------------------

arch_test, arch_table = march(
    residuals,
    arch_lags,
    K
)

print("\nMultivariate ARCH test:")
print(arch_table)


# ------------------------------------------
# 2. Multivariate normality test
# ------------------------------------------

normality_test, normality_table = multnorm(
    residuals
)

print("\nMultivariate normality test:")
print(normality_table)


# ------------------------------------------
# 3. Univariate diagnostic tests
# ------------------------------------------

variable_names = list(data_navne_question_1)

rows = []

for i in range(K):

    u_i = residuals[:, i]

    # Ljung-Box autocorrelation test
    lb_result = acorr_ljungbox(
        u_i,
        lags=[h],
        return_df=True
    )

    lb_stat = lb_result["lb_stat"].iloc[0]
    lb_pvalue = lb_result["lb_pvalue"].iloc[0]

    # ARCH-LM heteroskedasticity test
    arch_stat, arch_pvalue, _, _ = het_arch(
        u_i,
        nlags=arch_lags
    )

    # Jarque-Bera normality test
    jb_result = jarque_bera(u_i)

    jb_stat = jb_result.statistic
    jb_pvalue = jb_result.pvalue

    rows.append([
        variable_names[i],
        lb_stat,
        lb_pvalue,
        arch_stat,
        arch_pvalue,
        jb_stat,
        jb_pvalue
    ])


univariate_results = pd.DataFrame(
    rows,
    columns=[
        "Variable",
        "LB statistic",
        "LB p-value",
        "ARCH statistic",
        "ARCH p-value",
        "JB statistic",
        "JB p-value"
    ]
)
print("\nUnivariate residual diagnostics:")
print(univariate_results.round(4))


alpha = 0.05

for _, row in univariate_results.iterrows():

    print(f"\nVariable: {row['Variable']}")

    tests = {
        "Autocorrelation": row["LB p-value"],
        "ARCH": row["ARCH p-value"],
        "Normality": row["JB p-value"]
    }

    for test, pvalue in tests.items():
        conclusion = (
            "Reject H0"
            if pvalue < alpha
            else "Do not reject H0"
        )

        print(f"{test}: {conclusion} (p={pvalue:.4f})")


# Opgave 6

def comp_M(B, K, p):
    # Remove the constant term
    A = B[:, 1:1 + K*p]

    # Top row: [A1 A2 ... Ap]
    top = A

    # For VAR(1), the companion matrix is A1
    if p == 1:
        return top

    # Lower part: identity and zero matrices
    bottom = np.hstack([
        np.eye(K * (p - 1)),
        np.zeros((K * (p - 1), K))
    ])

    # Construct companion matrix
    F = np.vstack([top, bottom])

    return F

def VAR_stability(F):

    # Eigenvalues of the companion matrix
    eigenvalues = np.linalg.eigvals(F)

    # Absolute value of each eigenvalue
    modulus = np.abs(eigenvalues)

    # Largest absolute eigenvalue
    max_eigenvalue = np.max(modulus)

    # Stability condition
    stable = np.all(modulus < 1)

    return stable, eigenvalues, max_eigenvalue


# Estimate the VAR(3)
p = 3
T, K = y.shape

B, U, SIGMA, X, Y = Var_OLS(y, lags=p)

# Construct companion matrix
F = comp_M(B, K, p)

# Test stability
stable, eigenvalues, max_eigenvalue = VAR_stability(F)

print("Companion matrix shape:", F.shape)
print("Eigenvalues:", eigenvalues)
print("Maximum eigenvalue modulus:", max_eigenvalue)
print("Is VAR stable?", stable)

theta = np.linspace(0, 2*np.pi, 500)


plt.clf()

fig, ax = plt.subplots(figsize=(6, 6))

# Unit circle
ax.plot(
    np.cos(theta),
    np.sin(theta),
    linestyle="--",
    label="Unit circle"
)

# Eigenvalues
ax.scatter(
    eigenvalues.real,
    eigenvalues.imag,
    marker="x",
    s=70,
    label="Eigenvalues"
)

ax.axhline(0, linewidth=0.5)
ax.axvline(0, linewidth=0.5)

ax.set_xlabel("Real part")
ax.set_ylabel("Imaginary part")
ax.set_title("VAR(3) Companion Matrix Eigenvalues")
ax.set_aspect("equal")
ax.grid(True)
ax.legend(loc="upper right")

plt.show()

#Opgave 7

# Selected VAR lag length
p = 3

# Case 3: unrestricted constant
model = 3

# No exogenous variables in this initial specification
exog = 0

# Johansen cointegration test
(
    lr1, cval5, cval10, pval, l,
    beta, alpha, c0, c1, d0, d1
) = jcitestexog(
    y,
    p - 1,
    exog,
    model
)

# Construct results table
johansen_results = pd.DataFrame({
    "H0: rank <= r": range(y.shape[1]),
    "Trace statistic": np.asarray(lr1).ravel(),
    "5% critical value": np.asarray(cval5).ravel(),
    "p-value": np.asarray(pval).ravel()
})

print(johansen_results.round(4))

# Sequential rank decision at 5%
rank = 0

for i in range(len(johansen_results)):
    if johansen_results.loc[i, "p-value"] < 0.05:
        rank += 1
    else:
        break

print("Estimated cointegration rank:", rank)

print(Data["Dummy"])
for i in range(1,7):
    print(Data[f"Dummy.{i}"])

dummy_cols = [
    "Dummy", "Dummy.1", "Dummy.2",
    "Dummy.3", "Dummy.4", "Dummy.5",
    "Dummy.6"
]

for col in dummy_cols:
    print(f"\n{col}")
    print(Data.index[Data[col] == 1].tolist())


result = jcitestexog(y, p - 1, 0, 3)