"""Homework 1: VAR model specification and estimation (questions 1-7).

Own implementations: VAR OLS, information criteria, BG/ES tests,
companion matrix and stability. Course routines: march, multnorm,
jcitestexog (not submitted as own functions).

Requires VARSWE.xls and the course-provided s_master.py in the project.
"""

# %% 0. Imports and settings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, f, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from s_master import march, multnorm, jcitestexog

DATA_PATH = Path("Assignment 1/VARSWE.xls")
P_MAX = 6                # Maximum lag order for model selection
P = 3                    # VAR lag order investigated in questions 4-7
H_MAX = 4                # Autocorrelation test horizons
ARCH_LAGS = 2
JOHANSEN_MODEL = 3       # Course function's H1 ("ci") specification

VAR_COLS = [
    "Annual inflation rate",
    "Domestic interest rate",
    "Foreign interest rate",
    "GDP_log",
    "REER_log_diff",
]
DUMMY_COLS = [
    "Dummy.2",  # 1992Q3
    "Dummy",    # 1993Q1
    "Dummy.1",  # 1995Q4
    "Dummy.3",
    "Dummy.4",
    "Dummy.5",
    "Dummy.6",
]


# %% 1. Load, transform and align data (question 1)
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find {DATA_PATH}. Change DATA_PATH to the location of VARSWE.xls."
    )

data = pd.read_excel(DATA_PATH, index_col=0)
data.index = pd.to_datetime(data.index)
data = data.apply(pd.to_numeric, errors="raise")

if (data[["GDP", "Real effective exchange rate"]] <= 0).any().any():
    raise ValueError("GDP and the real effective exchange rate must be positive for logs.")

data["GDP_log"] = np.log(data["GDP"])
data["REER_log_diff"] = np.log(data["Real effective exchange rate"]).diff()

# One date index shared by the endogenous data and ALL THREE dummies.
model_df = data[VAR_COLS + DUMMY_COLS].dropna()
y = model_df[VAR_COLS].to_numpy(dtype=float)        # (91, 5)
exog = model_df[DUMMY_COLS].to_numpy(dtype=float)   # (91, 3)
T, K = y.shape

assert y.shape[0] == exog.shape[0]
assert np.isfinite(y).all() and np.isfinite(exog).all()
print(f"Sample: {model_df.index[0].date()} - {model_df.index[-1].date()}")
print(f"Endogenous y: {y.shape}; exogenous dummies: {exog.shape}")
print("Dummy quarters:")
for col in DUMMY_COLS:
    active_dates = model_df.index[model_df[col].eq(1)].date.tolist()
    print(f"  {col}: {active_dates}")

summary = model_df[VAR_COLS].agg(["count", "mean", "median", "std", "min", "max"]).T
print("\nQuestion 1: descriptive statistics")
print(summary.round(4))

fig, axes = plt.subplots(3, 2, figsize=(12, 9))
for ax, name in zip(axes.flat, VAR_COLS):
    model_df[name].plot(ax=ax, title=name)
    ax.grid(True, alpha=0.3)
for ax in list(axes.flat)[len(VAR_COLS):]:
    ax.axis("off")
fig.tight_layout()
plt.show()


# %% 2. OLS for a VAR(p) with optional exogenous regressors (question 2)
def VAR_OLS(y, p, exog=None, start=None):
    """Return B, U, Sigma, X, Y, with K x T_eff residuals.

    B columns: [constant | A_1 ... A_p | exogenous coefficients].
    If start > p, all models use the same estimation dates.
    """
    y = np.asarray(y, dtype=float)
    t, k = y.shape
    start = p if start is None else start
    if not (0 <= p <= start < t):
        raise ValueError("Require 0 <= p <= start < number of observations.")

    x_parts = [np.ones((1, t - start))]
    for lag in range(1, p + 1):
        x_parts.append(y[start - lag:t - lag].T)

    if exog is not None:
        exog = np.asarray(exog, dtype=float)
        if exog.ndim != 2 or exog.shape[0] != t:
            raise ValueError("exog must have shape (same T as y, number of dummies).")
        x_parts.append(exog[start:].T)

    X = np.vstack(x_parts)
    Y = y[start:].T
    t_eff = Y.shape[1]

    if np.linalg.matrix_rank(X) != X.shape[0]:
        raise ValueError(
            f"Singular VAR regression: {X.shape[0]} regressors and {t_eff} observations."
        )

    # Normal equations, not a ready-made VAR estimation function.
    B = np.linalg.solve(X @ X.T, X @ Y.T).T
    U = Y - B @ X
    Sigma = U @ U.T / t_eff
    return B, U, Sigma, X, Y


# %% 3. Information criteria and top-down LR tests (questions 2-3)
def pfind(y, pmax, exog=None):
    """Compare p=0,...,pmax on the SAME effective sample."""
    t, k = y.shape
    n_exog = 0 if exog is None else exog.shape[1]
    t_eff = t - pmax
    rows = []

    for p in range(pmax + 1):
        _, _, Sigma, _, _ = VAR_OLS(y, p, exog, start=pmax)
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            raise ValueError(f"Non-positive residual covariance determinant for p={p}.")
        n_params = k * (1 + k*p + n_exog)
        aic = logdet + 2*n_params/t_eff
        hq = logdet + 2*np.log(np.log(t_eff))*n_params/t_eff
        sic = logdet + np.log(t_eff)*n_params/t_eff
        rows.append([p, sic, hq, aic])

    table = pd.DataFrame(rows, columns=["Lag", "SIC", "HQ", "AIC"])
    chosen = pd.DataFrame([{criterion: int(table.loc[table[criterion].idxmin(), "Lag"])
                            for criterion in ("SIC", "HQ", "AIC")}])
    return table, chosen


def top_down(y, pmax, exog=None, alpha=0.05):
    """Sequential LR tests p vs p-1; stop at the first rejection from pmax."""
    t, k = y.shape
    t_eff = t - pmax
    logdets = {}
    for p in range(pmax + 1):
        _, _, Sigma, _, _ = VAR_OLS(y, p, exog, start=pmax)
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            raise ValueError(f"Non-positive residual covariance determinant for p={p}.")
        logdets[p] = logdet

    rows = []
    chosen_p = 0
    for p in range(pmax, 0, -1):
        stat = t_eff * (logdets[p - 1] - logdets[p])
        p_value = chi2.sf(stat, k*k)
        rows.append([p, p - 1, stat, p_value, p_value < alpha])
        if p_value < alpha and chosen_p == 0:
            chosen_p = p

    table = pd.DataFrame(rows, columns=["VAR(p)", "VAR(p-1)", "LR", "p-value", "Reject H0"])
    return table, chosen_p


ic_table, optimal_lags = pfind(y, P_MAX, exog)
lr_table, top_down_lag = top_down(y, P_MAX, exog)
print("\nQuestions 2-3: information criteria (including impulse dummies)")
print(ic_table.round(4))
print("Selected lags:")
print(optimal_lags.to_string(index=False))
print("\nTop-down LR tests (same sample and same dummies):")
print(lr_table.round(4))
print(f"Top-down selected lag: {top_down_lag}")
print(f"Lag order used in the remaining questions: p={P}")


# %% 4. BG LM and Edgerton-Shukur Rao F test (question 4)
def VAR_LM_test(y, p, h, exog=None):
    """System autocorrelation tests with h lagged VAR residual vectors.

    Full-sample (zero-filled) auxiliary regression; Rao F small-sample
    degrees of freedom include the constant and exogenous regressors.
    """
    _, U, Sigma_u, X, Y = VAR_OLS(y, p, exog)
    k, t_eff = U.shape
    if h < 1 or h >= t_eff:
        raise ValueError("Residual lag h must be between 1 and T_eff-1.")

    U_lags = np.zeros((k*h, t_eff))
    for lag in range(1, h + 1):
        U_lags[(lag - 1)*k:lag*k, lag:] = U[:, :-lag]

    X_aux = np.vstack([X, U_lags])
    if np.linalg.matrix_rank(X_aux) != X_aux.shape[0]:
        raise ValueError("Auxiliary BG regression has a singular regressor matrix.")
    B_aux = np.linalg.solve(X_aux @ X_aux.T, X_aux @ Y.T).T
    E = Y - B_aux @ X_aux
    Sigma_e = E @ E.T / t_eff

    LM = t_eff * (k - np.trace(np.linalg.solve(Sigma_u, Sigma_e)))
    p_LM = chi2.sf(max(0, LM), k*k*h)

    # Rao's F approximation (Edgerton-Shukur) incl. all deterministic regressors.
    restrictions = k*k*h
    s = np.sqrt((restrictions**2 - 4) / (k*k + (k*h)**2 - 5))
    n_deterministic = X.shape[0] - k*p  # constant + included exogenous dummies
    delta = t_eff - (n_deterministic + k*(p + h))
    delta1 = delta + (k*(h - 1) - 1)/2
    r = restrictions/2 - 1
    df2 = delta1*s - r
    if df2 <= 0:
        raise ValueError("Too few observations for Edgerton-Shukur approximation.")

    sign_u, logdet_u = np.linalg.slogdet(Sigma_u)
    sign_e, logdet_e = np.linalg.slogdet(Sigma_e)
    if sign_u <= 0 or sign_e <= 0:
        raise ValueError("Invalid residual covariance determinants.")
    F_ES = np.expm1((logdet_u - logdet_e)/s) * df2/restrictions
    p_ES = f.sf(max(0, F_ES), restrictions, df2)

    return pd.DataFrame({
        "h": [h, h],
        "Test": ["BG LM", "Edgerton-Shukur F"],
        "Statistic": [LM, F_ES],
        "p-value": [p_LM, p_ES],
    })


# %% 5. Selected VAR and residual diagnostics (question 5)
B, U, Sigma, X, Y = VAR_OLS(y, P, exog)
residuals = U.T  # lecture functions expect (T_eff, K)
print(f"\nSelected VAR({P}) with {exog.shape[1]} impulse dummies: {residuals.shape[0]} usable observations")

ac_tables = pd.concat(
    [VAR_LM_test(y, P, h, exog) for h in range(1, H_MAX + 1)],
    ignore_index=True,
)
print("\nMultivariate autocorrelation:")
print(ac_tables.round(4).to_string(index=False))

arch_test, arch_table = march(residuals, ARCH_LAGS, K)
normality_test, normality_table = multnorm(residuals)
print("\nMultivariate ARCH:")
print(arch_table)
print("\nMultivariate normality:")
print(normality_table)

uni_rows = []
for i, name in enumerate(VAR_COLS):
    u_i = residuals[:, i]
    lb = acorr_ljungbox(u_i, lags=[H_MAX], return_df=True)
    arch_stat, arch_p, _, _ = het_arch(u_i, nlags=ARCH_LAGS)
    jb = jarque_bera(u_i)
    uni_rows.append([name, lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0],
                     arch_stat, arch_p, jb.statistic, jb.pvalue])

uni_table = pd.DataFrame(uni_rows, columns=[
    "Variable", "LB stat", "LB p", "ARCH stat", "ARCH p", "JB stat", "JB p"
])
print("\nUnivariate residual diagnostics:")
print(uni_table.round(4).to_string(index=False))


# %% 6. Companion matrix and stability (question 6)
def comp_M(B, K, p):
    """B = [constant | A_1,...,A_p | exog], so ignore exog coefficients."""
    top = B[:, 1:1 + K*p]
    if p == 1:
        return top.copy()
    bottom = np.hstack([np.eye(K*(p - 1)), np.zeros((K*(p - 1), K))])
    return np.vstack([top, bottom])


def VAR_stability(F):
    eig = np.linalg.eigvals(F)
    modulus = np.abs(eig)
    return bool(np.all(modulus < 1)), eig, float(np.max(modulus))


F = comp_M(B, K, P)
stable, eigenvalues, max_eigenvalue = VAR_stability(F)
print("\nQuestion 6: stability")
print("Companion shape:", F.shape)
print("Maximum eigenvalue modulus:", max_eigenvalue)
print("VAR stable:", stable)

theta = np.linspace(0, 2*np.pi, 500)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(np.cos(theta), np.sin(theta), "--", label="Unit circle")
ax.scatter(eigenvalues.real, eigenvalues.imag, marker="x", label="Eigenvalues")
ax.axhline(0, linewidth=0.5)
ax.axvline(0, linewidth=0.5)
ax.set(xlabel="Real part", ylabel="Imaginary part",
       title=f"VAR({P}) companion eigenvalues")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("VAR_companion_eigenvalues.png", dpi=200)
plt.show()






# %% 7. Johansen cointegration test with impulse dummies (question 7)
# Pass the FULL exog sample, aligned one-to-one with y.
# jcitestexog internally removes the first P observations when building
# the VECM regressors. Passing exog[P:] here would trim the dummies twice
# and cause a dimension mismatch: 88 endogenous vs. 85 exogenous rows.
assert exog.shape[0] == y.shape[0]

(lr1, cval5, cval10, pval, eig_j,
 beta, alpha, c0, c1, d0, d1) = jcitestexog(
    y, P - 1, exog, JOHANSEN_MODEL
)

johansen_table = pd.DataFrame({
    "H0: rank <= r": np.arange(K),
    "Trace statistic": np.asarray(lr1).ravel(),
    "5% critical value": np.asarray(cval5).ravel(),
    "p-value": np.asarray(pval).ravel(),
})
rank = 0
for p_value in johansen_table["p-value"]:
    if p_value < 0.05:
        rank += 1
    else:
        break

print("\nQuestion 7: Johansen trace test with dummies")
print(johansen_table.round(4).to_string(index=False))
print("Estimated cointegration rank (5%):", rank)
print("Potential common trends if all series are I(1):", K - rank)


# %% Question 8: Cointegration vectors

# Model 2: constant in cointegration relation
# Model 4: constant and trend in cointegration relation
# Exogenous dummies are excluded in this question.

models = {
    2: "Constant",
    4: "Constant and trend"
}

johansen_q8 = {}

for model, name in models.items():

    result = jcitestexog(
        y,
        P - 1,
        0,
        model
    )

    (lr1, cval5, cval10, pval, eig,
     beta, alpha, c0, c1, d0, d1) = result

    p_values = np.asarray(pval).ravel()

    rank = 0
    for pv in p_values:
        if pv < 0.05:
            rank += 1
        else:
            break

    johansen_q8[model] = {
        "beta": beta,
        "alpha": alpha,
        "c0": c0,
        "d0": d0,
        "rank": rank
    }

    print(f"\n{name}")
    print("Estimated rank:", rank)
    print("Trace p-values:", p_values)


# %% Plot the first two cointegration relations

def get_deterministic_term(values, n):
    """Extract the first n coefficients or return zeros."""

    if values is None:
        return np.zeros(n)

    a = np.asarray(values, dtype=object).ravel()

    if a.size == 0 or all(v is None for v in a):
        return np.zeros(n)

    if a.size < n:
        raise ValueError("Too few deterministic coefficients.")

    a = np.asarray(a[:n], dtype=complex)

    if np.max(np.abs(a.imag)) > 1e-6:
        raise ValueError("Unexpected complex coefficients.")

    return a.real


def cointegration_relations(y, P, result, n_vectors=2):

    beta = np.real_if_close(np.asarray(result["beta"]))

    # beta is expected to be K x number_of_vectors
    if beta.shape[0] != y.shape[1]:
        beta = beta.T

    beta = beta[:, :n_vectors]

    if beta.shape[1] < n_vectors:
        raise ValueError("Fewer than two vectors returned.")

    # Lagged levels used in the VECM
    y_lag = y[P-1:-1, :]

    # First two cointegration combinations
    ect = np.real_if_close(y_lag @ beta).astype(float)

    # Restricted intercepts
    c0 = get_deterministic_term(
        result["c0"], n_vectors
    )

    # Restricted trend coefficients
    d0 = get_deterministic_term(
        result["d0"], n_vectors
    )

    t = np.arange(1, len(y_lag) + 1)

    ect = ect + c0[None, :] + t[:, None] * d0[None, :]

    return ect


dates = model_df.index[P-1:-1]

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 8),
    sharex=True
)

for row, (model, name) in enumerate(models.items()):

    ect = cointegration_relations(
        y, P, johansen_q8[model]
    )

    for col in range(2):

        ax = axes[row, col]

        ax.plot(dates, ect[:, col])
        ax.axhline(0, linestyle="--", linewidth=1)

        ax.set_title(
            f"{name}: vector {col + 1}"
        )
        ax.set_ylabel("Cointegration relation")
        ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "Johansen_cointegration_vectors.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Show first two estimated vectors for each model

for model, name in models.items():

    beta = np.real_if_close(
        np.asarray(johansen_q8[model]["beta"])
    )

    if beta.shape[0] != len(VAR_COLS):
        beta = beta.T

    beta_table = pd.DataFrame(
        beta[:, :2],
        index=VAR_COLS,
        columns=["Vector 1", "Vector 2"]
    )

    print(f"\n{name}")
    print(beta_table.round(4))
    print("Cointegration rank:", johansen_q8[model]["rank"])
    print("Restricted constants:", johansen_q8[model]["c0"])
    print("Restricted trends:", johansen_q8[model]["d0"])
