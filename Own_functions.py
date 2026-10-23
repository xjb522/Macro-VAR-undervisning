
import numpy as np
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import VECM


def johcontest_py(Y, r, test, cons, model, lags, alpha=0.05):
    """
    Python implementation of the Johansen LR test
    for a restricted linear trend in the cointegration
    relations (Question 9).

    Supported:
        model = "H*"
        test = ["bcon"]
        cons = [0, 0, ..., 0, 1]'

    Returns:
        LR, pvalue, dof, restricted_fit, unrestricted_fit
    """

    Y = np.asarray(Y, dtype=float)
    K = Y.shape[1]

    r = int(r)
    lags = int(lags)

    # Check the requested restriction
    expected_cons = np.zeros((K + 1, 1))
    expected_cons[-1, 0] = 1.0

    if (
        model != "H*"
        or list(test) != ["bcon"]
        or not np.array_equal(
            np.asarray(cons), expected_cons
        )
    ):
        raise NotImplementedError(
            "This function currently supports only "
            "the trend restriction in Question 9."
        )

    if not 0 < r < K:
        raise ValueError("Cointegration rank must be between 1 and K-1.")

    # Unrestricted: intercept outside,
    # linear trend inside cointegration relations
    unrestricted = VECM(
        Y,
        k_ar_diff=lags,
        coint_rank=r,
        deterministic="coli"
    ).fit()

    # Restricted: same intercept,
    # but no trend in cointegration relations
    restricted = VECM(
        Y,
        k_ar_diff=lags,
        coint_rank=r,
        deterministic="co"
    ).fit()

    # Likelihood ratio test
    LR = 2 * (unrestricted.llf - restricted.llf)

    if LR < -1e-7:
        raise RuntimeError(
            "Unrestricted likelihood is smaller. "
            "Check the model specifications."
        )

    LR = max(0.0, float(LR))

    # One trend restriction per cointegration relation
    dof = r
    pvalue = chi2.sf(LR, dof)

    print("=" * 50)
    print("Johansen LR test: Restricted linear trend")
    print(f"Cointegration rank: {r}")
    print(f"VECM lags: {lags}")
    print(f"Restricted log-likelihood: {restricted.llf:.4f}")
    print(f"Unrestricted log-likelihood: {unrestricted.llf:.4f}")
    print(f"LR statistic: {LR:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {pvalue:.4f}")

    if pvalue < alpha:
        print("Reject H0: linear trend is significant.")
    else:
        print("Do not reject H0: trend can be excluded.")

    return LR, pvalue, dof, restricted, unrestricted

"""Pure-Python Johansen VECM LR restriction tests for Question 10.

Specification: VAR(p), VECM(p-1), deterministic='co' (unrestricted
intercept outside the cointegration relation), *without* exogenous dummies.

This implements concentrated Gaussian reduced-rank likelihood ratios for:
1. Stationarity: e_i is in the cointegration space ("bvec").
2. Exclusion: row i of beta is zero ("bcon").
3. Weak exogeneity: row i of alpha is zero ("acon").

The tests are conditional on the specified cointegration rank r and
require the usual Johansen asymptotic assumptions.

Usage in the HW1 notebook after constructing y:
    from Q10_johansen_python import question10_tests
    q10 = question10_tests(y, p=3, r=3, names=VAR_COLS)
    for hypothesis, table in q10.items():
        print(f"\n{hypothesis}\n{table.to_string(index=False)}")

Neither MATLAB nor statsmodels is needed to run the tests.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2


def _ols_residuals(Y, X):
    """Residualize all columns of Y on columns of X."""
    if X.shape[1] == 0:
        return Y.copy()
    return Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]


def _concentrated_logdet(Y, X, rank):
    """ML log |Sigma| in multivariate reduced-rank regression Y = X B + E.

    B is restricted to rank <= rank; both factor matrices are reestimated.
    Based on canonical correlations after concentrating out other regressors.
    """
    T, k = Y.shape
    if X.shape[0] != T:
        raise ValueError("Y and X must have the same number of observations.")
    if not 0 <= rank <= min(k, X.shape[1]):
        raise ValueError("Impossible reduced-rank regression rank.")

    Syy = (Y.T @ Y) / T
    sign, logdet = np.linalg.slogdet(Syy)
    if sign <= 0:
        raise ValueError("Non-positive covariance determinant: check data and model.")
    if rank == 0:
        return float(logdet)

    Sxx = (X.T @ X) / T
    Sxy = (X.T @ Y) / T
    # Both covariance matrices must be positive definite for these data.
    Lx = np.linalg.cholesky(Sxx)
    Ly = np.linalg.cholesky(Syy)
    # Whitened cross-covariance: Lx^-1 Sxy Ly^-T.
    C = np.linalg.solve(Lx, Sxy)
    C = np.linalg.solve(Ly, C.T).T
    singular_values = np.linalg.svd(C, compute_uv=False)
    eig = np.clip(singular_values[:rank] ** 2, 0.0, 1.0 - 1e-12)
    return float(logdet + np.log1p(-eig).sum())


def _build_partialled_vecm(y, p):
    """Form residualized Delta y and y_{t-1} for a VAR(p) with intercept."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or not np.all(np.isfinite(y)):
        raise ValueError("y must be a finite T x K NumPy array.")
    n, k = y.shape
    if not 1 <= p < n:
        raise ValueError("p must be between 1 and len(y)-1.")
    dy = np.diff(y, axis=0)
    dependent = dy[p - 1:]      # Delta y_t, t=p,...,n-1
    lagged_levels = y[p - 1:-1] # y_{t-1}
    # All short-run differences and a single unrestricted intercept.
    parts = [np.ones((n-p, 1))]
    for lag in range(1, p):
        parts.append(dy[p-1-lag:n-1-lag])
    W = np.column_stack(parts)
    if np.linalg.matrix_rank(W) < W.shape[1]:
        raise ValueError("Short-run regressors are linearly dependent.")
    return (_ols_residuals(dependent, W),
            _ols_residuals(lagged_levels, W))


def question10_tests(y, p=3, r=3, names=None, significance=0.05):
    """Test stationarity, exclusion, weak exogeneity for each variable.

    Null hypotheses:
      Stationarity: e_i belongs to column space of beta (df = K-r).
      Exclusion:    e_i' beta = 0 (df = r).
      Weak exog.:   e_i' alpha = 0 (df = r).

    Uses Gaussian ML LR tests, all based on the same sample, lag order,
    cointegration rank, and an unrestricted constant outside beta.
    """
    y = np.asarray(y, dtype=float)
    n, k = y.shape
    p, r = int(p), int(r)
    if not 1 <= r < k:
        raise ValueError("Specify a reduced cointegration rank 1 <= r < K.")
    if not 0 < significance < 1:
        raise ValueError("significance must lie between 0 and 1.")
    if names is None:
        names = [f"y{i+1}" for i in range(k)]
    if len(names) != k:
        raise ValueError("names must have exactly K elements.")

    R0, R1 = _build_partialled_vecm(y, p)
    T = len(R0)
    ld_unrestricted = _concentrated_logdet(R0, R1, r)
    results = {"Stationarity": [], "Exclusion": [], "Weak exogeneity": []}

    def append(test_name, variable, ld_restricted, degrees):
        stat = T * (ld_restricted - ld_unrestricted)
        if stat < -1e-6:
            raise ArithmeticError(
                f"Constrained log likelihood exceeds unrestricted for {test_name}: "
                f"LR={stat:.5f}."
            )
        stat = max(0.0, float(stat))
        pvalue = float(chi2.sf(stat, degrees))
        results[test_name].append({
            "Variable": variable,
            "LR": stat,
            "df": degrees,
            "p-value": pvalue,
            "Reject H0 (5%)": pvalue < significance,
        })

    for i, name in enumerate(names):
        # EXCLUSION: all r beta vectors have zero coefficient on y_i.
        # Restrict beta to the K-1-dimensional subspace excluding e_i.
        other_levels = np.delete(R1, i, axis=1)
        ld_exclusion = _concentrated_logdet(R0, other_levels, r)
        append("Exclusion", name, ld_exclusion, r)

        # STATIONARITY: e_i is one of the r cointegrating directions.
        # Fit the coefficient on y_i,t-1 without a rank restriction;
        # reestimate the remaining r-1 directions in the other K-1 variables.
        own_level = R1[:, [i]]
        R0_cond = _ols_residuals(R0, own_level)
        R1_cond = _ols_residuals(other_levels, own_level)
        ld_stationarity = _concentrated_logdet(R0_cond, R1_cond, r-1)
        append("Stationarity", name, ld_stationarity, k-r)

        # WEAK EXOGENEITY: alpha[i, :] = 0.
        # The equation for Delta y_i has no error-correction loading.
        # Other equations may still be contemporaneously correlated with it.
        own_change = R0[:, [i]]
        other_changes = np.delete(R0, i, axis=1)
        R0_weak = _ols_residuals(other_changes, own_change)
        R1_weak = _ols_residuals(R1, own_change)
        own_var = float((own_change.T @ own_change).item() / T)
        if own_var <= 0:
            raise ValueError("Zero residual variance in a dependent variable.")
        ld_weak = np.log(own_var) + _concentrated_logdet(R0_weak, R1_weak, r)
        append("Weak exogeneity", name, ld_weak, r)

    return {test_name: pd.DataFrame(rows)
            for test_name, rows in results.items()}


"""HW1 Question 11: fixed cointegrating-vector LR tests and plots in pure Python.

Requires only numpy/pandas/matplotlib/scipy and the Q10_johansen_python.py
created for Question 10. Specification: VAR(3), H1/co (unrestricted
intercept), rank r=3, no dummy variables (as used for Q8-Q10).

Important economic caveats:
* beta_PPP = [0,0,0,0,1] is a test of *stationarity of changes in* log
  real exchange rate, NOT of purchasing power parity in real exchange
  rate levels.
* beta_UIP = [0,1,-1,0,-400] tests a *real-exchange-rate ex-post
  spread proxy*, NOT exact uncovered interest parity, which concerns
  expected nominal exchange-rate changes at matched horizons.
* -400 assumes domestic/foreign rates are annual percent values and the
  exchange-rate index rises for depreciation. Reverse the sign if needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2





def test_fixed_cointegration_vector(y, b, p=3, r=3):
    """H0: specified vector b lies in the cointegration space of rank r.

    Uses Gaussian concentrated likelihood and reestimates under H0.
    Model has unrestricted intercept outside beta, and no dummies.
    Asymptotic df = K-r. Any nonzero scaling of b gives the same test.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 2:
        raise ValueError("y must be a 2-D array: observations by variables")
    n, k = y.shape
    b = np.asarray(b, dtype=float).reshape(-1)
    if b.shape != (k,) or not np.all(np.isfinite(b)) or np.all(b == 0):
        raise ValueError("b must have K finite values and be nonzero")
    if not 1 <= r < k:
        raise ValueError("r must satisfy 1 <= r < K")

    R0, R1 = _build_partialled_vecm(y, p)
    Teff = len(R0)
    logdet_u = _concentrated_logdet(R0, R1, r)

    # Choose K independent regressors: [z_t=b'y_{t-1}, all y except
    # the component where b has its largest magnitude]. This spans the
    # same K-dimensional regressors as the original lagged levels.
    pivot = int(np.argmax(np.abs(b)))
    z = (R1 @ b).reshape(-1, 1)
    others = np.delete(R1, pivot, axis=1)

    # H0 fixes one cointegrating direction to b. Partial z out without
    # imposing an adjustment restriction; estimate remaining r-1
    # cointegrating directions by reduced-rank ML.
    R0_r = _ols_residuals(R0, z)
    others_r = _ols_residuals(others, z)
    logdet_r = _concentrated_logdet(R0_r, others_r, r - 1)

    LR = float(Teff * (logdet_r - logdet_u))
    if LR < -1e-6:
        raise ArithmeticError("Restricted likelihood exceeded unrestricted likelihood")
    LR = max(0.0, LR)
    df = k - r
    return {"LR": LR, "df": df, "p-value": float(chi2.sf(LR, df))}


def question11_tests(y, p=3, r=3, exchange_rate_rises_for_depreciation=True):
    """Test PPP-related stationarity and an ex-post UIP-related spread."""
    sign = -1 if exchange_rate_rises_for_depreciation else 1
    b_ppp = np.array([0, 0, 0, 0, 1.], dtype=float)
    b_uip = np.array([0, 1., -1., 0, sign * 400.], dtype=float)

    vectors = {"PPP proxy: stationary Δlog(REER)": b_ppp,
               "UIP proxy: interest spread − 400 Δlog(REER)": b_uip}
    rows = []
    for label, b in vectors.items():
        res = test_fixed_cointegration_vector(y, b, p=p, r=r)
        rows.append({"Test": label, **res,
                     "Reject H0 (5%)": res["p-value"] < 0.05})
    return pd.DataFrame(rows), vectors


def plot_question11_relations(y, dates, vectors, path=None):
    """Plot theoretical time series. Dates must match y rows exactly."""
    y = np.asarray(y, dtype=float)
    dates = pd.to_datetime(dates)
    if len(dates) != len(y):
        raise ValueError("dates must contain exactly one date for each row of y")

    fig, axes = plt.subplots(len(vectors), 1, figsize=(10, 6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (label, b) in zip(axes, vectors.items()):
        spread = y @ b
        ax.plot(dates, spread, label=label)
        ax.axhline(spread.mean(), linestyle="--", linewidth=1, label="Sample mean")
        ax.axvline(pd.Timestamp("1992-10-01"), linestyle=":", linewidth=1,
                   label="Swedish krona float (1992Q4)")
        ax.set_title(label)
        ax.set_ylabel("Δ log REER" if "PPP" in label else "Annual percentage points")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, axes


if __name__ == "__main__":
    print("Import this module from your HW1 script and pass your y, P, and dates.")



"""HW1 Q11: PPP/UIP proxy restrictions with 0, 3, or 7 exogenous dummies.

Pure Python. Uses Gaussian reduced-rank likelihood-ratio tests, and an OPTIONAL
parametric bootstrap under each null to calibrate critical values conditional
on the chosen rank, deterministic specification and fixed dummy paths.

Model: VAR(p) / VECM(p-1), unrestr. intercept outside cointegrating space.
Dummies enter the short-run VECM equations contemporaneously, not beta.

Caveats: this is a conditional parametric bootstrap under Gaussian innovation
and correctly specified VECM, not a proof of PPP/UIP, and not a bootstrap test
of Johansen cointegration *rank*. Especially the long step dummy is a different
kind of control from a single-period impulse dummy.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2

VAR_COLS_Q11 = [
    "Annual inflation rate", "Domestic interest rate", "Foreign interest rate",
    "GDP_log", "REER_log_diff"
]
DUMMY_3 = ["Dummy.2", "Dummy", "Dummy.1"]
DUMMY_7 = DUMMY_3 + ["Dummy.3", "Dummy.4", "Dummy.5", "Dummy.6"]


def _vecm_arrays(y, p, exog=None):
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or not np.all(np.isfinite(y)):
        raise ValueError("y must be a finite (n,K) array")
    n, k = y.shape
    if not 1 <= p < n:
        raise ValueError("Invalid VAR lag p")
    dy = np.diff(y, axis=0)
    dep = dy[p-1:]
    levels = y[p-1:-1]
    Wparts = [np.ones((n-p, 1))]
    for lag in range(1, p):
        Wparts.append(dy[p-1-lag:n-1-lag])
    if exog is not None:
        exog = np.asarray(exog, dtype=float)
        if exog.ndim != 2 or exog.shape[0] != n or not np.all(np.isfinite(exog)):
            raise ValueError("exog must be finite with exactly the same n rows as y")
        Wparts.append(exog[p:])  # IMPORTANT: only slice once, inside the function
    W = np.column_stack(Wparts)
    if np.linalg.matrix_rank(W) < W.shape[1]:
        raise ValueError("Short-run regressors (including dummies) are rank-deficient")
    R0 = _ols_residuals(dep, W)
    R1 = _ols_residuals(levels, W)
    return dep, levels, W, R0, R1


def fixed_vector_lr(y, b, p=3, r=3, exog=None):
    """Test H0: given K-vector b belongs to the rank-r cointegration space."""
    y = np.asarray(y, dtype=float)
    k = y.shape[1]
    b = np.asarray(b, dtype=float).reshape(-1)
    if b.shape != (k,) or not np.all(np.isfinite(b)) or np.all(b == 0):
        raise ValueError("b must be a finite, nonzero K-vector")
    if not 1 <= int(r) < k:
        raise ValueError("Require 1 <= r < K")
    _, _, _, R0, R1 = _vecm_arrays(y, p, exog)
    Teff = R0.shape[0]
    logdet_u = _concentrated_logdet(R0, R1, r)
    z = (R1 @ b)[:, None]
    pivot = np.argmax(np.abs(b))
    other = np.delete(R1, pivot, axis=1)
    logdet_r = _concentrated_logdet(
        _ols_residuals(R0, z), _ols_residuals(other, z), r-1
    )
    stat = Teff * (logdet_r - logdet_u)
    if stat < -1e-6:
        raise ArithmeticError(f"Unexpected negative LR statistic: {stat}")
    stat = max(0.0, float(stat))
    df = k - int(r)
    return {"LR": stat, "df": df, "p-asymptotic": float(chi2.sf(stat, df))}


def _rrr_coefficient(Y, X, rank):
    """ML rank-restricted coefficient X @ B approximating Y, no intercept."""
    T, k = Y.shape
    m = X.shape[1]
    if rank == 0:
        return np.zeros((m, k))
    Sxx, Syy, Sxy = X.T @ X / T, Y.T @ Y / T, X.T @ Y / T
    Lx, Ly = np.linalg.cholesky(Sxx), np.linalg.cholesky(Syy)
    C = np.linalg.solve(Lx, Sxy)
    C = np.linalg.solve(Ly, C.T).T
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    # Undo X and Y whitening to recover coefficients in original units.
    return np.linalg.solve(Lx.T, (U[:, :rank] * s[:rank]) @ Vt[:rank] @ Ly.T)


def _restricted_vecm_fit(y, b, p, r, exog):
    """MLE restricted VECM coefficients under beta-space inclusion of b."""
    dep, levels, W, R0, R1 = _vecm_arrays(y, p, exog)
    k = y.shape[1]
    b = np.asarray(b, dtype=float).reshape(k)
    pivot = int(np.argmax(np.abs(b)))
    other_ix = [i for i in range(k) if i != pivot]
    z = (R1 @ b)[:, None]
    others = R1[:, other_ix]

    if r > 1:
        r0_z = _ols_residuals(R0, z)
        others_z = _ols_residuals(others, z)
        coef_other = _rrr_coefficient(r0_z, others_z, r-1)
    else:
        coef_other = np.zeros((k-1, k))
    coef_z = np.linalg.lstsq(z, R0 - others @ coef_other, rcond=None)[0]
    pi_t = b[:, None] @ coef_z
    pi_t[other_ix, :] += coef_other

    coef_W = np.linalg.lstsq(W, dep - levels @ pi_t, rcond=None)[0]
    residuals = dep - levels @ pi_t - W @ coef_W
    residuals -= residuals.mean(axis=0, keepdims=True)
    Sigma = residuals.T @ residuals / len(residuals)
    if np.linalg.slogdet(Sigma)[0] <= 0:
        raise ValueError("Residual covariance not positive definite")
    return pi_t, coef_W, Sigma


def bootstrap_fixed_vector_lr(y, b, p=3, r=3, exog=None, reps=399, seed=2026):
    """Parametric LR bootstrap under H0, fixed lagged starting values/dummy path.

    This calibrates the *fixed vector* LR statistic, NOT the rank trace test.
    Re-estimates restricted and unrestricted rank-r models in each simulation.
    """
    y = np.asarray(y, dtype=float)
    n, k = y.shape
    if reps < 19:
        raise ValueError("Use at least 19 replications")
    obs = fixed_vector_lr(y, b, p, r, exog)["LR"]
    pi_t, coef_W, Sigma = _restricted_vecm_fit(y, b, p, r, exog)
    rng = np.random.default_rng(seed)
    simulated_stats = np.empty(reps)

    for j in range(reps):
        y_sim = np.empty_like(y)
        y_sim[:p] = y[:p]
        innovations = rng.multivariate_normal(np.zeros(k), Sigma, size=n-p)
        for t in range(p, n):
            w = [np.array([1.0])]
            for lag in range(1, p):
                w.append(y_sim[t-lag] - y_sim[t-lag-1])
            if exog is not None:
                w.append(np.asarray(exog, dtype=float)[t])
            w = np.concatenate(w)
            delta = y_sim[t-1] @ pi_t + w @ coef_W + innovations[t-p]
            y_sim[t] = y_sim[t-1] + delta
        simulated_stats[j] = fixed_vector_lr(y_sim, b, p, r, exog)["LR"]

    p_boot = (1 + (simulated_stats >= obs).sum()) / (reps + 1)
    return {"LR": obs, "p-bootstrap": float(p_boot),
            "bootstrap-95pct-critical": float(np.quantile(simulated_stats, 0.95)),
            "simulations": simulated_stats}


def compare_dummy_specifications(y, dummy3, dummy7, p=3, r=3,
                                 run_bootstrap=False, bootstrap_reps=399, seed=2026):
    """Compare same y/p/r using no dummies, Bjornland 3, and all 7.

    All exog inputs must be aligned (n x q) with y (do not manually drop p rows).
    """
    y = np.asarray(y, dtype=float)
    b_ppp = np.array([0, 0, 0, 0, 1.0])
    # Bjornland Appendix A: increase in REER = depreciation.
    b_uip = np.array([0, 1, -1, 0, -400.0])
    specs = [("No dummies", None), ("3 impulse dummies", dummy3),
             ("7 dummies (6 impulse, 1 step)", dummy7)]
    rows = []
    for label, x in specs:
        for test, b in [("PPP-related stationarity", b_ppp),
                        ("Ex-post UIP proxy", b_uip)]:
            res = fixed_vector_lr(y, b, p=p, r=r, exog=x)
            row = {"Specification": label, "Test": test, "Rank": r, **res}
            if run_bootstrap and x is not None:
                boots = bootstrap_fixed_vector_lr(
                    y, b, p=p, r=r, exog=x,
                    reps=bootstrap_reps, seed=seed)
                row["p-bootstrap"] = boots["p-bootstrap"]
                row["critical-95%-bootstrap"] = boots["bootstrap-95pct-critical"]
            rows.append(row)
    return pd.DataFrame(rows)
