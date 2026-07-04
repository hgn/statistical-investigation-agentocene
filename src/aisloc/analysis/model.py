"""Small statistics toolkit (numpy/scipy only).

Keeps the analysis free of statsmodels/sklearn so it runs on a stock scientific
Python stack. Provides OLS with cluster-robust (by repo) standard errors and a
regularised logistic regression for the PU propensity model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass
class OLSResult:
    beta: np.ndarray
    se: np.ndarray
    names: list[str]
    n: int
    resid: np.ndarray

    def ci(self, i: int, level: float = 0.95) -> tuple[float, float]:
        z = stats.norm.ppf(0.5 + level / 2)
        return self.beta[i] - z * self.se[i], self.beta[i] + z * self.se[i]

    def summary(self, level: float = 0.95) -> list[dict[str, float | str]]:
        out = []
        for i, name in enumerate(self.names):
            lo, hi = self.ci(i, level)
            se = self.se[i]
            z = self.beta[i] / se if se > 0 else np.nan
            out.append({
                "term": name, "coef": float(self.beta[i]), "se": float(se),
                "z": float(z), "ci_lo": float(lo), "ci_hi": float(hi),
                "p": float(2 * (1 - stats.norm.cdf(abs(z)))) if se > 0 else np.nan,
            })
        return out


def ols_cluster(
    X: np.ndarray, y: np.ndarray, names: list[str], groups: np.ndarray | None = None
) -> OLSResult:
    """OLS with optional cluster-robust covariance (CR0). ``groups`` is an array
    of cluster ids aligned with rows."""
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    n, k = X.shape

    if groups is None:
        sigma2 = (resid @ resid) / max(1, n - k)
        cov = sigma2 * XtX_inv
    else:
        meat = np.zeros((k, k))
        for g in np.unique(groups):
            m = groups == g
            xg, ug = X[m], resid[m]
            s = xg.T @ ug
            meat += np.outer(s, s)
        n_g = len(np.unique(groups))
        adj = (n_g / max(1, n_g - 1)) * ((n - 1) / max(1, n - k))
        cov = adj * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return OLSResult(beta, se, names, n, resid)


def logistic_fit(
    X: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 200
) -> np.ndarray:
    """L2-regularised logistic regression (intercept in column 0 assumed).

    Returns coefficient vector. Solved with L-BFGS on the penalised negative
    log-likelihood; robust to separable data thanks to the penalty."""
    n, k = X.shape

    def negll(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = X @ w
        # stable log(1+exp(z))
        ll = np.sum(np.logaddexp(0, z) - y * z)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (p - y)
        pen_mask = np.ones(k)
        pen_mask[0] = 0.0  # do not penalise intercept
        ll += 0.5 * l2 * np.sum((w * pen_mask) ** 2)
        grad += l2 * (w * pen_mask)
        return ll, grad

    w0 = np.zeros(k)
    res = optimize.minimize(negll, w0, jac=True, method="L-BFGS-B",
                            options={"maxiter": max_iter})
    return res.x


def logistic_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(X @ w)))


def zscore(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(a, axis=0)
    sd = np.nanstd(a, axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (a - mu) / sd, mu, sd
