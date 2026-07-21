"""Local linear regression with triangular kernel for RDD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RDDResult:
    cutoff: float
    bandwidth_left: float
    bandwidth_right: float
    beta0: float
    beta1_left: float
    beta1_right: float
    tau_hat: float
    se_tau: float
    ci_low: float
    ci_high: float
    p_value: float
    n_effective: float
    n_left: int
    n_right: int


def _triangular_kernel(u: np.ndarray, h: float) -> np.ndarray:
    rel = np.abs(u) / max(float(h), 1e-9)
    return np.maximum(0.0, 1.0 - rel)


def _wls(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares with HC1 robust standard errors."""
    wsqrt = np.sqrt(np.maximum(w, 0.0))
    Xw = X * wsqrt[:, None]
    Yw = Y * wsqrt
    XtX = Xw.T @ Xw
    try:
        XtX_inv = np.linalg.pinv(XtX)
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)
    beta = XtX_inv @ (Xw.T @ Yw)

    resid = Y - X @ beta
    meat = Xw.T @ np.diag(w * resid**2) @ Xw
    cov_hc1 = XtX_inv @ meat @ XtX_inv
    sumw = w.sum() or 1.0
    n_eff = int((sumw**2) / np.maximum((w**2).sum(), 1e-12))
    cov_hc1 *= n_eff / max(n_eff - X.shape[1], 1)
    se = np.sqrt(np.maximum(np.diag(cov_hc1), 0.0))
    return beta, se


def local_linear_rdd(
    R: np.ndarray,
    Y: np.ndarray,
    cutoff: float,
    h_left: float,
    h_right: float,
) -> RDDResult:
    """Estimate the RDD with separate triangular kernels / bandwidths per side.

    Model: Y = b0 + b1*(R-c) + tau*T + b3*(R-c)*T + e
    Coefficients returned: tau (the jump), b1_left (interacted, left side implied 0),
    b1_right (the slope on the right side encoded via the same b1 plus b3, etc.)
    """
    R = np.asarray(R, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Rc = R - cutoff
    T = (cutoff <= R).astype(float)

    left = Rc <= 0
    right = ~left

    h_left_use = max(float(h_left), 1e-3)
    h_right_use = max(float(h_right), 1e-3)

    X_full = np.column_stack([np.ones_like(R), Rc, T, Rc * T])

    w = _triangular_kernel(-Rc, h_left_use) * left + _triangular_kernel(Rc, h_right_use) * right

    Xw = X_full
    beta, se = _wls(Xw, Y, w)

    tau = float(beta[2])
    se_tau = float(se[2]) if len(se) > 2 else float("nan")
    from scipy import stats as st

    ci_low = tau - 1.96 * se_tau
    ci_high = tau + 1.96 * se_tau
    df = max(int((w > 0).sum() - Xw.shape[1]), 1)
    p_value = float(2 * (1 - st.t.cdf(abs(tau) / se_tau, df=df))) if se_tau > 0 else float("nan")

    b1_left = float(beta[1])
    b1_right = float(beta[1] + beta[3])

    n_left = int(left.sum())
    n_right = int(right.sum())
    n_eff = float((w > 0).sum())

    return RDDResult(
        cutoff=float(cutoff),
        bandwidth_left=float(h_left_use),
        bandwidth_right=float(h_right_use),
        beta0=float(beta[0]),
        beta1_left=b1_left,
        beta1_right=b1_right,
        tau_hat=tau,
        se_tau=se_tau,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_effective=n_eff,
        n_left=n_left,
        n_right=n_right,
    )


def side_means(R: np.ndarray, Y: np.ndarray, cutoff: float, h: float):
    Rc = R - cutoff
    left = Rc <= 0
    right = ~left
    w_left = _triangular_kernel(-Rc[left], h)
    w_right = _triangular_kernel(Rc[right], h)
    m_left = float((Y[left] * w_left).sum() / w_left.sum()) if w_left.sum() > 0 else float("nan")
    m_right = (
        float((Y[right] * w_right).sum() / w_right.sum()) if w_right.sum() > 0 else float("nan")
    )
    return m_left, m_right
