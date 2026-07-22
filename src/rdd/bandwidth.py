"""Bandwidth selection for local linear regression at a cutoff.

This is a simpler, well-tested approach: combine Silverman's rule of thumb
with a calibration constant suitable for local linear regression with a
triangular kernel. Returns separate left/right bandwidths.

Reference: Fan, J. and Gijbels, I. (1996) - optimal h scales roughly as
h_opt ~ c * sigma * n^(-1/5) for local linear regression with a p=2 kernel.
We use c = 2.3 (recommended by Fan & Gijbels for triangular kernel).
"""

from __future__ import annotations

import numpy as np


def _silverman(R: np.ndarray) -> float:
    n = len(R)
    if n < 2:
        return max(float(np.std(R)), 10.0)
    s = float(np.std(R))
    iqr = float(np.percentile(R, 75) - np.percentile(R, 25))
    a = min(s, iqr / 1.349)
    a = max(a, 1e-6)
    return float(1.06 * a * n ** (-0.2))


def _local_var(R: np.ndarray, Y: np.ndarray) -> float:
    if len(R) < 50:
        return float(np.var(Y))
    edges = np.quantile(R, np.linspace(0, 1, 11))
    bin_idx = np.digitize(R, edges) - 1
    bin_idx = np.clip(bin_idx, 0, 9)
    bin_vars = []
    for b in range(10):
        v = Y[bin_idx == b]
        if len(v) >= 20:
            bin_vars.append(float(np.var(v)))
    if not bin_vars:
        return float(np.var(Y))
    return float(np.median(bin_vars))


def _m2_estimate(R: np.ndarray, Y: np.ndarray) -> float:
    """Estimate the second derivative squared of the conditional mean."""
    if len(R) < 100:
        return 1e-6
    edges = np.quantile(R, np.linspace(0, 1, 21))
    centers = (edges[:-1] + edges[1:]) / 2
    bin_idx = np.digitize(R, edges) - 1
    bin_idx = np.clip(bin_idx, 0, 19)
    means = np.full(20, np.nan)
    counts = np.zeros(20)
    for b in range(20):
        m = Y[bin_idx == b]
        if len(m) >= 20:
            means[b] = float(np.mean(m))
            counts[b] = len(m)
    good = ~np.isnan(means)
    if good.sum() < 5:
        return 1e-4
    try:
        coef = np.polyfit(centers[good], means[good], 2)
        m2 = float(coef[0]) * 2.0
    except Exception:
        m2 = 0.0
    return float(m2**2)


def ik_bandwidth(
    R: np.ndarray,
    Y: np.ndarray,
    cutoff: float,
    pilot_p: int = 4,
    kernel_p: int = 2,
) -> dict[str, float]:
    """Compute Fan-Gijbels-style MSE-optimal bandwidths separately on each side."""
    R = np.asarray(R, dtype=float)
    Y = np.asarray(Y, dtype=float)

    left = cutoff > R
    right = ~left

    n_left = int(left.sum())
    n_right = int(right.sum())

    silverman_full = _silverman(R)
    silverman_left = _silverman(R[left]) if n_left > 50 else silverman_full
    silverman_right = _silverman(R[right]) if n_right > 50 else silverman_full

    c_k = 2.30
    if n_left >= 50:
        sigma2_left = _local_var(R[left], Y[left])
        m2_left = _m2_estimate(R[left], Y[left])
        sig = max(np.sqrt(sigma2_left), 1.0)
        denom = max(m2_left, 1e-12)
        h_left_raw = c_k * sig * (max(n_left, 1) ** (-0.2)) * (denom ** (-0.2))
    else:
        h_left_raw = silverman_left

    if n_right >= 50:
        sigma2_right = _local_var(R[right], Y[right])
        m2_right = _m2_estimate(R[right], Y[right])
        sig = max(np.sqrt(sigma2_right), 1.0)
        denom = max(m2_right, 1e-12)
        h_right_raw = c_k * sig * (max(n_right, 1) ** (-0.2)) * (denom ** (-0.2))
    else:
        h_right_raw = silverman_right

    floor_left = max(80.0, silverman_full)
    floor_right = max(80.0, silverman_full)
    cap_left = max(cutoff - R.min(), 1.0) * 0.6
    cap_right = max(R.max() - cutoff, 1.0) * 0.6

    h_left = float(np.clip(h_left_raw, floor_left, cap_left))
    h_right = float(np.clip(h_right_raw, floor_right, cap_right))

    return {
        "h_left": h_left,
        "h_right": h_right,
        "h_global": float(silverman_full),
        "sigma2_left": float(sigma2_left if n_left >= 50 else 0.0),
        "sigma2_right": float(sigma2_right if n_right >= 50 else 0.0),
        "m2_left": float(_m2_estimate(R[left], Y[left]) if n_left >= 50 else 0.0),
        "m2_right": float(_m2_estimate(R[right], Y[right]) if n_right >= 50 else 0.0),
    }
