"""Density test around the cutoff (no-manipulation check).

A simplified McCrary-style histogram comparison. Bins of the running variable
near the cutoff are compared across the two sides using a chi-square test.
If a treatment-induced sorting caused bunching the histogram would jump.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import stats as st


@dataclass
class DensityTestResult:
    cutoff: float
    h: float
    bins: np.ndarray
    counts_left: np.ndarray
    counts_right: np.ndarray
    chi2: float
    p_value: float
    n_left: int
    n_right: int


def density_test(
    R: np.ndarray,
    cutoff: float,
    n_bins: int = 20,
    h_fraction: float = 1.0,
) -> DensityTestResult:
    R = np.asarray(R, dtype=float)
    n = len(R)
    h = max(np.std(R) * 1.06 * n ** (-0.2), 1.0) * h_fraction

    left_mask = (R >= cutoff - h) & (R < cutoff)
    right_mask = (R >= cutoff) & (R < cutoff + h)

    bins = np.linspace(-h, h, n_bins + 1)
    counts_left, _ = np.histogram(R[left_mask] - cutoff, bins=bins)
    counts_right, _ = np.histogram(R[right_mask] - cutoff, bins=bins)

    table = np.vstack([counts_left, counts_right])
    table = np.where(table == 0, 1, table)
    chi2, p_value, _, _ = st.chi2_contingency(table)

    return DensityTestResult(
        cutoff=float(cutoff),
        h=float(h),
        bins=bins,
        counts_left=counts_left.astype(int),
        counts_right=counts_right.astype(int),
        chi2=float(chi2),
        p_value=float(p_value),
        n_left=int(left_mask.sum()),
        n_right=int(right_mask.sum()),
    )


def density_pair(t1: np.ndarray, t2: np.ndarray) -> Tuple[float, float]:
    """Simple z-test for two histograms."""
    n1, n2 = len(t1), len(t2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    bins = np.linspace(min(t1.min(), t2.min()), max(t1.max(), t2.max()), 11)
    c1, _ = np.histogram(t1, bins=bins)
    c2, _ = np.histogram(t2, bins=bins)
    p1 = c1 / c1.sum()
    p2 = c2 / c2.sum()
    diff = p1 - p2
    var = np.where((p1 + p2) > 0, (p1 + p2) * (1 - (p1 + p2) / 2) / np.sqrt(n1 * n2), 1.0)
    z = float(diff.sum() / (var.sum() + 1e-12))
    p = float(2 * (1 - st.norm.cdf(abs(z))))
    return z, p
