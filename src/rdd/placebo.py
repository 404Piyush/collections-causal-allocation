"""Placebo cutoff tests for RDD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from . import bandwidth, local_linear


@dataclass
class PlaceboResult:
    cutoff: float
    tau_hat: float
    se_tau: float
    ci_low: float
    ci_high: float
    h_left: float
    h_right: float


def run_placebo(
    R: np.ndarray,
    Y: np.ndarray,
    placebos: List[float],
    pilot_p: int = 4,
) -> List[PlaceboResult]:
    out: List[PlaceboResult] = []
    for c in placebos:
        if R.min() > c or R.max() < c:
            continue
        try:
            bw = bandwidth.ik_bandwidth(R, Y, c, pilot_p=pilot_p)
            res = local_linear.local_linear_rdd(R, Y, c, bw["h_left"], bw["h_right"])
        except Exception:
            h = max(np.std(R) * 1.06 * len(R) ** (-0.2), 50.0)
            res = local_linear.local_linear_rdd(R, Y, c, h, h)
        out.append(
            PlaceboResult(
                cutoff=float(c),
                tau_hat=res.tau_hat,
                se_tau=res.se_tau,
                ci_low=res.ci_low,
                ci_high=res.ci_high,
                h_left=res.bandwidth_left,
                h_right=res.bandwidth_right,
            )
        )
    return out
