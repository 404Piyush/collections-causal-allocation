"""Covariate balance checks at the cutoff."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bandwidth, local_linear


@dataclass
class CovariateResult:
    covariate: str
    cutoff: float
    h_left: float
    h_right: float
    tau_hat: float
    se_tau: float
    p_value: float


def covariate_balance_test(
    R: np.ndarray,
    covariates: dict[str, np.ndarray],
    cutoff: float,
) -> dict[str, CovariateResult]:
    out: dict[str, CovariateResult] = {}
    bw = bandwidth.ik_bandwidth(R, np.zeros_like(R), cutoff)
    for name, X in covariates.items():
        res = local_linear.local_linear_rdd(R, X, cutoff, bw["h_left"], bw["h_right"])
        out[name] = CovariateResult(
            covariate=name,
            cutoff=cutoff,
            h_left=res.bandwidth_left,
            h_right=res.bandwidth_right,
            tau_hat=res.tau_hat,
            se_tau=res.se_tau,
            p_value=res.p_value,
        )
    return out
