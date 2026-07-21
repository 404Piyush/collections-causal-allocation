"""Sensitivity analysis for the MILP optimizer."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .. import config
from . import optimizer

LOG = logging.getLogger(__name__)


@dataclass
class SensitivityRow:
    scenario: str
    budget: float | None
    capacity_scale: float
    status: str
    objective: float
    cost_total: float
    expected_recovery: float
    net_recovery: float
    n_assigned: int


def sensitivity_sweep(
    df,
    budgets: list[float] | None = None,
    capacity_scales: list[float] | None = None,
    time_limit_sec: int = 60,
) -> pd.DataFrame:
    if budgets is None:
        budgets = [config.TOTAL_BUDGET * f for f in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)]
    if capacity_scales is None:
        capacity_scales = [0.5, 1.0, 1.5]

    rows = []
    for b in budgets:
        for cs in capacity_scales:
            cap = {k: v * cs for k, v in config.AGENT_CAPACITY_HOURS.items()}
            LOG.info("Sweep budget=%.0f, capacity scale=%.2f", b, cs)
            res, _ = optimizer.optimize(
                df, budget=b, capacities=cap, time_limit_sec=time_limit_sec, gap=0.01
            )
            rows.append(
                SensitivityRow(
                    scenario=f"B={int(b):,}_cap={cs:.2f}",
                    budget=b,
                    capacity_scale=cs,
                    status=res.status,
                    objective=res.objective,
                    cost_total=res.cost_total,
                    expected_recovery=res.expected_recovery,
                    net_recovery=res.expected_recovery - res.cost_total,
                    n_assigned=res.n_assigned,
                )
            )
    return pd.DataFrame([r.__dict__ for r in rows])
