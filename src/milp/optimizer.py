"""Mixed-Integer Linear Programming collections allocator.

Decision variables:
  x[i, j, k] in {0, 1}   account i routed to channel j with agent tier k.

Objective (per spec):
  max  sum_i sum_j sum_k (P_ijk * Bal_i - C_jk) * x_ijk

Constraints:
  C1  sum_j sum_k x_ijk <= 1                       (one assignment per acct)
  C2  sum_i sum_j sum_k C_jk * x_ijk <= Budget
  C3  sum_i sum_j Time_j * x_ijk <= Cap_k * 60     (agent hours -> minutes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pulp import (
    PULP_CBC_CMD,
    LpInteger,
    LpMaximize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
    value,
)

from .. import config

LOG = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    status: str
    objective: float
    n_assigned: int
    cost_total: float
    expected_recovery: float
    channel_distribution: dict[str, int]
    tier_distribution: dict[str, int]


def _build_pijk(
    Bal: np.ndarray,
    channels: np.ndarray,
    tiers: np.ndarray,
    channel_pref: np.ndarray,
    tier_pref: np.ndarray,
) -> np.ndarray:
    """Per-account predicted payment probability, broadcast over (j, k).

    P_ijk = base_payment_prob_ij * tier_effect_k
    base_payment_prob grows with the score-derived payment probability but
    modulated by channel effectiveness.
    """
    n = len(Bal)
    P = np.zeros((n, len(config.CHANNELS), len(config.AGENT_TIERS)))
    for j in range(len(config.CHANNELS)):
        for k in range(len(config.AGENT_TIERS)):
            base = 0.10 + 0.40 * channel_pref[:, j] * tier_pref[:, k]
            base = np.minimum(base, 0.95)
            P[:, j, k] = base * 0.85
    return P


def optimize(
    df: pd.DataFrame,
    budget: float = config.TOTAL_BUDGET,
    capacities: dict[str, float] | None = None,
    time_limit_sec: int = 120,
    gap: float = 0.005,
    log: bool = False,
) -> tuple[OptimizationResult, pd.DataFrame]:
    """Run the MILP and return (result, assignment dataframe)."""
    if capacities is None:
        capacities = config.AGENT_CAPACITY_HOURS

    n = len(df)
    Bal = df["outstanding_balance"].to_numpy(dtype=float)
    channels = df["channel"].astype(str).to_numpy()
    tiers = df["agent_tier"].astype(str).to_numpy()
    R = df["expected_recovery_score"].to_numpy(dtype=float)

    R_norm = np.clip((R - 200) / 1500.0, 0, 1)
    channel_pref = np.array(
        [
            np.exp(-(((R_norm - 0.20) / 0.30) ** 2)),
            np.exp(-(((R_norm - 0.35) / 0.30) ** 2)),
            np.exp(-(((R_norm - 0.55) / 0.35) ** 2)),
            np.exp(-(((R_norm - 0.80) / 0.40) ** 2)),
        ]
    ).T
    channel_pref = channel_pref / channel_pref.sum(axis=1, keepdims=True)
    tier_pref = np.array(
        [
            np.ones(n),
            np.exp(-(((R_norm - 0.45) / 0.35) ** 2)),
            np.exp(-(((R_norm - 0.75) / 0.35) ** 2)),
        ]
    ).T
    tier_pref = tier_pref / tier_pref.sum(axis=1, keepdims=True)

    P = _build_pijk(Bal, channels, tiers, channel_pref, tier_pref)

    prob = LpProblem("collections_allocation", LpMaximize)

    x = {}
    for i in range(n):
        for j in range(len(config.CHANNELS)):
            for k in range(len(config.AGENT_TIERS)):
                x[i, j, k] = LpVariable(f"x_{i}_{j}_{k}", cat=LpInteger, lowBound=0, upBound=1)

    obj_terms = []
    for i in range(n):
        for j in range(len(config.CHANNELS)):
            ch = config.CHANNELS[j]
            for k in range(len(config.AGENT_TIERS)):
                tier = config.AGENT_TIERS[k]
                p_ijk = float(P[i, j, k])
                c_jk = float(config.COST_MATRIX[(ch, tier)])
                obj_terms.append(((p_ijk * Bal[i] - c_jk), x[i, j, k]))
    coeffs = np.array([t[0] for t in obj_terms])
    vars_ = [t[1] for t in obj_terms]
    prob += lpSum([c * v for c, v in zip(coeffs, vars_, strict=True)])

    for i in range(n):
        terms = [
            x[i, j, k] for j in range(len(config.CHANNELS)) for k in range(len(config.AGENT_TIERS))
        ]
        prob += lpSum(terms) <= 1, f"uniqueness_{i}"

    budget_terms = []
    for j in range(len(config.CHANNELS)):
        ch = config.CHANNELS[j]
        for k in range(len(config.AGENT_TIERS)):
            tier = config.AGENT_TIERS[k]
            c_jk = config.COST_MATRIX[(ch, tier)]
            for i in range(n):
                budget_terms.append((c_jk, x[i, j, k]))
    b_coeffs = np.array([t[0] for t in budget_terms])
    b_vars = [t[1] for t in budget_terms]
    prob += lpSum([c * v for c, v in zip(b_coeffs, b_vars, strict=True)]) <= budget, "budget"

    for k, tier in enumerate(config.AGENT_TIERS):
        cap_min = capacities[tier] * 60.0
        cap_terms = []
        for j, ch in enumerate(config.CHANNELS):
            t_j = config.TIME_MINUTES[ch]
            for i in range(n):
                cap_terms.append((t_j, x[i, j, k]))
        c_coeffs = np.array([t[0] for t in cap_terms])
        c_vars = [t[1] for t in cap_terms]
        prob += (
            lpSum([c * v for c, v in zip(c_coeffs, c_vars, strict=True)]) <= cap_min,
            f"cap_{tier}",
        )

    solver = PULP_CBC_CMD(
        msg=1 if log else 0,
        timeLimit=time_limit_sec,
        gapRel=gap,
    )
    prob.solve(solver)
    status = LpStatus[prob.status]

    n_assigned = 0
    cost_total = 0.0
    expected_recovery_total = 0.0
    records = []
    channel_dist = dict.fromkeys(config.CHANNELS, 0)
    tier_dist = dict.fromkeys(config.AGENT_TIERS, 0)

    for i in range(n):
        for j in range(len(config.CHANNELS)):
            for k in range(len(config.AGENT_TIERS)):
                v = int(round(value(x[i, j, k])))
                if v == 1:
                    ch = config.CHANNELS[j]
                    tier = config.AGENT_TIERS[k]
                    c_jk = config.COST_MATRIX[(ch, tier)]
                    p_ijk = float(P[i, j, k])
                    cost_total += c_jk
                    expected_recovery_total += p_ijk * Bal[i]
                    n_assigned += 1
                    channel_dist[ch] += 1
                    tier_dist[tier] += 1
                    records.append(
                        {
                            "account_id": int(df.iloc[i]["account_id"]),
                            "channel": ch,
                            "agent_tier": tier,
                            "cost_incurred": c_jk,
                            "p_ijk": round(p_ijk, 4),
                            "expected_recovery": round(p_ijk * Bal[i], 2),
                        }
                    )

    assignments = pd.DataFrame(records)
    result = OptimizationResult(
        status=status,
        objective=float(value(prob.objective) or 0.0),
        n_assigned=n_assigned,
        cost_total=round(cost_total, 2),
        expected_recovery=round(expected_recovery_total, 2),
        channel_distribution=channel_dist,
        tier_distribution=tier_dist,
    )
    return result, assignments
