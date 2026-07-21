"""Business KPI comparisons across three outreach scenarios.

Scenario A: All Level 0 (auto only)
Scenario B: All Level 1 (current practice)
Scenario C: MILP-optimized selection
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ScenarioMetrics:
    name: str
    n_accounts: int
    cost_total: float
    actual_recovery_total: float
    net_recovery: float
    recovery_rate: float
    avg_cost_per_account: float
    cost_per_dollar_recovered: float


def _recovery_with_treatment(
    df: pd.DataFrame, treat_indicator: np.ndarray, cost_per_treated: float
) -> tuple[float, float, int]:
    """Apply a treatment indicator to compute recovery and cost for the scenario.

    For each account:
      - if treated: actual_recovery_amount (synthesized with uplift) + cost
      - if not treated: actual_recovery_amount clipped to a self-curing baseline
    """
    Y = df["actual_recovery_amount"].to_numpy(dtype=float)
    rec_total = float((Y * (1 - 0.5 * (1 - treat_indicator))).sum())
    cost_total = float(treat_indicator.sum() * cost_per_treated)
    n_treated = int(treat_indicator.sum())
    return rec_total, cost_total, n_treated


def scenario_a_all_l0(df: pd.DataFrame, l0_cost: float = 1.5) -> ScenarioMetrics:
    Bal = df["outstanding_balance"].sum()
    Y_baseline = df["actual_recovery_amount"].to_numpy() * 0.85
    cost = float(len(df) * l0_cost)
    rec = float(Y_baseline.sum())
    return ScenarioMetrics(
        name="All Level 0 (auto only)",
        n_accounts=int(len(df)),
        cost_total=cost,
        actual_recovery_total=rec,
        net_recovery=rec - cost,
        recovery_rate=rec / float(Bal) if Bal > 0 else 0.0,
        avg_cost_per_account=cost / len(df),
        cost_per_dollar_recovered=cost / rec if rec > 0 else float("inf"),
    )


def scenario_b_all_l1(df: pd.DataFrame, l1_cost: float = 51.5) -> ScenarioMetrics:
    Bal = df["outstanding_balance"].sum()
    cost = float(len(df) * l1_cost)
    rec = float(df["actual_recovery_amount"].sum())
    return ScenarioMetrics(
        name="All Level 1 (current practice)",
        n_accounts=int(len(df)),
        cost_total=cost,
        actual_recovery_total=rec,
        net_recovery=rec - cost,
        recovery_rate=rec / float(Bal) if Bal > 0 else 0.0,
        avg_cost_per_account=cost / len(df),
        cost_per_dollar_recovered=cost / rec if rec > 0 else float("inf"),
    )


def scenario_c_milp(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    cost_total: float,
    expected_recovery: float,
) -> ScenarioMetrics:
    Bal = df["outstanding_balance"].sum()
    p_ij_avg = (assignments["p_ijk"] * assignments["expected_recovery"]).sum() / max(
        assignments["expected_recovery"].sum(), 1
    )
    actual_recovery_proxy = df["actual_recovery_amount"].sum() * (
        expected_recovery / max(Bal, 1.0) / max(p_ij_avg, 1e-6)
    )
    actual_recovery_proxy = min(actual_recovery_proxy, df["actual_recovery_amount"].sum())
    actual_recovery_proxy = max(actual_recovery_proxy, df["actual_recovery_amount"].sum() * 0.6)
    return ScenarioMetrics(
        name="MILP-optimized",
        n_accounts=int(assignments.shape[0]),
        cost_total=round(cost_total, 2),
        actual_recovery_total=round(actual_recovery_proxy, 2),
        net_recovery=round(actual_recovery_proxy - cost_total, 2),
        recovery_rate=actual_recovery_proxy / float(Bal) if Bal > 0 else 0.0,
        avg_cost_per_account=cost_total / max(assignments.shape[0], 1),
        cost_per_dollar_recovered=(
            cost_total / actual_recovery_proxy if actual_recovery_proxy > 0 else float("inf")
        ),
    )


def build_comparison_table(
    df: pd.DataFrame,
    milp_cost: float,
    milp_expected_recovery: float,
    milp_assignments: pd.DataFrame,
) -> pd.DataFrame:
    a = scenario_a_all_l0(df)
    b = scenario_b_all_l1(df)
    c = scenario_c_milp(df, milp_assignments, milp_cost, milp_expected_recovery)
    rows = []
    for sm in [a, b, c]:
        rows.append(
            {
                "scenario": sm.name,
                "n_accounts": sm.n_accounts,
                "cost_total_usd": round(sm.cost_total, 2),
                "actual_recovery_total_usd": round(sm.actual_recovery_total, 2),
                "net_recovery_usd": round(sm.net_recovery, 2),
                "recovery_rate": round(sm.recovery_rate, 4),
                "avg_cost_per_account_usd": round(sm.avg_cost_per_account, 2),
                "cost_per_dollar_recovered": round(sm.cost_per_dollar_recovered, 4),
            }
        )
    out = pd.DataFrame(rows)
    baseline = float(
        out.loc[out["scenario"].str.contains("All Level 1"), "net_recovery_usd"].iloc[0]
    )
    out["net_vs_current_pct"] = (
        (out["net_recovery_usd"] / baseline - 1.0) * 100.0 if baseline > 0 else 0.0
    )
    return out
