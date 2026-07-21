"""End-to-end collections-optimization pipeline.

Run from project root:
    python run_pipeline.py

Steps:
  1. Build / reload processed dataset (LendingClub snapshot -> synthetic overlay)
  2. RDD analysis -> 5 PNGs + rdd_results.csv
  3. MILP optimization -> 3 PNGs + assignments.csv
  4. Sensitivity sweep -> 1 PNG + milp_sensitivity.csv
  5. KPI comparison (A: All L0, B: All L1, C: MILP) -> 2 PNGs + scenario_comparison.csv
  6. Print executive summary to console
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # nosec
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # nosec

from src import config
from src.data_loader import load, make_rdd_arrays
from src.evaluation import metrics
from src.milp import optimizer, sensitivity
from src.rdd import bandwidth as bw_mod
from src.rdd import (
    covariate_balance,
    density_test,
)
from src.rdd import local_linear as rdd_ll
from src.rdd import placebo as rdd_placebo
from src.viz import kpi_plots, milp_plots, rdd_plots


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _ensure_dataset() -> None:
    from data.build_dataset import build

    res = build()
    LOG.info("Dataset ready: %s rows from %s", f"{res.n_rows:,}", res.source)


def _run_rdd(df: pd.DataFrame) -> None:
    LOG.info("=== RDD ANALYSIS ===")
    R, Y, _ = make_rdd_arrays(df, cutoff=config.CUTOFF)
    LOG.info("Running on %s observations, cutoff=%s", f"{len(R):,}", config.CUTOFF)
    bw = bw_mod.ik_bandwidth(R, Y, cutoff=config.CUTOFF)
    LOG.info(
        "Bandwidths: left=%.1f, right=%.1f, pilot=%.1f",
        bw["h_left"],
        bw["h_right"],
        bw["h_global"],
    )
    res = rdd_ll.local_linear_rdd(R, Y, config.CUTOFF, bw["h_left"], bw["h_right"])
    LOG.info(
        "τ̂ = $%.2f  (95%% CI [%.2f, %.2f], p=%.4g, n_eff=%.0f)",
        res.tau_hat,
        res.ci_low,
        res.ci_high,
        res.p_value,
        res.n_effective,
    )

    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rdd_plots.plot_binned_scatter(
        R,
        Y,
        config.CUTOFF,
        bw["h_left"],
        res,
        config.FIG_DIR / "rdd_binned_scatter.png",
    )

    rdd_plots.plot_bandwidth_sensitivity(
        R,
        Y,
        config.CUTOFF,
        base_h=bw["h_left"],
        output_path=config.FIG_DIR / "rdd_bandwidth_sensitivity.png",
    )

    placebos_ik = rdd_placebo.run_placebo(R, Y, config.PLACEBO_CUTOFFS)
    rdd_plots.plot_placebo(
        R,
        Y,
        placebos_ik,
        config.CUTOFF,
        res.tau_hat,
        res.ci_low,
        res.ci_high,
        config.FIG_DIR / "rdd_placebo.png",
    )

    dens = density_test.density_test(R, config.CUTOFF, n_bins=20, h_fraction=1.0)
    rdd_plots.plot_density(R, config.CUTOFF, dens, config.FIG_DIR / "rdd_density.png")

    cov_dict = {
        "loan_amnt": df["loan_amnt"].to_numpy(dtype=float),
        "int_rate": df["int_rate"].to_numpy(dtype=float),
        "dti": df["dti"].to_numpy(dtype=float),
        "outstanding_balance": df["outstanding_balance"].to_numpy(dtype=float),
    }
    cov_res = covariate_balance.covariate_balance_test(R, cov_dict, config.CUTOFF)
    rdd_plots.plot_covariate_balance(cov_res, config.FIG_DIR / "rdd_covariate_balance.png")

    rdd_summary = {
        "cutoff": res.cutoff,
        "bandwidth_left": res.bandwidth_left,
        "bandwidth_right": res.bandwidth_right,
        "tau_hat": res.tau_hat,
        "se_tau": res.se_tau,
        "ci_low": res.ci_low,
        "ci_high": res.ci_high,
        "p_value": res.p_value,
        "n_effective": res.n_effective,
        "n_left": res.n_left,
        "n_right": res.n_right,
    }
    pd.DataFrame([rdd_summary]).to_csv(config.REPORT_DIR / "rdd_results.csv", index=False)

    cov_rows = [
        {"covariate": r.covariate, "tau_hat": r.tau_hat, "se_tau": r.se_tau, "p_value": r.p_value}
        for r in cov_res.values()
    ]
    pd.DataFrame(cov_rows).to_csv(config.REPORT_DIR / "rdd_covariate_balance.csv", index=False)

    placebo_rows = [
        {
            "cutoff": p.cutoff,
            "tau_hat": p.tau_hat,
            "se_tau": p.se_tau,
            "ci_low": p.ci_low,
            "ci_high": p.ci_high,
        }
        for p in placebos_ik
    ]
    pd.DataFrame(placebo_rows).to_csv(config.REPORT_DIR / "rdd_placebo.csv", index=False)


def _run_milp(df: pd.DataFrame) -> tuple[optimizer.OptimizationResult, pd.DataFrame]:
    LOG.info("=== MILP OPTIMIZATION ===")
    res, assignments = optimizer.optimize(
        df, budget=config.TOTAL_BUDGET, log=False, time_limit_sec=120
    )
    LOG.info(
        "MILP status=%s  objective=$%.2f  assigned=%s/%s  total cost=$%.2f  expected recovery=$%.2f",
        res.status,
        res.objective,
        f"{res.n_assigned:,}",
        f"{len(df):,}",
        res.cost_total,
        res.expected_recovery,
    )
    LOG.info("Channel distribution: %s", res.channel_distribution)
    LOG.info("Tier distribution: %s", res.tier_distribution)

    assignments.to_csv(config.REPORT_DIR / "milp_assignments.csv", index=False)

    milp_plots.plot_allocation_heatmap(assignments, config.FIG_DIR / "milp_allocation.png")
    milp_plots.plot_tier_breakdown(assignments, config.FIG_DIR / "milp_tier_breakdown.png")
    return res, assignments


def _run_sensitivity(df: pd.DataFrame) -> None:
    LOG.info("=== MILP SENSITIVITY SWEEP ===")
    sens_df = sensitivity.sensitivity_sweep(
        df,
        budgets=[config.TOTAL_BUDGET * f for f in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)],
        capacity_scales=[0.5, 1.0, 1.5],
        time_limit_sec=30,
    )
    sens_df.to_csv(config.REPORT_DIR / "milp_sensitivity.csv", index=False)
    milp_plots.plot_pareto(sens_df, config.FIG_DIR / "milp_pareto.png")


def _run_kpis(df: pd.DataFrame, milp_res, milp_assign) -> None:
    LOG.info("=== KPI COMPARISON ===")
    comp = metrics.build_comparison_table(
        df, milp_res.cost_total, milp_res.expected_recovery, milp_assign
    )
    comp.to_csv(config.REPORT_DIR / "scenario_comparison.csv", index=False)
    LOG.info("\n%s", comp.to_string(index=False))

    kpi_plots.plot_kpi_summary(comp, config.FIG_DIR / "kpi_summary.png")
    kpi_plots.plot_waterfall_improvement(comp, config.FIG_DIR / "kpi_improvement_waterfall.png")


def main() -> None:
    _setup_logging()
    t0 = time.time()
    _ensure_dataset()
    df = load()
    LOG.info("Loaded %s rows, %s columns", f"{len(df):,}", df.shape[1])

    _run_rdd(df)
    milp_res, milp_assign = _run_milp(df)
    _run_sensitivity(df)
    _run_kpis(df, milp_res, milp_assign)

    elapsed = time.time() - t0
    LOG.info("Pipeline complete in %.1fs", elapsed)
    _print_summary(df, milp_res, milp_assign, elapsed)


def _print_summary(df, milp_res, milp_assign, elapsed):
    print("\n" + "=" * 78)
    print("COLLECTIONS OPTIMIZATION - EXECUTIVE SUMMARY")
    print("=" * 78)
    comp = pd.read_csv(config.REPORT_DIR / "scenario_comparison.csv")
    rdd_df = pd.read_csv(config.REPORT_DIR / "rdd_results.csv").iloc[0]
    tau = float(rdd_df["tau_hat"])
    ci_low = float(rdd_df["ci_low"])
    ci_high = float(rdd_df["ci_high"])
    p = float(rdd_df["p_value"])
    print(f"\nPortfolio      : {len(df):,} charged-off accounts")
    print(f"Outreach cutoff: c = ${float(rdd_df['cutoff']):.0f} expected recovery score")
    print(
        f"RDD causal lift: tau = ${tau:.1f} per account "
        f"(95% CI [${ci_low:.1f}, ${ci_high:.1f}], p={p:.3g})"
    )
    print(
        f"MILP optimizer : status = {milp_res.status}, "
        f"assigned {milp_res.n_assigned:,} of {len(df):,} accounts"
    )
    print()
    print(comp.to_string(index=False))
    print(f"\nOutputs saved to {config.OUTPUT_DIR}")
    print(f"Total runtime: {elapsed:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    LOG = logging.getLogger("pipeline")
    main()
