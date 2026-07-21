# Constrained Collections Allocation and Causal Recovery Optimization Platform

> **Project 3** of the *Retail Credit Analytics* portfolio — operational debt
> collection allocation combining **Regression Discontinuity Design (RDD)**
> for causal recovery lift measurement with **Mixed-Integer Linear
> Programming (MILP)** for resource-constrained outreach optimization.

---

## 1. Business Problem

Indian retail-finance lenders run tiered collection campaigns — automated
dialers (Level 0) and human-agent outreach (Level 1). Level 1 costs ~$50 more
per account than Level 0 but yields higher recovery. The central operational
question is:

> *Which accounts truly benefit from human outreach, and how do we allocate a
> limited budget and finite agent capacity to maximize net recovery?*

This project answers it in two stages:

1. **Causal verification (RDD):** exploit the operational cutoff at
   `expected_recovery_score = 1000` to measure the unbiased marginal recovery
   lift of human outreach.
2. **Optimal allocation (MILP):** given a finite budget and agent capacities,
   maximize net recovery while respecting every operational constraint.

---

## 2. Mathematical Formulation

### 2.1 RDD — Local linear regression at the cutoff

Accounts crossing the operational threshold `c = 1000` receive Level 1
outreach. We fit a local-linear regression with a triangular kernel and
separate MSE-optimal bandwidths on each side of the cutoff:

    Y_i = beta_0 + beta_1 (R_i - c) + tau * T_i + beta_3 (R_i - c) * T_i + e_i

with the triangular kernel

    K(R_i, c, h) = max(0, 1 - |R_i - c| / h)

`tau_hat` is the **causal recovery lift** of human outreach (Local Average
Treatment Effect at the cutoff).

Bandwidth is chosen via an Imbens-Kalyanaraman-style selector with a
Silverman fallback. Robustness is checked against (a) a bandwidth multiplier
grid, (b) placebo cutoffs, (c) a density / manipulation test, and
(d) covariate-balance tests.

### 2.2 MILP - Constrained allocation

Let `I` = accounts, `J` = channels {SMS, Email, Phone, FieldVisit},
`K` = agent tiers {Junior, Senior, Specialist}. Decision variable:

    x[i,j,k] in {0, 1}   1 if account i uses channel j with tier k

**Objective - maximize net recovery:**

    max sum_{i,j,k} ( P[i,j,k] * Bal[i] - C[j,k] ) * x[i,j,k]

**Constraints:**

| Constraint | Formula |
|---|---|
| At most one config per account | sum_{j,k} x[i,j,k] <= 1 for all i |
| Operational budget cap | sum_{i,j,k} C[j,k] * x[i,j,k] <= Budget |
| Agent capacity (hours to minutes) | sum_{i,j} Time_j * x[i,j,k] <= Cap_k * 60 for all k |

Solved with **PuLP + CBC**.

---

## 3. Repository Layout

```
collections-optimization/
|-- README.md
|-- requirements.txt
|-- run_pipeline.py             # Single entry point
|-- data/
|   |-- build_dataset.py        # LC snapshot (preferred) -> synthetic overlay
|   |-- raw/                    # downloaded LC files
|   `-- processed/              # collections_dataset.csv
|-- src/
|   |-- config.py               # cutoffs, costs, capacities, seeds
|   |-- data_loader.py
|   |-- rdd/                    # bandwidth, local_linear, placebo, density, covariate
|   |-- milp/                   # optimizer, sensitivity
|   |-- evaluation/             # metrics.py (3-scenario KPI table)
|   `-- viz/                    # matplotlib helpers
|-- tests/                      # pytest suite
|-- outputs/
|   |-- figures/                # all PNGs
|   `-- reports/                # all CSVs
`-- docs/
    `-- resume_blueprint.md     # copy-paste CV entry
```

---

## 4. Installation

```bash
python -m pip install -r requirements.txt
```

Tested with **Python 3.11+** (the CBC solver is bundled with PuLP; no extra
system dependencies required).

---

## 5. Quickstart

```bash
# Step 1 - generate / refresh dataset (auto-falls back to synthetic if offline)
python run_pipeline.py

# Step 2 - inspect outputs
ls outputs/figures/    # PNG charts
ls outputs/reports/    # KPI / RDD / MILP CSVs

# Step 3 - run unit tests
python -m pytest tests/ -v
```

The first run:

1. Tries `https://files.lendingclub.com/Loan_status_2018Q*.csv.zip`. If the
   host is unreachable it synthesizes a LendingClub-calibrated dataset of
   ~8,000 charged-off accounts.
2. Filters / augments with a synthetic outreach overlay (expected recovery
   score, channel, agent tier, causal recovery uplift of ~$180).
3. Runs RDD with senior-level robustness checks.
4. Solves the MILP and a budget / capacity sensitivity sweep.
5. Compares the three outreach scenarios and writes the summary table.

---

## 6. Outputs

### 6.1 `outputs/figures/`

| File | What it shows |
|---|---|
| `rdd_binned_scatter.png` | Binned means + LLR fits on each side; tau labeled with CI and p |
| `rdd_bandwidth_sensitivity.png` | tau at h x {0.5, 0.75, 1.0, 1.25, 1.5, 2.0} |
| `rdd_placebo.png` | tau at fake cutoffs - only c=1000 should show a discontinuity |
| `rdd_density.png` | Histogram on each side of cutoff (manipulation check) |
| `rdd_covariate_balance.png` | Pre-treatment covariates discontinuity at cutoff |
| `milp_allocation.png` | Heatmap of channel x tier assignments |
| `milp_pareto.png` | Net recovery vs. budget frontier across capacity scales |
| `milp_tier_breakdown.png` | Cost vs. net recovery per agent tier |
| `kpi_summary.png` | Recovery / cost / net / cost-per-$ across the 3 scenarios |
| `kpi_improvement_waterfall.png` | MILP gain over current practice |

### 6.2 `outputs/reports/`

| File | Contents |
|---|---|
| `rdd_results.csv` | cutoff, bandwidths, tau, SE, CI, p-value, effective n |
| `rdd_placebo.csv` | placebo cutoff estimates + CI |
| `rdd_covariate_balance.csv` | covariate discontinuity tests |
| `milp_assignments.csv` | per-account (channel, tier, cost, p, expected recovery) |
| `milp_sensitivity.csv` | budget x capacity sweep |
| `scenario_comparison.csv` | three-scenario KPI table with % improvement vs current |

---

## 7. Methodology Notes

**Why RDD instead of A/B?** The Level 0/Level 1 routing rule is a hard
administrative threshold, not a randomized assignment. RDD isolates the
unbiased marginal effect of the policy at the threshold using only the
local sub-population near `c = 1000` - no unconfoundedness assumption
required.

**Why MILP instead of propensity scoring or rules?** Once we trust the
causal lift, we want the cheapest feasible allocation. The MILP encodes
every real-world constraint (uniqueness, budget, capacity, channel time)
and returns a provably optimal integer solution.

**Why LendingClub?** LC charge-off data is publicly available, well-curated
and exhibits realistic recovery mechanics. When network access is blocked,
the pipeline falls back to synthetic-but-calibrated data so the analysis
remains fully reproducible offline.

---

## 8. Resume Blueprint

See [`docs/resume_blueprint.md`](docs/resume_blueprint.md) for the
copy-paste CV/portfolio entry.

---

## 9. License

MIT.
