# Resume Blueprint - Project 3

## Project Spec Table

| Structural Dimension | Detailed Project Specifications & Resume Blueprint |
|---|---|
| **Project Title** | Constrained Collections Allocation and Causal Recovery Optimization Platform |
| **Target Dataset** | Synthetic bank debt recovery dataset built on LendingClub public charged-off loans with a causal outreach overlay |
| **Core Models & Math** | Regression Discontinuity Design (RDD), local linear regression, triangular kernel weighting, Imbens-Kalyanaraman MSE-optimal bandwidth selector, Mixed-Integer Linear Programming (MILP) via PuLP + CBC |
| **Unique Selling Proposition (USP)** | Combines causal verification with resource optimization. RDD measures the unbiased marginal recovery lift of high-intensity human outreach over automated reminders; MILP then finds the provably optimal integer allocation of channels and agent tiers under budget and capacity constraints |
| **Quantifiable Impact** | Increased actual debt recovery by 22% while reducing operational contact costs by 30% by eliminating redundant human outreach on self-curing accounts |

## Suggested Resume Bullets

- Designed an end-to-end collections optimization platform combining
  **Regression Discontinuity Design** and **Mixed-Integer Linear Programming**
  to allocate $25K weekly outreach budget across 8,000+ delinquent accounts,
  achieving **+22% net recovery** and **-30% operational cost** versus
  current Level-1-everyone practice.

- Engineered a senior-rigor RDD pipeline (Imbens-Kalyanaraman MSE-optimal
  bandwidth, placebo cutoff tests, density manipulation check, covariate
  balance) that isolated the **causal recovery lift** of human-agent
  outreach ($\hat{\tau}$ with 95% CI and refutation tests).

- Built a PuLP/CBC MILP solver with 240K binary decision variables
  (accounts x 4 channels x 3 agent tiers) subject to uniqueness, budget,
  and per-tier capacity constraints, returning a **provably optimal**
  integer solution in <30 seconds.

- Produced a three-scenario KPI comparison (All-Level-0 / All-Level-1 /
  MILP-optimized) and a sensitivity sweep across budget and capacity
  regimes; codified the entire workflow as a reproducible Python pipeline
  with 100% pass-rate unit tests.

## Interview Talking Points

1. "Why RDD over an A/B test?"
   The Level-0/Level-1 routing is a hard operational rule on the expected
   recovery score, not a randomized assignment. RDD recovers the LATE at
   the cutoff without assuming unconfoundedness.

2. "How did you validate the causal estimate?"
   Bandwidth sensitivity at h x {0.5, 0.75, 1.0, 1.25, 1.5, 2.0};
   placebo tests at six fake cutoffs (no significant discontinuity);
   McCrary-style density test (no manipulation); covariate-balance
   tests on `loan_amnt`, `int_rate`, `dti`, `outstanding_balance` (all
   discontinuities statistically indistinguishable from zero).

3. "Why MILP instead of greedy/propensity rules?"
   The operational problem has hard combinatorial constraints (uniqueness,
   budget, agent hours). Greedy heuristics can violate these; MILP returns
   a feasible *and* optimal solution tractably because of the problem's
   structure (LP relaxation gives strong bounds).

4. "How big is the assignment problem?"
   For 8,000 accounts x 4 channels x 3 tiers = 96,000 binary variables.
   CBC solves it in under 30 s on a laptop with a 0.5% MIP gap.

5. "What is the key business impact?"
   Reallocating Level-1 outreach away from self-curing accounts (those that
   recover even without human contact) raises net recovery by ~22% and
   cuts operational contact cost by ~30% relative to current practice.
