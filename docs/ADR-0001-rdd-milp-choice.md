# ADR-0001: Combining Regression Discontinuity Design with Mixed-Integer Linear Programming

- **Status:** Accepted
- **Date:** 2026-07-21
- **Decision-makers:** Piyush Utkar

## Context

Banks run tiered collection campaigns and want to know (a) whether human
outreach causally improves recovery and (b) how to allocate finite budget
and agent capacity across millions of delinquent accounts. The two
questions are usually answered separately by analytics and operations
teams, which creates a hand-off gap: causal estimates never inform
operational allocation.

## Decision

We couple **Regression Discontinuity Design (RDD)** with **Mixed-Integer
Linear Programming (MILP)** in a single pipeline.

- **RDD** exploits the operational cutoff at `expected_recovery_score = 1000`
  to estimate the unbiased marginal recovery lift `τ̂` of human outreach.
  Implementation uses Imbens-Kalyanaraman bandwidth selection, local linear
  regression, and a triangular kernel, with robustness checks (placebo
  cutoffs, density test, covariate balance).
- **MILP** uses `τ̂` and per-borrower payment-propensity scores as inputs
  to a constrained 0/1 optimization that selects channel × agent-tier for
  every account subject to budget and capacity.

## Alternatives considered

1. **A/B test on top of the cutoff** — randomizes outreach within the
   marginal zone. Cleaner causally but requires holding back collection
   actions, which is operationally and ethically costly in a regulated
   collections context. **Rejected.**
2. **Pure uplift modeling + greedy thresholding** — easier to deploy but
   does not respect joint budget/capacity constraints and is not
   causally identified. **Rejected.**
3. **Propensity-score matching + hand-tuned rules** — what the business
   was doing before. Identifies correlation, not causation, and cannot
   adapt to changing capacity. **Rejected as baseline.**

## Consequences

- ✅ Causal identification (RDD) and operational optimization (MILP) in
  one pipeline.
- ✅ Jointly optimal allocations under realistic constraints.
- ✅ Reuses existing operational data — no randomized experiment required.
- ⚠️ RDD's external validity is bounded to the local neighborhood of the
  cutoff. The MILP can extrapolate further, but `τ̂` itself does not.
- ⚠️ The synthetic dataset in this repo has an injected uplift of ~$180;
  reviewers should re-fit on real lender data before relying on the
  point estimate.

## References

- Imbens, G. & Kalyanaraman, K. (2012). *Optimal bandwidth choice for
  RDD.* Review of Economic Studies.
- Hahn, J., Todd, P. & Van der Klaauw, W. (2001). *Identification and
  estimation of treatment effects with a RDD.* Econometrica.
- McCrary, J. (2008). *Manipulation of the running variable in the RDD.*
  Journal of Econometrics.
- See `references.md` for the full bibliography.