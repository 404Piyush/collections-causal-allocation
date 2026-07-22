"""RDD visualizations."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .. import config
from ..rdd import density_test, local_linear, placebo


def _triangular_kernel(u: np.ndarray, h: float) -> np.ndarray:
    return np.asarray(np.maximum(0.0, 1.0 - np.abs(u) / max(float(h), 1e-9)))


def plot_binned_scatter(
    R: np.ndarray,
    Y: np.ndarray,
    cutoff: float,
    h: float,
    result: local_linear.RDDResult,
    output_path,
):
    n_bins = config.RDD_N_BINS
    edges = np.linspace(R.min(), R.max(), n_bins + 1)
    bin_idx = np.digitize(R, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_means_Y = np.zeros(n_bins)
    bin_means_R = np.zeros(n_bins)
    bin_n = np.zeros(n_bins)
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.sum() >= 5:
            bin_means_Y[b] = Y[sel].mean()
            bin_means_R[b] = R[sel].mean()
            bin_n[b] = sel.sum()

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sizes = np.clip(bin_n, 5, 200)
    ax.scatter(
        bin_means_R[bin_n > 0],
        bin_means_Y[bin_n > 0],
        s=sizes[bin_n > 0],
        c="#4575b4",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
        label="Bin means (size = n)",
    )

    R_grid = np.linspace(cutoff - h, cutoff, 200)
    pred_left = result.beta0 + result.beta1_left * (R_grid - cutoff)
    ax.plot(R_grid, pred_left, color="#d73027", lw=2.5, label=f"Left fit (h={h:.0f})")

    R_grid2 = np.linspace(cutoff, cutoff + h, 200)
    pred_right = result.beta0 + result.tau_hat + result.beta1_right * (R_grid2 - cutoff)
    ax.plot(R_grid2, pred_right, color="#1a9850", lw=2.5, label=f"Right fit (h={h:.0f})")

    ax.axvline(cutoff, color="black", ls="--", lw=1)
    ymin, ymax = ax.get_ylim()
    ymin = np.nanmin(Y) - 5
    ymax = np.nanmax(Y) + 5
    ax.set_ylim(min(ymin, -50), max(ymax, 50))
    ax.set_xlabel("Expected Recovery Score R", fontsize=12)
    ax.set_ylabel("Actual Recovery Amount ($)", fontsize=12)
    ax.set_title(
        f"RDD at cutoff c = {cutoff:.0f}  |  τ̂ = ${result.tau_hat:.1f}  "
        f"(95% CI ${result.ci_low:.1f}, ${result.ci_high:.1f},  p={result.p_value:.3f})",
        fontsize=12,
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_bandwidth_sensitivity(
    R: np.ndarray,
    Y: np.ndarray,
    cutoff: float,
    base_h: float,
    output_path,
    multipliers: Sequence[float] = config.RDD_BANDWIDTH_GRID,
):
    h_left_grid = base_h * np.array(multipliers)
    h_right_grid = base_h * np.array(multipliers)

    rows = []
    for hl, hr in zip(h_left_grid, h_right_grid, strict=True):
        res = local_linear.local_linear_rdd(R, Y, cutoff, hl, hr)
        rows.append((hl, res.tau_hat, res.ci_low, res.ci_high, res.p_value))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    hs = [r[0] for r in rows]
    taus = [r[1] for r in rows]
    ci_l = [r[2] for r in rows]
    ci_h = [r[3] for r in rows]
    ax.plot(hs, taus, "o-", lw=2, color="#d73027", label="τ̂ estimate")
    ax.fill_between(hs, ci_l, ci_h, alpha=0.20, color="#d73027", label="95% CI")
    ax.axhline(0, color="black", lw=1, ls=":")
    ax.axvline(base_h, color="black", lw=1.5, ls="--", label=f"Optimal bandwidth h = {base_h:.0f}")
    ax.set_xlabel("Bandwidth h (applied symmetrically)", fontsize=12)
    ax.set_ylabel("Causal Recovery Lift τ̂ ($)", fontsize=12)
    ax.set_title("Bandwidth sensitivity — robustness of τ̂ to bandwidth choice", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

    return rows


def plot_placebo(
    R: np.ndarray,
    Y: np.ndarray,
    placebos: list[placebo.PlaceboResult],
    real_cutoff: float,
    real_tau: float,
    real_ci_low: float,
    real_ci_high: float,
    output_path,
):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cs = [p.cutoff for p in placebos]
    ts = [p.tau_hat for p in placebos]
    lcl = [p.ci_low for p in placebos]
    ucl = [p.ci_high for p in placebos]
    ax.errorbar(
        cs,
        ts,
        yerr=[np.array(ts) - np.array(lcl), np.array(ucl) - np.array(ts)],
        fmt="o",
        color="#4575b4",
        ecolor="#4575b4",
        elinewidth=1.5,
        capsize=4,
        ms=9,
        label="Placebo cutoffs",
    )
    ax.axhline(0, color="black", lw=1, ls=":")
    ax.axvline(
        real_cutoff, color="#d73027", lw=2, ls="--", label=f"Real cutoff c={real_cutoff:.0f}"
    )
    ax.fill_betweenx(
        [-50, 50],
        real_cutoff - 5,
        real_cutoff + 5,
        alpha=0.10,
        color="#d73027",
    )
    ax.set_xlabel("Fake cutoff value (placebo)", fontsize=12)
    ax.set_ylabel("Placebo τ̂", fontsize=12)
    ax.set_title("Placebo test — only the true cutoff should show a discontinuity", fontsize=12)
    ax.set_ylim(min(min(lcl) - 5, -20), max(max(ucl) + 5, 20))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_density(
    R: np.ndarray,
    cutoff: float,
    density: density_test.DensityTestResult,
    output_path,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    width = (density.bins[1] - density.bins[0]) * 0.45
    centers = (density.bins[:-1] + density.bins[1:]) / 2
    ax.bar(
        centers - width / 2,
        density.counts_left,
        width=width,
        color="#4575b4",
        alpha=0.75,
        label=f"Left  (n={density.n_left})",
    )
    ax.bar(
        centers + width / 2,
        density.counts_right,
        width=width,
        color="#d73027",
        alpha=0.75,
        label=f"Right (n={density.n_right})",
    )
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("R - cutoff", fontsize=12)
    ax.set_ylabel("Bin counts", fontsize=12)
    ax.set_title(
        f"Density (manipulation) test around cutoff — χ²={density.chi2:.2f}, "
        f"p={density.p_value:.3f}  (no manipulation expected)",
        fontsize=12,
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_covariate_balance(cov_results, output_path):
    names = list(cov_results.keys())
    taus = [cov_results[n].tau_hat for n in names]
    ses = [cov_results[n].se_tau for n in names]
    ps = [cov_results[n].p_value for n in names]

    fig, ax = plt.subplots(figsize=(10, max(4.0, len(names) * 0.6)))
    ypos = np.arange(len(names))
    ax.errorbar(
        taus,
        ypos,
        xerr=[1.96 * s for s in ses],
        fmt="o",
        color="#4575b4",
        ecolor="#4575b4",
        elinewidth=2,
        capsize=3,
        ms=8,
    )
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Covariate discontinuity τ̂ at cutoff (95% CI)", fontsize=12)
    ax.set_title("Covariate balance at cutoff — should be ≈ 0", fontsize=12)
    for i, p in enumerate(ps):
        ax.text(
            max(taus[i] + 1.96 * ses[i], 0.01),
            i,
            f"p={p:.3f}",
            va="center",
            fontsize=9,
            color="gray",
        )
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
