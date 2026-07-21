"""KPI dashboard visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_kpi_summary(comparison_df: pd.DataFrame, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    scens = comparison_df["scenario"].tolist()
    colors = ["#4575b4", "#d73027", "#1a9850"]

    def _bar(ax, values, title, ylabel, fmt="${:,.0f}"):
        bars = ax.bar(scens, values, color=colors[: len(scens)])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=9, rotation=15)
        for bar, v in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                fmt.format(v),
                ha="center",
                fontsize=9,
            )

    _bar(axes[0, 0], comparison_df["actual_recovery_total_usd"], "Actual Recovery Total", "USD")
    _bar(axes[0, 1], comparison_df["cost_total_usd"], "Operational Cost", "USD")
    _bar(axes[1, 0], comparison_df["net_recovery_usd"], "Net Recovery", "USD")
    _bar(
        axes[1, 1],
        comparison_df["cost_per_dollar_recovered"],
        "Cost per $1 Recovered",
        "USD/$",
        "${:.3f}",
    )

    fig.suptitle("Business KPI comparison across outreach strategies", fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_waterfall_improvement(comparison_df: pd.DataFrame, output_path):
    if not any(comparison_df["scenario"].str.contains("All Level 1")):
        return
    baseline = float(
        comparison_df.loc[
            comparison_df["scenario"].str.contains("All Level 1"), "net_recovery_usd"
        ].iloc[0]
    )
    milp = float(
        comparison_df.loc[comparison_df["scenario"].str.contains("MILP"), "net_recovery_usd"].iloc[
            0
        ]
    )
    delta = milp - baseline
    pct = (milp / baseline - 1.0) * 100.0 if baseline > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    x = ["Current practice\n(All Level 1)", "MILP-optimized\n(MILP)"]
    y = [baseline, milp]
    bars = ax.bar(x, y, color=["#d73027", "#1a9850"], width=0.45)
    for bar, v in zip(bars, y, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + abs(baseline) * 0.01,
            f"${v:,.0f}",
            ha="center",
            fontsize=11,
        )
    ax.annotate(
        "",
        xy=(1, milp),
        xytext=(0, baseline),
        arrowprops={"arrowstyle": "->", "lw": 2, "color": "black"},
    )
    ax.text(
        0.5,
        (baseline + milp) / 2,
        f"+${delta:,.0f}  (+{pct:.1f}%)",
        ha="center",
        fontsize=12,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#fef6b8", "edgecolor": "black"},
    )
    ax.set_ylabel("Net Recovery ($)", fontsize=12)
    ax.set_title("MILP Impact vs. Current Outreach Practice", fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
