"""MILP visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_allocation_heatmap(assignments: pd.DataFrame, output_path):
    pivot = (
        assignments.groupby(["channel", "agent_tier"]).size().unstack(fill_value=0)
    )
    for tier in ["Junior", "Senior", "Specialist"]:
        if tier not in pivot.columns:
            pivot[tier] = 0
    pivot = pivot[["Junior", "Senior", "Specialist"]]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=11)
    ax.set_xlabel("Agent Tier", fontsize=12)
    ax.set_ylabel("Channel", fontsize=12)
    ax.set_title("MILP-optimal allocation (account counts)", fontsize=12)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, int(pivot.iloc[i, j]),
                    ha="center", va="center",
                    color="black" if pivot.iloc[i, j] < pivot.values.max() * 0.6 else "white",
                    fontsize=11)
    fig.colorbar(im, ax=ax, label="Accounts assigned")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_pareto(sensitivity_df: pd.DataFrame, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for cs in sorted(sensitivity_df["capacity_scale"].unique()):
        sub = sensitivity_df[sensitivity_df["capacity_scale"] == cs].sort_values("budget")
        ax.plot(sub["budget"], sub["net_recovery"],
                "o-", lw=2, ms=8, label=f"Capacity scale = {cs:.2f}")
    ax.set_xlabel("Total Operational Budget ($)", fontsize=12)
    ax.set_ylabel("Net Recovery (Recovery $- Cost $)", fontsize=12)
    ax.set_title("Pareto frontier — net recovery vs. budget", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_tier_breakdown(assignments: pd.DataFrame, output_path):
    by_tier = assignments.groupby("agent_tier").agg(
        n=("account_id", "count"),
        cost=("cost_incurred", "sum"),
        expected=("expected_recovery", "sum"),
    ).reset_index()
    by_tier["net"] = by_tier["expected"] - by_tier["cost"]
    by_tier = by_tier.sort_values("net", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(by_tier))
    w = 0.35
    ax.bar(x - w / 2, by_tier["cost"], width=w, color="#4575b4",
           label="Operational Cost ($)")
    ax.bar(x + w / 2, by_tier["net"], width=w, color="#1a9850",
           label="Net Recovery ($)")
    ax.set_xticks(x)
    ax.set_xticklabels(by_tier["agent_tier"], fontsize=11)
    ax.set_xlabel("Agent Tier", fontsize=12)
    ax.set_ylabel("Amount ($)", fontsize=12)
    ax.set_title("Cost vs. Net Recovery per Agent Tier", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    for i, (c, n) in enumerate(zip(by_tier["cost"], by_tier["net"])):
        ax.text(i - w / 2, c + max(by_tier["cost"]) * 0.01,
                f"${c:,.0f}", ha="center", fontsize=9)
        ax.text(i + w / 2, n + max(by_tier["cost"]) * 0.01,
                f"${n:,.0f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
