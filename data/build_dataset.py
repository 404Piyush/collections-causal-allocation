"""
Build / refresh the collections dataset.

Source preference order:
  1. LendingClub accepted-loans public snapshot (downloaded to data/raw/).
  2. If download fails (offline / blocked), falls back to a fully synthetic
     dataset whose marginal distributions are calibrated to publicly known
     LendingClub charge-off statistics.

The processed file is augmented with the synthetic outreach overlay that the
RDD and MILP modules consume:
  - expected_recovery_score   R_i  (running variable)
  - outreach_tier             T_i  in {0=Level 0 automated, 1=Level 1 human}
  - channel                   j    in {SMS, Email, Phone, FieldVisit}
  - agent_tier                k    in {Junior, Senior, Specialist}
  - actual_recovery_amount    Y_i
  - cost_incurred             C_i
  - predicted_payment_prob    P_ijk  per (channel x tier)
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

LOG = logging.getLogger("build_dataset")

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_URLS = [
    "https://files.lendingclub.com/Loan_status_2018Q4.csv.zip",
    "https://files.lendingclub.com/Loan_status_2018Q3.csv.zip",
    "https://files.lendingclub.com/Loan_status_2018Q2.csv.zip",
    "https://files.lendingclub.com/Loan_status_2018Q1.csv.zip",
    "https://files.lendingclub.com/Loan_status_2017Q4.csv.zip",
]

CUTOFF_SCORE = 1000.0
TREATMENT_UPLIFT_DOLLARS = 180.0  # average causal recovery lift of human outreach
SEED = 20260521


@dataclass
class BuildResult:
    output_path: Path
    n_rows: int
    source: str


def _try_download_lendingclub(timeout: int = 4) -> Optional[Path]:
    LOG.info("Attempting LendingClub download (max %ss each)...", timeout)
    for url in CANDIDATE_URLS:
        target = RAW_DIR / Path(url).name
        if target.exists():
            LOG.info("Cache hit: %s", target.name)
            return target
        try:
            LOG.info("Trying %s", url)
            r = requests.get(
                url, timeout=(timeout, timeout), stream=True,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if r.status_code != 200:
                LOG.warning("HTTP %s on %s", r.status_code, url)
                r.close()
                continue
            with open(target, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
            LOG.info("Downloaded %s (%s MB)",
                     target.name, round(target.stat().st_size / 1e6, 1))
            return target
        except Exception as exc:
            LOG.warning("Skipping %s: %s", url.split("/")[-1], exc)
            continue
    return None


def _extract_chargeoff(csv_path: Path) -> pd.DataFrame:
    cols = [
        "loan_amnt", "term", "int_rate", "grade", "sub_grade",
        "annual_inc", "dti", "issue_d", "loan_status",
        "total_rec_prncp", "total_pymnt", "recoveries",
        "last_pymnt_amnt", "last_pymnt_d", "emp_length",
    ]
    LOG.info("Reading %s and filtering to Charged Off...", csv_path.name)
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False, dtype={"term": "string"})
    df = df[df["loan_status"].fillna("") == "Charged Off"].copy()
    if df.empty:
        raise RuntimeError("LendingClub file had no Charged Off rows")
    LOG.info("Filtered to %s charged-off loans", f"{len(df):,}")
    return df


def _synthetic_lendingclub_like(n: int, rng: np.random.Generator) -> pd.DataFrame:
    LOG.info("Generating %s synthetic charge-off records (LC-calibrated)...", f"{n:,}")
    grades = list("ABCDEFG")
    grade_probs = np.array([0.04, 0.10, 0.18, 0.22, 0.20, 0.14, 0.12])
    grade_probs = grade_probs / grade_probs.sum()
    grade = rng.choice(grades, size=n, p=grade_probs)

    grade_int_lo = {"A": 0.060, "B": 0.085, "C": 0.115, "D": 0.150,
                    "E": 0.180, "F": 0.225, "G": 0.275}
    grade_int_jitter = {"A": 0.025, "B": 0.030, "C": 0.035, "D": 0.040,
                        "E": 0.045, "F": 0.050, "G": 0.055}

    int_rate = np.empty(n)
    for g in grades:
        mask = grade == g
        lo, jit = grade_int_lo[g], grade_int_jitter[g]
        int_rate[mask] = lo + jit * rng.random(mask.sum())

    loan_amnt = np.round(np.clip(rng.lognormal(mean=9.55, sigma=0.55, size=n), 1000, 40000) / 50) * 50
    annual_inc = np.clip(rng.lognormal(mean=10.95, sigma=0.55, size=n), 8000, 500000)
    dti = np.clip(rng.normal(loc=18, scale=8, size=n), 0, 60)
    term = rng.choice([" 36 months", " 60 months"], size=n, p=[0.75, 0.25])
    emp_length = rng.choice(
        ["< 1 year", "1 year", "2 years", "3 years", "4 years",
         "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"],
        size=n,
    )
    issue_year = rng.integers(2012, 2017, size=n)
    issue_month = rng.integers(1, 13, size=n)
    issue_d = [f"{m:02d}-{y}" for m, y in zip(issue_month, issue_year)]

    total_paid_frac = np.clip(
        rng.beta(2.5, 5.5, size=n)
        - 0.10 * np.array([grade_int_lo[g] for g in grade]),
        0.02, 0.92,
    )
    total_rec_prncp = np.round(loan_amnt * total_paid_frac, 2)
    total_pymnt = total_rec_prncp + np.maximum(
        0, rng.normal(loc=loan_amnt * 0.03, scale=loan_amnt * 0.02, size=n)
    )
    last_pymnt_amnt = np.round(np.clip(
        rng.normal(loc=loan_amnt * 0.04, scale=loan_amnt * 0.02, size=n),
        0, loan_amnt,
    ), 2)

    rec_rate = np.clip(rng.beta(1.6, 5.0, size=n), 0, 1)
    recoveries = np.round(
        np.maximum(loan_amnt - total_rec_prncp, 0) * rec_rate
        + rng.normal(0, 50, size=n),
        2,
    )
    recoveries = np.clip(recoveries, 0, None)

    return pd.DataFrame({
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": np.round(int_rate, 3),
        "grade": grade,
        "sub_grade": [f"{g}{i % 5 + 1}" for g, i in zip(grade, rng.integers(0, 5, size=n))],
        "annual_inc": np.round(annual_inc, 2),
        "dti": np.round(dti, 2),
        "issue_d": issue_d,
        "loan_status": "Charged Off",
        "total_rec_prncp": total_rec_prncp,
        "total_pymnt": total_pymnt,
        "recoveries": recoveries,
        "last_pymnt_amnt": last_pymnt_amnt,
        "last_pymnt_d": "Jan-2017",
        "emp_length": emp_length,
    })


def _synthesize_outreach_overlay(
    df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Add the synthetic outreach overlay so that:
      - expected_recovery_score R_i is continuous and smooth around c=1000
      - T_i = 1[R_i >= 1000]
      - Y_i has a structural causal uplift of ~TREATMENT_UPLIFT_DOLLARS
        on top of the (real) recoveries baseline
      - channel, agent_tier, cost, predicted_payment_prob assigned per account
    """
    int_rate = df["int_rate"].astype(float).to_numpy()
    grade = df["grade"].astype(str).to_numpy()
    dti = df["dti"].astype(float).to_numpy()
    loan_amnt = df["loan_amnt"].astype(float).to_numpy()
    recoveries = df["recoveries"].astype(float).to_numpy()

    grade_score = np.array([ord(g) - ord("A") for g in grade])

    z_int = (int_rate - 0.155) / 0.055
    z_grade = (grade_score - 3.0) / 1.8
    z_dti = (dti - 18.0) / 8.0
    z_loan = (np.log(loan_amnt) - 9.55) / 0.55

    linpred = (
        1.6 * z_int
        + 1.3 * z_grade
        + 0.6 * z_dti
        - 0.4 * z_loan
        + rng.normal(0, 1.4, size=len(df))
    )
    R = 1000.0 + 450.0 * (1 / (1 + np.exp(-linpred))) - 225.0 + rng.normal(0, 35.0, size=len(df))

    df["expected_recovery_score"] = np.round(R, 2)
    df["outreach_tier"] = (R >= CUTOFF_SCORE).astype(int)

    base_recovery = recoveries + np.clip(rng.normal(0, 40, size=len(df)), -200, 200)
    uplift = TREATMENT_UPLIFT_DOLLARS / np.sqrt(np.clip(loan_amnt, 1, None) / 15000.0)
    Y = base_recovery + uplift * df["outreach_tier"].to_numpy()
    Y = np.round(np.clip(Y, 0, loan_amnt * 1.05), 2)
    df["actual_recovery_amount"] = Y

    outstanding = np.round(loan_amnt * (1 - df["total_rec_prncp"].to_numpy() / loan_amnt), 2)
    df["outstanding_balance"] = np.clip(outstanding, 200, None)

    grade_num = np.array([ord(g) - ord("A") for g in grade])
    dpd = np.clip(30 + 25 * grade_num + rng.normal(0, 30, size=len(df)), 1, 365).astype(int)
    df["days_past_due"] = dpd

    channels = np.array(["SMS", "Email", "Phone", "FieldVisit"])
    tiers = np.array(["Junior", "Senior", "Specialist"])

    safe_R = np.clip(R, 1, None)
    ch_pref = np.stack([
        np.exp(-((safe_R - 400) / 250) ** 2),
        np.exp(-((safe_R - 600) / 250) ** 2),
        np.exp(-((safe_R - 900) / 300) ** 2),
        np.exp(-((safe_R - 1300) / 350) ** 2),
    ], axis=1)
    ch_pref = ch_pref / ch_pref.sum(axis=1, keepdims=True)
    channel_idx = np.array([rng.choice(4, p=p) for p in ch_pref])
    df["channel"] = channels[channel_idx]

    tier_pref = np.stack([
        np.ones_like(safe_R),
        np.exp(-((safe_R - 600) / 350) ** 2),
        np.exp(-((safe_R - 1100) / 350) ** 2),
    ], axis=1)
    tier_pref = tier_pref / tier_pref.sum(axis=1, keepdims=True)
    tier_idx = np.array([rng.choice(3, p=p) for p in tier_pref])
    df["agent_tier"] = tiers[tier_idx]

    cost_matrix = np.array([
        [0.50, 0.50, 0.50],
        [0.30, 0.30, 0.30],
        [5.00, 10.00, 15.00],
        [30.00, 60.00, 100.00],
    ])
    base_cost_L1 = 50.0
    base_cost_L0 = 1.5
    cost = np.where(
        df["outreach_tier"].to_numpy() == 1,
        cost_matrix[channel_idx, tier_idx] + base_cost_L1,
        cost_matrix[channel_idx, tier_idx] + base_cost_L0,
    )
    cost = np.round(cost * np.round(1 + rng.normal(0, 0.05, size=len(df)), 3), 2)
    df["cost_incurred"] = cost

    R_norm = (R - 200) / 1500.0
    channel_eff = np.array([0.55, 0.50, 0.75, 0.85])[channel_idx]
    tier_eff = np.array([0.80, 0.95, 1.10])[tier_idx]
    P = 0.20 + 0.45 * (1 / (1 + np.exp(-1.6 * R_norm))) * channel_eff * tier_eff
    P = np.clip(P + rng.normal(0, 0.02, size=len(df)), 0.02, 0.95)
    df["predicted_payment_prob"] = np.round(P, 4)

    df["customer_segment"] = np.where(
        loan_amnt >= 25000, "MSME",
        np.where(annual_inc := df["annual_inc"].astype(float).to_numpy() >= 80000,
                 "Salaried", "SelfEmployed"),
    )

    df["loan_amnt"] = loan_amnt
    df["int_rate"] = np.round(int_rate, 4)

    return df


def build(n_synthetic: int = 8000) -> BuildResult:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    rng = np.random.default_rng(SEED)

    output = PROCESSED_DIR / "collections_dataset.csv"
    if output.exists() and output.stat().st_size > 1000:
        LOG.info("Processed dataset already exists at %s, skipping rebuild", output)
        return BuildResult(output, int(pd.read_csv(output).shape[0]), "cache")

    t0 = time.time()
    csv_path = _try_download_lendingclub()
    source = "synthetic"
    if csv_path is not None:
        try:
            with zipfile.ZipFile(csv_path, "r") as zf:
                inner = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not inner:
                    raise RuntimeError("No CSV inside zip")
                with zf.open(inner[0]) as fh:
                    df = pd.read_csv(fh, low_memory=False, nrows=200000)
            charged = df[df["loan_status"].fillna("") == "Charged Off"]
            if len(charged) >= 1500:
                LOG.info("Loaded %s charged-off rows from snapshot", f"{len(charged):,}")
                source = "lendingclub"
            else:
                LOG.info("Snapshot too thin (%s rows); using synthetic fallback",
                         f"{len(charged):,}")
                df = _synthetic_lendingclub_like(n_synthetic, rng)
        except Exception as exc:
            LOG.warning("LC parse failed (%s); using synthetic fallback", exc)
            df = _synthetic_lendingclub_like(n_synthetic, rng)
    else:
        LOG.info("All LendingClub URLs unreachable; using synthetic fallback")
        df = _synthetic_lendingclub_like(n_synthetic, rng)

    if source != "lendingclub":
        LOG.info("Using synthetic LC-like dataset (%s rows)", f"{len(df):,}")

    df = df.reset_index(drop=True)
    df["account_id"] = np.arange(1, len(df) + 1)

    df = _synthesize_outreach_overlay(df, rng)

    cols_order = [
        "account_id", "loan_amnt", "outstanding_balance", "int_rate", "grade",
        "term", "annual_inc", "dti", "customer_segment", "days_past_due",
        "expected_recovery_score", "outreach_tier", "channel", "agent_tier",
        "actual_recovery_amount", "cost_incurred", "predicted_payment_prob",
        "total_rec_prncp", "recoveries", "loan_status", "issue_d", "emp_length",
    ]
    df = df[cols_order]
    df.to_csv(output, index=False)

    LOG.info("Wrote %s rows -> %s  (source=%s, %.1fs)",
             f"{len(df):,}", output, source, time.time() - t0)
    return BuildResult(output, len(df), source)


if __name__ == "__main__":
    res = build()
    print(f"OK: {res.source} -> {res.output_path}  ({res.n_rows:,} rows)")
