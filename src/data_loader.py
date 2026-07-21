"""Load and pre-validate the processed collections dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from . import config

LOG = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "account_id",
    "loan_amnt",
    "outstanding_balance",
    "expected_recovery_score",
    "outreach_tier",
    "actual_recovery_amount",
    "channel",
    "agent_tier",
    "cost_incurred",
    "predicted_payment_prob",
]


def load(path: Path = config.PROCESSED_DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["outreach_tier"] = df["outreach_tier"].astype(int)
    return df


def make_rdd_arrays(
    df: pd.DataFrame, cutoff: float = config.CUTOFF
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = df["expected_recovery_score"].to_numpy(dtype=float)
    Y = df["actual_recovery_amount"].to_numpy(dtype=float)
    T = (R >= cutoff).astype(float)
    return R, Y, T


def make_milp_arrays(df: pd.DataFrame):
    Bal = df["outstanding_balance"].to_numpy(dtype=float)
    channels = df["channel"].astype(str).to_numpy()
    tiers = df["agent_tier"].astype(str).to_numpy()
    return Bal, channels, tiers
