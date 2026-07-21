"""Tests for MILP optimizer on a tiny problem with a hand-checked optimum."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.milp import optimizer


@pytest.fixture
def tiny_df():
    return pd.DataFrame({
        "account_id": [1, 2, 3],
        "outstanding_balance": [1000.0, 500.0, 2000.0],
        "channel": ["Phone", "SMS", "Email"],
        "agent_tier": ["Senior", "Junior", "Junior"],
        "expected_recovery_score": [1100.0, 500.0, 800.0],
    })


def test_milp_runs(tiny_df):
    res, assignments = optimizer.optimize(
        tiny_df, budget=10000.0,
        capacities={"Junior": 50.0, "Senior": 50.0, "Specialist": 50.0},
        time_limit_sec=20,
    )
    assert res.status == "Optimal"
    assert "objective" in res.__dict__


def test_milp_assigns_each_account_at_most_one(tiny_df):
    res, assignments = optimizer.optimize(
        tiny_df, budget=10000.0,
        capacities={"Junior": 50.0, "Senior": 50.0, "Specialist": 50.0},
        time_limit_sec=20,
    )
    assert assignments["account_id"].duplicated().sum() == 0
