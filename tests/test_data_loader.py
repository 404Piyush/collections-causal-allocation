"""Sanity tests for the synthetic dataset loader / schema."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load, make_milp_arrays, make_rdd_arrays


def test_data_loader_returns_dataframe():
    csv = Path(__file__).resolve().parents[1] / "data" / "processed" / "collections_dataset.csv"
    if not csv.exists():
        pytest.skip("processed dataset not built yet")
    df = load(csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 100
    assert "outreach_tier" in df.columns
    assert df["outreach_tier"].isin([0, 1]).all()


def test_rdd_arrays_have_correct_shapes():
    csv = Path(__file__).resolve().parents[1] / "data" / "processed" / "collections_dataset.csv"
    if not csv.exists():
        pytest.skip("processed dataset not built yet")
    df = load(csv)
    R, Y, T = make_rdd_arrays(df)
    assert len(R) == len(Y) == len(T)
    assert ((R >= 1000).astype(int) == T.astype(int)).all()


def test_milp_arrays_align():
    csv = Path(__file__).resolve().parents[1] / "data" / "processed" / "collections_dataset.csv"
    if not csv.exists():
        pytest.skip("processed dataset not built yet")
    df = load(csv)
    Bal, channels, tiers = make_milp_arrays(df)
    assert len(Bal) == len(channels) == len(tiers) == len(df)
