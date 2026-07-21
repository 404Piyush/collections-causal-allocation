"""Tests for RDD module on a synthetic dataset with KNOWN jump."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rdd import bandwidth as bw_mod
from src.rdd import local_linear as rdd


@pytest.fixture
def rdd_world():
    rng = np.random.default_rng(42)
    n = 2000
    R = rng.uniform(0, 2000, size=n)
    intercept = 100.0
    slope = 0.6
    true_tau = 75.0
    T = (R >= 1000).astype(float)
    Y = intercept + slope * R + true_tau * T + rng.normal(0, 30, size=n)
    return R, Y, true_tau


def test_bandwidth_returns_positive(rdd_world):
    R, Y, _ = rdd_world
    info = bw_mod.ik_bandwidth(R, Y, cutoff=1000.0)
    assert info["h_left"] > 0
    assert info["h_right"] > 0


def test_rdd_recovers_true_jump(rdd_world):
    R, Y, true_tau = rdd_world
    info = bw_mod.ik_bandwidth(R, Y, cutoff=1000.0)
    res = rdd.local_linear_rdd(R, Y, 1000.0, info["h_left"], info["h_right"])
    assert res.tau_hat == pytest.approx(true_tau, abs=15.0)
    assert res.p_value < 0.01
    assert res.ci_low < true_tau < res.ci_high


def test_rdd_breaks_with_bogus_bandwidth(rdd_world):
    R, Y, _ = rdd_world
    res = rdd.local_linear_rdd(R, Y, 1000.0, 5.0, 5.0)
    assert res.n_effective < 200


def test_rdd_no_jump_when_none(rdd_world):
    rng = np.random.default_rng(7)
    R = rng.uniform(0, 2000, size=1000)
    Y = 100 + 0.5 * R + rng.normal(0, 30, size=1000)
    res = rdd.local_linear_rdd(R, Y, 1000.0, 200.0, 200.0)
    assert abs(res.tau_hat) < 15.0
