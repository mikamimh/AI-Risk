"""Tests for compute_nri_with_ci, compute_idi_with_ci, and p-value behavior."""

import numpy as np
import pytest
from stats_compare import compute_nri_with_ci, compute_idi_with_ci


RNG = np.random.default_rng(0)
N = 300
Y = RNG.integers(0, 2, N)
P_OLD = RNG.random(N)
P_NEW = RNG.random(N)


def test_nri_with_ci_returns_ci_keys():
    result = compute_nri_with_ci(Y, P_OLD, P_NEW, n_boot=200, seed=0)
    assert "NRI_CI_low" in result
    assert "NRI_CI_high" in result
    assert "NRI_p" in result


def test_nri_with_ci_returns_event_nonevents_ci_keys():
    result = compute_nri_with_ci(Y, P_OLD, P_NEW, n_boot=200, seed=0)
    assert "NRI_events_CI_low" in result
    assert "NRI_events_CI_high" in result
    assert "NRI_nonevents_CI_low" in result
    assert "NRI_nonevents_CI_high" in result


def test_nri_with_ci_p_value_for_identical_scores():
    """P-value for identical old and new should be near 1.0 (no NRI)."""
    result = compute_nri_with_ci(Y, P_OLD, P_OLD, n_boot=500, seed=42)
    assert result["NRI total"] == pytest.approx(0.0, abs=1e-10)
    assert result["NRI_p"] == pytest.approx(1.0, abs=0.15)


def test_nri_with_ci_interval_contains_point():
    result = compute_nri_with_ci(Y, P_OLD, P_NEW, n_boot=500, seed=0)
    assert result["NRI_CI_low"] <= result["NRI total"] <= result["NRI_CI_high"]


def test_nri_with_ci_preserves_original_keys():
    result = compute_nri_with_ci(Y, P_OLD, P_NEW, n_boot=200, seed=0)
    assert "NRI events" in result
    assert "NRI non-events" in result
    assert "NRI total" in result


def test_idi_with_ci_returns_ci_keys():
    result = compute_idi_with_ci(Y, P_OLD, P_NEW, n_boot=200, seed=0)
    assert "IDI_CI_low" in result
    assert "IDI_CI_high" in result
    assert "IDI_p" in result


def test_idi_with_ci_p_value_for_identical_scores():
    """P-value for identical old and new should be near 1.0 (IDI=0)."""
    result = compute_idi_with_ci(Y, P_OLD, P_OLD, n_boot=500, seed=42)
    assert result["IDI"] == pytest.approx(0.0, abs=1e-10)
    assert result["IDI_p"] == pytest.approx(1.0, abs=0.15)


def test_idi_with_ci_interval_contains_point():
    result = compute_idi_with_ci(Y, P_OLD, P_NEW, n_boot=500, seed=0)
    assert result["IDI_CI_low"] <= result["IDI"] <= result["IDI_CI_high"]
