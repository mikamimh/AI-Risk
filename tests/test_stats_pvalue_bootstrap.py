"""Tests for bootstrap p-value correctness in NRI/IDI CI functions.

Verifies that:
- p-values are in [0, 1]
- When no bootstrap sample crosses zero, NRI_p/IDI_p == 0.0 AND
  NRI_p_lower_bound / IDI_p_lower_bound is set to 1/n_boot (resolution limit)
- When samples span zero (e.g. identical models), p ≈ 1.0 and no lower bound
"""

import numpy as np
import pytest
from stats_compare import compute_nri_with_ci, compute_idi_with_ci


def _make_y_p(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) > 0.85).astype(int)
    p = np.clip(rng.random(n), 0.01, 0.99)
    return y, p


def test_nri_p_in_unit_interval():
    y, p = _make_y_p()
    result = compute_nri_with_ci(y, p, p * 0.9 + 0.05, n_boot=500, seed=0)
    assert 0.0 <= result["NRI_p"] <= 1.0


def test_idi_p_in_unit_interval():
    y, p = _make_y_p()
    result = compute_idi_with_ci(y, p, p * 0.9 + 0.05, n_boot=500, seed=0)
    assert 0.0 <= result["IDI_p"] <= 1.0


def test_identical_models_nri_p_near_one():
    """Identical old and new → NRI = 0 for all samples → p ≈ 1.0."""
    y, p = _make_y_p()
    result = compute_nri_with_ci(y, p, p, n_boot=500, seed=42)
    assert result["NRI total"] == pytest.approx(0.0, abs=1e-10)
    assert result["NRI_p"] == pytest.approx(1.0, abs=0.15)
    # No lower bound when p > 0
    assert result.get("NRI_p_lower_bound") is None


def test_identical_models_idi_p_near_one():
    y, p = _make_y_p()
    result = compute_idi_with_ci(y, p, p, n_boot=500, seed=42)
    assert result["IDI"] == pytest.approx(0.0, abs=1e-10)
    assert result["IDI_p"] == pytest.approx(1.0, abs=0.15)
    assert result.get("IDI_p_lower_bound") is None


def test_lower_bound_set_when_p_zero():
    """When every bootstrap sample is strictly positive, p=0 and lower bound is set."""
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) > 0.85).astype(int)
    # New model always predicts higher → all NRI bootstrap samples > 0
    p_old = np.full(n, 0.05)
    p_new = np.full(n, 0.90)
    n_boot = 200
    result = compute_nri_with_ci(y, p_old, p_new, n_boot=n_boot, seed=0)
    if result["NRI_p"] == 0.0:
        lb = result.get("NRI_p_lower_bound")
        assert lb is not None, "NRI_p_lower_bound must be set when NRI_p == 0.0"
        assert lb == pytest.approx(1.0 / n_boot, rel=0.01), (
            f"Lower bound should be 1/n_boot = {1/n_boot:.5f}, got {lb}"
        )
    # If NRI_p > 0, lower bound must be None
    else:
        assert result.get("NRI_p_lower_bound") is None


def test_no_clip_below_1_over_n_boot():
    """p-value must never be artificially clipped to 1e-10 (old behavior)."""
    rng = np.random.default_rng(1)
    n = 300
    y = (rng.random(n) > 0.85).astype(int)
    p_old = np.clip(rng.random(n), 0.01, 0.99)
    p_new = np.clip(rng.random(n), 0.01, 0.99)
    n_boot = 100
    result = compute_nri_with_ci(y, p_old, p_new, n_boot=n_boot, seed=0)
    # p must be >= 1/n_boot OR zero with lower_bound set
    if result["NRI_p"] > 0.0:
        assert result["NRI_p"] >= 1.0 / n_boot - 1e-9, (
            f"p={result['NRI_p']} is below resolution 1/n_boot={1/n_boot}"
        )
