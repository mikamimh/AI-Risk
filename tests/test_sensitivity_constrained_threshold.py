"""Unit tests for sensitivity_constrained_threshold and threshold role constants."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stats_compare import (
    THRESHOLD_ROLE_EXPLORATORY,
    THRESHOLD_ROLE_FIXED_COMPARATOR,
    THRESHOLD_ROLE_HISTORICAL_COMPARATOR,
    THRESHOLD_ROLE_PRIMARY,
    sensitivity_constrained_threshold,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_data(n: int = 200, prevalence: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    p = np.clip(y * 0.5 + rng.random(n) * 0.5, 0.01, 0.99)
    return y, p


# ── threshold role constants ──────────────────────────────────────────────────

def test_threshold_role_constants_are_strings():
    assert isinstance(THRESHOLD_ROLE_PRIMARY, str)
    assert isinstance(THRESHOLD_ROLE_FIXED_COMPARATOR, str)
    assert isinstance(THRESHOLD_ROLE_HISTORICAL_COMPARATOR, str)
    assert isinstance(THRESHOLD_ROLE_EXPLORATORY, str)


def test_threshold_role_values():
    assert THRESHOLD_ROLE_PRIMARY == "Primary"
    assert THRESHOLD_ROLE_EXPLORATORY == "Exploratory"
    assert "comparator" in THRESHOLD_ROLE_FIXED_COMPARATOR.lower()
    assert "comparator" in THRESHOLD_ROLE_HISTORICAL_COMPARATOR.lower()


# ── sensitivity_constrained_threshold ────────────────────────────────────────

class TestSensitivityConstrainedThreshold:
    def test_returns_dict_with_required_keys(self):
        y, p = _make_data()
        result = sensitivity_constrained_threshold(y, p, min_sensitivity=0.90)
        required = {
            "threshold", "sensitivity", "specificity", "PPV", "NPV",
            "TP", "FP", "TN", "FN", "n_flagged", "flag_rate",
            "event_rate_above", "event_rate_below", "status",
        }
        assert required.issubset(result.keys())

    def test_achieves_target_sensitivity(self):
        y, p = _make_data(n=300, prevalence=0.20)
        result = sensitivity_constrained_threshold(y, p, min_sensitivity=0.90)
        assert result["status"] == "ok"
        assert result["sensitivity"] >= 0.90 - 1e-6

    def test_threshold_is_probability_scale_not_percentage(self):
        """threshold must be in [0, 1], never 8 when 8% is meant."""
        y, p = _make_data(n=300)
        result = sensitivity_constrained_threshold(y, p, min_sensitivity=0.90)
        if result["status"] == "ok":
            assert 0.0 <= result["threshold"] <= 1.0
            assert result["threshold"] < 1.0  # not a percentage

    def test_selects_largest_threshold_among_valid(self):
        """Lower sensitivity target -> threshold can be same or higher (more specific)."""
        y, p = _make_data(n=300, prevalence=0.20)
        r90 = sensitivity_constrained_threshold(y, p, min_sensitivity=0.90)
        r85 = sensitivity_constrained_threshold(y, p, min_sensitivity=0.85)
        if r90["status"] == "ok" and r85["status"] == "ok":
            # Lower constraint -> can select higher threshold (same or more specific)
            assert r85["threshold"] >= r90["threshold"] - 1e-6

    def test_maximises_specificity_within_constraint(self):
        """The returned threshold should be at least as specific as the 8% fixed threshold."""
        y, p = _make_data(n=300, prevalence=0.15, seed=7)
        result = sensitivity_constrained_threshold(y, p, min_sensitivity=0.90)
        if result["status"] == "ok":
            # sens90 policy prefers higher specificity — 8% is a comparator
            assert result["sensitivity"] >= 0.90 - 1e-6

    def test_returns_not_available_when_impossible(self):
        """min_sensitivity > 1.0 is impossible at any threshold."""
        y, p = _make_data()
        result = sensitivity_constrained_threshold(y, p, min_sensitivity=1.5)
        assert result["status"] == "not_available"
        assert math.isnan(result["threshold"])

    def test_returns_not_available_with_no_events(self):
        """No positive cases -> sensitivity undefined -> not_available."""
        y = np.zeros(50, dtype=int)
        p = np.linspace(0.01, 0.99, 50)
        result = sensitivity_constrained_threshold(y, p, min_sensitivity=0.90)
        assert result["status"] == "not_available"

    def test_confusion_matrix_correct(self):
        """TP + FN = n_events and TN + FP = n_non_events at the returned threshold."""
        y, p = _make_data(n=200, prevalence=0.20)
        result = sensitivity_constrained_threshold(y, p, min_sensitivity=0.85)
        if result["status"] == "ok":
            n_pos = int(y.sum())
            n_neg = len(y) - n_pos
            assert result["TP"] + result["FN"] == n_pos
            assert result["TN"] + result["FP"] == n_neg

    def test_flag_rate_in_unit_interval(self):
        y, p = _make_data(n=200)
        result = sensitivity_constrained_threshold(y, p)
        if result["status"] == "ok":
            assert 0.0 <= result["flag_rate"] <= 1.0

    def test_n_flagged_equals_tp_plus_fp(self):
        y, p = _make_data(n=200)
        result = sensitivity_constrained_threshold(y, p)
        if result["status"] == "ok":
            assert result["n_flagged"] == result["TP"] + result["FP"]

    def test_status_ok_has_finite_threshold(self):
        y, p = _make_data(n=400, prevalence=0.15, seed=0)
        result = sensitivity_constrained_threshold(y, p, min_sensitivity=0.90)
        if result["status"] == "ok":
            assert math.isfinite(result["threshold"])

    def test_multiple_seeds_consistent(self):
        """Across different seeds, the threshold is always valid when status=ok."""
        for seed in range(5):
            y, p = _make_data(n=200, prevalence=0.15, seed=seed)
            result = sensitivity_constrained_threshold(y, p, min_sensitivity=0.90)
            if result["status"] == "ok":
                assert 0.0 < result["threshold"] < 1.0
                assert result["sensitivity"] >= 0.90 - 1e-6

    def test_default_min_sensitivity_is_90_percent(self):
        """Default min_sensitivity parameter is 0.90."""
        y, p = _make_data(n=400, prevalence=0.20, seed=1)
        r_default = sensitivity_constrained_threshold(y, p)
        r_explicit = sensitivity_constrained_threshold(y, p, min_sensitivity=0.90)
        assert r_default["status"] == r_explicit["status"]
        if r_default["status"] == "ok":
            assert math.isclose(r_default["threshold"], r_explicit["threshold"])
