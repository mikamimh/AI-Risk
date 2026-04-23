"""Tests for calibration_in_the_large (CIL)."""

import numpy as np
import pytest
from stats_compare import calibration_in_the_large


def test_cil_zero_when_predicted_equals_observed():
    """When predicted probabilities = empirical rate, CIL should be ~0."""
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    # predicted = observed mean (0.5)
    p = np.full(len(y), 0.5)
    result = calibration_in_the_large(y, p, n_boot=200, seed=0)
    assert result["CIL"] == pytest.approx(0.0, abs=1e-10)


def test_cil_positive_when_overestimating():
    """Positive CIL = model overestimates risk on average."""
    y = np.zeros(100, dtype=int)  # all non-events
    p = np.full(100, 0.3)         # predicted 30%
    result = calibration_in_the_large(y, p, n_boot=200, seed=0)
    assert result["CIL"] == pytest.approx(0.3, abs=1e-10)
    assert result["CIL"] > 0


def test_cil_negative_when_underestimating():
    """Negative CIL = model underestimates risk on average."""
    y = np.ones(100, dtype=int)  # all events
    p = np.full(100, 0.2)        # predicted 20%
    result = calibration_in_the_large(y, p, n_boot=200, seed=0)
    assert result["CIL"] < 0


def test_cil_has_ci_keys():
    y = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    p = np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.1, 0.9, 0.2, 0.6, 0.8])
    result = calibration_in_the_large(y, p, n_boot=200, seed=0)
    assert "CIL_CI_low" in result
    assert "CIL_CI_high" in result
    assert result["CIL_CI_low"] <= result["CIL"] <= result["CIL_CI_high"]
