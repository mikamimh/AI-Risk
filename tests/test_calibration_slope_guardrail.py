"""Tests for Guardrail C: calibration slope sanity in _select_best_model."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from modeling import _select_best_model


def _make_lb(*names):
    """Leaderboard with decreasing AUC so first entry is preferred by ranking."""
    return pd.DataFrame([
        {"Modelo": name, "AUC": 0.80 - i * 0.01, "AUPRC": 0.40 - i * 0.01}
        for i, name in enumerate(names)
    ])


def _make_passable_oof(n: int = 300, seed: int = 0) -> tuple:
    """Return (y, p) where p passes all guardrails A–B but has controllable slope."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) > 0.85).astype(int)
    # Predictions that pass A (coverage below 8%), B1 (AUC≥0.6), B2 (Brier skill), B3 (range)
    p = np.clip(0.05 + 0.30 * y + rng.normal(0, 0.07, n), 0.01, 0.99)
    return y, p


# ---------------------------------------------------------------------------
# Mock-based tests — directly verify guardrail C branch logic
# ---------------------------------------------------------------------------

def test_pathological_slope_rejected_via_mock():
    """Model returning slope 0.22 from calibration_intercept_slope is rejected."""
    lb = _make_lb("LGBMock", "RFMock")
    y, p = _make_passable_oof()
    oof = {"LGBMock": p.copy(), "RFMock": p.copy()}

    call_count = [0]

    def _mock_cal(y_, p_):
        call_count[0] += 1
        # First call → LGBMock: pathological slope (LightGBM isotonic artefact)
        if call_count[0] == 1:
            return {"Calibration intercept": -1.29, "Calibration slope": 0.22}
        # Second call → RFMock: well-calibrated
        return {"Calibration intercept": 0.034, "Calibration slope": 1.013}

    with patch("modeling.calibration_intercept_slope", side_effect=_mock_cal):
        best = _select_best_model(lb, oof_predictions=oof, y=y)

    assert best == "RFMock", (
        f"Expected RFMock (good calibration), got {best}. "
        "LGBMock with slope 0.22 should be rejected by Guardrail C."
    )


def test_overdispersed_slope_rejected_via_mock():
    """Model returning slope 3.50 (above 2.50 ceiling) is rejected."""
    lb = _make_lb("OverDisp", "GoodCalib")
    y, p = _make_passable_oof(seed=1)
    oof = {"OverDisp": p.copy(), "GoodCalib": p.copy()}

    call_count = [0]

    def _mock_cal(y_, p_):
        call_count[0] += 1
        if call_count[0] == 1:
            return {"Calibration intercept": 0.5, "Calibration slope": 3.50}
        return {"Calibration intercept": 0.1, "Calibration slope": 0.90}

    with patch("modeling.calibration_intercept_slope", side_effect=_mock_cal):
        best = _select_best_model(lb, oof_predictions=oof, y=y)

    assert best == "GoodCalib", (
        f"Expected GoodCalib, got {best}. "
        "OverDisp with slope 3.50 > 2.50 should be rejected by Guardrail C."
    )


def test_slope_at_lower_boundary_passes():
    """Slope exactly at min_cal_slope (0.40) is accepted (not strictly less than)."""
    lb = _make_lb("OnlyModel")
    y, p = _make_passable_oof(seed=2)
    oof = {"OnlyModel": p}

    with patch("modeling.calibration_intercept_slope",
               return_value={"Calibration intercept": -0.1, "Calibration slope": 0.40}):
        best = _select_best_model(lb, oof_predictions=oof, y=y)

    assert best == "OnlyModel"


def test_slope_at_upper_boundary_passes():
    """Slope exactly at max_cal_slope (2.50) is accepted."""
    lb = _make_lb("OnlyModel")
    y, p = _make_passable_oof(seed=3)
    oof = {"OnlyModel": p}

    with patch("modeling.calibration_intercept_slope",
               return_value={"Calibration intercept": 0.2, "Calibration slope": 2.50}):
        best = _select_best_model(lb, oof_predictions=oof, y=y)

    assert best == "OnlyModel"


def test_nan_slope_does_not_penalise():
    """NaN slope (calibration could not be computed) passes Guardrail C."""
    lb = _make_lb("OnlyModel")
    y, p = _make_passable_oof(seed=4)
    oof = {"OnlyModel": p}

    with patch("modeling.calibration_intercept_slope",
               return_value={"Calibration intercept": np.nan, "Calibration slope": np.nan}):
        best = _select_best_model(lb, oof_predictions=oof, y=y)

    assert best == "OnlyModel"


def test_exception_in_calibration_does_not_crash():
    """Exception raised by calibration_intercept_slope is silently ignored."""
    lb = _make_lb("OnlyModel")
    y, p = _make_passable_oof(seed=5)
    oof = {"OnlyModel": p}

    with patch("modeling.calibration_intercept_slope", side_effect=RuntimeError("mock error")):
        best = _select_best_model(lb, oof_predictions=oof, y=y)

    assert best == "OnlyModel"


def test_custom_slope_bounds_respected():
    """Custom min_cal_slope=0.60 rejects slope 0.45 that would pass default 0.40."""
    lb = _make_lb("Borderline", "GoodCalib")
    y, p = _make_passable_oof(seed=6)
    oof = {"Borderline": p.copy(), "GoodCalib": p.copy()}

    call_count = [0]

    def _mock_cal(y_, p_):
        call_count[0] += 1
        if call_count[0] == 1:
            return {"Calibration intercept": -0.5, "Calibration slope": 0.45}
        return {"Calibration intercept": 0.05, "Calibration slope": 1.10}

    # With default floor (0.40): slope 0.45 passes
    call_count[0] = 0
    with patch("modeling.calibration_intercept_slope", side_effect=_mock_cal):
        best_default = _select_best_model(
            lb, oof_predictions=oof, y=y, min_cal_slope=0.40
        )
    assert best_default == "Borderline"

    # With tighter floor (0.60): slope 0.45 fails → GoodCalib wins
    call_count[0] = 0
    with patch("modeling.calibration_intercept_slope", side_effect=_mock_cal):
        best_tight = _select_best_model(
            lb, oof_predictions=oof, y=y, min_cal_slope=0.60
        )
    assert best_tight == "GoodCalib"


def test_fallback_when_all_fail_guardrail_c():
    """When all models fail Guardrail C, top-AUC model is returned as fallback."""
    lb = _make_lb("ModelA", "ModelB")
    y, p = _make_passable_oof(seed=7)
    oof = {"ModelA": p.copy(), "ModelB": p.copy()}

    # Both models have pathological slope
    with patch("modeling.calibration_intercept_slope",
               return_value={"Calibration intercept": -2.0, "Calibration slope": 0.10}):
        best = _select_best_model(lb, oof_predictions=oof, y=y)

    # Should not raise; returns top by AUC
    assert best in ("ModelA", "ModelB")
