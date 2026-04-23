"""Tests for the calibration-aware tiebreaker in _select_best_model."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from modeling import _select_best_model


def _make_lb(*entries):
    """Build leaderboard from (name, auc, auprc) tuples."""
    return pd.DataFrame([
        {"Modelo": name, "AUC": auc, "AUPRC": auprc}
        for name, auc, auprc in entries
    ])


def _make_oof(y: np.ndarray, seed: int) -> np.ndarray:
    """Distinct OOF array for a given seed (no NaNs, passes all guardrails)."""
    rng = np.random.default_rng(seed)
    return np.clip(0.04 + 0.30 * y + rng.normal(0, 0.07, len(y)), 0.01, 0.99)


def _make_y(n: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(n) > 0.85).astype(int)


def _slope_router(slopes: dict, oof: dict):
    """Return a mock side_effect that identifies the model by array mean proximity."""
    means = {name: arr.mean() for name, arr in oof.items()}

    def _side_effect(y_, p_):
        p_mean = float(np.asarray(p_).mean())
        # Nearest mean → this model's slope
        name = min(means, key=lambda n: abs(means[n] - p_mean))
        slope = slopes[name]
        return {"Calibration intercept": 0.0, "Calibration slope": slope}

    return _side_effect


# ── Core tiebreaker tests ─────────────────────────────────────────────────

def test_tiebreaker_selects_better_calibration():
    """Within AUC tie margin, model with slope closer to 1.0 wins."""
    y = _make_y()
    # Distinct OOF arrays so the router can distinguish models
    p_xgb = _make_oof(y, seed=10)
    p_rf = _make_oof(y, seed=20)

    lb = _make_lb(
        ("XGBoost", 0.746, 0.35),
        ("RandomForest", 0.745, 0.34),
    )
    oof = {"XGBoost": p_xgb, "RandomForest": p_rf}

    slopes = {"XGBoost": 0.52, "RandomForest": 1.01}
    mock_fn = _slope_router(slopes, oof)

    with patch("modeling.calibration_intercept_slope", side_effect=mock_fn):
        best = _select_best_model(lb, oof_predictions=oof, y=y)

    assert best == "RandomForest", (
        f"Expected RandomForest (slope≈1.0), got {best}. "
        "ΔAUC=0.001 is within tie margin; RF should win on calibration."
    )


def test_tiebreaker_exact_tie_auc():
    """Identical AUC: calibration slope is the sole tiebreaker."""
    y = _make_y(seed=1)
    p_a = _make_oof(y, seed=11)
    p_b = _make_oof(y, seed=21)

    lb = _make_lb(("ModelA", 0.745, 0.35), ("ModelB", 0.745, 0.34))
    oof = {"ModelA": p_a, "ModelB": p_b}

    # ModelB slope 0.98 is closer to 1.0 than ModelA slope 0.60
    slopes = {"ModelA": 0.60, "ModelB": 0.98}
    with patch("modeling.calibration_intercept_slope",
               side_effect=_slope_router(slopes, oof)):
        best = _select_best_model(lb, oof_predictions=oof, y=y)

    assert best == "ModelB"


def test_tiebreaker_does_not_activate_when_gap_exceeds_margin():
    """ΔAUC > 0.01 → plain AUC ranking wins, regardless of calibration."""
    y = _make_y(seed=2)
    p_high = _make_oof(y, seed=12)
    p_low = _make_oof(y, seed=22)

    lb = _make_lb(
        ("HighAUC", 0.780, 0.40),   # 0.04 above LowAUC
        ("LowAUC", 0.740, 0.35),
    )
    oof = {"HighAUC": p_high, "LowAUC": p_low}

    # Both have valid slopes — only AUC matters here
    with patch("modeling.calibration_intercept_slope",
               return_value={"Calibration intercept": 0.0, "Calibration slope": 1.0}):
        best = _select_best_model(lb, oof_predictions=oof, y=y)

    assert best == "HighAUC", (
        f"Expected HighAUC (ΔAUC=0.04 > margin 0.01), got {best}"
    )


def test_tiebreaker_three_way_tie():
    """Three models within margin: the one with slope closest to 1.0 wins."""
    y = _make_y(seed=3)
    p_a = _make_oof(y, seed=13)
    p_b = _make_oof(y, seed=23)
    p_c = _make_oof(y, seed=33)

    lb = _make_lb(
        ("ModelA", 0.748, 0.40),
        ("ModelB", 0.746, 0.38),
        ("ModelC", 0.745, 0.36),
    )
    oof = {"ModelA": p_a, "ModelB": p_b, "ModelC": p_c}

    # ModelB has slope 1.02, closest to 1.0
    slopes = {"ModelA": 0.30, "ModelB": 1.02, "ModelC": 0.70}
    with patch("modeling.calibration_intercept_slope",
               side_effect=_slope_router(slopes, oof)):
        best = _select_best_model(lb, oof_predictions=oof, y=y)

    assert best == "ModelB", f"Expected ModelB (slope 1.02 ≈ 1.0), got {best}"


def test_tiebreaker_custom_margin():
    """Custom auc_tie_margin=0.02 catches gaps that default 0.01 does not."""
    y = _make_y(seed=4)
    p_a = _make_oof(y, seed=14)
    p_b = _make_oof(y, seed=24)

    lb = _make_lb(
        ("ModelA", 0.760, 0.40),   # 0.015 above ModelB
        ("ModelB", 0.745, 0.35),
    )
    oof = {"ModelA": p_a, "ModelB": p_b}
    slopes = {"ModelA": 0.45, "ModelB": 0.99}

    # Default margin 0.01: gap 0.015 exceeds it → ModelA wins by AUC
    with patch("modeling.calibration_intercept_slope",
               side_effect=_slope_router(slopes, oof)):
        best_default = _select_best_model(
            lb, oof_predictions=oof, y=y, auc_tie_margin=0.01
        )
    assert best_default == "ModelA"

    # Wide margin 0.02: gap 0.015 is within it → ModelB wins on calibration
    with patch("modeling.calibration_intercept_slope",
               side_effect=_slope_router(slopes, oof)):
        best_wide = _select_best_model(
            lb, oof_predictions=oof, y=y, auc_tie_margin=0.02
        )
    assert best_wide == "ModelB"


def test_tiebreaker_nan_slope_treated_as_worst():
    """Model returning NaN slope gets inf distance — loses tiebreaker."""
    y = _make_y(seed=5)
    p_nan = _make_oof(y, seed=15)
    p_good = _make_oof(y, seed=25)

    lb = _make_lb(("NaNSlope", 0.746, 0.36), ("GoodSlope", 0.745, 0.35))
    oof = {"NaNSlope": p_nan, "GoodSlope": p_good}

    slopes = {"NaNSlope": float("nan"), "GoodSlope": 0.95}
    with patch("modeling.calibration_intercept_slope",
               side_effect=_slope_router(slopes, oof)):
        best = _select_best_model(lb, oof_predictions=oof, y=y)

    assert best == "GoodSlope", (
        "NaN slope should be treated as infinite distance, so GoodSlope wins."
    )


def test_single_usable_model_skips_tiebreaker():
    """Only one usable model → returned directly without tiebreaker."""
    y = _make_y(seed=6)
    p = _make_oof(y, seed=16)
    lb = _make_lb(("OnlyUsable", 0.750, 0.38))
    oof = {"OnlyUsable": p}

    with patch("modeling.calibration_intercept_slope",
               return_value={"Calibration intercept": 0.0, "Calibration slope": 1.0}):
        best = _select_best_model(lb, oof_predictions=oof, y=y)

    assert best == "OnlyUsable"


def test_tiebreaker_exception_in_slope_treated_as_worst():
    """Exception during slope computation → inf distance, model loses."""
    y = _make_y(seed=7)
    p_err = _make_oof(y, seed=17)
    p_ok = _make_oof(y, seed=27)

    lb = _make_lb(("ErrorModel", 0.746, 0.36), ("OKModel", 0.745, 0.35))
    oof = {"ErrorModel": p_err, "OKModel": p_ok}

    # Identify ErrorModel by exact array equality (p_ is p_arr[mask], mask=all True)
    err_ref = p_err.copy()

    def _mock_with_exception(y_, p_):
        if np.array_equal(p_, err_ref):
            raise RuntimeError("simulated cal error")
        return {"Calibration intercept": 0.0, "Calibration slope": 1.0}

    with patch("modeling.calibration_intercept_slope",
               side_effect=_mock_with_exception):
        best = _select_best_model(lb, oof_predictions=oof, y=y)

    assert best == "OKModel", (
        "ErrorModel's exception should be caught; OKModel wins tiebreaker."
    )
