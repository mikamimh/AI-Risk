"""Statistical regression test — catches pipeline changes that degrade model performance.

Uses a deterministic synthetic dataset that exercises the same preprocessing
paths as the real pipeline: numeric, binary, categorical, ordinal, and
missingness indicators. Bounds are intentionally loose (AUC > 0.55, Brier <
0.30) — the goal is to catch catastrophic regressions, not validate clinical
performance.

The test_binary_encoding_preserves_categories test would have caught the V2
bug immediately (Diabetes 'Insulin' → NaN destroyed clinical information).
"""

import numpy as np
import pandas as pd
import pytest

from modeling import (
    _BINARY_DIRECT_ENCODE_COLS,
    _encode_binary_direct,
    build_preprocessor,
    clean_features,
)


def _make_synthetic_dataset(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic dataset that exercises all preprocessing paths."""
    rng = np.random.default_rng(seed)

    age = rng.integers(30, 90, size=n).astype(float)
    lvef = rng.normal(55, 15, size=n).clip(10, 80)

    # Verified binary cols (in _BINARY_DIRECT_ENCODE_COLS)
    hypertension = rng.choice(["Yes", "No"], size=n, p=[0.6, 0.4])
    dialysis = rng.choice(["Yes", "No"], size=n, p=[0.05, 0.95])

    # Multi-category cols (NOT in binary encode set)
    diabetes = rng.choice(
        ["No", "Oral", "Insulin", "Diet Only"], size=n, p=[0.5, 0.2, 0.2, 0.1]
    )
    cva = rng.choice(
        ["No", "TIA", "≤ 30 days", "≥ 30 days"], size=n, p=[0.85, 0.05, 0.05, 0.05]
    )

    # Ordinal valve severity
    aortic_stenosis = rng.choice(
        ["None", "Mild", "Moderate", "Severe"], size=n, p=[0.5, 0.2, 0.2, 0.1]
    )

    # Surgery text (TargetEncoder categorical)
    surgery = rng.choice(
        ["CABG", "AVR", "MVR", "OPCAB", "AVR, CABG"],
        size=n,
        p=[0.4, 0.2, 0.15, 0.15, 0.1],
    )

    # Outcome with known signal from age and LVEF
    logit = -3.0 + 0.04 * age - 0.03 * lvef + 0.5 * (diabetes == "Insulin").astype(float)
    prob = 1.0 / (1.0 + np.exp(-logit))
    morte_30d = (rng.random(n) < prob).astype(int)

    # Add some LVEF missingness
    lvef_col = lvef.copy().astype(object)
    lvef_col[rng.random(n) < 0.10] = np.nan

    return pd.DataFrame({
        "Age (years)": age,
        "Pré-LVEF, %": lvef_col,
        "Hypertension": hypertension,
        "Dialysis": dialysis,
        "Diabetes": diabetes,
        "CVA": cva,
        "Aortic Stenosis": aortic_stenosis,
        "Surgery": surgery,
        "morte_30d": morte_30d,
        "_patient_key": [f"P{i:04d}" for i in range(n)],
    })


# ── Encoding correctness ──────────────────────────────────────────────────────

def test_binary_encoding_preserves_multi_category_columns():
    """Multi-category columns (Diabetes Insulin, CVA TIA) must survive unchanged."""
    df = _make_synthetic_dataset()
    features = [c for c in df.columns if c not in ("morte_30d", "_patient_key")]
    result = _encode_binary_direct(df[features])

    # Diabetes must keep 'Insulin' — would have been NaN'd in the original V2 bug
    assert "Insulin" in result["Diabetes"].dropna().unique(), (
        "Diabetes 'Insulin' was destroyed by _encode_binary_direct. "
        "Ensure Diabetes is NOT in _BINARY_DIRECT_ENCODE_COLS."
    )
    # CVA must keep 'TIA'
    assert "TIA" in result["CVA"].dropna().unique(), (
        "CVA 'TIA' was destroyed by _encode_binary_direct."
    )


def test_binary_encoding_converts_verified_binaries():
    """Columns in _BINARY_DIRECT_ENCODE_COLS become 0/1 float after clean_features."""
    df = _make_synthetic_dataset()
    features = [c for c in df.columns if c not in ("morte_30d", "_patient_key")]
    cleaned = clean_features(df[features])

    for col in _BINARY_DIRECT_ENCODE_COLS:
        if col not in cleaned.columns:
            continue
        assert pd.api.types.is_numeric_dtype(cleaned[col]), (
            f"{col} should be numeric after binary encoding, got {cleaned[col].dtype}"
        )
        unique = set(cleaned[col].dropna().unique())
        assert unique <= {0.0, 1.0}, (
            f"{col}: expected only {{0.0, 1.0}}, got {unique}"
        )


def test_hypertension_converted_not_diabetes():
    """Hypertension (binary) becomes 0/1; Diabetes (categorical) stays string."""
    df = _make_synthetic_dataset()
    features = [c for c in df.columns if c not in ("morte_30d", "_patient_key")]
    result = _encode_binary_direct(df[features])

    assert pd.api.types.is_numeric_dtype(result["Hypertension"]), (
        "Hypertension should be numeric after binary encoding"
    )
    assert result["Diabetes"].dtype == object, (
        f"Diabetes should remain object, got {result['Diabetes'].dtype}"
    )


# ── Pipeline integrity ────────────────────────────────────────────────────────

def test_preprocessor_builds_without_error():
    """build_preprocessor should handle mixed dtypes (numeric + categorical) without crash."""
    df = _make_synthetic_dataset()
    features = [c for c in df.columns if c not in ("morte_30d", "_patient_key")]
    cleaned = clean_features(df[features])
    prep = build_preprocessor(cleaned)
    assert prep is not None


def test_full_pipeline_auc_above_floor():
    """Full train_and_select_model on synthetic data: AUC > 0.55, Brier < 0.30."""
    from modeling import train_and_select_model

    df = _make_synthetic_dataset(n=200, seed=42)
    features = [c for c in df.columns if c not in ("morte_30d", "_patient_key")]

    artifacts = train_and_select_model(df, features)

    assert artifacts is not None
    assert artifacts.best_model_name is not None
    assert not artifacts.leaderboard.empty

    best_auc = float(artifacts.leaderboard.iloc[0]["AUC"])
    assert best_auc > 0.55, (
        f"Best model AUC {best_auc:.3f} ≤ 0.55 on synthetic data with known signal "
        "— possible preprocessing regression"
    )

    best_brier = float(artifacts.leaderboard.iloc[0]["Brier"])
    assert best_brier < 0.30, (
        f"Best model Brier {best_brier:.3f} ≥ 0.30 — possible calibration regression"
    )
