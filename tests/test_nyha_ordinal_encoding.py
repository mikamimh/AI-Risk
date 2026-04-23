"""Tests for NYHA ordinal encoding in build_preprocessor."""

import pandas as pd
import pytest
from modeling import build_preprocessor, _NYHA_COLS, _NYHA_ORDER


def _make_df():
    return pd.DataFrame({
        "Age (years)": [65.0, 70.0, 55.0, 80.0],
        "Preoperative NYHA": ["I", "II", "III", "IV"],
        "Surgery": ["CABG", "AVR", "CABG", "MVR"],
    })


def test_nyha_has_own_ordinal_transformer():
    """NYHA should be in a dedicated 'nyha' transformer, not in 'cat'."""
    prep = build_preprocessor(_make_df())
    transformer_names = [name for name, _, _ in prep.transformers]
    assert "nyha" in transformer_names, (
        f"Expected 'nyha' transformer, got {transformer_names}"
    )


def test_nyha_not_in_categorical_transformer():
    """Preoperative NYHA must not be in the TargetEncoder categorical pipe."""
    prep = build_preprocessor(_make_df())
    for name, _, cols in prep.transformers:
        if name == "cat":
            assert "Preoperative NYHA" not in cols, (
                "NYHA should not be in the categorical (TargetEncoder) pipe"
            )


def test_nyha_absent_column_skipped():
    """If NYHA is not present in X, no 'nyha' transformer is added."""
    df = pd.DataFrame({
        "Age (years)": [65.0, 70.0],
        "Surgery": ["CABG", "AVR"],
    })
    prep = build_preprocessor(df)
    transformer_names = [name for name, _, _ in prep.transformers]
    assert "nyha" not in transformer_names


def test_nyha_constants():
    """_NYHA_COLS and _NYHA_ORDER must have expected values."""
    assert _NYHA_COLS == ["Preoperative NYHA"]
    assert _NYHA_ORDER == ["I", "II", "III", "IV"]
