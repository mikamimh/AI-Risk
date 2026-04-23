"""Tests for V2: binary direct 0/1 encoding in clean_features."""

import numpy as np
import pandas as pd
import pytest
from modeling import clean_features, _BINARY_DIRECT_ENCODE_COLS, _encode_binary_direct


def test_yes_no_converted_to_numeric():
    """Yes/No string values → 1.0/0.0 float."""
    df = pd.DataFrame({
        "Hypertension": ["Yes", "No", "Yes", "No"],
        "Age (years)": [65, 70, 55, 80],
    })
    result = clean_features(df)
    assert pd.api.types.is_numeric_dtype(result["Hypertension"])
    assert list(result["Hypertension"]) == [1.0, 0.0, 1.0, 0.0]


def test_portuguese_tokens_converted():
    """Sim/Não → 1.0/0.0 for a column in the binary set."""
    df = pd.DataFrame({
        "Hypertension": ["Sim", "Não", "sim", "nao"],
    })
    result = _encode_binary_direct(df)
    assert list(result["Hypertension"]) == [1.0, 0.0, 1.0, 0.0]


def test_unknown_value_becomes_nan():
    """Values not in positive or negative token sets → NaN."""
    df = pd.DataFrame({
        "Hypertension": ["Yes", None, "Unknown", "No"],
    })
    result = clean_features(df)
    assert result["Hypertension"].iloc[0] == 1.0
    assert pd.isna(result["Hypertension"].iloc[1])
    assert pd.isna(result["Hypertension"].iloc[2])
    assert result["Hypertension"].iloc[3] == 0.0


def test_already_numeric_column_untouched():
    """Columns already numeric (from XLSX float read) are not re-converted."""
    df = pd.DataFrame({
        "Hypertension": [1.0, 0.0, 1.0, np.nan],
        "Age (years)": [65, 70, 55, 80],
    })
    result = clean_features(df)
    # Values must be unchanged
    assert result["Hypertension"].iloc[0] == 1.0
    assert result["Hypertension"].iloc[1] == 0.0
    assert pd.isna(result["Hypertension"].iloc[3])


def test_non_binary_column_not_affected():
    """_encode_binary_direct should not touch Surgery (not in binary set)."""
    df = pd.DataFrame({
        "Surgery": ["CABG", "AVR", "MVR"],
        "Hypertension": ["Yes", "No", "Yes"],
    })
    result = _encode_binary_direct(df)
    # Surgery untouched by binary encoder
    assert list(result["Surgery"]) == ["CABG", "AVR", "MVR"]
    # Hypertension converted
    assert list(result["Hypertension"]) == [1.0, 0.0, 1.0]


def test_all_binary_cols_covered():
    """All _BINARY_DIRECT_ENCODE_COLS are converted when present."""
    data = {col: ["Yes", "No"] for col in _BINARY_DIRECT_ENCODE_COLS}
    df = pd.DataFrame(data)
    result = _encode_binary_direct(df)
    for col in _BINARY_DIRECT_ENCODE_COLS:
        assert pd.api.types.is_numeric_dtype(result[col]), f"{col} not numeric"
        assert result[col].iloc[0] == 1.0
        assert result[col].iloc[1] == 0.0


def test_non_binary_columns_excluded():
    """Columns with >2 clinical categories must NOT be in binary encode set."""
    excluded = {
        "Diabetes", "CVA", "IE", "Cancer ≤ 5 yrs",
        "Anticoagulation/ Antiaggregation", "Pneumonia",
        "Family Hx of CAD",
    }
    for col in excluded:
        assert col not in _BINARY_DIRECT_ENCODE_COLS, (
            f"{col} should NOT be in _BINARY_DIRECT_ENCODE_COLS — "
            "it has multi-level clinical categories that must not be collapsed to NaN"
        )


def test_diabetes_insulin_preserved():
    """Diabetes 'Insulin' must survive _encode_binary_direct as a string."""
    df = pd.DataFrame({
        "Diabetes": ["No", "Oral", "Insulin", "Diet Only"],
        "Age (years)": [65, 70, 55, 80],
    })
    result = _encode_binary_direct(df)
    # Diabetes not in binary set → unchanged string
    assert list(result["Diabetes"]) == ["No", "Oral", "Insulin", "Diet Only"]


def test_encode_binary_direct_missing_column_ignored():
    """Columns not in the DataFrame are silently skipped."""
    df = pd.DataFrame({"Age (years)": [65, 70]})
    result = _encode_binary_direct(df)
    assert list(result.columns) == ["Age (years)"]


def test_integer_dtype_already_numeric_skipped():
    """int64 columns (already numeric) are not re-processed."""
    df = pd.DataFrame({
        "Hypertension": pd.array([1, 0, 1], dtype="int64"),
    })
    result = _encode_binary_direct(df)
    assert list(result["Hypertension"]) == [1, 0, 1]
