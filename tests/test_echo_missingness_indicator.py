"""Tests for V4: echo missingness indicator in MISSINGNESS_INDICATOR_SPECS."""

import pandas as pd
import numpy as np
from risk_data import (
    MISSINGNESS_INDICATOR_SPECS,
    MISSINGNESS_INDICATOR_COLUMNS,
    add_missingness_indicators,
)


def test_missing_echo_key_in_specs():
    assert "missing_echo_key" in MISSINGNESS_INDICATOR_SPECS


def test_missing_echo_key_in_columns():
    assert "missing_echo_key" in MISSINGNESS_INDICATOR_COLUMNS


def test_missing_echo_key_correct_source_cols():
    assert MISSINGNESS_INDICATOR_SPECS["missing_echo_key"] == (
        "Pré-LVEF, %",
        "Aortic Stenosis",
        "Mitral Regurgitation",
    )


def test_indicator_1_when_all_echo_missing():
    """When all three key echo vars are NaN, missing_echo_key = 1."""
    df = pd.DataFrame({
        "Pré-LVEF, %": [np.nan],
        "Aortic Stenosis": [np.nan],
        "Mitral Regurgitation": [np.nan],
        "morte_30d": [0],
    })
    result = add_missingness_indicators(df)
    assert result["missing_echo_key"].iloc[0] == 1


def test_indicator_1_when_any_echo_missing():
    """OR logic: even one missing echo var triggers the indicator."""
    df = pd.DataFrame({
        "Pré-LVEF, %": [np.nan],    # missing
        "Aortic Stenosis": ["Mild"], # present
        "Mitral Regurgitation": ["None"],  # present (valid categorical)
        "morte_30d": [0],
    })
    result = add_missingness_indicators(df)
    # LVEF is missing → OR → indicator = 1
    assert result["missing_echo_key"].iloc[0] == 1


def test_indicator_0_when_all_echo_present():
    """When all three vars are present and non-missing, indicator = 0."""
    df = pd.DataFrame({
        "Pré-LVEF, %": [55.0],
        "Aortic Stenosis": ["Mild"],
        "Mitral Regurgitation": ["None"],
        "morte_30d": [1],
    })
    result = add_missingness_indicators(df)
    assert result["missing_echo_key"].iloc[0] == 0


def test_indicator_mixed_rows():
    """Multiple rows — OR logic per row."""
    df = pd.DataFrame({
        "Pré-LVEF, %": [np.nan, 60.0, 55.0],
        "Aortic Stenosis": [np.nan, "None", "Mild"],
        "Mitral Regurgitation": [np.nan, np.nan, "None"],
        "morte_30d": [0, 1, 0],
    })
    result = add_missingness_indicators(df)
    # Row 0: all NaN → 1
    assert result["missing_echo_key"].iloc[0] == 1
    # Row 1: LVEF present, AS present, MR missing → 1 (OR logic)
    assert result["missing_echo_key"].iloc[1] == 1
    # Row 2: all present → 0
    assert result["missing_echo_key"].iloc[2] == 0


def test_existing_indicators_still_present():
    """Existing indicators (renal, CBC, coagulation) must still be computed."""
    df = pd.DataFrame({
        "Creatinine (mg/dL)": [np.nan],
        "Cr clearance, ml/min *": [np.nan],
        "Hematocrit (%)": [38.0],
        "WBC Count (10³/μL)": [np.nan],
        "Platelet Count (cells/μL)": [np.nan],
        "INR": [np.nan],
        "PTT": [np.nan],
        "Pré-LVEF, %": [np.nan],
        "Aortic Stenosis": [np.nan],
        "Mitral Regurgitation": [np.nan],
        "morte_30d": [0],
    })
    result = add_missingness_indicators(df)
    assert "missing_renal_labs" in result.columns
    assert "missing_cbc_labs" in result.columns
    assert "missing_coagulation_labs" in result.columns
    assert "missing_echo_key" in result.columns
