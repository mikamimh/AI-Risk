"""Tests enforcing the Race exclusion policy for the AI Risk predictive model.

Race is excluded from AI Risk model features (EXCLUDED_ETHICAL_COLUMNS /
NEVER_FEATURE_COLUMNS) but retained in the analytical dataset for:
  - STS Score input mapping (raceblack / raceasian fields)
  - Cohort description and fairness/subgroup analyses
  - Audit exports

These tests lock the policy in place so any accidental re-inclusion of Race
as a model feature is caught immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import risk_data
from risk_data import (
    EXCLUDED_ETHICAL_COLUMNS,
    NEVER_FEATURE_COLUMNS,
    FLAT_PREOP_ALLOWED_COLUMNS,
    prepare_flat_dataset,
)
from variable_contract import VARIABLE_CONTRACT
from variable_dictionary import VARIABLE_DICTIONARY


# ---------------------------------------------------------------------------
# 1. Policy enforcement — exclusion sets
# ---------------------------------------------------------------------------

def test_race_in_excluded_ethical_columns():
    """Race must be in EXCLUDED_ETHICAL_COLUMNS — the dedicated policy set
    for variables excluded on statistical/ethical grounds."""
    assert "Race" in EXCLUDED_ETHICAL_COLUMNS


def test_race_in_never_feature_columns():
    """EXCLUDED_ETHICAL_COLUMNS feeds into NEVER_FEATURE_COLUMNS; Race must
    therefore be barred from becoming a model feature by the belt-and-suspenders
    guard that runs in both prepare_flat_dataset and prepare_master_dataset."""
    assert "Race" in NEVER_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 2. Data retention — Race must stay available for STS and audit
# ---------------------------------------------------------------------------

def test_race_retained_in_flat_preop_allowed_columns():
    """Race must remain in FLAT_PREOP_ALLOWED_COLUMNS so it is loaded into
    the analytical DataFrame — needed for STS mapping, audit, and fairness."""
    assert "Race" in FLAT_PREOP_ALLOWED_COLUMNS


def test_race_retained_in_variable_contract():
    """Race must remain in VARIABLE_CONTRACT to be parsed and ingested
    correctly from source files (canonical name resolution, tolerant parsing,
    unknown-blank semantics)."""
    assert "Race" in VARIABLE_CONTRACT


# ---------------------------------------------------------------------------
# 3. Documentation — variable dictionary must reflect exclusion
# ---------------------------------------------------------------------------

def test_race_not_marked_as_active_predictor_in_variable_dictionary():
    """The variable dictionary must not list Race as an active predictor
    (in_model == 'Yes').  The entry must start with 'No' to clearly
    communicate the exclusion to consumers of the dictionary export."""
    race_entry = next(
        (e for e in VARIABLE_DICTIONARY if e["variable"] == "Race"), None
    )
    assert race_entry is not None, "Race entry missing from VARIABLE_DICTIONARY"
    assert race_entry["in_model"].startswith("No"), (
        f"Race in_model should start with 'No', got: {race_entry['in_model']!r}"
    )


# ---------------------------------------------------------------------------
# 4. STS mapping — Race must still be usable by sts_calculator
# ---------------------------------------------------------------------------

def test_sts_input_builder_maps_race_black_to_raceblack_yes():
    """When the patient row contains a value with 'Black' in it (e.g.
    'Black/African American'), the STS input builder must set raceblack='Yes'.
    The STS calculator checks ``'Black' in race`` — 'African American' alone
    does NOT contain that substring and maps to empty."""
    from sts_calculator import build_sts_input_from_row
    row = {"Race": "Black/African American"}
    result = build_sts_input_from_row(row)
    assert result.get("raceblack") == "Yes"
    assert result.get("raceasian") == ""


def test_sts_input_builder_maps_race_asian_to_raceasian_yes():
    """When the patient row contains 'Asian', the STS input builder must
    set raceasian='Yes' and leave raceblack empty."""
    from sts_calculator import build_sts_input_from_row
    row = {"Race": "Asian"}
    result = build_sts_input_from_row(row)
    assert result.get("raceasian") == "Yes"
    assert result.get("raceblack") == ""


def test_sts_input_builder_handles_missing_race_gracefully():
    """When Race is absent from the row (e.g. a new patient without Race
    recorded), raceblack and raceasian must both be empty strings — not raise."""
    from sts_calculator import build_sts_input_from_row
    row = {}
    result = build_sts_input_from_row(row)
    assert result.get("raceblack") == ""
    assert result.get("raceasian") == ""


# ---------------------------------------------------------------------------
# 5. Inference pipeline — feature_columns must not contain Race
# ---------------------------------------------------------------------------

def _make_race_source_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 60
    race_values = ["White"] * 50 + ["African American"] * 7 + ["Asian"] * 3
    rng.shuffle(race_values)
    return pd.DataFrame({
        "Name": [f"P{i}" for i in range(n)],
        "morte_30d": (rng.random(n) > 0.85).astype(int),
        "Age (years)": rng.uniform(40, 80, n),
        "Pré-LVEF, %": rng.uniform(30, 70, n),
        "Creatinine (mg/dL)": rng.uniform(0.5, 3.0, n),
        "Sex": rng.choice(["Male", "Female"], n),
        "Race": race_values,
        "Surgery": ["CABG"] * n,
        "Surgical Priority": ["Elective"] * n,
        "Procedure Date": ["2024-01-01"] * n,
    })


def test_prepare_flat_dataset_excludes_race_from_feature_columns(monkeypatch):
    """prepare_flat_dataset must not include Race in feature_columns even
    when Race is present in the input DataFrame with non-trivial variance."""
    source_df = _make_race_source_df(seed=0)
    monkeypatch.setattr(risk_data, "_read_csv_auto", lambda _path: source_df.copy())

    prepared = prepare_flat_dataset("synthetic.csv")
    assert "Race" not in prepared.feature_columns, (
        "Race must not appear in feature_columns — EXCLUDED_ETHICAL_COLUMNS "
        "should have blocked it."
    )


def test_prepare_flat_dataset_retains_race_in_data_columns(monkeypatch):
    """Even though Race is excluded from feature_columns, it must remain
    in prepared.data so it is available for STS mapping and audit."""
    source_df = _make_race_source_df(seed=1)
    monkeypatch.setattr(risk_data, "_read_csv_auto", lambda _path: source_df.copy())

    prepared = prepare_flat_dataset("synthetic.csv")
    assert "Race" in prepared.data.columns, (
        "Race must remain in prepared.data.columns for STS mapping and audit — "
        "only feature_columns should exclude it."
    )
