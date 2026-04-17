"""Shared pytest fixtures for AI Risk test suite.

MIMIC-like synthetic fixture
----------------------------
``mimic_like_df`` / ``mimic_like_raw_df`` provide a compact synthetic dataset
designed to exercise all external-normalization edge cases in one place:

  * Imperial heights (inches, median ≈ 67 in → should be converted to cm)
  * Imperial weights (lb, median ≈ 185 lb → should be converted to kg)
  * Pediatric rows (age < 18) — must be excluded from adult STS scope
  * Mixed token variants (Sim/Não, oui/non, ja/nein alongside Yes/No)
  * Out-of-scope STS surgeries (Bentall, dissection, Ross, transplant, homograft)
  * Supported adult STS surgeries (ISOLATED CABG, AVR, MVR, AVR + CABG)

Expected normalization outcomes:
  - height_converted = True  (all heights are in inches range < 100)
  - weight_converted = True  (all weights are in lb range, median ≈ 185)
  - n_pediatric ≥ 2          (rows with age 14 and 16)
  - n_sts_scope_excluded ≥ 5 (Bentall, dissection, ROSS, transplant, homograft)
  - n_sts_ready ≤ n_total - 2 (at minimum pediatric rows excluded)
"""
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Raw (un-normalized) MIMIC-like DataFrame
# ──────────────────────────────────────────────────────────────────────────────

_MIMIC_ROWS = [
    # Supported adult surgeries — should end up STS-ready (if fields complete)
    {
        "Age (years)": 65, "Sex": "Male",
        "Surgery": "ISOLATED CABG", "Surgical Priority": "Elective",
        "Height (cm)": 67.0, "Weight (kg)": 185.0,
        "Hypertension": "Sim",       # Portuguese Yes → Yes
        "Diabetes": "Yes",
        "COPD": "No",
        "Mortality30": 0,
    },
    {
        "Age (years)": 72, "Sex": "Female",
        "Surgery": "AVR", "Surgical Priority": "Urgent",
        "Height (cm)": 63.0, "Weight (kg)": 155.0,
        "Hypertension": "oui",       # French Yes → Yes
        "Diabetes": "non",           # French No → No
        "COPD": "No",
        "Mortality30": 0,
    },
    {
        "Age (years)": 58, "Sex": "Male",
        "Surgery": "MVR", "Surgical Priority": "Elective",
        "Height (cm)": 70.0, "Weight (kg)": 200.0,
        "Hypertension": "ja",        # German Yes → Yes
        "Diabetes": "nein",          # German No → No
        "COPD": "No",
        "Mortality30": 1,
    },
    {
        "Age (years)": 68, "Sex": "Female",
        "Surgery": "AVR + CABG", "Surgical Priority": "Elective",
        "Height (cm)": 62.0, "Weight (kg)": 165.0,
        "Hypertension": "Não",       # Portuguese No → No
        "Diabetes": "No",
        "COPD": "Yes",
        "Mortality30": 0,
    },
    {
        "Age (years)": 55, "Sex": "Male",
        "Surgery": "ISOLATED CABG", "Surgical Priority": "Urgent",
        "Height (cm)": 69.0, "Weight (kg)": 265.0,  # heavy patient, still in lb range
        "Hypertension": "Yes",
        "Diabetes": "Yes",
        "COPD": "No",
        "Mortality30": 0,
    },
    # Out-of-scope surgeries — sts_scope_excluded must be True
    {
        "Age (years)": 60, "Sex": "Male",
        "Surgery": "BENTALL PROCEDURE", "Surgical Priority": "Elective",
        "Height (cm)": 71.0, "Weight (kg)": 190.0,
        "Hypertension": "Yes", "Diabetes": "No", "COPD": "No",
        "Mortality30": 0,
    },
    {
        "Age (years)": 55, "Sex": "Female",
        "Surgery": "AORTIC DISSECTION REPAIR", "Surgical Priority": "Emergency",
        "Height (cm)": 65.0, "Weight (kg)": 145.0,
        "Hypertension": "No", "Diabetes": "No", "COPD": "No",
        "Mortality30": 1,
    },
    {
        "Age (years)": 48, "Sex": "Male",
        "Surgery": "ROSS PROCEDURE", "Surgical Priority": "Elective",
        "Height (cm)": 72.0, "Weight (kg)": 175.0,
        "Hypertension": "Yes", "Diabetes": "No", "COPD": "No",
        "Mortality30": 0,
    },
    {
        "Age (years)": 52, "Sex": "Male",
        "Surgery": "CARDIAC TRANSPLANT", "Surgical Priority": "Urgent",
        "Height (cm)": 68.0, "Weight (kg)": 170.0,
        "Hypertension": "Yes", "Diabetes": "Yes", "COPD": "No",
        "Mortality30": 1,
    },
    {
        "Age (years)": 63, "Sex": "Female",
        "Surgery": "HOMOGRAFT REPLACEMENT", "Surgical Priority": "Elective",
        "Height (cm)": 61.0, "Weight (kg)": 150.0,
        "Hypertension": "No", "Diabetes": "No", "COPD": "No",
        "Mortality30": 0,
    },
    # Pediatric rows — must be flagged is_pediatric=True and excluded from STS
    {
        "Age (years)": 14, "Sex": "Male",
        "Surgery": "ISOLATED CABG", "Surgical Priority": "Elective",
        "Height (cm)": 60.0, "Weight (kg)": 120.0,
        "Hypertension": "No", "Diabetes": "No", "COPD": "No",
        "Mortality30": 0,
    },
    {
        "Age (years)": 16, "Sex": "Female",
        "Surgery": "AVR", "Surgical Priority": "Urgent",
        "Height (cm)": 58.0, "Weight (kg)": 110.0,
        "Hypertension": "No", "Diabetes": "No", "COPD": "No",
        "Mortality30": 0,
    },
]

_MIMIC_DF = pd.DataFrame(_MIMIC_ROWS)


@pytest.fixture
def mimic_like_raw_df() -> pd.DataFrame:
    """Raw (un-normalized) MIMIC-like DataFrame with imperial units and mixed tokens."""
    return _MIMIC_DF.copy()


@pytest.fixture
def mimic_like_normalized():
    """Normalized DataFrame + ExternalNormalizationReport from the MIMIC-like fixture.

    Runs the full ``normalize_external_dataset`` pipeline over ``_MIMIC_DF``.
    Use this fixture when you need the post-normalization state and report
    without re-running the pipeline in every test.
    """
    from risk_data import normalize_external_dataset, ExternalReadMeta
    read_meta = ExternalReadMeta(
        encoding_used="utf-8",
        delimiter=",",
        rows_loaded=len(_MIMIC_DF),
        columns_loaded=len(_MIMIC_DF.columns),
    )
    df_norm, report = normalize_external_dataset(
        _MIMIC_DF.copy(), source_name="mimic_like_fixture.csv", read_meta=read_meta
    )
    return df_norm, report
