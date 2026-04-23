"""Canonical variable semantic contract for AI Risk.

Each entry declares parse mode, missing-value semantics, plausibility
ranges, and categorical valid values for a canonical preoperative column.

Usage in risk_data.py:
    from variable_contract import VARIABLE_CONTRACT

Boolean fields:
    none_is_valid              -- "None" is a valid category (LITERAL set)
    ordinal_encoding_none_valid -- valve disease columns only; "None" = no
                                   disease, participates in ordinal encoding
                                   (NONE_IS_VALID_COLUMNS set)
    blank_impute_no            -- blank cells → "No" during ingestion
                                   (registry convention: condition absent
                                   when not documented; dtype-independent)
"""

from typing import Any, Dict

# Sentinel: resolve to the global MISSING_TOKENS from risk_data at call time.
_GLOBAL_MISSING = "global"

VARIABLE_CONTRACT: Dict[str, Dict[str, Any]] = {

    # ── Demographic ──────────────────────────────────────────────────────
    "Age (years)": {
        "dtype": "numeric",
        "parse_mode": "strict",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0, 120),
        "unit": "years",
    },
    "Sex": {
        "dtype": "categorical",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "valid_categories": ["Male", "Female"],
    },
    "Height (cm)": {
        "dtype": "numeric",
        "parse_mode": "strict",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (50, 250),
        "unit": "cm",
    },
    "Weight (kg)": {
        "dtype": "numeric",
        "parse_mode": "strict",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (10, 500),
        "unit": "kg",
    },
    "BSA, m2": {
        "dtype": "numeric",
        "parse_mode": "strict",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0.5, 5.0),
        "unit": "m²",
    },
    "Race": {
        "dtype": "categorical",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },

    # ── Surgical context ─────────────────────────────────────────────────
    "Surgical Priority": {
        "dtype": "categorical",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "valid_categories": ["Elective", "Urgent", "Emergency", "Salvage"],
    },
    "Surgery": {
        "dtype": "text",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "Previous surgery": {
        "dtype": "text",
        "parse_mode": "tolerant",
        "blank_semantics": "absent",   # blank = no prior surgery → "None"
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": True,
    },

    # ── Cardiac history ──────────────────────────────────────────────────
    "IE": {
        "dtype": "categorical",  # No / Yes / Possible — not binary
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "No. of Diseased Vessels": {
        "dtype": "numeric",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0, 3),
    },
    "Left Main Stenosis ≥ 50%": {
        "dtype": "binary",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "Proximal LAD Stenosis ≥ 70%": {
        "dtype": "binary",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "Coronary Symptom": {
        "dtype": "categorical",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "valid_categories": [
            "No coronary symptoms", "Stable Angina", "Unstable Angina",
            "Non-STEMI", "STEMI", "Angina Equivalent", "Other",
        ],
    },
    "Preoperative NYHA": {
        "dtype": "ordinal",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "valid_categories": ["I", "II", "III", "IV"],
    },
    "CCS4": {
        "dtype": "binary",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "Pré-LVEF, %": {
        "dtype": "numeric",
        "parse_mode": "strict",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (1.0, 100.0),
        "unit": "%",
    },
    "Classification of Heart Failure According to Ejection Fraction": {
        "dtype": "categorical",
        "parse_mode": "tolerant",
        "blank_semantics": "not_applicable",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "HF": {
        "dtype": "categorical",
        "parse_mode": "tolerant",
        "blank_semantics": "absent",   # blank = no HF → "None"
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": True,
    },
    "Arrhythmia Remote": {
        "dtype": "categorical",
        "parse_mode": "tolerant",
        "blank_semantics": "absent",   # blank = no prior arrhythmia → "None"
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": True,
    },
    "Arrhythmia Recent": {
        "dtype": "categorical",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": True,
    },

    # ── Comorbidities ────────────────────────────────────────────────────
    "Hypertension": {
        "dtype": "binary",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "Diabetes": {
        "dtype": "categorical",  # No / Oral / Insulin / Diet Only / No Control Method
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "Dyslipidemia": {
        "dtype": "binary",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "CVA": {
        "dtype": "categorical",  # No / TIA / ≤ 30 days / ≥ 30 days
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "PVD": {
        "dtype": "binary",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "Alcohol": {
        "dtype": "binary",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "Smoking (Pack-year)": {
        "dtype": "categorical",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "valid_categories": ["Never", "Former", "Current"],
    },
    "Cancer ≤ 5 yrs": {
        "dtype": "categorical",  # No / plus specific cancer types (Bowel, Breast, …)
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "Family Hx of CAD": {
        "dtype": "binary",
        "parse_mode": "tolerant",
        "blank_semantics": "absent",   # registry convention: documented when positive
        "blank_impute_no": True,       # blank → "No" regardless of dtype
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "Anticoagulation/ Antiaggregation": {
        "dtype": "categorical",  # No / plus medication regimens (AAS, Clopidogrel, …)
        "parse_mode": "tolerant",
        "blank_semantics": "absent",   # registry convention: documented when positive
        "blank_impute_no": True,       # blank → "No" regardless of dtype
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "Suspension of Anticoagulation (day)": {
        "dtype": "numeric",
        "parse_mode": "tolerant",
        "blank_semantics": "not_applicable",  # conditional on Anticoagulation=Yes
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0, 365),
        "unit": "days",
    },
    "Pneumonia": {
        "dtype": "categorical",  # No / Treated / Under treatment
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "Dialysis": {
        "dtype": "binary",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "KDIGO †": {
        "dtype": "categorical",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "Chronic Lung Disease": {
        "dtype": "binary",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "Critical preoperative state": {
        "dtype": "binary",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },
    "Poor mobility": {
        "dtype": "binary",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
    },

    # ── Laboratory values ────────────────────────────────────────────────
    "Cr clearance, ml/min *": {
        "dtype": "numeric",
        "parse_mode": "strict",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0.0, 500.0),
        "unit": "ml/min",
    },
    "Creatinine (mg/dL)": {
        "dtype": "numeric",
        "parse_mode": "strict",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0.05, 50.0),
        "unit": "mg/dL",
    },
    "Hematocrit (%)": {
        "dtype": "numeric",
        "parse_mode": "strict",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (1.0, 100.0),
        "unit": "%",
    },
    "WBC Count (10³/μL)": {
        "dtype": "numeric",
        "parse_mode": "strict",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0.1, 500.0),
        "unit": "10³/μL",
    },
    "Platelet Count (cells/μL)": {
        "dtype": "numeric",
        "parse_mode": "strict",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (1.0, 2_000_000.0),
        "unit": "cells/μL",
    },
    "INR": {
        "dtype": "numeric",
        "parse_mode": "strict",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0.5, 20.0),
        "unit": "ratio",
    },
    "PTT": {
        "dtype": "numeric",
        "parse_mode": "strict",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (1.0, 500.0),
        "unit": "seconds",
    },

    # ── Echocardiographic — valve disease ────────────────────────────────
    # ordinal_encoding_none_valid=True: "None" = no disease, participates in
    # ordinal TargetEncoder (NONE_IS_VALID_COLUMNS in risk_data.py).
    "Aortic Stenosis": {
        "dtype": "ordinal",
        "parse_mode": "categorical",
        "blank_semantics": "absent",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": True,
        "ordinal_encoding_none_valid": True,
        "valid_categories": ["None", "Trivial", "Mild", "Moderate", "Severe"],
    },
    "Aortic Mean gradient (mmHg)": {
        "dtype": "numeric",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0.0, 250.0),
        "unit": "mmHg",
    },
    "AVA (cm²)": {
        "dtype": "numeric",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0.01, 10.0),
        "unit": "cm²",
    },
    "Aortic Regurgitation": {
        "dtype": "ordinal",
        "parse_mode": "categorical",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": True,
        "ordinal_encoding_none_valid": True,
        "valid_categories": ["None", "Trivial", "Mild", "Moderate", "Severe"],
    },
    "Vena contracta": {
        "dtype": "numeric",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0.0, 50.0),
        "unit": "cm",
    },
    "PHT Aortic": {
        "dtype": "numeric",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "unit": "ms",
    },
    "Mitral Stenosis": {
        "dtype": "ordinal",
        "parse_mode": "categorical",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": True,
        "ordinal_encoding_none_valid": True,
        "valid_categories": ["None", "Trivial", "Mild", "Moderate", "Severe"],
    },
    "Mitral Mean gradient (mmHg)": {
        "dtype": "numeric",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0.0, 100.0),
        "unit": "mmHg",
    },
    "MVA (cm²)": {
        "dtype": "numeric",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0.01, 15.0),
        "unit": "cm²",
    },
    "Mitral Regurgitation": {
        "dtype": "ordinal",
        "parse_mode": "categorical",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": True,
        "ordinal_encoding_none_valid": True,
        "valid_categories": ["None", "Trivial", "Mild", "Moderate", "Severe"],
    },
    "Vena contracta (mm)": {
        "dtype": "numeric",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0.0, 50.0),
        "unit": "mm",
    },
    "PHT Mitral": {
        "dtype": "numeric",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "unit": "ms",
    },
    "Tricuspid Regurgitation": {
        "dtype": "ordinal",
        "parse_mode": "categorical",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": True,
        "ordinal_encoding_none_valid": True,
        "valid_categories": ["None", "Trivial", "Mild", "Moderate", "Severe"],
    },
    "PSAP": {
        "dtype": "numeric",
        "parse_mode": "strict",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0.0, 250.0),
        "unit": "mmHg",
    },
    "TAPSE": {
        "dtype": "numeric",
        "parse_mode": "strict",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": False,
        "plausible_range": (0.0, 60.0),
        "unit": "mm",
    },
    "Aortic Root Abscess": {
        "dtype": "categorical",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": True,
    },

    # ── Medication ───────────────────────────────────────────────────────
    "Preoperative Medications": {
        "dtype": "text",
        "parse_mode": "tolerant",
        "blank_semantics": "unknown",
        "missing_tokens": _GLOBAL_MISSING,
        "none_is_valid": True,
    },
}
