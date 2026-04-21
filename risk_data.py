"""Data loading, validation, and preparation for AI Risk.

This module handles:
- Reading Excel, CSV, SQLite, and Parquet data sources
- Column mapping and standardization
- Data validation and eligibility criteria
- Patient matching across multiple data tables
- Feature engineering and preprocessing

Example:
    >>> from risk_data import prepare_master_dataset
    >>> prepared = prepare_master_dataset("patient_data.xlsx")
    >>> df = prepared.data
    >>> features = prepared.feature_columns
"""

import csv as _csv_module
import re
import sqlite3
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sts_calculator import STS_UNSUPPORTED_SURGERY_KEYWORDS


MISSING_TOKENS = {
    "",
    "-",
    "--",
    "nan",
    "none",
    "na",
    "n/a",
    "null",
    "not applicable",
    "unknown",
    "not informed",
    "não informado",
}

# Pandas treats the literal string "None" as NA by default.  That is unsafe for
# clinical categorical fields where "None" means absence of disease/condition.
PANDAS_PRESERVE_NONE_READ_KWARGS = {"keep_default_na": False}

NONE_IS_VALID_COLUMNS = {
    "Aortic Stenosis", "Aortic Regurgitation",
    "Mitral Stenosis", "Mitral Regurgitation",
    "Tricuspid Regurgitation",
    "aortic_stenosis_pre", "aortic_regurgitation_pre",
    "mitral_stenosis_pre", "mitral_regurgitation_pre",
    "tricuspid_regurgitation_pre",
    "aortic_stenosis_post", "aortic_regurgitation_post",
}

LITERAL_NONE_IS_VALID_COLUMNS = NONE_IS_VALID_COLUMNS | {
    "Arrhythmia Recent",
    "Arrhythmia Remote",
    "Aortic Root Abscess",
    "HF",
    "Preoperative Medications",
    "Previous surgery",
}

# Binary history variables where a blank cell in the source data means the
# condition is absent (implicit negative), NOT that the information is unknown.
# This convention is standard in cardiac surgery registries (STS, EuroSCORE II):
# if a historical flag is positive, it is always explicitly documented; blank
# entries are used as shorthand for "No" by the data entry workflow.
#
# Populated by _impute_blank_as_no() AFTER normalize_dataframe() so that all
# MISSING_TOKENS have already been standardised to NaN.
#
# Excluded from this set (intentionally):
#   "Suspension of Anticoagulation (day)" — numeric, conditional on
#   Anticoagulation=Yes; blank = N/A, not zero days.
BLANK_MEANS_NO_COLUMNS: frozenset = frozenset({
    "Family Hx of CAD",
    "Anticoagulation/ Antiaggregation",
})


BLANK_MEANS_NONE_COLUMNS: frozenset = frozenset({
    "Aortic Stenosis",
    "Arrhythmia Remote",
    "HF",
    "Previous surgery",
})

CORONARY_SYMPTOM_CANONICAL_VALUES: Dict[str, str] = {
    "none": "No coronary symptoms",
    "no symptoms": "No coronary symptoms",
    "no coronary symptoms": "No coronary symptoms",
    "sem sintomas coronarianos": "No coronary symptoms",
    "stable angina": "Stable Angina",
    "angina estavel": "Stable Angina",
    "angina estável": "Stable Angina",
    "unstable angina": "Unstable Angina",
    "angina instavel": "Unstable Angina",
    "angina instável": "Unstable Angina",
    "nstemi": "Non-STEMI",
    "non-stemi": "Non-STEMI",
    "iam sem supra de st": "Non-STEMI",
    "stemi": "STEMI",
    "iam com supra de st": "STEMI",
    "angina equivalent": "Angina Equivalent",
    "equivalente anginoso": "Angina Equivalent",
    "other": "Other",
    "outro": "Other",
}


def normalize_coronary_symptom_value(value: object) -> object:
    """Return the canonical Coronary Symptom category for known exact labels.

    The literal string ``"None"`` is a valid coronary-presentation category
    meaning no coronary symptoms. It must be canonicalized before generic
    missing-token handling, where ``"none"`` otherwise means missing for most
    columns. True blanks and textual unknown tokens are intentionally left for
    the regular missing-value pipeline.
    """
    if pd.isna(value):
        return value
    text = str(value).strip()
    if text == "":
        return value
    return CORONARY_SYMPTOM_CANONICAL_VALUES.get(text.lower(), value)


def _normalize_coronary_symptom_column(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize only the Coronary Symptom field before missing normalization."""
    if "Coronary Symptom" not in df.columns:
        return df
    out = df.copy()
    out["Coronary Symptom"] = out["Coronary Symptom"].map(normalize_coronary_symptom_value)
    return out


ARRHYTHMIA_RECENT_CANONICAL_VALUES: Dict[str, str] = {
    "none": "None",
    "no": "None",
    "no recent arrhythmia": "None",
    "sem arritmia recente": "None",
    "atrial fibrillation": "Atrial Fibrillation",
    "af": "Atrial Fibrillation",
    "fa": "Atrial Fibrillation",
    "atrial flutter": "Atrial Flutter",
    "flutter": "Atrial Flutter",
    "v tach / v fib": "V. Tach / V. Fib",
    "vt/vf": "V. Tach / V. Fib",
    "ventricular tachycardia": "V. Tach / V. Fib",
    "ventricular fibrillation": "V. Tach / V. Fib",
    "third degree block": "3rd Degree Block",
    "3rd degree block": "3rd Degree Block",
    "3 degree block": "3rd Degree Block",
}


def _normalize_arrhythmia_recent_column(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize Arrhythmia Recent before missing-token normalization."""
    if "Arrhythmia Recent" not in df.columns:
        return df
    out = df.copy()
    out["Arrhythmia Recent"] = out["Arrhythmia Recent"].map(normalize_arrhythmia_recent_value)
    return out


# Severity order for multi-value resolution (highest index = most severe)
_ARRHYTHMIA_SEVERITY = ["None", "Atrial Flutter", "Atrial Fibrillation", "3rd Degree Block", "V. Tach / V. Fib"]

_ARRHYTHMIA_CANONICAL_MAP: Dict[str, str] = {
    "none": "None",
    "no": "None",
    "no remote arrhythmia": "None",
    "no recent arrhythmia": "None",
    "sem arritmia remota": "None",
    "sem arritmia recente": "None",
    "atrial fibrillation": "Atrial Fibrillation",
    "af": "Atrial Fibrillation",
    "fa": "Atrial Fibrillation",
    "atrial flutter": "Atrial Flutter",
    "flutter": "Atrial Flutter",
    "v. tach / v. fib": "V. Tach / V. Fib",
    "v tach / v fib": "V. Tach / V. Fib",
    "vt/vf": "V. Tach / V. Fib",
    "ventricular tachycardia": "V. Tach / V. Fib",
    "ventricular fibrillation": "V. Tach / V. Fib",
    "third degree block": "3rd Degree Block",
    "3rd degree block": "3rd Degree Block",
    "3 degree block": "3rd Degree Block",
    # Encoding-corrupted variants: superscript "rd" → "??" via latin-1
    "3?? degree block": "3rd Degree Block",
    "3? degree block": "3rd Degree Block",
}

# Keep old name pointing to shared map for backwards compat
ARRHYTHMIA_REMOTE_CANONICAL_VALUES = _ARRHYTHMIA_CANONICAL_MAP
ARRHYTHMIA_RECENT_CANONICAL_VALUES = _ARRHYTHMIA_CANONICAL_MAP


def _canonicalize_single_arrhythmia(token: str) -> str:
    """Canonicalize one arrhythmia token; unknown tokens returned as-is."""
    lower = token.strip().lower()
    if lower in MISSING_TOKENS and lower != "none":
        return ""
    # Excel preserves superscript ordinal markers such as "3ʳᵈ"; CSV exports
    # may corrupt them to "3??". Normalize both to the same canonical token.
    lower = lower.translate(str.maketrans({
        "ʳ": "r",
        "ᵈ": "d",
        "ᵗ": "t",
        "ʰ": "h",
        "ˢ": "s",
        "ⁿ": "n",
        "ᵒ": "o",
    }))
    # Regex fallback: any '3' followed by non-alpha chars + 'degree block'
    import re as _re
    if _re.match(r"3\W{0,4}degree\s+block", lower):
        return "3rd Degree Block"
    return _ARRHYTHMIA_CANONICAL_MAP.get(lower, token.strip())


def _resolve_arrhythmia(value: object) -> object:
    """Canonicalize an arrhythmia field that may contain comma-separated values.

    Multi-value entries (e.g. 'Atrial Fibrillation, Atrial Flutter') are
    resolved to the single most severe canonical category.
    """
    if pd.isna(value):
        return value
    text = str(value).strip()
    if not text or text.lower() in MISSING_TOKENS and text.lower() != "none":
        return np.nan

    parts = [p.strip() for p in text.split(",") if p.strip()]
    canonical_parts = [_canonicalize_single_arrhythmia(p) for p in parts]
    # Drop empty (missing tokens) and "None" unless everything is None
    non_none = [c for c in canonical_parts if c and c != "None"]
    if not non_none:
        return "None"
    # Return the most severe
    def _severity(c: str) -> int:
        try:
            return _ARRHYTHMIA_SEVERITY.index(c)
        except ValueError:
            return len(_ARRHYTHMIA_SEVERITY)  # unknown → treat as most severe
    return max(non_none, key=_severity)


def normalize_arrhythmia_remote_value(value: object) -> object:
    return _resolve_arrhythmia(value)


def normalize_arrhythmia_recent_value(value: object) -> object:
    return _resolve_arrhythmia(value)


def _normalize_arrhythmia_remote_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Arrhythmia Remote" not in df.columns:
        return df
    out = df.copy()
    out["Arrhythmia Remote"] = out["Arrhythmia Remote"].map(normalize_arrhythmia_remote_value)
    return out


HF_CANONICAL_VALUES: Dict[str, str] = {
    "none": "None",
    "no": "None",
    "nenhuma": "None",
    "nao": "None",
    "não": "None",
    "acute": "Acute",
    "aguda": "Acute",
    "chronic": "Chronic",
    "cronica": "Chronic",
    "crônica": "Chronic",
    "both": "Both",
    "ambas": "Both",
    "both acute and chronic": "Both",
    "yes": "Both",
    "sim": "Both",
}


def normalize_hf_value(value: object) -> object:
    """Canonicalize HF categorical labels."""
    if pd.isna(value):
        return value
    text = str(value).strip()
    lower = text.lower()
    if lower in MISSING_TOKENS and lower != "none":
        return np.nan
    return HF_CANONICAL_VALUES.get(lower, value)


def _normalize_hf_column(df: pd.DataFrame) -> pd.DataFrame:
    if "HF" not in df.columns:
        return df
    out = df.copy()
    out["HF"] = out["HF"].map(normalize_hf_value)
    return out


NONE_ABSENCE_CANONICAL_VALUES: Dict[str, str] = {
    "none": "None",
    "no": "None",
    "nao": "None",
    "nÃ£o": "None",
    "yes": "Yes",
    "sim": "Yes",
}


def normalize_previous_surgery_value(value: object) -> object:
    """Canonicalize explicit no-prior-surgery labels without touching redo text."""
    if pd.isna(value):
        return value
    text = str(value).strip()
    lower = text.lower()
    if lower in MISSING_TOKENS and lower != "none":
        return np.nan
    return NONE_ABSENCE_CANONICAL_VALUES.get(lower, value)


def _normalize_previous_surgery_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Previous surgery" not in df.columns:
        return df
    out = df.copy()
    out["Previous surgery"] = out["Previous surgery"].map(normalize_previous_surgery_value)
    return out


def normalize_aortic_root_abscess_value(value: object) -> object:
    """Canonicalize absent aortic-root abscess as the dataset's ``None`` label."""
    if pd.isna(value):
        return value
    text = str(value).strip()
    lower = text.lower()
    if lower in MISSING_TOKENS and lower != "none":
        return np.nan
    return NONE_ABSENCE_CANONICAL_VALUES.get(lower, value)


def _normalize_aortic_root_abscess_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Aortic Root Abscess" not in df.columns:
        return df
    out = df.copy()
    out["Aortic Root Abscess"] = out["Aortic Root Abscess"].map(normalize_aortic_root_abscess_value)
    return out


CVA_CANONICAL_VALUES: Dict[str, str] = {
    "no": "No",
    "nao": "No",
    "não": "No",
    # <= 30 days — ASCII canonical + Unicode + encoding-corrupted variants
    "<= 30 days": "<= 30 days",
    "≤ 30 days": "<= 30 days",
    "? 30 days": "<= 30 days",   # ≤ corrupted by latin-1 read
    "le 30 days": "<= 30 days",
    "< 30 days": "<= 30 days",
    "30 days or less": "<= 30 days",
    "yes": "<= 30 days",          # legacy binary Yes → most conservative assumption
    "sim": "<= 30 days",
    # >= 30 days — ASCII canonical + Unicode variants
    ">= 30 days": ">= 30 days",
    "≥ 30 days": ">= 30 days",
    "ge 30 days": ">= 30 days",
    "> 30 days": ">= 30 days",
    "more than 30 days": ">= 30 days",
    # Other categories
    "timing unk": "Timing unk",
    "timing unknown": "Timing unk",
    "unknown timing": "Timing unk",
    "tia": "TIA",
    "transient ischemic attack": "TIA",
    "other cvd": "Other CVD",
    "other cerebrovascular disease": "Other CVD",
    "outra dvd": "Other CVD",
}


def normalize_cva_value(value: object) -> object:
    """Canonicalize CVA categorical labels; legacy 'Yes' maps to '≤ 30 days'."""
    if pd.isna(value):
        return value
    text = str(value).strip()
    lower = text.lower()
    if lower in MISSING_TOKENS:
        return np.nan
    return CVA_CANONICAL_VALUES.get(lower, value)


def _normalize_cva_column(df: pd.DataFrame) -> pd.DataFrame:
    if "CVA" not in df.columns:
        return df
    out = df.copy()
    out["CVA"] = out["CVA"].map(normalize_cva_value)
    return out


PNEUMONIA_CANONICAL_VALUES: Dict[str, str] = {
    "no": "No",
    "nao": "No",
    "não": "No",
    "yes": "Under treatment",
    "sim": "Under treatment",
    "under treatment": "Under treatment",
    "em tratamento": "Under treatment",
    "treated": "Treated",
    "tratada": "Treated",
}


def normalize_pneumonia_value(value: object) -> object:
    """Canonicalize Pneumonia categorical labels; legacy 'Yes' maps to 'Under treatment'."""
    if pd.isna(value):
        return value
    text = str(value).strip()
    lower = text.lower()
    if lower in MISSING_TOKENS:
        return np.nan
    return PNEUMONIA_CANONICAL_VALUES.get(lower, value)


def _normalize_pneumonia_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Pneumonia" not in df.columns:
        return df
    out = df.copy()
    out["Pneumonia"] = out["Pneumonia"].map(normalize_pneumonia_value)
    return out


def parse_suspension_anticoagulation_days(value: object) -> float:
    """Parse the conditional days-since-anticoagulation-suspension field.

    Blanks and unknown/not-applicable tokens remain missing. Plain numeric
    values are preserved, and only simple recoverable text forms are accepted:
    ``"> 5"``, ``"5 days"``, ``"2d"``. Ambiguous free text remains missing.
    """
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == "" or text.lower() in MISSING_TOKENS:
        return np.nan

    parsed = parse_number(text, strict=True)
    if pd.notna(parsed) and float(parsed) >= 0:
        return float(parsed)

    match = re.fullmatch(
        r"[<>~]?\s*([-+]?\d+(?:[.,]\d+)?)\s*(?:d|day|days|dia|dias)?\.?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return np.nan
    recovered = parse_number(match.group(1), strict=True)
    if pd.isna(recovered) or float(recovered) < 0:
        return np.nan
    return float(recovered)


def _normalize_suspension_anticoagulation_days_column(df: pd.DataFrame) -> pd.DataFrame:
    """Apply narrow numeric parsing to Suspension of Anticoagulation (day)."""
    col = "Suspension of Anticoagulation (day)"
    if col not in df.columns:
        return df
    out = df.copy()
    out[col] = out[col].map(parse_suspension_anticoagulation_days)
    return out


def _impute_blank_as_no(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN with 'No' for BLANK_MEANS_NO_COLUMNS after missing-token normalisation.

    Must be called AFTER normalize_dataframe() so that every MISSING_TOKEN
    (empty string, '-', 'nan', etc.) has already been converted to NaN.

    Semantics preserved:
    - Existing "Yes" or "No" values are never overwritten.
    - Columns absent from the DataFrame are silently skipped.
    - Conditional/numeric fields (e.g. "Suspension of Anticoagulation (day)")
      are not in BLANK_MEANS_NO_COLUMNS and are therefore untouched.
    - True missing values in continuous/lab/echo fields are unaffected.

    Returns a copy of *df*.
    """
    out = df.copy()
    for col in BLANK_MEANS_NO_COLUMNS:
        if col not in out.columns:
            continue
        n_filled = int(out[col].isna().sum())
        if n_filled > 0:
            out[col] = out[col].fillna("No")
    return out


def _impute_blank_as_none(df: pd.DataFrame) -> pd.DataFrame:
    """Fill source blanks with 'None' only for explicitly listed fields.

    This runs BEFORE normalize_dataframe() so that true blank cells become the
    valid clinical category "None", while textual unknown tokens (e.g.
    "Unknown", "N/A", "-") still flow through the usual missing-token path.
    """
    out = df.copy()
    for col in BLANK_MEANS_NONE_COLUMNS:
        if col not in out.columns:
            continue
        blank_mask = out[col].isna() | out[col].astype(str).str.strip().eq("")
        if bool(blank_mask.any()):
            out.loc[blank_mask, col] = "None"
    return out


def add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add conservative panel-level missingness indicators.

    The indicators are intentionally limited to routinely available
    preoperative laboratory panels. They encode whether a clinical
    information block was partially unavailable before median/mode imputation,
    without adding one indicator per raw variable.
    """
    out = df.copy()
    for indicator, source_cols in MISSINGNESS_INDICATOR_SPECS.items():
        missing_any = pd.Series(False, index=out.index)
        for col in source_cols:
            if col in out.columns:
                missing_any = missing_any | out[col].apply(
                    lambda v, c=col: is_missing(v, column=c)
                )
            else:
                missing_any = pd.Series(True, index=out.index)
        out[indicator] = missing_any.astype(int)
    return out


REQUIRED_SOURCE_TABLES = [
    "Preoperative",
    "Pre-Echocardiogram",
    "Postoperative",
]

OPTIONAL_SOURCE_TABLES = [
    "EuroSCORE II",
    "EuroSCORE II Automático",
    "STS Score",
]

MISSINGNESS_INDICATOR_SPECS: Dict[str, tuple[str, ...]] = {
    "missing_renal_labs": (
        "Creatinine (mg/dL)",
        "Cr clearance, ml/min *",
    ),
    "missing_cbc_labs": (
        "Hematocrit (%)",
        "WBC Count (10³/μL)",
        "Platelet Count (cells/μL)",
    ),
    "missing_coagulation_labs": (
        "INR",
        "PTT",
    ),
}

MISSINGNESS_INDICATOR_COLUMNS: frozenset = frozenset(MISSINGNESS_INDICATOR_SPECS)

FLAT_ALIAS_TO_APP_COLUMNS = {
    "patient_id": "Name",
    "patient_name": "Name",
    "sex": "Sex",
    "age_years": "Age (years)",
    "height_cm": "Height (cm)",
    "weight_kg": "Weight (kg)",
    "bsa_m2": "BSA, m2",
    "race": "Race",
    "surgical_priority": "Surgical Priority",
    "surgery_pre": "Surgery",
    "surgery": "Surgery",
    "procedure_date": "Procedure Date",
    "infective_endocarditis_pre": "IE",
    "infective_endocarditis": "IE",
    "diseased_vessels_count": "No. of Diseased Vessels",
    "num_diseased_vessels": "No. of Diseased Vessels",
    "left_main_stenosis_ge_50_pct": "Left Main Stenosis ≥ 50%",
    "left_main_stenosis_ge_50": "Left Main Stenosis ≥ 50%",
    "proximal_lad_stenosis_ge_70_pct": "Proximal LAD Stenosis ≥ 70%",
    "proximal_lad_stenosis_ge_70": "Proximal LAD Stenosis ≥ 70%",
    "coronary_symptom": "Coronary Symptom",
    "nyha_pre": "Preoperative NYHA",
    "preoperative_nyha": "Preoperative NYHA",
    "ccs4": "CCS4",
    "pre_lvef_pct": "Pré-LVEF, %",
    "lvef_pre_pct": "Pré-LVEF, %",
    "hf_class_by_ef": "Classification of Heart Failure According to Ejection Fraction",
    "previous_surgery": "Previous surgery",
    "preoperative_medications": "Preoperative Medications",
    "heart_failure": "HF",
    "hf": "HF",
    "arrhythmia_remote": "Arrhythmia Remote",
    "arrhythmia_recent": "Arrhythmia Recent",
    "hypertension": "Hypertension",
    "diabetes": "Diabetes",
    "dyslipidemia": "Dyslipidemia",
    "stroke_history": "CVA",
    "cva": "CVA",
    "peripheral_vascular_disease": "PVD",
    "pvd": "PVD",
    "alcohol_use": "Alcohol",
    "alcohol": "Alcohol",
    "smoking_pack_years": "Smoking (Pack-year)",
    "smoking": "_smoking_status_csv",
    "smoking_status": "_smoking_status_csv",
    "cancer_le_5_years": "Cancer ≤ 5 yrs",
    "family_history_cad": "Family Hx of CAD",
    "anticoagulation_antiplatelet": "Anticoagulation/ Antiaggregation",
    "days_anticoagulation_suspension": "Suspension of Anticoagulation (day)",
    "pneumonia_pre": "Pneumonia",
    "other_information": "Others informations",
    "dialysis_pre": "Dialysis",
    "kdigo_stage": "KDIGO †",
    "copd": "Chronic Lung Disease",
    "dpoc": "Chronic Lung Disease",
    "chronic_lung_disease_pre": "Chronic Lung Disease",
    "dpoc_pre": "Chronic Lung Disease",
    "critical_preop_state": "Critical preoperative state",
    "critical_state_pre": "Critical preoperative state",
    "critical_condition": "Critical preoperative state",
    "critical_condition_pre": "Critical preoperative state",
    "estado_critico_pre": "Critical preoperative state",
    "poor_mobility": "Poor mobility",
    "reduced_mobility": "Poor mobility",
    "mobilidade_reduzida": "Poor mobility",
    "creatinine_clearance_ml_min": "Cr clearance, ml/min *",
    "creatinine_mg_dl": "Creatinine (mg/dL)",
    "creatinine_pre_mg_dl": "Creatinine (mg/dL)",
    "hematocrit_pct": "Hematocrit (%)",
    "hematocrit_pre_pct": "Hematocrit (%)",
    "wbc_count_10e3_ul": "WBC Count (10³/μL)",
    "wbc_count_pre_10e3_ul": "WBC Count (10³/μL)",
    "platelet_count_cells_ul": "Platelet Count (cells/μL)",
    "platelet_count_pre_cells_ul": "Platelet Count (cells/μL)",
    "inr": "INR",
    "ptt": "PTT",
    "exam_date": "Exam date",
    "lvef_pct": "Pré-LVEF, %",
    "aortic_stenosis": "Aortic Stenosis",
    "aortic_stenosis_pre": "Aortic Stenosis",
    "aortic_mean_gradient_mmhg": "Aortic Mean gradient (mmHg)",
    "aortic_mean_gradient_pre_mmhg": "Aortic Mean gradient (mmHg)",
    "ava_cm2": "AVA (cm²)",
    "aortic_valve_area_pre_cm2": "AVA (cm²)",
    "aortic_regurgitation": "Aortic Regurgitation",
    "aortic_regurgitation_pre": "Aortic Regurgitation",
    "vena_contracta": "Vena contracta",
    "vena_contracta_pre": "Vena contracta",
    "pht_aortic": "PHT Aortic",
    "pht_aortic_pre": "PHT Aortic",
    "mitral_stenosis": "Mitral Stenosis",
    "mitral_stenosis_pre": "Mitral Stenosis",
    "mitral_mean_gradient_mmhg": "Mitral Mean gradient (mmHg)",
    "mitral_mean_gradient_pre_mmhg": "Mitral Mean gradient (mmHg)",
    "mva_cm2": "MVA (cm²)",
    "mitral_valve_area_pre_cm2": "MVA (cm²)",
    "mitral_regurgitation": "Mitral Regurgitation",
    "mitral_regurgitation_pre": "Mitral Regurgitation",
    "vena_contracta_mm": "Vena contracta (mm)",
    "vena_contracta_mm_pre": "Vena contracta (mm)",
    "pht_mitral": "PHT Mitral",
    "pht_mitral_pre": "PHT Mitral",
    "tricuspid_regurgitation": "Tricuspid Regurgitation",
    "tricuspid_regurgitation_pre": "Tricuspid Regurgitation",
    "psap": "PSAP",
    "psap_pre": "PSAP",
    "tapse": "TAPSE",
    "tapse_pre": "TAPSE",
    "aortic_root_abscess": "Aortic Root Abscess",
    "aortic_root_abscess_pre": "Aortic Root Abscess",
    "death": "Death",
    "euroscore_ii": "EuroSCORE II",
    "euroscore_ii_automatic": "EuroSCORE II Automático",
    "operative_mortality_sts": "Operative Mortality",
    "morbidity_mortality_sts": "Morbidity & Mortality",
    "euroscore": "EuroSCORE II",
    "operative": "Operative Mortality",
    "morbidity": "Morbidity & Mortality",
    "stroke_sts_short": "Stroke",
    "renal_fail": "Renal Failure",
    "reoperati": "Reoperation",
    "prolonged": "Prolonged Ventilation",
    "deep_ster": "Deep Sternal Wound Infection",
    "long_hosp": "Long Hospital Stay (>14 days)",
    "short_hos": "Short Hospital Stay (<6 days)",
    "stroke_sts": "Stroke",
    "renal_failure_sts": "Renal Failure",
    "reoperation_sts": "Reoperation",
    "prolonged_ventilation_sts": "Prolonged Ventilation",
    "deep_sternal_wound_infection_sts": "Deep Sternal Wound Infection",
    "long_hospital_stay_gt_14_days_sts": "Long Hospital Stay (>14 days)",
    "short_hospital_stay_lt_6_days_sts": "Short Hospital Stay (<6 days)",
}

FLAT_PREOP_ALLOWED_COLUMNS = {
    "Sex",
    "Age (years)",
    "Height (cm)",
    "Weight (kg)",
    "BSA, m2",
    "Race",
    "Surgical Priority",
    "Surgery",
    "IE",
    "No. of Diseased Vessels",
    "Left Main Stenosis ≥ 50%",
    "Proximal LAD Stenosis ≥ 70%",
    "Coronary Symptom",
    "Preoperative NYHA",
    "CCS4",
    "Pré-LVEF, %",
    "Classification of Heart Failure According to Ejection Fraction",
    "Previous surgery",
    "HF",
    "Arrhythmia Remote",
    "Arrhythmia Recent",
    "Hypertension",
    "Diabetes",
    "Dyslipidemia",
    "CVA",
    "PVD",
    "Alcohol",
    "Smoking (Pack-year)",
    "Cancer ≤ 5 yrs",
    "Family Hx of CAD",
    "Anticoagulation/ Antiaggregation",
    "Suspension of Anticoagulation (day)",
    "Pneumonia",
    "Dialysis",
    "KDIGO †",
    "Chronic Lung Disease",
    "Critical preoperative state",
    "Poor mobility",
    "Cr clearance, ml/min *",
    "Creatinine (mg/dL)",
    "Hematocrit (%)",
    "WBC Count (10³/μL)",
    "Platelet Count (cells/μL)",
    "INR",
    "PTT",
    "Aortic Stenosis",
    "Aortic Mean gradient (mmHg)",
    "AVA (cm²)",
    "Aortic Regurgitation",
    "Vena contracta",
    "PHT Aortic",
    "Mitral Stenosis",
    "Mitral Mean gradient (mmHg)",
    "MVA (cm²)",
    "Mitral Regurgitation",
    "Vena contracta (mm)",
    "PHT Mitral",
    "Tricuspid Regurgitation",
    "PSAP",
    "TAPSE",
    "Aortic Root Abscess",
}

# ── Never-feature column policy ───────────────────────────────────────────────
# Columns that must never enter AI Risk training, validation, or inference
# as predictors. Applied as belt-and-suspenders on top of the allowlist
# (flat path) and sheet-level structural separation (multi-sheet path).
# Names are canonical app names: post-_normalize_flat_columns for CSV,
# sheet column names for Excel.

EXCLUDED_OUTCOME_COLUMNS: frozenset = frozenset({
    "Death",
    "morte_30d",
})

EXCLUDED_POSTOPERATIVE_COLUMNS: frozenset = frozenset({
    # Post-op functional status
    "nyha_post",
    # Drain, ICU, length of stay
    "drain_debit_day_1", "drain_debit_day_2", "drain_debit_day_3",
    "icu_days",
    "postoperative_hospitalization_days",
    "hospital_stay_days",
    "transfusion",
    # Post-op labs
    "creatinine_post_mg_dl",
    "hematocrit_post_pct",
    "wbc_count_post_10e3_ul",
    "platelet_count_post_cells_ul",
    # Post-op complications
    "acute_myocardial_infarction_post",
    "pneumonia_post",
    "stroke_post",
    "atrial_fibrillation_post",
    "av_block_post",
    "vasoplegic_syndrome_post",
    "cardiogenic_shock_post",
    "hemorrhagic_shock_post",
    "septic_shock_post",
    "infective_endocarditis_post",
    "pulmonary_embolism_post",
    "surgical_site_infection_post",
    "mediastinitis_post",
    "delirium_post",
    "dialysis_post",
    "reoperated",
    "rehospitalization_lt_30_days",
    "other_complications",
    # Post-op echocardiography
    "lvef_post_pct",
    "aortic_stenosis_post",
    "aortic_mean_gradient_post_mmhg",
    "aortic_valve_area_post_cm2",
    "aortic_regurgitation_post",
    "vena_contracta_post",
    "pht_aortic_post",
    "mitral_mean_gradient_post_mmhg",
    "mitral_valve_area_post_cm2",
    "mitral_regurgitation_post",
    "vena_contracta_mm_post",
    "pht_mitral_post",
    "tricuspid_regurgitation_post",
    "psap_post",
    "tapse_post",
})

EXCLUDED_COMPARATOR_SCORE_COLUMNS: frozenset = frozenset({
    # EuroSCORE II (raw and derived)
    "EuroSCORE II",
    "EuroSCORE II Automático",
    "euroscore_sheet",
    "euroscore_auto_sheet",
    "euroscore_calc",
    "euroscore_sheet_clean",
    "euroscore_auto_sheet_clean",
    # STS Score predicted endpoints
    "sts_score",
    "sts_score_sheet",
    "Operative Mortality",
    "Morbidity & Mortality",
    "Stroke",
    "Renal Failure",
    "Reoperation",
    "Prolonged Ventilation",
    "Deep Sternal Wound Infection",
    "Long Hospital Stay (>14 days)",
    "Short Hospital Stay (<6 days)",
})

EXCLUDED_METADATA_COLUMNS: frozenset = frozenset({
    # Patient identity and temporal reference
    "Name",
    "_patient_key",
    "Procedure Date",
    "_proc_date",
    "patient_id",
    "surgery_year",
    "surgery_quarter",
    "days_pre_echo_to_surgery",
    "days_surgery_to_post_echo",
    # AI Risk model output columns (circular leakage)
    "ia_risk_oof",
    "ia_risk_fullfit",
    "classe_ia",
    "classe_euro",
    "classe_sts",
})

NEVER_FEATURE_COLUMNS: frozenset = (
    EXCLUDED_OUTCOME_COLUMNS
    | EXCLUDED_POSTOPERATIVE_COLUMNS
    | EXCLUDED_COMPARATOR_SCORE_COLUMNS
    | EXCLUDED_METADATA_COLUMNS
)


def _norm_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_patient(value: object) -> str:
    txt = _norm_text(value).upper()
    txt = re.sub(r"\s+", " ", txt)
    return txt


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def parse_number(
    value: object,
    *,
    _warn_ambiguous: list | None = None,
    strict: bool = False,
) -> float:
    """Parse a numeric value handling BR and EN formats.

    Handles comma-decimal (``1,08``), BR thousands (``1.234,56``),
    EN thousands (``1,234.56``), and percentage suffixes (``64,7%``).

    **EN safety guarantee**: any value that ``float(value)`` already
    parses correctly will produce the identical result.

    Parameters
    ----------
    _warn_ambiguous : list, optional
        If provided, ambiguous values (e.g. ``69,227``) are appended
        so the caller can log them.
    strict : bool
        If ``True``, disable the regex fallback that extracts embedded
        digits from non-numeric strings (e.g. ``"ID0001"`` → ``1``).
        Use ``True`` for column-level conversion to avoid false positives.
    """
    if pd.isna(value):
        return np.nan
    txt = str(value).strip()
    if txt.lower() in MISSING_TOKENS:
        return np.nan
    # Strip trailing percentage
    if txt.endswith("%"):
        txt = txt[:-1].strip()
    if not txt:
        return np.nan

    # Fast path: direct float (integers, EN decimals like "64.7")
    try:
        return float(txt)
    except ValueError:
        pass

    has_comma = "," in txt
    has_dot = "." in txt

    if has_comma and has_dot:
        last_comma = txt.rfind(",")
        last_dot = txt.rfind(".")
        if last_comma > last_dot:
            # BR thousands: 1.234,56 → remove dots, comma→dot
            cleaned = txt.replace(".", "").replace(",", ".")
        else:
            # EN thousands: 1,234.56 → remove commas
            cleaned = txt.replace(",", "")
    elif has_comma:
        # Only comma — disambiguate thousands vs decimal
        magnitude = txt.lstrip("-+ ")
        parts = magnitude.split(",")
        if (
            len(parts) >= 2
            and parts[0].isdigit()
            and len(parts[0]) <= 3
            and all(len(p) == 3 and p.isdigit() for p in parts[1:])
        ):
            # EN thousands without decimal: 1,234 or 69,227
            cleaned = txt.replace(",", "")
            if _warn_ambiguous is not None and len(parts) == 2:
                _warn_ambiguous.append(txt)
        else:
            # BR decimal: 1,08 or 64,7
            cleaned = txt.replace(",", ".")
    else:
        cleaned = txt

    try:
        return float(cleaned)
    except ValueError:
        if strict:
            return np.nan
        # Last resort: extract first numeric substring
        found = re.findall(r"[-+]?\d*\.?\d+", cleaned)
        if found:
            try:
                return float(found[0])
            except ValueError:
                pass
        return np.nan


# ──────────────────────────────────────────────────────────────────────
# Clinical plausibility layer (narrow post-parse correction)
# ──────────────────────────────────────────────────────────────────────
#
# The generic parse_number() handles BR/EN numeric formats well, but a
# handful of clinically ambiguous values slip through — for example the
# raw token "69,227" in an LVEF cell is treated as EN thousands
# (69227) because it matches the 3-digit-group pattern, even though
# 69227 % LVEF is clinically impossible and the intended value is
# almost certainly 69.227.
#
# This layer runs ONLY for the listed clinical columns and:
#   1. checks the parsed value against a wide clinical safety range,
#   2. if out of range, attempts ONE reinterpretation (treat every
#      comma as a decimal point) using the original raw string,
#   3. accepts the reinterpretation only if it lands inside the range,
#   4. otherwise sets NaN.
#
# Every action is reported through the IngestionReport warnings bucket
# so the Phase 3 observability layer surfaces it automatically.
#
# Ranges are deliberately wide safety bounds — the goal is to catch
# parsing errors (orders-of-magnitude off), NOT to reject physiological
# outliers. A value that is merely abnormal must still pass.
_CLINICAL_PLAUSIBILITY_RANGES: Dict[str, Tuple[float, float]] = {
    "Pré-LVEF, %": (1.0, 100.0),
    "Hematocrit (%)": (1.0, 100.0),
    "Creatinine (mg/dL)": (0.05, 50.0),
    "PSAP": (0.0, 250.0),
    "Aortic Mean gradient (mmHg)": (0.0, 250.0),
    "Mitral Mean gradient (mmHg)": (0.0, 100.0),
    "AVA (cm\u00b2)": (0.01, 10.0),
    "MVA (cm\u00b2)": (0.01, 15.0),
    "Vena contracta": (0.0, 50.0),
    "Vena contracta (mm)": (0.0, 50.0),
    "TAPSE": (0.0, 60.0),
}


def _reinterpret_as_decimal(raw: object) -> float:
    """One-shot reinterpretation: treat every comma as a decimal point.

    Used ONLY by the clinical plausibility layer as a narrow fallback
    when parse_number()'s output is clinically impossible.  This helper
    deliberately does NOT replicate parse_number's full logic — it
    applies exactly one transformation (``,`` → ``.``) and, if that
    leaves multiple dots, keeps only the last as the decimal point.

    Returns np.nan if the reinterpretation cannot produce a float.
    """
    if raw is None:
        return np.nan
    if isinstance(raw, float) and np.isnan(raw):
        return np.nan
    txt = str(raw).strip()
    if not txt or txt.lower() in MISSING_TOKENS:
        return np.nan
    if txt.endswith("%"):
        txt = txt[:-1].strip()
    cleaned = txt.replace(",", ".")
    if cleaned.count(".") > 1:
        head, _, tail = cleaned.rpartition(".")
        cleaned = head.replace(".", "") + "." + tail
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _apply_clinical_plausibility(
    out: pd.DataFrame,
    raw_df: pd.DataFrame,
    report_warnings: List["ColumnAction"],
    *,
    ambiguous_by_col: Optional[Dict[str, Dict[Any, str]]] = None,
    correction_records: Optional[List["CorrectionRecord"]] = None,
) -> Dict[str, set]:
    """Apply column-specific plausibility correction to ``out`` in place.

    For every column listed in :data:`_CLINICAL_PLAUSIBILITY_RANGES`
    that exists in ``out``:

    1. Find rows where the parsed value is finite but outside the
       plausible range ``[low, high]``.
    2. Read the original raw string from ``raw_df`` for the same row
       and attempt one reinterpretation via
       :func:`_reinterpret_as_decimal`.
    3. If the reinterpretation lies inside the range, overwrite the
       parsed value.  Corrected rows whose raw token was previously
       flagged as ambiguous by :func:`parse_number` (i.e. a comma-
       thousands reading such as ``69,227``) are emitted as a single
       **consolidated** warning that names both the ambiguity AND the
       plausibility rescue in one line.  Other corrected rows keep the
       regular ``corrected_plausibility`` warning.
    4. Otherwise, set the cell to NaN and append a
       ``cleared_implausible`` warning.

    Values already inside the range (or already NaN) are left
    untouched, guaranteeing that valid inputs are never modified.

    Parameters
    ----------
    ambiguous_by_col : dict, optional
        ``{column: {row_index: raw_text}}`` tracking tokens that
        :func:`parse_number` flagged as EN-thousands-ambiguous during
        the numeric conversion step.  Used to produce the consolidated
        warning described above.
    correction_records : list, optional
        If provided, one :class:`CorrectionRecord` is appended per
        corrected or cleared row for the exportable audit table.

    Returns
    -------
    dict
        ``{column: set of row indices}`` listing the ambiguous rows
        that were successfully rescued by plausibility correction.
        The caller uses this to avoid emitting a redundant generic
        ``flagged_ambiguous`` warning for those rows.
    """
    ambiguous_by_col = ambiguous_by_col or {}
    covered_ambiguous_by_col: Dict[str, set] = {}

    for col, (lo, hi) in _CLINICAL_PLAUSIBILITY_RANGES.items():
        if col not in out.columns:
            continue

        parsed_series = pd.to_numeric(out[col], errors="coerce")
        out_of_range_mask = parsed_series.notna() & (
            (parsed_series < lo) | (parsed_series > hi)
        )
        if not out_of_range_mask.any():
            continue

        col_ambiguous = ambiguous_by_col.get(col, {})

        consolidated_examples: List[str] = []
        plain_corrected_examples: List[str] = []
        cleared_examples: List[str] = []
        n_consolidated = 0
        n_plain_corrected = 0
        n_cleared = 0
        covered_ambiguous_idxs: set = set()

        for idx in out.index[out_of_range_mask]:
            parsed = float(parsed_series.loc[idx])
            raw_val = (
                raw_df.at[idx, col]
                if (col in raw_df.columns and idx in raw_df.index)
                else parsed
            )
            raw_str = "" if raw_val is None else str(raw_val)
            reinterpreted = _reinterpret_as_decimal(raw_val)

            if (
                isinstance(reinterpreted, float)
                and np.isfinite(reinterpreted)
                and lo <= reinterpreted <= hi
            ):
                out.at[idx, col] = reinterpreted
                was_ambiguous = idx in col_ambiguous
                if was_ambiguous:
                    covered_ambiguous_idxs.add(idx)
                    n_consolidated += 1
                    if len(consolidated_examples) < 5:
                        consolidated_examples.append(
                            f"raw {raw_val!r} parsed as {parsed:g} and "
                            f"corrected to {reinterpreted:g} by clinical plausibility"
                        )
                    action_name = "consolidated_ambiguity_correction"
                    reason = (
                        f"ambiguous raw value parsed as EN thousands, then "
                        f"reinterpreted as decimal within plausible range "
                        f"[{lo:g}, {hi:g}]"
                    )
                else:
                    n_plain_corrected += 1
                    if len(plain_corrected_examples) < 5:
                        plain_corrected_examples.append(
                            f"raw={raw_val!r} parsed={parsed:g} "
                            f"-> corrected={reinterpreted:g}"
                        )
                    action_name = "corrected_plausibility"
                    reason = (
                        f"value outside plausible range [{lo:g}, {hi:g}]; "
                        f"reinterpreted as decimal"
                    )
                if correction_records is not None:
                    correction_records.append(CorrectionRecord(
                        column=col,
                        row_index=idx,
                        raw_value=raw_str,
                        parsed_value=parsed,
                        action=action_name,
                        final_value=float(reinterpreted),
                        reason=reason,
                    ))
            else:
                out.at[idx, col] = np.nan
                n_cleared += 1
                if len(cleared_examples) < 5:
                    cleared_examples.append(
                        f"raw={raw_val!r} parsed={parsed:g} -> NaN"
                    )
                if correction_records is not None:
                    correction_records.append(CorrectionRecord(
                        column=col,
                        row_index=idx,
                        raw_value=raw_str,
                        parsed_value=parsed,
                        action="cleared_implausible",
                        final_value=None,
                        reason=(
                            f"value outside plausible range [{lo:g}, {hi:g}] "
                            "and no safe reinterpretation; set to NaN"
                        ),
                    ))

        if covered_ambiguous_idxs:
            covered_ambiguous_by_col[col] = covered_ambiguous_idxs

        if n_consolidated > 0:
            report_warnings.append(ColumnAction(
                column=col,
                action="consolidated_ambiguity_correction",
                detail=(
                    f"{n_consolidated} ambiguous value(s) corrected by "
                    f"clinical plausibility [range {lo:g}, {hi:g}]: "
                    + "; ".join(consolidated_examples)
                ),
                count=n_consolidated,
            ))
        if n_plain_corrected > 0:
            report_warnings.append(ColumnAction(
                column=col,
                action="corrected_plausibility",
                detail=(
                    f"{n_plain_corrected} value(s) outside clinical range "
                    f"[{lo:g}, {hi:g}] reinterpreted as decimal: "
                    + "; ".join(plain_corrected_examples)
                ),
                count=n_plain_corrected,
            ))
        if n_cleared > 0:
            report_warnings.append(ColumnAction(
                column=col,
                action="cleared_implausible",
                detail=(
                    f"{n_cleared} value(s) outside clinical range "
                    f"[{lo:g}, {hi:g}] still implausible after "
                    "reinterpretation; set to NaN: "
                    + "; ".join(cleared_examples)
                ),
                count=n_cleared,
            ))

    return covered_ambiguous_by_col


def is_missing(value: object, column: str | None = None) -> bool:
    """Context-aware missing value detection.

    Returns ``True`` if *value* should be treated as missing data.
    For columns in :data:`LITERAL_NONE_IS_VALID_COLUMNS`,
    ``"none"`` is **not** treated as missing because it is a valid
    clinical value.
    """
    if pd.isna(value):
        return True
    txt = str(value).strip()
    if not txt:
        return True
    txt_lower = txt.lower()
    if column and column in LITERAL_NONE_IS_VALID_COLUMNS and txt_lower == "none":
        return False
    return txt_lower in MISSING_TOKENS


def contextual_missing_mask(s: pd.Series, column: str | None = None) -> pd.Series:
    """Return a missing-value mask using the dataset's column semantics."""
    return s.apply(lambda v: is_missing(v, column=column))


@dataclass
class PostopTiming:
    """Structured representation of a postoperative timing value.

    Preserves the full semantics of values like ``"Operative"``,
    ``"0"``, ``"> 30"``, ``"Death"``, and day-counts without
    flattening them to a single boolean.
    """
    raw_value: str
    category: str       # survivor | operative | day_of_surgery | days_to_event
                        # | beyond_threshold | event_occurred_no_day | unknown
    days: Optional[int]
    threshold: Optional[int]
    event_occurred: bool
    within_30d: bool

    @property
    def is_operative(self) -> bool:
        return self.category == "operative"

    @property
    def is_early(self) -> bool:
        """Event within 48 h (operative, day 0, days 1-2)."""
        if self.category in ("operative", "day_of_surgery"):
            return True
        return self.days is not None and self.days <= 2


_SURVIVOR_TOKENS = {"-", "--"}

# Boolean-style outcome labels used in flat CSV / batch files.
# Applied in map_death_30d as a fallback *after* timing-based parsing
# returns unknown — the canonical timing logic (operative, day counts,
# >threshold) always takes precedence.
_BOOLEAN_EVENT_TOKENS = {"yes", "y", "true", "sim"}          # → event = 1
_BOOLEAN_SURVIVOR_TOKENS = {"no", "n", "false", "não", "nao"}  # → no event = 0


def parse_postop_timing(value: object) -> PostopTiming:
    """Parse a postoperative timing value into a :class:`PostopTiming`.

    Recognises the patterns found in the dataset's Death, ICU-days,
    and complication-timing columns.
    """
    if pd.isna(value):
        return PostopTiming("", "unknown", None, None, False, False)

    txt = str(value).strip()
    txt_lower = txt.lower()

    if not txt or txt_lower in (MISSING_TOKENS - _SURVIVOR_TOKENS):
        return PostopTiming(txt, "unknown", None, None, False, False)

    if txt_lower in _SURVIVOR_TOKENS:
        return PostopTiming(txt, "survivor", None, None, False, False)

    if txt_lower == "operative":
        return PostopTiming(txt, "operative", 0, None, True, True)

    if txt_lower == "death":
        return PostopTiming(txt, "event_occurred_no_day", None, None, True, True)

    # "> 15", "> 30", ">15", ">30"
    m = re.match(r">\s*(\d+)", txt)
    if m:
        threshold = int(m.group(1))
        return PostopTiming(txt, "beyond_threshold", None, threshold, True, threshold < 30)

    num = parse_number(txt)
    if not np.isnan(num):
        days = int(num)
        if days == 0:
            return PostopTiming(txt, "day_of_surgery", 0, None, True, True)
        return PostopTiming(txt, "days_to_event", days, None, True, days <= 30)

    return PostopTiming(txt, "unknown", None, None, False, False)


def map_death_30d(value: object) -> int:
    """Map a Death column value to 30-day mortality (0 or 1).

    Resolution order:
    1. Canonical timing logic via :func:`parse_postop_timing`:
       survivor tokens ("-", "--") → 0; operative / day 0 → 1;
       numeric day 1-30 → 1; day >30 or ">30" → 0; "death" → 1.
    2. Boolean-style fallback (only when timing returns unknown):
       yes / y / true / sim → 1;  no / n / false / não / nao → 0.
    3. Truly unrecognised: warn and return 0 (legacy safe default).
    """
    timing = parse_postop_timing(value)
    if timing.category == "survivor":
        return 0
    if timing.category != "unknown":
        return int(timing.within_30d)
    # Timing parser could not interpret the value — try boolean labels.
    raw_lower = (timing.raw_value or "").lower()
    if raw_lower in _BOOLEAN_EVENT_TOKENS:
        return 1
    if raw_lower in _BOOLEAN_SURVIVOR_TOKENS:
        return 0
    # Truly unrecognised: warn and fall back to 0.
    if timing.raw_value and raw_lower not in MISSING_TOKENS:
        warnings.warn(
            f"map_death_30d: unrecognised value '{timing.raw_value}' "
            "— treated as 0 (survivor). Please review this record.",
            stacklevel=2,
        )
    return 0


def is_combined_surgery(text: object) -> int:
    s = _norm_text(text)
    if not s:
        return 0
    return int("," in s or ";" in s or "+" in s)


MAJOR_PROCEDURES = {
    "cabg",
    "opcab",
    "avr",
    "av repair",
    "mv repair",
    "mvr",
    "tv repair",
    "tvr",
    "asd closure",
    "vsd correction",
    "aortic aneurism repair",
    "aortic dissection repair",
    "bentall-de bono procedure",
    "valve sparing aortic root replacement (david procedure)",
    "ross",
    "pericardiectomy",
    "heart transplant",
    "intracardiac tumor resection",
    "resection of intracardiac and/or pulmonary artery thrombus",
    "pulmonary homograft implantation",
    "left ventricular aneurysmectomy",
    "myectomy",
    "surgical treatment of anomalous aortic origin of coronary",
}

MINOR_PROCEDURES = {
    "pacemaker implantation",
    "pfo closure",
    "laao",
    "thrombus removal",
    "pacemaker electrode extraction",
    "tevar",
    "removal of panus in aortic valve",
}

THORACIC_AORTA_PROCEDURES = {
    "aortic aneurism repair",
    "aortic dissection repair",
    "bentall-de bono procedure",
    "valve sparing aortic root replacement (david procedure)",
}

NON_AORTA_EXCLUSIONS = {
    "surgical treatment of anomalous aortic origin of coronary",
    "removal of panus in aortic valve",
}

# ── Procedure intermediate-group classification ───────────────────────────────
# Replaces the coarser procedure_macro_group with a clinically richer taxonomy
# that separates aortic valve from mitral/tricuspid, aortic root reconstruction
# from aneurysm/dissection repair, and retains all other distinctions.
#
# NOTE: procedure_group is derived and stored in the dataset but is NOT used as
# a model feature. A controlled ablation (n=454, 68 events) showed consistent
# degradation when including it as a TargetEncoder input (AUC −0.017,
# AUPRC −0.020, Brier +0.002, calibration slope 0.954 vs 1.028, feature
# importance rank 54/62). TargetEncoder is unstable for 11-category taxonomies
# at this cohort size. The feature is retained exclusively for Data Quality
# auditing (procedure_group_dist, audit_surgery_coverage).
#
# Priority list: higher index = highest-risk / dominant for combined surgeries.
# Semantics:
#   UNKNOWN    — surgery field absent, blank, or a recognised missing token
#   OTHER      — surgery text present but not matched in PROCEDURE_INTERMEDIATE_GROUP_MAP
_INTERMEDIATE_GROUP_PRIORITY: List[str] = [
    "UNKNOWN",
    "OTHER",
    "OTHER_CARDIAC",
    "CONGENITAL_STRUCTURAL",
    "CABG_OPCAB",
    "MITRAL_TRICUSPID",
    "AORTIC_VALVE",
    "AORTA_ANEURYSM",
    "AORTA_ROOT",
    "CARDIAC_MASS_THROMBUS",
    "HF_TRANSPLANT",
]

PROCEDURE_INTERMEDIATE_GROUP_MAP: Dict[str, str] = {
    # CABG / coronary bypass
    "cabg": "CABG_OPCAB",
    "opcab": "CABG_OPCAB",
    "surgical treatment of anomalous aortic origin of coronary": "CABG_OPCAB",
    "left ventricular aneurysmectomy": "CABG_OPCAB",   # post-MI LV aneurysm
    # Aortic valve (AVR, repair, homografts)
    "avr": "AORTIC_VALVE",
    "av repair": "AORTIC_VALVE",
    "ross": "AORTIC_VALVE",
    "pulmonary homograft implantation": "AORTIC_VALVE",
    "aortic homograft implantation": "AORTIC_VALVE",
    "removal of panus in aortic valve": "AORTIC_VALVE",
    # Mitral / tricuspid valve
    "mvr": "MITRAL_TRICUSPID",
    "mv repair": "MITRAL_TRICUSPID",
    "tvr": "MITRAL_TRICUSPID",
    "tv repair": "MITRAL_TRICUSPID",
    "myectomy": "MITRAL_TRICUSPID",   # septal myectomy; LVOT / mitral contact
    # Aortic root reconstruction (valve + root composite)
    "bentall-de bono procedure": "AORTA_ROOT",
    "valve sparing aortic root replacement (david procedure)": "AORTA_ROOT",
    # Thoracic aortic aneurysm, dissection, endovascular
    "aortic aneurism repair": "AORTA_ANEURYSM",
    "aortic dissection repair": "AORTA_ANEURYSM",
    "debranching": "AORTA_ANEURYSM",
    "tevar": "AORTA_ANEURYSM",
    "evar": "AORTA_ANEURYSM",
    # HF / transplant
    "heart transplant": "HF_TRANSPLANT",
    # Congenital / structural
    "asd closure": "CONGENITAL_STRUCTURAL",
    "vsd correction": "CONGENITAL_STRUCTURAL",
    "pfo closure": "CONGENITAL_STRUCTURAL",
    "percutaneous closure of pfo": "CONGENITAL_STRUCTURAL",
    "laao": "CONGENITAL_STRUCTURAL",
    # Cardiac mass / thrombus
    "intracardiac tumor resection": "CARDIAC_MASS_THROMBUS",
    "resection of intracardiac and/or pulmonary artery thrombus": "CARDIAC_MASS_THROMBUS",
    "thrombus removal": "CARDIAC_MASS_THROMBUS",
    # Other cardiac (devices, pericardial)
    "pericardiectomy": "OTHER_CARDIAC",
    "pacemaker implantation": "OTHER_CARDIAC",
    "pacemaker electrode extraction": "OTHER_CARDIAC",
}


def procedure_group(text: object) -> str:
    """Return the intermediate procedure group for a Surgery field value.

    Provides finer clinical resolution than a broad macro-group while avoiding
    the noise of raw procedure text.  For combined surgeries the highest-priority
    group (by clinical risk) is returned.

    Returns:
        'UNKNOWN'     — field absent, blank, or a recognised missing token.
        'OTHER'       — text present but not in PROCEDURE_INTERMEDIATE_GROUP_MAP.
        'OTHER_CARDIAC' — pacemaker, pericardiectomy, and other non-classified cardiac.
        <GROUP>       — one of the nine clinical procedure groups.
    """
    s = _norm_text(text)
    if not s or s.lower() in MISSING_TOKENS:
        return "UNKNOWN"
    normalized = s.replace(";", ",").replace("+", ",")
    parts = [p.strip().lower() for p in normalized.split(",") if p.strip()]
    groups = [PROCEDURE_INTERMEDIATE_GROUP_MAP.get(p, "OTHER") for p in parts]
    if not groups:
        return "UNKNOWN"
    return max(
        groups,
        key=lambda g: _INTERMEDIATE_GROUP_PRIORITY.index(g) if g in _INTERMEDIATE_GROUP_PRIORITY else 0,
    )


def audit_surgery_coverage(surgery_series: "pd.Series") -> dict:
    """Return procedure-group mapping coverage statistics for a Surgery column.

    Pure function — no side-effects, no external state.

    Returns
    -------
    dict with keys:
        total           : int   — total rows evaluated
        n_mapped        : int   — rows resolved to a known procedure group
        n_unknown       : int   — rows with blank/missing surgery info
        n_other         : int   — rows with text present but not in taxonomy
        coverage_rate   : float — n_mapped / total
        top_unrecognized: list[tuple[str, int]] — top-10 raw Surgery values
                          that resolved to 'OTHER', by frequency
    """
    import pandas as _pd
    total = len(surgery_series)
    groups = surgery_series.map(procedure_group)
    n_unknown = int((groups == "UNKNOWN").sum())
    n_other = int((groups == "OTHER").sum())
    n_mapped = total - n_unknown - n_other
    coverage_rate = n_mapped / total if total > 0 else 0.0
    other_mask = groups == "OTHER"
    top_unrecognized: list = []
    if n_other > 0:
        top_unrecognized = [
            (str(v), int(c))
            for v, c in surgery_series[other_mask].value_counts().head(10).items()
        ]
    return {
        "total": total,
        "n_mapped": n_mapped,
        "n_unknown": n_unknown,
        "n_other": n_other,
        "coverage_rate": coverage_rate,
        "top_unrecognized": top_unrecognized,
    }


_PREV_SURG_YEAR_PAT = re.compile(r"\(\s*\d{4}\s*\)")
_PREV_SURG_REPEAT_PAT = re.compile(r"\(\s*x\s*(\d+)\s*\)", re.IGNORECASE)


def parse_previous_surgery(text: object) -> dict:
    """Parse a Previous surgery free-text value into structured audit fields.

    Grammar conventions observed in the dataset:
        ;      → separates distinct surgical episodes (different operating times)
        +      → same-time combined procedures within one episode
        (YYYY) → year annotation for an episode
        (xN)   → repetition marker: procedure was performed N times

    "No", blank, or any MISSING_TOKEN → no prior surgery.

    Returns
    -------
    dict with keys:
        any               : bool — True if any prior surgery is documented
        count_est         : int  — estimated episode count (;-segments, xN expanded)
        has_combined      : bool — True if '+' joins procedures in an episode
        has_repeat_marker : bool — True if (xN) pattern is present
        has_year_marker   : bool — True if (YYYY) pattern is present
    """
    _default = {
        "any": False, "count_est": 0,
        "has_combined": False, "has_repeat_marker": False, "has_year_marker": False,
    }
    if pd.isna(text):
        return _default
    txt = str(text).strip()
    if not txt or txt.lower() in MISSING_TOKENS or txt.lower() == "no":
        return _default

    has_year_marker = bool(_PREV_SURG_YEAR_PAT.search(txt))
    has_repeat_marker = bool(_PREV_SURG_REPEAT_PAT.search(txt))
    has_combined = "+" in txt

    episodes = [ep.strip() for ep in txt.split(";") if ep.strip()]
    count_est = 0
    for ep in episodes:
        m = _PREV_SURG_REPEAT_PAT.search(ep)
        if m:
            count_est += int(m.group(1))
        else:
            count_est += 1

    return {
        "any": True,
        "count_est": count_est,
        "has_combined": has_combined,
        "has_repeat_marker": has_repeat_marker,
        "has_year_marker": has_year_marker,
    }


_PREV_SURG_AUDIT_COLS = [
    "previous_surgery_any",
    "previous_surgery_count_est",
    "previous_surgery_has_combined",
    "previous_surgery_has_repeat_marker",
    "previous_surgery_has_year_marker",
]


def _add_previous_surgery_audit_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Derive 5 audit-only columns from Previous surgery. Not model features."""
    if "Previous surgery" not in df.columns:
        return df
    out = df.copy()
    parsed = out["Previous surgery"].map(parse_previous_surgery)
    out["previous_surgery_any"] = parsed.map(lambda d: d["any"])
    out["previous_surgery_count_est"] = parsed.map(lambda d: d["count_est"])
    out["previous_surgery_has_combined"] = parsed.map(lambda d: d["has_combined"])
    out["previous_surgery_has_repeat_marker"] = parsed.map(lambda d: d["has_repeat_marker"])
    out["previous_surgery_has_year_marker"] = parsed.map(lambda d: d["has_year_marker"])
    return out


def split_surgery_procedures(text: object) -> List[str]:
    s = _norm_text(text)
    if not s:
        return []
    normalized = s.replace(";", ",").replace("+", ",")
    parts = [p.strip().lower() for p in normalized.split(",") if p.strip()]
    return [p for p in parts if p not in MINOR_PROCEDURES]


def procedure_weight(text: object) -> str:
    parts = split_surgery_procedures(text)
    major = [p for p in parts if p in MAJOR_PROCEDURES]
    if not major:
        return "1_non_cabg"
    if len(major) == 1 and major[0] in {"cabg", "opcab"}:
        return "isolated_cabg"
    if len(major) == 1:
        return "1_non_cabg"
    if len(major) == 2:
        return "2_procedures"
    return "3plus_procedures"


def thoracic_aorta_surgery(text: object) -> int:
    parts = split_surgery_procedures(text)
    for p in parts:
        if p in NON_AORTA_EXCLUSIONS:
            continue
        if p in THORACIC_AORTA_PROCEDURES:
            return 1
    return 0


def _choose_echo_for_patient(patient_surgeries: pd.DataFrame, patient_echo: pd.DataFrame) -> pd.DataFrame:
    if patient_echo.empty:
        out = pd.DataFrame(index=patient_surgeries.index)
        return out

    eco = patient_echo.copy().sort_values("_echo_date")
    eco_valid = eco[eco["_echo_date"].notna()]
    rows = []
    for idx, surg_row in patient_surgeries.iterrows():
        d = surg_row["_proc_date"]
        if pd.isna(d) or eco_valid.empty:
            chosen = eco.iloc[-1]
        else:
            eligible = eco_valid[eco_valid["_echo_date"] <= d]
            if not eligible.empty:
                chosen = eligible.iloc[-1]
            else:
                dist = (pd.to_datetime(eco_valid["_echo_date"]) - pd.to_datetime(d)).abs()
                if len(dist) == 0:
                    chosen = eco.iloc[-1]
                else:
                    chosen = eco_valid.loc[dist.idxmin()]
        chosen = chosen.copy()
        chosen.name = idx
        rows.append(chosen)

    return pd.DataFrame(rows)


def _aggregate_score_by_patient(df: pd.DataFrame, patient_col: str, score_col: str) -> pd.Series:
    temp = df.copy()
    temp["_patient_key"] = temp[patient_col].map(normalize_patient)
    temp["_score"] = pd.to_numeric(temp[score_col], errors="coerce")
    return temp.groupby("_patient_key")["_score"].median()


@dataclass
class ColumnAction:
    """Record of a single normalization action on a column."""
    column: str
    action: str     # converted_numeric | normalized_missing | dropped_sparse
                    # | dropped_constant | flagged_ambiguous | required_missing
    detail: str
    count: int = 0


@dataclass
class CorrectionRecord:
    """Per-row normalization action for the exportable audit table.

    One record is appended for every value that the plausibility layer
    either corrected (via decimal reinterpretation) or cleared (set to
    NaN because no safe interpretation exists).  Records are aggregated
    into :attr:`IngestionReport.correction_records` and can be rendered
    as a DataFrame via :meth:`IngestionReport.audit_dataframe`.
    """
    column: str
    row_index: Any                       # DataFrame index label (patient ID when set as index)
    raw_value: str
    parsed_value: Optional[float]
    action: str                          # corrected_plausibility | consolidated_ambiguity_correction | cleared_implausible
    final_value: Optional[float]
    reason: str


@dataclass
class IngestionReport:
    """Structured report of all normalization actions applied during ingestion."""
    columns_converted: List[ColumnAction]
    missing_normalized: List[ColumnAction]
    columns_dropped: List[ColumnAction]       # reported only, not removed from df
    required_failures: List[ColumnAction]
    warnings: List[ColumnAction]
    n_rows_input: int
    n_rows_output: int
    n_columns_input: int
    n_columns_output: int
    correction_records: List[CorrectionRecord] = field(default_factory=list)

    def has_errors(self) -> bool:
        """True if any required columns are missing."""
        return len(self.required_failures) > 0

    def summary_lines(self) -> List[str]:
        """One-line-per-action summary suitable for UI display."""
        lines: List[str] = []
        for a in self.columns_converted:
            lines.append(f"[CONVERTED] {a.column}: {a.detail}")
        for a in self.missing_normalized:
            lines.append(f"[MISSING] {a.column}: {a.detail}")
        for a in self.columns_dropped:
            lines.append(f"[SPARSE] {a.column}: {a.detail}")
        for a in self.required_failures:
            lines.append(f"[ERROR] {a.column}: {a.detail}")
        for a in self.warnings:
            lines.append(f"[WARNING] {a.column}: {a.detail}")
        return lines

    def audit_dataframe(self) -> pd.DataFrame:
        """Exportable per-row audit table of every normalization action.

        Columns: ``column``, ``row_index``, ``raw_value``, ``parsed_value``,
        ``action``, ``final_value``, ``reason``.  Empty DataFrame (with the
        correct schema) when no corrections were applied.
        """
        cols = [
            "column", "row_index", "raw_value", "parsed_value",
            "action", "final_value", "reason",
        ]
        if not self.correction_records:
            return pd.DataFrame(columns=cols)
        rows = [
            {
                "column": r.column,
                "row_index": r.row_index,
                "raw_value": r.raw_value,
                "parsed_value": r.parsed_value,
                "action": r.action,
                "final_value": r.final_value,
                "reason": r.reason,
            }
            for r in self.correction_records
        ]
        return pd.DataFrame(rows, columns=cols)


@dataclass
class PreparedData:
    data: pd.DataFrame
    feature_columns: List[str]
    info: Dict[str, object]
    ingestion_report: Optional[IngestionReport] = None


@dataclass
class ExternalReadMeta:
    """Metadata captured during external CSV ingestion.

    Produced by :func:`read_external_table_with_fallback`.
    """
    encoding_used: str
    delimiter: str
    rows_loaded: int
    columns_loaded: int


@dataclass
class ExternalNormalizationReport:
    """Structured report of all normalization actions for an external dataset.

    Produced by :func:`normalize_external_dataset`.  Each field corresponds
    to the output of one pipeline stage; the full object is persisted in
    session state by the Temporal Validation tab.
    """
    source_name: Optional[str]
    read_meta: Optional[ExternalReadMeta]
    column_mapping: Dict[str, str]
    token_summary: Dict[str, dict]
    unit_summary: dict
    scope_summary: dict
    sts_readiness_summary: dict
    warnings: List[str]

    def summary_lines(self) -> List[str]:
        """One-line-per-topic summary suitable for UI display."""
        lines: List[str] = []
        if self.read_meta:
            lines.append(
                f"encoding: {self.read_meta.encoding_used}"
                f" \u00b7 delimiter: {self.read_meta.delimiter!r}"
                f" \u00b7 {self.read_meta.rows_loaded} rows"
                f" \u00b7 {self.read_meta.columns_loaded} columns"
            )
        renamed = {k: v for k, v in self.column_mapping.items() if k != v}
        if renamed:
            lines.append(
                f"columns renamed: {len(renamed)} alias(es) mapped to canonical names"
            )
        if self.token_summary:
            total_tok = sum(
                s.get("yes_converted", 0) + s.get("no_converted", 0)
                for s in self.token_summary.values()
            )
            lines.append(
                f"tokens normalized: {total_tok} value(s) in {len(self.token_summary)} column(s)"
            )
        u = self.unit_summary
        if u.get("height_converted"):
            lines.append(
                f"height: {u['n_height_converted']} value(s) converted"
                f" from inches to cm (original median {u['height_original_median']} in)"
            )
        if u.get("weight_converted"):
            lines.append(
                f"weight: {u['n_weight_converted']} value(s) converted"
                f" from lb to kg (original median {u['weight_original_median']} lb)"
            )
        s = self.scope_summary
        if s.get("n_pediatric", 0) > 0:
            lines.append(
                f"pediatric rows (age < 18): {s['n_pediatric']}"
                " \u2014 excluded from adult STS ACSD scope"
            )
        if s.get("n_sts_scope_excluded", 0) > 0:
            lines.append(
                f"STS-scope-excluded surgeries: {s['n_sts_scope_excluded']}"
                " (dissection / aneurysm / Bentall / Ross / transplant / homograft)"
            )
        sr = self.sts_readiness_summary
        if sr:
            lines.append(
                f"STS-ready rows after normalization:"
                f" {sr.get('n_ready', 0)}/{sr.get('n_total', 0)}"
                f" ({sr.get('n_ready_pct', 0):.1f}%)"
            )
        for w in self.warnings:
            lines.append(f"[WARNING] {w}")
        return lines

    def to_export_rows(self) -> List[Dict[str, Any]]:
        """Structured rows for the Normalization_Summary XLSX sheet.

        Returns a list of ``{"Field": str, "Value": ...}`` dicts covering
        ingestion metadata, scope/readiness counts, unit conversions, warnings,
        and all ``summary_lines()`` entries.  Never mutates the parent DataFrame.
        """
        rows: List[Dict[str, Any]] = []
        if self.read_meta is not None:
            rm = self.read_meta
            rows += [
                {"Field": "source_name",       "Value": str(self.source_name or "")},
                {"Field": "encoding_used",      "Value": rm.encoding_used},
                {"Field": "delimiter",          "Value": repr(rm.delimiter)},
                {"Field": "rows_loaded",        "Value": rm.rows_loaded},
                {"Field": "columns_loaded",     "Value": rm.columns_loaded},
            ]
        sc = self.scope_summary or {}
        rs = self.sts_readiness_summary or {}
        us = self.unit_summary or {}
        rows += [
            {"Field": "n_pediatric",            "Value": sc.get("n_pediatric", 0)},
            {"Field": "n_sts_scope_excluded",   "Value": sc.get("n_sts_scope_excluded", 0)},
            {"Field": "n_sts_ready",            "Value": rs.get("n_ready", "")},
            {"Field": "n_sts_ready_pct",        "Value": rs.get("n_ready_pct", "")},
            {"Field": "height_converted",       "Value": us.get("height_converted", False)},
            {"Field": "weight_converted",       "Value": us.get("weight_converted", False)},
        ]
        for i, warning in enumerate(self.warnings, 1):
            rows.append({"Field": f"warning_{i}", "Value": warning})
        for i, line in enumerate(self.summary_lines(), 1):
            rows.append({"Field": f"summary_line_{i}", "Value": line})
        return rows


def _load_source_tables(source_path: str) -> Dict[str, pd.DataFrame]:
    _COL_FIXES = {
        "Surgical Priorit": "Surgical Priority",
    }

    def _strip_col_whitespace(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
        df.columns = [_COL_FIXES.get(c, c) for c in df.columns]
        return df

    ext = Path(source_path).suffix.lower()
    if ext in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(source_path)
        return {
            name: _strip_col_whitespace(
                pd.read_excel(
                    source_path,
                    sheet_name=name,
                    **PANDAS_PRESERVE_NONE_READ_KWARGS,
                )
            )
            for name in xls.sheet_names
        }
    if ext in {".db", ".sqlite", ".sqlite3"}:
        conn = sqlite3.connect(source_path)
        try:
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
            return {name: _strip_col_whitespace(pd.read_sql_query(f'SELECT * FROM "{name}"', conn)) for name in tables}
        finally:
            conn.close()
    raise ValueError(f"Unsupported multi-table source format: {ext}")


def _normalize_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    rename_map = {col: FLAT_ALIAS_TO_APP_COLUMNS[col] for col in renamed.columns if col in FLAT_ALIAS_TO_APP_COLUMNS}
    renamed = renamed.rename(columns=rename_map)
    return renamed


def _read_csv_auto(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Read a CSV with automatic separator sniffing and encoding fallback.

    Brazilian data exported from Windows Excel is frequently saved as
    CP1252 (aka Windows-1252), not UTF-8.  Passing such a file to
    ``pd.read_csv`` with the default UTF-8 codec raises on the first
    accented character (``ç``, ``á``, ``ã`` …).  We try a short
    prioritized chain of encodings and return the first that succeeds:

    1. ``utf-8-sig`` — modern default, also strips a BOM if present.
    2. ``cp1252``   — Windows-1252, the most common Brazilian Excel export.
    3. ``latin-1``  — last resort; accepts any single byte so it cannot
       raise ``UnicodeDecodeError``, at the cost of slightly wrong
       rendering for characters in the 0x80-0x9F range.

    The separator is still sniffed by ``engine="python"`` / ``sep=None``
    so comma and semicolon files both work unchanged.
    """
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                nrows=nrows,
                encoding=enc,
                **PANDAS_PRESERVE_NONE_READ_KWARGS,
            )
        except UnicodeDecodeError as e:
            last_err = e
            continue
    # Unreachable in practice (latin-1 accepts any byte), but re-raise the
    # last real error so failures are still observable.
    raise last_err if last_err is not None else RuntimeError(
        f"Failed to read CSV with any known encoding: {path}"
    )


def _read_flat_excel(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Read an Excel workbook that represents a single flat cohort table.

    Multi-sheet XLSX files with the canonical source tables are handled by
    :func:`prepare_master_dataset`. This helper is for exported CSV-like Excel
    files, commonly saved with a default sheet name such as ``Planilha1``.
    """
    xls = pd.ExcelFile(path)
    if not xls.sheet_names:
        raise ValueError("Excel workbook has no sheets")
    if len(xls.sheet_names) > 1:
        raise ValueError(
            "Flat Excel files must contain exactly one sheet; multi-sheet "
            "workbooks must include Preoperative, Pre-Echocardiogram, and "
            "Postoperative sheets."
        )
    df = pd.read_excel(
        path,
        sheet_name=xls.sheet_names[0],
        nrows=nrows,
        **PANDAS_PRESERVE_NONE_READ_KWARGS,
    )
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df


def normalize_dataframe(
    df: pd.DataFrame,
    *,
    numeric_hint: set | None = None,
    required_columns: set | None = None,
    min_completion: float = 0.05,
    source_label: str = "unknown",
) -> Tuple[pd.DataFrame, IngestionReport]:
    """Unified normalization pipeline for training and validation data.

    Steps (in order):
    1. Missing-token normalization (context-aware for valve columns).
    2. Numeric conversion via :func:`parse_number` for object columns
       where >= 60 % of non-null values parse as numeric.
    3. Sparsity / constant-column analysis (reported, **not** dropped).
    4. Required-column validation.

    Parameters
    ----------
    df : DataFrame
        Data after format-specific loading and column mapping.
    numeric_hint : set of str, optional
        Columns forced through numeric conversion regardless of the
        auto-detection threshold.
    required_columns : set of str, optional
        Columns whose absence is recorded as a required-column failure.
    min_completion : float
        Minimum fraction of non-missing values.  Columns below this
        threshold are flagged as sparse (default 0.05 → >95 % missing).
    source_label : str
        Provenance tag included in the report.

    Returns
    -------
    (DataFrame, IngestionReport)
    """
    out = df.copy()
    n_rows = len(out)

    cols_converted: List[ColumnAction] = []
    missing_normalized: List[ColumnAction] = []
    cols_dropped: List[ColumnAction] = []
    required_failures: List[ColumnAction] = []
    report_warnings: List[ColumnAction] = []
    correction_records: List[CorrectionRecord] = []
    # Per-column map of row indices whose raw token was flagged as
    # EN-thousands-ambiguous by parse_number.  Passed into
    # _apply_clinical_plausibility so it can merge ambiguity +
    # successful plausibility correction into a single warning.
    ambiguous_by_col: Dict[str, Dict[Any, str]] = {}

    # ── 1. Missing-token normalization ──────────────────────────────
    for col in out.columns:
        if out[col].dtype != object:
            continue
        before_na = int(out[col].isna().sum())
        mask = out[col].apply(lambda v, c=col: is_missing(v, column=c))
        after_na = int(mask.sum())
        n_new = after_na - before_na
        if n_new > 0:
            out.loc[mask, col] = np.nan
            missing_normalized.append(ColumnAction(
                column=col,
                action="normalized_missing",
                detail=f"{n_new} missing token(s) \u2192 NaN",
                count=n_new,
            ))

    # ── 2. Numeric conversion ──────────────────────────────────────
    _force = numeric_hint or set()
    for col in out.columns:
        if out[col].dtype != object:
            continue
        non_null = out[col].dropna()
        if non_null.empty:
            continue

        # Per-row scan: capture ambiguous tokens with their row index
        # so they can later be merged with plausibility corrections.
        parsed_values: Dict[Any, float] = {}
        col_ambiguous: Dict[Any, str] = {}
        for idx, v in non_null.items():
            local_ambig: list = []
            parsed_values[idx] = parse_number(
                v, _warn_ambiguous=local_ambig, strict=True
            )
            if local_ambig:
                col_ambiguous[idx] = local_ambig[0]
        n_parsed = sum(1 for x in parsed_values.values() if pd.notna(x))
        pct = n_parsed / len(non_null)

        if col in _force or pct >= 0.6:
            out[col] = out[col].map(
                lambda v: parse_number(v, strict=True)
            )
            cols_converted.append(ColumnAction(
                column=col,
                action="converted_numeric",
                detail=f"{n_parsed}/{len(non_null)} non-null values parsed ({pct:.0%})",
                count=n_parsed,
            ))
            if col_ambiguous:
                ambiguous_by_col[col] = col_ambiguous

    # ── 2b. Clinical plausibility correction (narrow, column-specific) ──
    # Runs ONLY for the clinical columns listed in
    # _CLINICAL_PLAUSIBILITY_RANGES.  Values already inside the
    # plausible range (or NaN) are untouched; out-of-range values get
    # one reinterpretation attempt (comma → decimal point) and are
    # otherwise set to NaN.  Every action is logged via report_warnings
    # and, per-row, into ``correction_records`` for the audit table.
    covered_ambiguous_by_col = _apply_clinical_plausibility(
        out,
        df,
        report_warnings,
        ambiguous_by_col=ambiguous_by_col,
        correction_records=correction_records,
    )

    # ── 2c. Emit generic ambiguity warnings ONLY for tokens not already
    # consolidated by a successful plausibility correction.  This avoids
    # the redundant "flagged_ambiguous + corrected_plausibility" pair
    # that previously described the same value twice.
    for col, idx_to_raw in ambiguous_by_col.items():
        covered = covered_ambiguous_by_col.get(col, set())
        uncovered = {i: t for i, t in idx_to_raw.items() if i not in covered}
        if not uncovered:
            continue
        examples = list(uncovered.values())[:5]
        report_warnings.append(ColumnAction(
            column=col,
            action="flagged_ambiguous",
            detail=(
                f"Ambiguous values treated as EN thousands: {examples}"
            ),
            count=len(uncovered),
        ))

    # ── 3. Sparsity / constant-column analysis (report only) ──────
    for col in out.columns:
        miss_count = int(out[col].isna().sum())
        miss_rate = miss_count / n_rows if n_rows > 0 else 0
        if miss_rate > (1 - min_completion):
            cols_dropped.append(ColumnAction(
                column=col,
                action="dropped_sparse",
                detail=f"{miss_rate:.0%} missing ({miss_count}/{n_rows})",
                count=miss_count,
            ))
        elif out[col].dropna().nunique() <= 1:
            cols_dropped.append(ColumnAction(
                column=col,
                action="dropped_constant",
                detail="\u22641 unique non-missing value",
                count=0,
            ))

    # ── 4. Required-column check ──────────────────────────────────
    if required_columns:
        for col in sorted(required_columns):
            if col not in out.columns:
                required_failures.append(ColumnAction(
                    column=col,
                    action="required_missing",
                    detail=f"Required column not found in data",
                    count=0,
                ))

    report = IngestionReport(
        columns_converted=cols_converted,
        missing_normalized=missing_normalized,
        columns_dropped=cols_dropped,
        required_failures=required_failures,
        warnings=report_warnings,
        n_rows_input=len(df),
        n_rows_output=n_rows,
        n_columns_input=len(df.columns),
        n_columns_output=len(out.columns),
        correction_records=correction_records,
    )
    return out, report


# ══════════════════════════════════════════════════════════════════════════
# External-dataset normalization pipeline
# ══════════════════════════════════════════════════════════════════════════
#
# Staged normalization for external CSV datasets before they reach
# prepare_master_dataset / STS / temporal validation.  Every step is
# explicit, auditable, and logged.  No correction is made silently.
#
# Orchestrating entry point:  normalize_external_dataset(df_raw, ...)
#
# Stages (in order):
#   1. read_external_table_with_fallback — robust CSV reader + encoding meta
#   2. canonicalize_external_columns     — trim/alias column names
#   3. normalize_external_tokens         — Yes/No/Sim/Não variant normalization
#   4. normalize_external_units          — height/weight unit detection
#   5. apply_external_scope_rules        — pediatric flag, surgery-text cleaning
#   6. build_sts_readiness_flags         — per-row STS preflight assessment
#
# Auto-corrected (logged in report):
#   - Encoding fallback:  utf-8-sig → utf-8 → cp1252 → latin-1
#   - Column aliases:     snake_case → canonical display name
#   - Token variants:     Sim → Yes, Não → No, oui → Yes, etc.
#   - Anthropometric:     inches → cm, lb → kg (heuristic thresholds)
#
# Flagged only (NOT auto-corrected):
#   - Pediatric patients (age < 18)
#   - Out-of-scope STS surgeries (Bentall, dissection, aneurysm, …)
#   - Missing/invalid required STS fields
# ══════════════════════════════════════════════════════════════════════════


def _sniff_csv_delimiter(path: str, encoding: str) -> str:
    """Sniff the field delimiter of a CSV file using :mod:`csv.Sniffer`."""
    try:
        with open(path, encoding=encoding, errors="replace", newline="") as f:
            sample = f.read(8192)
        dialect = _csv_module.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return ","


def read_external_table_with_fallback(
    path: str,
) -> Tuple[pd.DataFrame, ExternalReadMeta]:
    """Read a CSV with a prioritized encoding fallback chain.

    Encoding chain (tried in order):

    1. ``utf-8-sig`` — modern default; also strips a BOM if present.
    2. ``utf-8``     — plain UTF-8.
    3. ``cp1252``    — Windows-1252, the most common Brazilian Excel export.
    4. ``latin-1``   — accepts any single byte; last resort.

    The delimiter is auto-detected via :mod:`csv.Sniffer`.

    Unlike the internal :func:`_read_csv_auto`, this function:

    * includes plain ``utf-8`` in the chain
    * returns :class:`ExternalReadMeta` (encoding, delimiter, shape)
    * is intended only for **external** datasets, not the training pipeline.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    (DataFrame, ExternalReadMeta)
    """
    _ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
    last_err: Exception | None = None
    for enc in _ENCODINGS:
        try:
            df = pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=enc,
                **PANDAS_PRESERVE_NONE_READ_KWARGS,
            )
            delim = _sniff_csv_delimiter(path, enc)
            meta = ExternalReadMeta(
                encoding_used=enc,
                delimiter=delim,
                rows_loaded=len(df),
                columns_loaded=len(df.columns),
            )
            return df, meta
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err if last_err is not None else RuntimeError(
        f"Failed to read CSV with any known encoding: {path}"
    )


def canonicalize_external_columns(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Normalize column names for external datasets.

    Steps applied to every column name:

    1. Strip leading/trailing whitespace.
    2. Collapse consecutive internal spaces to a single space.
    3. Map to a canonical app column name via :data:`FLAT_ALIAS_TO_APP_COLUMNS`
       (exact match first, then case-insensitive).

    Parameters
    ----------
    df : DataFrame
        Raw external dataframe (not modified in place).

    Returns
    -------
    (normalized_df, original_to_canonical)
        ``original_to_canonical`` maps every original column name to its
        canonical counterpart (equal to original when no renaming applies).
    """
    out = df.copy()
    _alias_lower = {k.lower(): v for k, v in FLAT_ALIAS_TO_APP_COLUMNS.items()}
    original_to_canonical: Dict[str, str] = {}
    rename_map: Dict[str, str] = {}

    for orig_col in list(df.columns):
        cleaned = re.sub(r"\s+", " ", str(orig_col)).strip()
        canonical = (
            FLAT_ALIAS_TO_APP_COLUMNS.get(cleaned)
            or _alias_lower.get(cleaned.lower())
            or cleaned
        )
        original_to_canonical[orig_col] = canonical
        if canonical != orig_col:
            rename_map[orig_col] = canonical

    out = out.rename(columns=rename_map)
    return out, original_to_canonical


# ── Token normalization ──────────────────────────────────────────────────
#
# Maps linguistic Yes/No variants to canonical English tokens.  Only columns
# where ≥ 50 % of non-null values match any known binary indicator token are
# processed — prevents accidental rewrites in free-text or multi-value columns.
#
# Auto-corrected:
#   YES variants → "Yes":  sim, sí, oui, ja  (+ case fix for "yes")
#   NO  variants → "No" :  não/nao, non, nein, nee  (+ case fix for "no")
#
# NOT handled here (already covered by normalize_dataframe / MISSING_TOKENS):
#   Unknown / N/A / "" / "-" / "--" → NaN

_TOKEN_NORM_MAP: Dict[str, str] = {
    # Case normalization (English)
    "yes":  "Yes",
    "no":   "No",
    # Portuguese
    "sim":  "Yes",
    "não":  "No",
    "nao":  "No",
    # Spanish (accented only — "si" alone is too ambiguous)
    "sí":   "Yes",
    # French
    "oui":  "Yes",
    "non":  "No",
    # German
    "ja":   "Yes",
    "nein": "No",
    # Dutch
    "nee":  "No",
}

# Broader set for column-type detection only (not used as output values)
_BINARY_INDICATOR_TOKENS: frozenset = frozenset(
    list(_TOKEN_NORM_MAP.keys()) + ["y", "n", "true", "false", "1", "0"]
)


def normalize_external_tokens(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Normalize Yes/No/Unknown token variants in object-dtype columns.

    For each object column (excluding valve-severity columns and already-
    converted numerics), linguistic variants are mapped to canonical English:

    * ``"Sim"`` / ``"sim"`` → ``"Yes"``; ``"Não"`` / ``"nao"`` → ``"No"``
    * ``"oui"`` → ``"Yes"``; ``"non"`` / ``"nein"`` → ``"No"``
    * Case-only fixes: ``"yes"`` → ``"Yes"``; ``"no"`` → ``"No"``

    A column is normalized only when ≥ 50 % of its non-null values appear in
    :data:`_BINARY_INDICATOR_TOKENS` — this avoids polluting free-text or
    multi-value columns with spurious Yes/No rewrites.

    Missing tokens (``"unknown"``, ``"n/a"``, ``""`` …) are left as-is; the
    downstream :func:`normalize_dataframe` step handles them.

    Returns
    -------
    (normalized_df, token_summary)
        ``token_summary`` is ``{column: {"yes_converted": int, "no_converted": int,
        "total": int}}``.
    """
    out = df.copy()
    token_summary: Dict[str, dict] = {}

    for col in out.columns:
        if out[col].dtype != object:
            continue
        if col in LITERAL_NONE_IS_VALID_COLUMNS:
            # Never overwrite columns where literal "None" is a clinical value.
            continue

        non_null = out[col].dropna()
        if non_null.empty:
            continue

        lower_vals = non_null.astype(str).str.strip().str.lower()
        hit_rate = lower_vals.isin(_BINARY_INDICATOR_TOKENS).sum() / len(non_null)
        if hit_rate < 0.50:
            continue  # Not enough binary tokens — skip to avoid polluting non-binary columns

        n_yes = n_no = 0
        for idx, val in non_null.items():
            lower = str(val).strip().lower()
            canonical = _TOKEN_NORM_MAP.get(lower)
            if canonical is None:
                continue
            if str(val).strip() != canonical:
                out.at[idx, col] = canonical
                if canonical == "Yes":
                    n_yes += 1
                else:
                    n_no += 1

        if n_yes + n_no > 0:
            token_summary[col] = {
                "yes_converted": n_yes,
                "no_converted": n_no,
                "total": n_yes + n_no,
            }

    return out, token_summary


# ── Anthropometric unit normalization ────────────────────────────────────
#
# Height heuristic:  median(height_cm) in (0, 100)  → suspect inches → × 2.54
#   Adult height in cm:     150–200  |  in inches: 59–79
#   Median below 100 is clinically impossible in cm for an adult population.
#
# Weight heuristic:  median(weight_kg) > 140 AND max > 250  → suspect lbs → / 2.205
#   Adult weight in kg:  40–200  |  in lbs: 88–440
#   Median above 140 combined with max above 250 strongly implies pounds.
#
# Optional BSA cross-check: DuBois formula; >20 % divergence logged as warning.


def normalize_external_units(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """Detect and correct common anthropometric unit mismatches.

    Targets ``Height (cm)`` / ``height_cm`` and ``Weight (kg)`` /
    ``weight_kg``.  See inline comments for the heuristic thresholds.

    An optional BSA cross-check is performed when ``BSA, m2`` / ``bsa_m2``
    is present and a conversion was applied.

    Returns
    -------
    (normalized_df, unit_summary)
        Keys: ``height_converted`` (bool), ``height_original_median``,
        ``n_height_converted``, ``weight_converted`` (bool),
        ``weight_original_median``, ``n_weight_converted``, ``warnings``.
    """
    out = df.copy()
    unit_summary: dict = {
        "height_converted": False,
        "height_conversion_factor": None,
        "height_original_median": None,
        "n_height_converted": 0,
        "weight_converted": False,
        "weight_conversion_factor": None,
        "weight_original_median": None,
        "n_weight_converted": 0,
        "warnings": [],
    }

    # ── Height ──────────────────────────────────────────────────────────
    h_col: Optional[str] = next(
        (c for c in ("Height (cm)", "height_cm") if c in out.columns), None
    )
    if h_col is not None:
        h_series = pd.to_numeric(out[h_col], errors="coerce")
        h_valid = h_series.dropna()
        if not h_valid.empty:
            h_median = float(h_valid.median())
            unit_summary["height_original_median"] = round(h_median, 1)
            if 0 < h_median < 100:
                factor = 2.54
                n_conv = int(h_series.notna().sum())
                out[h_col] = (h_series * factor).round(1)
                unit_summary.update(
                    height_converted=True,
                    height_conversion_factor=factor,
                    n_height_converted=n_conv,
                )

    # ── Weight ──────────────────────────────────────────────────────────
    w_col: Optional[str] = next(
        (c for c in ("Weight (kg)", "weight_kg") if c in out.columns), None
    )
    if w_col is not None:
        w_series = pd.to_numeric(out[w_col], errors="coerce")
        w_valid = w_series.dropna()
        if not w_valid.empty:
            w_median = float(w_valid.median())
            w_max = float(w_valid.max())
            unit_summary["weight_original_median"] = round(w_median, 1)
            if w_median > 140 and w_max > 250:
                factor = 1.0 / 2.205
                n_conv = int(w_series.notna().sum())
                out[w_col] = (w_series * factor).round(1)
                unit_summary.update(
                    weight_converted=True,
                    weight_conversion_factor=round(factor, 5),
                    n_weight_converted=n_conv,
                )

    # ── Mixed-unit heterogeneity check ─────────────────────────────────────
    # Emit a warning when the same column contains a suspicious mix of
    # metric and imperial values (e.g. some rows in cm, others in inches).
    # Threshold: each "unit regime" must represent at least 5 % of valid rows.
    _MIX_MIN_FRAC = 0.05
    if h_col is not None:
        h_series_raw = pd.to_numeric(df[h_col], errors="coerce")
        h_valid_raw = h_series_raw.dropna()
        if len(h_valid_raw) >= 10:
            n_imperial = int(((h_valid_raw >= 45) & (h_valid_raw <= 90)).sum())
            n_metric = int(((h_valid_raw >= 130) & (h_valid_raw <= 250)).sum())
            frac_imperial = n_imperial / len(h_valid_raw)
            frac_metric = n_metric / len(h_valid_raw)
            if frac_imperial >= _MIX_MIN_FRAC and frac_metric >= _MIX_MIN_FRAC:
                unit_summary["warnings"].append(
                    f"Mixed height units suspected in {h_col!r}: "
                    f"{n_imperial} value(s) in adult-inches range (45–90) and "
                    f"{n_metric} value(s) in adult-cm range (130–250). "
                    "Verify source data before relying on auto-conversion."
                )
    if w_col is not None:
        w_series_raw = pd.to_numeric(df[w_col], errors="coerce")
        w_valid_raw = w_series_raw.dropna()
        if len(w_valid_raw) >= 10:
            n_lb = int((w_valid_raw > 200).sum())       # >200 → almost certainly lbs
            n_kg = int((w_valid_raw < 100).sum())       # <100 → almost certainly kg
            frac_lb = n_lb / len(w_valid_raw)
            frac_kg = n_kg / len(w_valid_raw)
            if frac_lb >= _MIX_MIN_FRAC and frac_kg >= _MIX_MIN_FRAC:
                unit_summary["warnings"].append(
                    f"Mixed weight units suspected in {w_col!r}: "
                    f"{n_lb} value(s) > 200 (likely lb) and "
                    f"{n_kg} value(s) < 100 (likely kg). "
                    "Verify source data before relying on auto-conversion."
                )

    # ── Optional BSA cross-check ─────────────────────────────────────────
    bsa_col: Optional[str] = next(
        (c for c in ("BSA, m2", "bsa_m2") if c in out.columns), None
    )
    if bsa_col is not None and (
        unit_summary["height_converted"] or unit_summary["weight_converted"]
    ):
        if h_col is not None and w_col is not None:
            bsa_s = pd.to_numeric(out[bsa_col], errors="coerce")
            h_s = pd.to_numeric(out[h_col], errors="coerce")
            w_s = pd.to_numeric(out[w_col], errors="coerce")
            valid = h_s.notna() & w_s.notna() & bsa_s.notna()
            if valid.sum() >= 3:
                try:
                    computed = 0.007184 * (h_s[valid] ** 0.725) * (w_s[valid] ** 0.425)
                    ratio = float((computed / bsa_s[valid]).median())
                    if abs(ratio - 1.0) > 0.20:
                        unit_summary["warnings"].append(
                            f"BSA cross-check: computed BSA diverges from recorded BSA by "
                            f"{abs(ratio - 1.0) * 100:.0f}% (median ratio {ratio:.2f}). "
                            "Review unit conversion."
                        )
                except Exception:
                    pass

    return out, unit_summary


# ── Clinical scope rules ─────────────────────────────────────────────────
#
# STS_UNSUPPORTED_SURGERY_KEYWORDS is imported from sts_calculator — single
# canonical source shared by classify_sts_eligibility and apply_external_scope_rules.


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first *candidates* column name present in *df*, else ``None``."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def apply_external_scope_rules(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """Apply clinical scope rules and flag out-of-scope patients.

    Actions (in order):

    1. **Pediatric flag** — rows with ``Age (years)`` < 18 receive
       ``is_pediatric = True``.  These rows are outside adult STS ACSD
       scope and will be marked ``sts_input_ready = False`` downstream.

    2. **Surgery-text cleaning** — the surgery column is stripped, internal
       spaces collapsed, and separators standardised (``;`` / ``+`` → ``,``).
       The result is stored in ``_surgery_cleaned`` for downstream matching.

    3. **STS scope exclusion flag** — rows whose cleaned surgery text
       contains any keyword in :data:`STS_UNSUPPORTED_SURGERY_KEYWORDS` receive
       ``sts_scope_excluded = True`` and a human-readable ``sts_scope_reason``.

    Returns
    -------
    (df, scope_summary)
        Keys: ``n_pediatric``, ``n_sts_scope_excluded``, ``n_surgery_cleaned``,
        ``age_column_found``, ``surgery_column_found``, ``warnings``.
    """
    out = df.copy()
    scope_summary: dict = {
        "n_pediatric": 0,
        "n_sts_scope_excluded": 0,
        "n_surgery_cleaned": 0,
        "age_column_found": False,
        "surgery_column_found": False,
        "warnings": [],
    }

    # ── 1. Pediatric flag ─────────────────────────────────────────────────
    age_col = _find_col(out, ["Age (years)", "age_years", "age"])
    if age_col is not None:
        scope_summary["age_column_found"] = True
        ages = pd.to_numeric(out[age_col], errors="coerce")
        is_ped = (ages < 18) & ages.notna()
        out["is_pediatric"] = is_ped
        scope_summary["n_pediatric"] = int(is_ped.sum())
        if scope_summary["n_pediatric"] > 0:
            scope_summary["warnings"].append(
                f"{scope_summary['n_pediatric']} patient(s) with age < 18 flagged "
                "as pediatric — excluded from adult STS ACSD processing."
            )
    else:
        out["is_pediatric"] = False

    # ── 2. Surgery-text cleaning ──────────────────────────────────────────
    surg_col = _find_col(out, ["Surgery", "surgery_pre", "surgery"])
    if surg_col is not None:
        scope_summary["surgery_column_found"] = True
        raw_surg = out[surg_col].astype(str)
        cleaned_surg = (
            raw_surg
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(";", ",", regex=False)
            .str.replace("+", ",", regex=False)
        )
        scope_summary["n_surgery_cleaned"] = int((cleaned_surg != raw_surg).sum())
        out["_surgery_cleaned"] = cleaned_surg
    else:
        out["_surgery_cleaned"] = pd.Series("", index=out.index, dtype=object)

    # ── 3. STS scope exclusion flag ───────────────────────────────────────
    excluded = pd.Series(False, index=out.index, dtype=bool)
    reasons_series = pd.Series("", index=out.index, dtype=object)

    surg_upper = out["_surgery_cleaned"].astype(str).str.upper()
    for idx, s in surg_upper.items():
        for kw in sorted(STS_UNSUPPORTED_SURGERY_KEYWORDS):  # sorted for determinism
            if kw in s:
                excluded.at[idx] = True
                reasons_series.at[idx] = (
                    f"procedure outside STS ACSD scope: keyword '{kw}' found"
                )
                break

    out["sts_scope_excluded"] = excluded
    out["sts_scope_reason"] = reasons_series
    scope_summary["n_sts_scope_excluded"] = int(excluded.sum())
    if scope_summary["n_sts_scope_excluded"] > 0:
        scope_summary["warnings"].append(
            f"{scope_summary['n_sts_scope_excluded']} row(s) with surgery outside "
            "STS ACSD scope (dissection / aneurysm / Bentall / Ross / transplant / homograft)."
        )

    return out, scope_summary


def build_sts_readiness_flags(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """Build per-row STS preflight readiness flags.

    For each row, determines whether the patient is ready for an STS ACSD
    query by checking in order:

    1. **Pediatric** (``is_pediatric == True``) → ``sts_input_ready = False``
    2. **Scope excluded** (``sts_scope_excluded == True``) → ``False``
    3. **Missing required fields** (age, sex, surgery, surgical_priority) → ``False``
    4. **Invalid age** (outside [1, 110]) → ``False``
    5. All checks pass → ``sts_input_ready = True``

    If :func:`apply_external_scope_rules` has not been run first, the
    ``is_pediatric`` and ``sts_scope_excluded`` flags are computed inline.

    Adds columns:

    * ``sts_input_ready`` (bool)
    * ``sts_missing_required_fields`` (str, comma-separated field names)
    * ``sts_invalid_required_fields`` (str, comma-separated descriptions)
    * ``sts_readiness_reason`` (str)

    Returns
    -------
    (df, sts_readiness_summary)
        Keys: ``n_total``, ``n_ready``, ``n_pediatric_excluded``,
        ``n_scope_excluded``, ``n_missing_fields``, ``n_invalid_fields``,
        ``n_ready_pct``, ``required_fields_checked``.
    """
    out = df.copy()

    # Ensure scope flags exist (populated by apply_external_scope_rules if called)
    if "is_pediatric" not in out.columns:
        _ac = _find_col(out, ["Age (years)", "age_years"])
        if _ac is not None:
            _ages = pd.to_numeric(out[_ac], errors="coerce")
            out["is_pediatric"] = (_ages < 18) & _ages.notna()
        else:
            out["is_pediatric"] = False

    if "sts_scope_excluded" not in out.columns:
        out["sts_scope_excluded"] = False
        out["sts_scope_reason"] = ""

    _STS_REQ: Dict[str, List[str]] = {
        "age":               ["Age (years)", "age_years"],
        "sex":               ["Sex", "sex"],
        "surgery":           ["Surgery", "surgery_pre"],
        "surgical_priority": ["Surgical Priority", "surgical_priority"],
    }
    _resolved: Dict[str, Optional[str]] = {
        k: _find_col(out, v) for k, v in _STS_REQ.items()
    }

    ready_flags: List[bool] = []
    missing_lists: List[str] = []
    invalid_lists: List[str] = []
    reasons: List[str] = []

    for _, row in out.iterrows():
        is_ped = bool(row.get("is_pediatric", False))
        scope_exc = bool(row.get("sts_scope_excluded", False))
        scope_reason = str(row.get("sts_scope_reason", ""))

        missing: List[str] = []
        invalid: List[str] = []

        # age
        age_r = _resolved["age"]
        if age_r:
            age_val = pd.to_numeric(row.get(age_r), errors="coerce")
            if pd.isna(age_val):
                missing.append("age")
            elif not (1 <= float(age_val) <= 110):
                invalid.append(f"age={age_val:.0f} (must be 1\u2013110)")
        else:
            missing.append("age")

        # sex
        sex_r = _resolved["sex"]
        if sex_r:
            sv = str(row.get(sex_r) or "").strip()
            if not sv or sv.lower() in ("nan", "none", ""):
                missing.append("sex")
        else:
            missing.append("sex")

        # surgery
        surg_r = _resolved["surgery"]
        if surg_r:
            sv = str(row.get(surg_r) or "").strip()
            if not sv or sv.lower() in ("nan", "none", ""):
                missing.append("surgery")
        else:
            missing.append("surgery")

        # surgical_priority
        prio_r = _resolved["surgical_priority"]
        if prio_r:
            sv = str(row.get(prio_r) or "").strip()
            if not sv or sv.lower() in ("nan", "none", ""):
                missing.append("surgical_priority")
        else:
            missing.append("surgical_priority")

        # Determine readiness
        if is_ped:
            ready = False
            reason = "pediatric patient (age < 18) \u2014 outside adult STS ACSD scope"
        elif scope_exc:
            ready = False
            reason = scope_reason or "surgery outside STS ACSD scope"
        elif missing:
            ready = False
            reason = f"missing required fields: {', '.join(missing)}"
        elif invalid:
            ready = False
            reason = f"invalid required fields: {', '.join(invalid)}"
        else:
            ready = True
            reason = "all required fields present and valid"

        ready_flags.append(ready)
        missing_lists.append(", ".join(missing))
        invalid_lists.append(", ".join(invalid))
        reasons.append(reason)

    out["sts_input_ready"] = ready_flags
    out["sts_missing_required_fields"] = missing_lists
    out["sts_invalid_required_fields"] = invalid_lists
    out["sts_readiness_reason"] = reasons

    n_total = len(out)
    n_ready = int(sum(ready_flags))

    return out, {
        "n_total": n_total,
        "n_ready": n_ready,
        "n_pediatric_excluded": int(out["is_pediatric"].sum()),
        "n_scope_excluded": int(out["sts_scope_excluded"].sum()),
        "n_missing_fields": int(sum(1 for m in missing_lists if m)),
        "n_invalid_fields": int(sum(1 for m in invalid_lists if m)),
        "n_ready_pct": round(n_ready / n_total * 100, 1) if n_total > 0 else 0.0,
        "required_fields_checked": [k for k, v in _resolved.items() if v is not None],
    }


def normalize_external_dataset(
    df_raw: pd.DataFrame,
    source_name: Optional[str] = None,
    read_meta: Optional[ExternalReadMeta] = None,
) -> Tuple[pd.DataFrame, ExternalNormalizationReport]:
    """Orchestrating entry point for the external-dataset normalization pipeline.

    Runs the full staged pipeline:

    1. :func:`canonicalize_external_columns` — trim and map column aliases
    2. :func:`normalize_external_tokens`     — Yes/No linguistic normalization
    3. :func:`normalize_external_units`      — height/weight unit detection
    4. :func:`apply_external_scope_rules`    — pediatric flag + surgery cleaning
    5. :func:`build_sts_readiness_flags`     — per-row STS preflight assessment

    Call this function **before** passing an external CSV/Parquet dataset to
    :func:`prepare_master_dataset` or the temporal-validation STS batch.

    Parameters
    ----------
    df_raw : DataFrame
        Raw external dataframe (not modified in place; a copy is returned).
    source_name : str, optional
        Human-readable source identifier (e.g. filename) for audit purposes.
    read_meta : ExternalReadMeta, optional
        Encoding/delimiter metadata from :func:`read_external_table_with_fallback`.

    Returns
    -------
    (normalized_df, ExternalNormalizationReport)
    """
    df = df_raw.copy()

    df, column_mapping = canonicalize_external_columns(df)
    df = _impute_blank_as_none(df)
    df = _normalize_previous_surgery_column(df)
    df = _normalize_arrhythmia_recent_column(df)
    df = _normalize_arrhythmia_remote_column(df)
    df = _normalize_hf_column(df)
    df = _normalize_aortic_root_abscess_column(df)
    df = _normalize_cva_column(df)
    df = _normalize_pneumonia_column(df)
    df = _normalize_suspension_anticoagulation_days_column(df)
    df, token_summary = normalize_external_tokens(df)
    df, unit_summary = normalize_external_units(df)
    df, scope_summary = apply_external_scope_rules(df)
    df, sts_readiness_summary = build_sts_readiness_flags(df)

    all_warnings: List[str] = (
        list(scope_summary.get("warnings", []))
        + list(unit_summary.get("warnings", []))
    )

    report = ExternalNormalizationReport(
        source_name=source_name,
        read_meta=read_meta,
        column_mapping=column_mapping,
        token_summary=token_summary,
        unit_summary=unit_summary,
        scope_summary=scope_summary,
        sts_readiness_summary=sts_readiness_summary,
        warnings=all_warnings,
    )
    return df, report


def dry_run_external_ingestion(path: str) -> dict:
    """Read and normalize an external file without running any model inference.

    Performs file read (encoding auto-detection), column canonicalization, token
    and unit normalization, clinical scope checks, and STS preflight assessment.
    **Does not** call the AI Risk model, EuroSCORE II, or the STS ACSD web
    calculator.

    Parameters
    ----------
    path : str
        Absolute or relative path to a ``.csv`` or ``.parquet`` file.

    Returns
    -------
    dict with keys:
        ``path``                 — original path argument (str)
        ``read_meta``            — :class:`ExternalReadMeta` from file read
        ``normalized_df``        — normalized :class:`pandas.DataFrame`
        ``normalization_report`` — :class:`ExternalNormalizationReport`
        ``summary_lines``        — list[str] from ``report.summary_lines()``
        ``n_sts_ready``          — int: rows ready for STS processing
        ``n_sts_not_ready``      — int: rows excluded from STS processing
        ``warnings``             — list[str]: all normalization warnings
        ``error``                — str or None: read/parse error, if any

    Raises
    ------
    Does not raise — errors are captured in the ``error`` key so callers
    can surface them without a try/except.
    """
    result: dict = {
        "path": path,
        "read_meta": None,
        "normalized_df": None,
        "normalization_report": None,
        "summary_lines": [],
        "n_sts_ready": 0,
        "n_sts_not_ready": 0,
        "warnings": [],
        "error": None,
    }
    try:
        ext = Path(path).suffix.lower()
        if ext == ".parquet":
            df_raw = pd.read_parquet(path)
            read_meta = ExternalReadMeta(
                encoding_used="parquet",
                delimiter="N/A",
                rows_loaded=len(df_raw),
                columns_loaded=len(df_raw.columns),
            )
        elif ext == ".csv":
            df_raw, read_meta = read_external_table_with_fallback(path)
        else:
            result["error"] = (
                f"Unsupported file type {ext!r}. dry_run_external_ingestion"
                " accepts .csv and .parquet only."
            )
            return result

        df_norm, report = normalize_external_dataset(
            df_raw, source_name=Path(path).name, read_meta=read_meta
        )
        rs = report.sts_readiness_summary or {}
        result.update(
            read_meta=read_meta,
            normalized_df=df_norm,
            normalization_report=report,
            summary_lines=report.summary_lines(),
            n_sts_ready=int(rs.get("n_ready", 0)),
            n_sts_not_ready=int(rs.get("n_total", 0)) - int(rs.get("n_ready", 0)),
            warnings=list(report.warnings),
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    return result


def prepare_flat_dataset(source_path: str) -> PreparedData:
    ext = Path(source_path).suffix.lower()
    if ext == ".csv":
        df = _read_csv_auto(source_path)
    elif ext in {".xlsx", ".xls"}:
        df = _read_flat_excel(source_path)
    elif ext == ".parquet":
        df = pd.read_parquet(source_path)
    else:
        raise ValueError(f"Unsupported flat source format: {ext}")

    data = _normalize_flat_columns(df)

    # Expand simplified smoking status into the canonical status column
    if "_smoking_status_csv" in data.columns:
        _status = data["_smoking_status_csv"].astype(str).str.strip().str.lower()
        if "Smoking (Pack-year)" not in data.columns:
            data["Smoking (Pack-year)"] = _status.map(
                lambda v: "Current" if v == "current"
                else "Former" if v in {"former", "ex-smoker"}
                else "Never"
            )
        data.drop(columns=["_smoking_status_csv"], inplace=True)

    if "morte_30d" not in data.columns:
        if "Death" in data.columns:
            data["morte_30d"] = data["Death"].map(map_death_30d)
        else:
            raise ValueError("Flat dataset must include a 'morte_30d' column or a 'Death' column that can be converted into the outcome.")

    def _parse_pct_score(series: pd.Series) -> pd.Series:
        """Parse score values like '0,91%', '1.23%', '2,06' -> float (as %).

        Excel stores cells formatted as percentages as fractions (6.46% ->
        0.0646). CSV exports usually contain the visible percentage-point
        value (6.46). Normalize both representations to percentage points.
        """
        raw = series.copy()
        numeric_mask = raw.map(
            lambda v: isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, bool)
        )

        result = pd.Series(np.nan, index=series.index, dtype=float)
        if numeric_mask.any():
            result.loc[numeric_mask] = pd.to_numeric(raw.loc[numeric_mask], errors="coerce")
            numeric_vals = result.loc[numeric_mask].dropna()
            if not numeric_vals.empty and numeric_vals.between(0, 1).mean() >= 0.9:
                result.loc[numeric_mask] = result.loc[numeric_mask] * 100.0

        if (~numeric_mask).any():
            cleaned = (
                raw.loc[~numeric_mask].astype(str)
                .str.strip()
                .str.rstrip("%")
                .str.strip()
                .str.replace(",", ".", regex=False)
            )
            result.loc[~numeric_mask] = pd.to_numeric(cleaned, errors="coerce")
        return result

    if "euroscore_sheet" not in data.columns:
        if "EuroSCORE II" in data.columns:
            data["euroscore_sheet"] = _parse_pct_score(data["EuroSCORE II"])
        else:
            data["euroscore_sheet"] = np.nan
    if "euroscore_auto_sheet" not in data.columns:
        if "EuroSCORE II Automático" in data.columns:
            data["euroscore_auto_sheet"] = _parse_pct_score(data["EuroSCORE II Automático"])
        elif "EuroSCORE II" in data.columns:
            data["euroscore_auto_sheet"] = _parse_pct_score(data["EuroSCORE II"])
        else:
            data["euroscore_auto_sheet"] = np.nan
    if "sts_score_sheet" not in data.columns:
        sts_candidates = [c for c in data.columns if "Operative Mortality" in str(c)]
        if not sts_candidates:
            sts_candidates = [c for c in data.columns if c == "STS Score"]
        if sts_candidates:
            data["sts_score_sheet"] = _parse_pct_score(data[sts_candidates[0]])
        else:
            data["sts_score_sheet"] = np.nan

    if "Surgery" in data.columns:
        data["cirurgia_combinada"] = data["Surgery"].map(is_combined_surgery)
        data["peso_procedimento"] = data["Surgery"].map(procedure_weight)
        data["thoracic_aorta_flag"] = data["Surgery"].map(thoracic_aorta_surgery)
        data["procedure_group"] = data["Surgery"].map(procedure_group)

    if "_patient_key" not in data.columns:
        if "Name" in data.columns:
            data["_patient_key"] = data["Name"].map(normalize_patient)
        else:
            data["_patient_key"] = pd.Series(range(len(data))).astype(str)

    # ── Unified normalization ──
    data = _normalize_coronary_symptom_column(data)
    data = _impute_blank_as_none(data)
    data = _normalize_previous_surgery_column(data)
    data = _normalize_arrhythmia_recent_column(data)
    data = _normalize_arrhythmia_remote_column(data)
    data = _normalize_hf_column(data)
    data = _normalize_aortic_root_abscess_column(data)
    data = _normalize_cva_column(data)
    data = _normalize_pneumonia_column(data)
    data = _normalize_suspension_anticoagulation_days_column(data)
    data, ingestion_report = normalize_dataframe(data, source_label="flat")
    data = add_missingness_indicators(data)
    # Interpret blank as implicit "No" for binary history columns where the
    # source data convention is: present → documented; absent → left blank.
    data = _impute_blank_as_no(data)
    data = _add_previous_surgery_audit_cols(data)

    exclude_cols = set(NEVER_FEATURE_COLUMNS)
    allowed_cols = (
        set(FLAT_PREOP_ALLOWED_COLUMNS)
        | {"cirurgia_combinada", "peso_procedimento", "thoracic_aorta_flag", "_patient_key"}
        | set(MISSINGNESS_INDICATOR_COLUMNS)
    )
    _noise_cols_flat = {
        "Others informations", "Others",
        "Classification of Heart Failure According to Ejection Fraction",
        "Preoperative Medications",
        # Semantically retained in the analytical dataframe/Data Quality, but
        # not promoted into the active feature set without a dedicated ablation.
        "Arrhythmia Recent",
        "Aortic Root Abscess",
    }
    _engineered = {
        "cirurgia_combinada",
        "peso_procedimento",
        "thoracic_aorta_flag",
        *MISSINGNESS_INDICATOR_COLUMNS,
    }

    def _should_exclude_flat(col_name: str) -> bool:
        if col_name in _engineered:
            return False
        s = data[col_name]
        # Count true missing with column context: literal "None" is a valid
        # clinical category for valve/HF/arrhythmia fields.
        is_missing_mask = contextual_missing_mask(s, column=col_name)
        if is_missing_mask.sum() / len(s) > 0.95:
            return True
        # Zero variance (only 1 unique non-missing value)
        real_values = s[~is_missing_mask]
        if real_values.nunique() <= 1:
            return True
        return False

    feature_columns = [
        c for c in data.columns
        if c in allowed_cols and c not in exclude_cols and c != "_patient_key"
        and c not in _noise_cols_flat and not _should_exclude_flat(c)
    ]
    _nf_in_data = sorted(c for c in data.columns if c in NEVER_FEATURE_COLUMNS)
    info = {
        "n_rows": int(len(data)),
        "n_features": int(len(feature_columns)),
        "positive_rate": float(pd.to_numeric(data["morte_30d"], errors="coerce").mean()),
        "source_type": "flat",
        "never_feature_columns_in_source": _nf_in_data,
    }
    return PreparedData(data=data, feature_columns=feature_columns, info=info, ingestion_report=ingestion_report)


def prepare_master_dataset(xlsx_path: str, require_surgery_and_date: bool = True) -> PreparedData:
    ext = Path(xlsx_path).suffix.lower()
    if ext in {".csv", ".parquet"}:
        return prepare_flat_dataset(xlsx_path)

    tables = _load_source_tables(xlsx_path)
    missing_tables = [t for t in REQUIRED_SOURCE_TABLES if t not in tables]
    if missing_tables:
        if ext in {".xlsx", ".xls"} and len(tables) == 1:
            return prepare_flat_dataset(xlsx_path)
        raise ValueError(f"Missing required tables/sheets: {', '.join(missing_tables)}")

    pre = tables["Preoperative"]
    eco = tables["Pre-Echocardiogram"]
    post = tables["Postoperative"]
    eu = tables.get("EuroSCORE II")
    eu_auto = tables.get("EuroSCORE II Automático")
    sts = tables.get("STS Score")

    pre = pre.copy()
    post = post.copy()
    eco = eco.copy()

    pre["_patient_key"] = pre["Name"].map(normalize_patient)
    post["_patient_key"] = post["Patient"].map(normalize_patient)
    eco["_patient_key"] = eco["Patient"].map(normalize_patient)

    pre["_proc_date"] = _to_date(pre["Procedure Date"])
    post["_proc_date"] = _to_date(post["Procedure Date"])
    eco["_echo_date"] = _to_date(eco["Exam date"])

    pre_rows_before = len(pre)
    if require_surgery_and_date:
        pre["_surgery_txt"] = pre["Surgery"].astype(str).str.strip()
        pre = pre[(pre["_proc_date"].notna()) & (pre["_surgery_txt"].str.lower().ne("nan")) & (pre["_surgery_txt"] != "")]
        pre = pre.drop(columns=["_surgery_txt"])
    pre_rows_after = len(pre)
    pre_unique_after = int(pre[["_patient_key", "_proc_date"]].drop_duplicates().shape[0])
    post_unique = int(post[["_patient_key", "_proc_date"]].drop_duplicates().shape[0])

    pre_post = pre.merge(
        post[["_patient_key", "_proc_date", "Death"]],
        on=["_patient_key", "_proc_date"],
        how="inner",
    )
    pre_post_rows = len(pre_post)
    pre_post["morte_30d"] = pre_post["Death"].map(map_death_30d)

    eco_cols = [c for c in eco.columns if c not in {"Patient", "Exam date", "_patient_key", "_echo_date"}]
    echo_joined_rows = []
    for patient_key, grp in pre_post.groupby("_patient_key", sort=False):
        egrp = eco[eco["_patient_key"] == patient_key][["_echo_date"] + eco_cols]
        joined = _choose_echo_for_patient(grp[["_proc_date"]], egrp)
        echo_joined_rows.append(joined)

    if echo_joined_rows:
        echo_aligned = pd.concat(echo_joined_rows).sort_index()
    else:
        echo_aligned = pd.DataFrame(index=pre_post.index)

    for c in eco_cols:
        if c in echo_aligned.columns:
            pre_post[c] = echo_aligned[c]
        else:
            pre_post[c] = np.nan

    pre_post["cirurgia_combinada"] = pre_post["Surgery"].map(is_combined_surgery)
    pre_post["peso_procedimento"] = pre_post["Surgery"].map(procedure_weight)
    pre_post["thoracic_aorta_flag"] = pre_post["Surgery"].map(thoracic_aorta_surgery)
    pre_post["procedure_group"] = pre_post["Surgery"].map(procedure_group)

    eu_series = pd.Series(dtype=float)
    if eu is not None and "Patient" in eu.columns and "EuroSCORE II" in eu.columns:
        eu_series = _aggregate_score_by_patient(eu, "Patient", "EuroSCORE II")

    eu_auto_series = pd.Series(dtype=float)
    if eu_auto is not None and "Patient" in eu_auto.columns:
        eu_auto_cols = [c for c in eu_auto.columns if c != "Patient"]
        if eu_auto_cols:
            eu_auto_series = _aggregate_score_by_patient(eu_auto, "Patient", eu_auto_cols[-1])

    sts_series = pd.Series(dtype=float)
    if sts is not None and "Patient" in sts.columns:
        sts_candidates = [c for c in sts.columns if "Operative Mortality" in c]
        if sts_candidates:
            sts_series = _aggregate_score_by_patient(sts, "Patient", sts_candidates[0])

    pre_post["euroscore_sheet"] = pre_post["_patient_key"].map(eu_series)
    pre_post["euroscore_auto_sheet"] = pre_post["_patient_key"].map(eu_auto_series)
    pre_post["sts_score_sheet"] = pre_post["_patient_key"].map(sts_series)

    # ── Unified normalization ──
    pre_post = _normalize_coronary_symptom_column(pre_post)
    pre_post = _impute_blank_as_none(pre_post)
    pre_post = _normalize_previous_surgery_column(pre_post)
    pre_post = _normalize_arrhythmia_recent_column(pre_post)
    pre_post = _normalize_arrhythmia_remote_column(pre_post)
    pre_post = _normalize_hf_column(pre_post)
    pre_post = _normalize_aortic_root_abscess_column(pre_post)
    pre_post = _normalize_cva_column(pre_post)
    pre_post = _normalize_pneumonia_column(pre_post)
    pre_post = _normalize_suspension_anticoagulation_days_column(pre_post)
    pre_post, ingestion_report = normalize_dataframe(pre_post, source_label="master")
    pre_post = add_missingness_indicators(pre_post)
    # Interpret blank as implicit "No" for binary history columns where the
    # source data convention is: present → documented; absent → left blank.
    pre_post = _impute_blank_as_no(pre_post)
    pre_post = _add_previous_surgery_audit_cols(pre_post)

    pre_cols_exclude = {
        "Name",
        "Procedure Date",
        "Death",
        "_patient_key",
        "_proc_date",
    }
    echo_cols_exclude = {"Patient", "Exam date", "_patient_key", "_echo_date"}

    pre_features = [c for c in pre.columns if c not in pre_cols_exclude]
    echo_features = [c for c in eco.columns if c not in echo_cols_exclude and c not in pre_features]
    engineered = [
        "cirurgia_combinada",
        "peso_procedimento",
        "thoracic_aorta_flag",
        *MISSINGNESS_INDICATOR_COLUMNS,
    ]
    feature_columns = pre_features + echo_features + engineered
    feature_columns = [c for c in feature_columns if c in pre_post.columns]

    # Exclude features that add noise: >95% missing or known non-informative columns
    _noise_cols = {
        "Others informations", "Others",
        "Classification of Heart Failure According to Ejection Fraction",
        "Preoperative Medications",
        # Semantically retained in the analytical dataframe/Data Quality, but
        # not promoted into the active feature set without a dedicated ablation.
        "Arrhythmia Recent",
        "Aortic Root Abscess",
    }
    def _too_sparse_or_constant(col_name: str) -> bool:
        if col_name in engineered:
            return False
        s = pre_post[col_name]
        is_missing_mask = contextual_missing_mask(s, column=col_name)
        if is_missing_mask.sum() / len(s) > 0.95:
            return True
        real_values = s[~is_missing_mask]
        if real_values.nunique() <= 1:
            return True
        return False

    feature_columns = [
        c for c in feature_columns
        if c not in _noise_cols and not _too_sparse_or_constant(c)
    ]

    # Belt-and-suspenders: block any never-feature column that may have slipped
    # through sheet-level structural separation (e.g. a preop sheet that grew
    # unexpected columns).
    _nf_hits = sorted(c for c in feature_columns if c in NEVER_FEATURE_COLUMNS)
    if _nf_hits:
        warnings.warn(
            f"Never-feature policy intercepted {len(_nf_hits)} column(s) in "
            f"multi-sheet path: {_nf_hits}. Check source sheet for unexpected columns.",
            stacklevel=2,
        )
    feature_columns = [c for c in feature_columns if c not in NEVER_FEATURE_COLUMNS]

    info = {
        "n_rows": len(pre_post),
        "n_features": len(feature_columns),
        "positive_rate": float(pre_post["morte_30d"].mean()),
        "pre_rows_before_criteria": int(pre_rows_before),
        "pre_rows_after_criteria": int(pre_rows_after),
        "excluded_missing_surgery_or_date": int(pre_rows_before - pre_rows_after),
        "require_surgery_and_date": bool(require_surgery_and_date),
        "pre_unique_patient_date_after_criteria": pre_unique_after,
        "post_unique_patient_date": post_unique,
        "matched_pre_post_rows": int(pre_post_rows),
        "excluded_no_pre_post_match": int(pre_unique_after - pre_post_rows),
        "echo_rows": int(len(eco)),
        "available_optional_tables": [t for t in OPTIONAL_SOURCE_TABLES if t in tables],
        "never_feature_intercepted": _nf_hits,
    }

    return PreparedData(data=pre_post, feature_columns=feature_columns, info=info, ingestion_report=ingestion_report)
