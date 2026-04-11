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

import re
import sqlite3
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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

NONE_IS_VALID_COLUMNS = {
    "Aortic Stenosis", "Aortic Regurgitation",
    "Mitral Stenosis", "Mitral Regurgitation",
    "Tricuspid Regurgitation",
    "aortic_stenosis_pre", "aortic_regurgitation_pre",
    "mitral_stenosis_pre", "mitral_regurgitation_pre",
    "tricuspid_regurgitation_pre",
}

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
    "pre_lvef_pct": "LVEF, %",
    "lvef_pre_pct": "LVEF, %",
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
    "ex_smoker_pack_years": "Ex-Smoker (Pack-year)",
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
    "lvef_pct": "LVEF, %",
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
    "LVEF, %",
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
    "Ex-Smoker (Pack-year)",
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
    "LVEF, %": (1.0, 100.0),
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
    For columns in :data:`NONE_IS_VALID_COLUMNS` (valve severity),
    ``"none"`` is **not** treated as missing because it is a valid
    clinical value meaning *no disease*.
    """
    if pd.isna(value):
        return True
    txt = str(value).strip()
    if not txt:
        return True
    txt_lower = txt.lower()
    if column and column in NONE_IS_VALID_COLUMNS and txt_lower == "none":
        return False
    return txt_lower in MISSING_TOKENS


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

    Delegates to :func:`parse_postop_timing` for parsing, then returns
    ``1`` if the event occurred within 30 days, ``0`` otherwise.
    Preserves legacy behaviour: unrecognised values map to 0 with warning.
    """
    timing = parse_postop_timing(value)
    if timing.category == "survivor":
        return 0
    if timing.category == "unknown":
        if timing.raw_value and timing.raw_value.lower() not in MISSING_TOKENS:
            warnings.warn(
                f"map_death_30d: unrecognised value '{timing.raw_value}' "
                "— treated as 0 (survivor). Please review this record.",
                stacklevel=2,
            )
        return 0
    return int(timing.within_30d)


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
        return {name: _strip_col_whitespace(pd.read_excel(source_path, sheet_name=name)) for name in xls.sheet_names}
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
    return pd.read_csv(path, sep=None, engine="python", nrows=nrows)


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


def prepare_flat_dataset(source_path: str) -> PreparedData:
    ext = Path(source_path).suffix.lower()
    if ext == ".csv":
        df = _read_csv_auto(source_path)
    elif ext == ".parquet":
        df = pd.read_parquet(source_path)
    else:
        raise ValueError(f"Unsupported flat source format: {ext}")

    data = _normalize_flat_columns(df)

    # Expand simplified smoking status into the two pack-year columns
    if "_smoking_status_csv" in data.columns:
        _status = data["_smoking_status_csv"].astype(str).str.strip().str.lower()
        if "Smoking (Pack-year)" not in data.columns:
            data["Smoking (Pack-year)"] = _status.map(
                lambda v: "30" if v == "current" else "Never"
            )
        if "Ex-Smoker (Pack-year)" not in data.columns:
            data["Ex-Smoker (Pack-year)"] = _status.map(
                lambda v: "20" if v in {"former", "ex-smoker"} else "Never"
            )
        data.drop(columns=["_smoking_status_csv"], inplace=True)

    if "morte_30d" not in data.columns:
        if "Death" in data.columns:
            data["morte_30d"] = data["Death"].map(map_death_30d)
        else:
            raise ValueError("Flat dataset must include a 'morte_30d' column or a 'Death' column that can be converted into the outcome.")

    def _parse_pct_score(series: pd.Series) -> pd.Series:
        """Parse score values like '0,91%', '1.23%', '2,06' → float (as %)."""
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.rstrip("%")
            .str.strip()
            .str.replace(",", ".", regex=False)
        )
        result = pd.to_numeric(cleaned, errors="coerce")
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

    if "_patient_key" not in data.columns:
        if "Name" in data.columns:
            data["_patient_key"] = data["Name"].map(normalize_patient)
        else:
            data["_patient_key"] = pd.Series(range(len(data))).astype(str)

    # ── Unified normalization ──
    data, ingestion_report = normalize_dataframe(data, source_label="flat")

    exclude_cols = {
        "morte_30d",
        "Death",
        "euroscore_sheet",
        "euroscore_auto_sheet",
        "euroscore_calc",
        "sts_score",
        "sts_score_sheet",
        "euroscore_sheet_clean",
        "euroscore_auto_sheet_clean",
        "ia_risk_oof",
        "ia_risk_fullfit",
        "classe_ia",
        "classe_euro",
        "classe_sts",
    }
    allowed_cols = set(FLAT_PREOP_ALLOWED_COLUMNS) | {"cirurgia_combinada", "peso_procedimento", "thoracic_aorta_flag", "_patient_key"}
    _noise_cols_flat = {
        "Others informations", "Others",
        "Classification of Heart Failure According to Ejection Fraction",
        "Preoperative Medications",
    }
    _engineered = {"cirurgia_combinada", "peso_procedimento", "thoracic_aorta_flag"}

    def _should_exclude_flat(col_name: str) -> bool:
        if col_name in _engineered:
            return False
        s = data[col_name]
        # Count true missing: NaN + MISSING_TOKENS (e.g. "-", "nan", "unknown")
        is_missing = s.isna() | s.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)
        if is_missing.sum() / len(s) > 0.95:
            return True
        # Zero variance (only 1 unique non-missing value)
        real_values = s[~is_missing]
        if real_values.nunique() <= 1:
            return True
        return False

    feature_columns = [
        c for c in data.columns
        if c in allowed_cols and c not in exclude_cols and c != "_patient_key"
        and c not in _noise_cols_flat and not _should_exclude_flat(c)
    ]
    info = {
        "n_rows": int(len(data)),
        "n_features": int(len(feature_columns)),
        "positive_rate": float(pd.to_numeric(data["morte_30d"], errors="coerce").mean()),
        "source_type": "flat",
    }
    return PreparedData(data=data, feature_columns=feature_columns, info=info, ingestion_report=ingestion_report)


def prepare_master_dataset(xlsx_path: str, require_surgery_and_date: bool = True) -> PreparedData:
    ext = Path(xlsx_path).suffix.lower()
    if ext in {".csv", ".parquet"}:
        return prepare_flat_dataset(xlsx_path)

    tables = _load_source_tables(xlsx_path)
    missing_tables = [t for t in REQUIRED_SOURCE_TABLES if t not in tables]
    if missing_tables:
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
    pre_post, ingestion_report = normalize_dataframe(pre_post, source_label="master")

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
    engineered = ["cirurgia_combinada", "peso_procedimento", "thoracic_aorta_flag"]
    feature_columns = pre_features + echo_features + engineered
    feature_columns = [c for c in feature_columns if c in pre_post.columns]

    # Exclude features that add noise: >95% missing or known non-informative columns
    _noise_cols = {
        "Others informations", "Others",
        "Classification of Heart Failure According to Ejection Fraction",
        "Preoperative Medications",
    }
    def _too_sparse_or_constant(col_name: str) -> bool:
        if col_name in engineered:
            return False
        s = pre_post[col_name]
        is_missing = s.isna() | s.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)
        if is_missing.sum() / len(s) > 0.95:
            return True
        real_values = s[~is_missing]
        if real_values.nunique() <= 1:
            return True
        return False

    feature_columns = [
        c for c in feature_columns
        if c not in _noise_cols and not _too_sparse_or_constant(c)
    ]

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
    }

    return PreparedData(data=pre_post, feature_columns=feature_columns, info=info, ingestion_report=ingestion_report)
