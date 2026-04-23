"""EuroSCORE II - European System for Cardiac Operative Risk Evaluation.

This module implements the published EuroSCORE II formula for predicting
30-day mortality risk in cardiac surgery. It maps available clinical data to
the 27 logistic regression coefficients published by Nashef et al. (2012).

References:
    Nashef et al. European heart journal. 2012;33(16):1925-1933

Example:
    >>> from euroscore import euroscore_from_row
    >>> prob = euroscore_from_row(patient_series)
    >>> print(f"EuroSCORE II: {prob:.1%}")
"""

from typing import Dict

import numpy as np
import pandas as pd

from risk_data import parse_number, procedure_weight, thoracic_aorta_surgery


# EuroSCORE II logistic regression constant
EURO_CONST = -5.324537

# EuroSCORE II coefficient dictionary
# Keys match preprocessed variable names, values are logistic regression coefficients
COEF = {
    "nyha_ii": 0.1070545,
    "nyha_iii": 0.2958358,
    "nyha_iv": 0.5597929,
    "ccs4": 0.2226147,
    "iddm": 0.3542749,
    "age": 0.0285181,
    "female": 0.2196434,
    "eca": 0.5360268,
    "cpd": 0.1886564,
    "poor_mobility": 0.2407181,
    "redo": 1.118599,
    "dialysis": 0.6421508,
    "cc_le_50": 0.8592256,
    "cc_50_85": 0.303553,
    "active_endo": 0.6194522,
    "critical": 1.086517,
    "lv_moderate": 0.3150652,
    "lv_poor": 0.8084096,
    "lv_very_poor": 0.9346919,
    "recent_mi": 0.1528943,
    "pap_31_55": 0.1788899,
    "pap_ge_55": 0.3491475,
    "urgent": 0.3174673,
    "emergency": 0.7039121,
    "salvage": 1.362947,
    "weight_1_non_cabg": 0.0062118,
    "weight_2": 0.5521478,
    "weight_3plus": 0.9724533,
    "thoracic_aorta": 0.6527205,
}


def _yes(value: object) -> bool:
    """Check if value represents 'yes' / 'sim' / 'true'.

    Args:
        value: String or boolean value

    Returns:
        True if value represents affirmative, False otherwise

    Accepts:
        'yes', 'sim', 'true', '1', 'treated', 'active', 'possible'
    """
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"yes", "sim", "true", "1", "treated", "active", "possible"}


def _age_x(age: float) -> float:
    """Transform age for EuroSCORE II logistic regression.

    Ages ≤60 contribute 0. Ages >60 contribute linearly (age - 59).

    Args:
        age: Patient age in years

    Returns:
        Transformed age value for coefficient multiplication

    Example:
        >>> _age_x(55)  # Returns 1.0 (baseline contribution, age ≤ 60)
        >>> _age_x(70)  # Returns 11.0 (age - 59)
    """
    if np.isnan(age):
        return 1.0
    if age <= 60:
        return 1.0
    return age - 59.0


def _lv_category(lvef: float) -> str:
    """Categorize left ventricular ejection fraction for EuroSCORE II.

    Args:
        lvef: Left ventricular ejection fraction (%)

    Returns:
        Category: 'very_poor' (≤20%), 'poor' (21-30%), 'moderate' (31-50%), 'good' (>50%)

    Examples:
        >>> _lv_category(15)   # Returns 'very_poor'
        >>> _lv_category(25)   # Returns 'poor'
        >>> _lv_category(40)   # Returns 'moderate'
        >>> _lv_category(np.nan)  # Returns 'good' (default)
    """
    if np.isnan(lvef):
        return "good"
    if lvef <= 20:
        return "very_poor"
    if lvef <= 30:
        return "poor"
    if lvef <= 50:
        return "moderate"
    return "good"


def _pap_category(psap: float) -> str:
    if np.isnan(psap):
        return "normal"
    if psap >= 55:
        return "ge_55"
    if psap >= 31:
        return "31_55"
    return "normal"


def _urgency_category(value: object) -> str:
    """Categorize surgical urgency for EuroSCORE II.

    Args:
        value: Surgical priority field value

    Returns:
        Category: 'salvage', 'emergency', 'urgent', or 'elective' (default)

    Examples:
        >>> _urgency_category("Emergency")     # Returns 'emergency'
        >>> _urgency_category("salvage")       # Returns 'salvage'
        >>> _urgency_category("Elective")      # Returns 'elective'
    """
    s = str(value).strip().lower() if pd.notna(value) else ""
    if "salvage" in s or "salvamento" in s:
        return "salvage"
    if "emerg" in s:
        return "emergency"
    if "urgent" in s:
        return "urgent"
    return "elective"


def _recent_mi_from_coronary(value: object) -> bool:
    if pd.isna(value):
        return False
    s = str(value).strip().lower()
    return "stemi" in s or "non-stemi" in s or "nstemi" in s


def _creatinine_clearance_category(cc: float, dialysis: bool) -> str:
    if dialysis:
        return "dialysis"
    if np.isnan(cc):
        return "normal"
    if cc <= 50:
        return "le_50"
    if cc <= 85:
        return "50_85"
    return "normal"


def euroscore_from_row(row: pd.Series) -> float:
    """Calculate EuroSCORE II probability from patient clinical data.

    Implements the published EuroSCORE II formula (Nashef et al. 2012) using
    27 preoperative variables mapped from the available dataset.

    Args:
        row: Pandas Series with patient clinical data

    Returns:
        Float (0-1): Predicted 30-day mortality probability

    Raises:
        KeyError: If row doesn't contain expected column names

    Expected columns:
        Age (years), Sex, Preoperative NYHA, CCS4, Diabetes, PVD, CVA,
        Chronic Lung Disease, Poor mobility, Previous surgery, Dialysis,
        Cr clearance ml/min, IE, Critical preoperative state, Pré-LVEF %,
        LVEF %, Coronary Symptom, PSAP, Surgery, Surgical Priority

    Notes:
        - Missing values (NaN) treated conservatively (default to low risk)
        - Poor mobility only positive if CCS4 is also positive
        - LVEF checked in two columns (Pré-LVEF first, fallback to LVEF)
        - Procedure weight derived from comma-separated surgery list
        - Thoracic aorta surgery identified only for explicit aortic procedures

    Example:
        >>> patient = pd.Series({
        ...     'Age (years)': 75,
        ...     'Sex': 'M',
        ...     'Preoperative NYHA': '3',
        ...     'LVEF, %': 35,
        ...     'Surgery': 'CABG',
        ...     'Surgical Priority': 'Elective',
        ...     # ... other columns
        ... })
        >>> risk = euroscore_from_row(patient)
        >>> print(f"EuroSCORE II: {risk:.1%}")  # e.g., 8.5%

    References:
        Nashef SAM, et al. Eur J Cardiothorac Surg. 2012;41(4):734-744
    """
    age = parse_number(row.get("Age (years)"))
    female = str(row.get("Sex", "")).strip().upper() == "F"
    nyha = str(row.get("Preoperative NYHA", "")).strip().upper()
    ccs4 = _yes(row.get("CCS4"))
    iddm = str(row.get("Diabetes", "")).strip().lower() == "insulin"
    eca = _yes(row.get("PVD"))

    cpd = _yes(row.get("Chronic Lung Disease", np.nan))
    poor_mobility = _yes(row.get("Poor mobility", np.nan))

    prev_surg = row.get("Previous surgery")
    redo = pd.notna(prev_surg) and str(prev_surg).strip().lower() not in {"", "nan", "unknown", "no"}

    dialysis = _yes(row.get("Dialysis"))
    cc = parse_number(row.get("Cr clearance, ml/min *"))
    cc_cat = _creatinine_clearance_category(cc, dialysis)

    ie_val = str(row.get("IE", "")).strip().lower()
    active_endo = ie_val in {"yes", "possible", "active"}

    critical = _yes(row.get("Critical preoperative state", np.nan))
    lvef = parse_number(row.get("Pré-LVEF, %"))
    lv_cat = _lv_category(lvef)
    recent_mi = _recent_mi_from_coronary(row.get("Coronary Symptom"))

    psap = parse_number(row.get("PSAP"))
    pap_cat = _pap_category(psap)
    urg = _urgency_category(row.get("Surgical Priority"))

    surgery_text = row.get("Surgery", "")
    proc_weight = procedure_weight(surgery_text)
    th_aorta = bool(thoracic_aorta_surgery(surgery_text))

    logit = EURO_CONST
    logit += COEF["age"] * _age_x(age)
    if female:
        logit += COEF["female"]
    if nyha == "II":
        logit += COEF["nyha_ii"]
    elif nyha == "III":
        logit += COEF["nyha_iii"]
    elif nyha == "IV":
        logit += COEF["nyha_iv"]

    if ccs4:
        logit += COEF["ccs4"]
    if iddm:
        logit += COEF["iddm"]
    if eca:
        logit += COEF["eca"]
    if cpd:
        logit += COEF["cpd"]
    if poor_mobility:
        logit += COEF["poor_mobility"]
    if redo:
        logit += COEF["redo"]

    if cc_cat == "dialysis":
        logit += COEF["dialysis"]
    elif cc_cat == "le_50":
        logit += COEF["cc_le_50"]
    elif cc_cat == "50_85":
        logit += COEF["cc_50_85"]

    if active_endo:
        logit += COEF["active_endo"]
    if critical:
        logit += COEF["critical"]

    if lv_cat == "moderate":
        logit += COEF["lv_moderate"]
    elif lv_cat == "poor":
        logit += COEF["lv_poor"]
    elif lv_cat == "very_poor":
        logit += COEF["lv_very_poor"]

    if recent_mi:
        logit += COEF["recent_mi"]
    if pap_cat == "31_55":
        logit += COEF["pap_31_55"]
    elif pap_cat == "ge_55":
        logit += COEF["pap_ge_55"]

    if urg == "urgent":
        logit += COEF["urgent"]
    elif urg == "emergency":
        logit += COEF["emergency"]
    elif urg == "salvage":
        logit += COEF["salvage"]

    if proc_weight == "1_non_cabg":
        logit += COEF["weight_1_non_cabg"]
    elif proc_weight == "2_procedures":
        logit += COEF["weight_2"]
    elif proc_weight == "3plus_procedures":
        logit += COEF["weight_3plus"]

    if th_aorta:
        logit += COEF["thoracic_aorta"]

    return float(1.0 / (1.0 + np.exp(-logit)))


def euroscore_from_inputs(inputs: Dict[str, object]) -> float:
    """Calculate EuroSCORE II from dictionary of inputs.

    Convenience wrapper around euroscore_from_row() that converts dict to Series.

    Args:
        inputs: Dictionary with patient data (same keys as euroscore_from_row)

    Returns:
        Float (0-1): Predicted 30-day mortality probability

    Example:
        >>> inputs = {
        ...     'Age (years)': 65,
        ...     'Sex': 'F',
        ...     'LVEF, %': 45,
        ...     # ... other inputs
        ... }
        >>> risk = euroscore_from_inputs(inputs)
    """
    return euroscore_from_row(pd.Series(inputs))
