"""Regression test for EuroSCORE II implementation.

Validates that euroscore.euroscore_from_inputs produces results within
tolerance of the official EuroSCORE II calculator
(https://www.euroscore.org/calc.html) for 15 reference cases covering
low/intermediate/high-risk profiles, elective/urgent/emergency/salvage
urgencies, isolated/combined procedures, and specific comorbidities
(dialysis, endocarditis, redo, poor LV function, aortic surgery).

If this test fails after a change to euroscore.py or risk_data helpers
(procedure_weight, thoracic_aorta_surgery), inspect which case diverged —
a legitimate refactor should NOT change these outputs.

Reference values were obtained from the official EuroSCORE II calculator
and documented in euroscore_reference_cases_template.md at the repo root.
"""

import pytest
from euroscore import euroscore_from_inputs

# ---------------------------------------------------------------------------
# Shared "No" inputs reused across all cases to keep each dict concise.
# These represent the "nothing remarkable" baseline for every variable that
# is not explicitly set in a case.
# ---------------------------------------------------------------------------
_NO = "No"
_NONE_LVEF = 60          # LVEF >50% → "good" → no coefficient
_NONE_PSAP = None        # no pulmonary hypertension
_NONE_CS = ""            # no recent MI / coronary symptom

# ---------------------------------------------------------------------------
# Reference cases: (description, inputs_dict, expected_pct, abs_tol_pp)
#
# Tolerance policy:
#   0.15 pp  — default (difference ≤ calculator UI rounding)
#   0.50 pp  — high-risk cases (expected >30%) where small coefficient
#              errors cause larger absolute swings
# ---------------------------------------------------------------------------
REFERENCE_CASES = [
    (
        "Caso 1 — Jovem, CABG eletivo, baixo risco",
        {
            "Age (years)": 55,
            "Sex": "M",
            "Cr clearance, ml/min *": 90,
            "Dialysis": _NO,
            "PVD": _NO,
            "Poor mobility": _NO,
            "Previous surgery": _NO,
            "Chronic Lung Disease": _NO,
            "IE": _NO,
            "Critical preoperative state": _NO,
            "Diabetes": _NO,
            "Preoperative NYHA": "I",
            "CCS4": _NO,
            "Pré-LVEF, %": _NONE_LVEF,
            "Coronary Symptom": _NONE_CS,
            "PSAP": _NONE_PSAP,
            "Surgical Priority": "Elective",
            "Surgery": "CABG",
        },
        0.50,
        0.15,
    ),
    (
        "Caso 2 — Idoso, CABG eletivo, comorbidades leves",
        {
            "Age (years)": 72,
            "Sex": "M",
            "Cr clearance, ml/min *": 65,
            "Dialysis": _NO,
            "PVD": "Yes",
            "Poor mobility": _NO,
            "Previous surgery": _NO,
            "Chronic Lung Disease": "Yes",
            "IE": _NO,
            "Critical preoperative state": _NO,
            "Diabetes": _NO,
            "Preoperative NYHA": "II",
            "CCS4": _NO,
            "Pré-LVEF, %": _NONE_LVEF,
            "Coronary Symptom": _NONE_CS,
            "PSAP": _NONE_PSAP,
            "Surgical Priority": "Elective",
            "Surgery": "CABG",
        },
        2.15,
        0.15,
    ),
    (
        "Caso 3 — Mulher idosa, AVR eletivo, LVEF moderada",
        {
            "Age (years)": 78,
            "Sex": "F",
            "Cr clearance, ml/min *": 65,
            "Dialysis": _NO,
            "PVD": _NO,
            "Poor mobility": _NO,
            "Previous surgery": _NO,
            "Chronic Lung Disease": _NO,
            "IE": _NO,
            "Critical preoperative state": _NO,
            "Diabetes": _NO,
            "Preoperative NYHA": "II",
            "CCS4": _NO,
            "Pré-LVEF, %": 40,
            "Coronary Symptom": _NONE_CS,
            "PSAP": _NONE_PSAP,
            "Surgical Priority": "Elective",
            "Surgery": "AVR",
        },
        2.12,
        0.15,
    ),
    (
        "Caso 4 — Redo, AVR após CABG prévio",
        {
            "Age (years)": 68,
            "Sex": "M",
            "Cr clearance, ml/min *": 65,
            "Dialysis": _NO,
            "PVD": _NO,
            "Poor mobility": _NO,
            "Previous surgery": "Yes",
            "Chronic Lung Disease": _NO,
            "IE": _NO,
            "Critical preoperative state": _NO,
            "Diabetes": _NO,
            "Preoperative NYHA": "II",
            "CCS4": _NO,
            "Pré-LVEF, %": _NONE_LVEF,
            "Coronary Symptom": _NONE_CS,
            "PSAP": _NONE_PSAP,
            "Surgical Priority": "Elective",
            "Surgery": "AVR",
        },
        2.84,
        0.15,
    ),
    (
        "Caso 5 — Diálise, CABG eletivo",
        {
            "Age (years)": 65,
            "Sex": "M",
            "Cr clearance, ml/min *": None,
            "Dialysis": "Yes",
            "PVD": "Yes",
            "Poor mobility": _NO,
            "Previous surgery": _NO,
            "Chronic Lung Disease": _NO,
            "IE": _NO,
            "Critical preoperative state": _NO,
            "Diabetes": "Insulin",
            "Preoperative NYHA": "III",
            "CCS4": _NO,
            "Pré-LVEF, %": 40,
            "Coronary Symptom": _NONE_CS,
            "PSAP": 40,
            "Surgical Priority": "Elective",
            "Surgery": "CABG",
        },
        5.57,
        0.15,
    ),
    (
        "Caso 6 — Emergência, CABG, IAM recente",
        {
            "Age (years)": 70,
            "Sex": "M",
            "Cr clearance, ml/min *": 65,
            "Dialysis": _NO,
            "PVD": _NO,
            "Poor mobility": _NO,
            "Previous surgery": _NO,
            "Chronic Lung Disease": _NO,
            "IE": _NO,
            "Critical preoperative state": _NO,
            "Diabetes": _NO,
            "Preoperative NYHA": "III",
            "CCS4": "Yes",
            "Pré-LVEF, %": 40,
            "Coronary Symptom": "STEMI",
            "PSAP": _NONE_PSAP,
            "Surgical Priority": "Emergency",
            "Surgery": "CABG",
        },
        4.67,
        0.15,
    ),
    (
        "Caso 7 — Salvage, estado crítico, alto risco",
        {
            "Age (years)": 75,
            "Sex": "M",
            "Cr clearance, ml/min *": 40,
            "Dialysis": _NO,
            "PVD": "Yes",
            "Poor mobility": "Yes",
            "Previous surgery": _NO,
            "Chronic Lung Disease": "Yes",
            "IE": _NO,
            "Critical preoperative state": "Yes",
            "Diabetes": "Insulin",
            "Preoperative NYHA": "IV",
            "CCS4": "Yes",
            "Pré-LVEF, %": 25,
            "Coronary Symptom": "STEMI",
            "PSAP": 60,
            "Surgical Priority": "Salvage",
            "Surgery": "CABG",
        },
        86.45,
        0.50,  # high-risk case: larger abs tolerance
    ),
    (
        "Caso 8 — LVEF muito ruim (≤20%), MVR eletivo",
        {
            "Age (years)": 67,
            "Sex": "F",
            "Cr clearance, ml/min *": 65,
            "Dialysis": _NO,
            "PVD": _NO,
            "Poor mobility": _NO,
            "Previous surgery": _NO,
            "Chronic Lung Disease": _NO,
            "IE": _NO,
            "Critical preoperative state": _NO,
            "Diabetes": _NO,
            "Preoperative NYHA": "III",
            "CCS4": _NO,
            "Pré-LVEF, %": 15,
            "Coronary Symptom": _NONE_CS,
            "PSAP": 60,
            "Surgical Priority": "Elective",
            "Surgery": "MVR",
        },
        4.80,
        0.15,
    ),
    (
        "Caso 9 — Endocardite ativa, AVR urgente",
        {
            "Age (years)": 60,
            "Sex": "M",
            "Cr clearance, ml/min *": 65,
            "Dialysis": _NO,
            "PVD": _NO,
            "Poor mobility": _NO,
            "Previous surgery": _NO,
            "Chronic Lung Disease": _NO,
            "IE": "Yes",
            "Critical preoperative state": _NO,
            "Diabetes": _NO,
            "Preoperative NYHA": "III",
            "CCS4": _NO,
            "Pré-LVEF, %": 40,
            "Coronary Symptom": _NONE_CS,
            "PSAP": 40,
            "Surgical Priority": "Urgent",
            "Surgery": "AVR",
        },
        3.70,
        0.15,
    ),
    (
        "Caso 10 — Cirurgia de aorta torácica (2 procedimentos)",
        {
            "Age (years)": 65,
            "Sex": "M",
            "Cr clearance, ml/min *": 90,
            "Dialysis": _NO,
            "PVD": _NO,
            "Poor mobility": _NO,
            "Previous surgery": _NO,
            "Chronic Lung Disease": _NO,
            "IE": _NO,
            "Critical preoperative state": _NO,
            "Diabetes": _NO,
            "Preoperative NYHA": "II",
            "CCS4": _NO,
            "Pré-LVEF, %": _NONE_LVEF,
            "Coronary Symptom": _NONE_CS,
            "PSAP": _NONE_PSAP,
            "Surgical Priority": "Elective",
            # aortic aneurism repair = thoracic aorta flag + 1 major procedure;
            # AVR = second major procedure → weight_2 + thoracic_aorta
            "Surgery": "aortic aneurism repair, AVR",
        },
        2.10,
        0.15,
    ),
    (
        "Caso 11 — Duplo procedimento CABG + AVR",
        {
            "Age (years)": 70,
            "Sex": "M",
            "Cr clearance, ml/min *": 65,
            "Dialysis": _NO,
            "PVD": _NO,
            "Poor mobility": _NO,
            "Previous surgery": _NO,
            "Chronic Lung Disease": _NO,
            "IE": _NO,
            "Critical preoperative state": _NO,
            "Diabetes": _NO,
            "Preoperative NYHA": "II",
            "CCS4": _NO,
            "Pré-LVEF, %": 40,
            "Coronary Symptom": _NONE_CS,
            "PSAP": _NONE_PSAP,
            "Surgical Priority": "Elective",
            "Surgery": "CABG, AVR",
        },
        2.34,
        0.15,
    ),
    (
        "Caso 12 — Triplo procedimento, alto risco",
        {
            "Age (years)": 73,
            "Sex": "F",
            "Cr clearance, ml/min *": 40,
            "Dialysis": _NO,
            "PVD": "Yes",
            "Poor mobility": _NO,
            "Previous surgery": _NO,
            "Chronic Lung Disease": "Yes",
            "IE": _NO,
            "Critical preoperative state": _NO,
            "Diabetes": "Insulin",
            "Preoperative NYHA": "III",
            "CCS4": "Yes",
            "Pré-LVEF, %": 25,
            "Coronary Symptom": _NONE_CS,
            "PSAP": 40,
            "Surgical Priority": "Urgent",
            "Surgery": "CABG, AVR, MVR",
        },
        50.71,
        0.50,  # high-risk case: larger abs tolerance
    ),
    (
        "Caso 13 — Mulher jovem, AVR eletivo, sanity check",
        {
            "Age (years)": 45,
            "Sex": "F",
            "Cr clearance, ml/min *": 90,
            "Dialysis": _NO,
            "PVD": _NO,
            "Poor mobility": _NO,
            "Previous surgery": _NO,
            "Chronic Lung Disease": _NO,
            "IE": _NO,
            "Critical preoperative state": _NO,
            "Diabetes": _NO,
            "Preoperative NYHA": "I",
            "CCS4": _NO,
            "Pré-LVEF, %": _NONE_LVEF,
            "Coronary Symptom": _NONE_CS,
            "PSAP": _NONE_PSAP,
            "Surgical Priority": "Elective",
            "Surgery": "AVR",
        },
        0.62,
        0.15,
    ),
    (
        "Caso 14 — Homem 80+, CABG eletivo, múltiplas comorbidades",
        {
            "Age (years)": 82,
            "Sex": "M",
            "Cr clearance, ml/min *": 40,
            "Dialysis": _NO,
            "PVD": "Yes",
            "Poor mobility": "Yes",
            "Previous surgery": _NO,
            "Chronic Lung Disease": "Yes",
            "IE": _NO,
            "Critical preoperative state": _NO,
            "Diabetes": "Insulin",
            "Preoperative NYHA": "III",
            "CCS4": _NO,
            "Pré-LVEF, %": 40,
            "Coronary Symptom": _NONE_CS,
            "PSAP": _NONE_PSAP,
            "Surgical Priority": "Elective",
            "Surgery": "CABG",
        },
        13.25,
        0.15,
    ),
    (
        "Caso 15 — Intermediário limpo, âncora média",
        {
            "Age (years)": 60,
            "Sex": "M",
            "Cr clearance, ml/min *": 65,
            "Dialysis": _NO,
            "PVD": _NO,
            "Poor mobility": _NO,
            "Previous surgery": _NO,
            "Chronic Lung Disease": _NO,
            "IE": _NO,
            "Critical preoperative state": _NO,
            "Diabetes": _NO,
            "Preoperative NYHA": "II",
            "CCS4": _NO,
            "Pré-LVEF, %": _NONE_LVEF,
            "Coronary Symptom": _NONE_CS,
            "PSAP": _NONE_PSAP,
            "Surgical Priority": "Elective",
            "Surgery": "CABG",
        },
        0.75,
        0.15,
    ),
]


@pytest.mark.parametrize("description,inputs,expected_pct,tol_pp", REFERENCE_CASES)
def test_euroscore_matches_official_calculator(description, inputs, expected_pct, tol_pp):
    """EuroSCORE II output must match official calculator within tolerance."""
    actual_prob = euroscore_from_inputs(inputs)
    actual_pct = actual_prob * 100
    diff = actual_pct - expected_pct
    assert abs(diff) <= tol_pp, (
        f"{description}: expected {expected_pct:.2f}% (± {tol_pp} pp), "
        f"got {actual_pct:.2f}% (diff = {diff:+.3f} pp). "
        f"Inputs: {inputs}"
    )
