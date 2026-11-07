"""Regression tests: "Previous surgery" absence tokens must not trigger the
EuroSCORE II redo coefficient.

risk_data.normalize_previous_surgery_value canonicalizes blank / "No" to the
literal string "None". The previous rule in euroscore_from_row did not list
"none" as absence, so every patient without prior surgery received the redo
coefficient (+1.118599 on the logit).
"""

import math

import numpy as np
import pandas as pd
import pytest

from euroscore import (
    COEF,
    _has_previous_cardiac_surgery,
    euroscore_from_inputs,
    euroscore_from_row,
)
from tests.test_euroscore_reference import REFERENCE_CASES

_BASE = {
    "Age (years)": 65,
    "Sex": "M",
    "Preoperative NYHA": "II",
    "CCS4": "No",
    "Diabetes": "No",
    "PVD": "No",
    "Chronic Lung Disease": "No",
    "Poor mobility": "No",
    "Dialysis": "No",
    "Cr clearance, ml/min *": 90,
    "IE": "No",
    "Critical preoperative state": "No",
    "Pré-LVEF, %": 60,
    "Coronary Symptom": "",
    "PSAP": None,
    "Surgery": "CABG",
    "Surgical Priority": "Elective",
}

_ABSENCE_VALUES = [
    "None", "none", " NONE ", np.nan, None, "", "No", "Unknown",
    "nenhuma", "Não", "nao", "-", "--",
]
_REDO_VALUES = ["CABG", "AVR", "MV repair", "Yes", "AVR + CABG; MVR"]


def _score(prev):
    return euroscore_from_row(pd.Series(dict(_BASE, **{"Previous surgery": prev})))


def _logit(p):
    return math.log(p / (1.0 - p))


@pytest.mark.parametrize("value", _ABSENCE_VALUES)
def test_absence_tokens_match_no(value):
    assert _has_previous_cardiac_surgery(value) is False
    assert _score(value) == pytest.approx(_score("No"), abs=1e-12)


@pytest.mark.parametrize("value", _REDO_VALUES)
def test_redo_text_adds_redo_coefficient(value):
    assert _has_previous_cardiac_surgery(value) is True
    assert _logit(_score(value)) - _logit(_score("No")) == pytest.approx(COEF["redo"], abs=1e-9)


def test_reference_case_with_none_matches_official_value():
    """Caso 1 (official calculator, Previous surgery = No) must give the same
    expected value when the field carries the canonical "None"."""
    desc, inputs, expected_pct, tol_pp = REFERENCE_CASES[0]
    assert inputs["Previous surgery"] == "No"
    result_pct = euroscore_from_inputs(dict(inputs, **{"Previous surgery": "None"})) * 100
    assert abs(result_pct - expected_pct) <= tol_pp, desc


def test_prepared_dataframe_none_is_not_redo():
    df = pd.DataFrame([dict(_BASE, **{"Previous surgery": v}) for v in ["None", "None", "AVR"]])
    scores = df.apply(euroscore_from_row, axis=1)
    assert scores.iloc[0] == pytest.approx(_score("No"))
    assert scores.iloc[1] == pytest.approx(_score("No"))
    assert scores.iloc[2] > scores.iloc[0]
