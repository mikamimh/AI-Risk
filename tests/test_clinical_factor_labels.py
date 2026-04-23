"""Verify clinical interpretation factor labels are human-readable."""

import pytest
from tabs.prediction import _patient_factor_label


def _tr(en, pt):
    return en


def test_missingness_indicators_have_clinical_labels():
    """No missingness indicator should expose its raw snake_case name."""
    indicators = [
        "missing_renal_labs",
        "missing_cbc_labs",
        "missing_coagulation_labs",
        "missing_echo_key",
    ]
    for ind in indicators:
        label = _patient_factor_label(ind, {}, _tr)
        assert "missing_" not in label.lower(), (
            f"{ind} label '{label}' still contains raw snake_case — "
            "should be a clinical phrasing"
        )
        assert "_" not in label, (
            f"{ind} label '{label}' contains underscore — not human-readable"
        )


def test_derived_flags_have_clinical_labels():
    """Procedure-derived internal names should map to clinical phrases."""
    derived = [
        "peso_procedimento",
        "thoracic_aorta_flag",
        "cirurgia_combinada",
    ]
    for d in derived:
        label = _patient_factor_label(d, {}, _tr)
        assert "_" not in label, (
            f"{d} label '{label}' still contains underscore"
        )


def test_suspension_of_anticoagulation_is_human_readable():
    """'Suspension of Anticoagulation (day)' should become clinical text."""
    label_recent = _patient_factor_label(
        "Suspension of Anticoagulation (day)",
        {"Suspension of Anticoagulation (day)": 1.0},
        _tr,
    )
    assert "(day)" not in label_recent
    assert "_" not in label_recent

    label_remote = _patient_factor_label(
        "Suspension of Anticoagulation (day)",
        {"Suspension of Anticoagulation (day)": 5.0},
        _tr,
    )
    assert "(day)" not in label_remote
    assert "_" not in label_remote


def test_wbc_platelet_hematocrit_inr_readable():
    """CBC and coagulation lab labels should be clinical, not column names."""
    cases = [
        ("WBC Count (10³/μL)", {"WBC Count (10³/μL)": 12.0}),
        ("Platelet Count (cells/μL)", {"Platelet Count (cells/μL)": 100000.0}),
        ("Hematocrit (%)", {"Hematocrit (%)": 30.0}),
        ("INR", {"INR": 2.0}),
    ]
    for feature, form_map in cases:
        label = _patient_factor_label(feature, form_map, _tr)
        assert label != feature, (
            f"Feature '{feature}' returned itself as label — not human-readable"
        )
        assert "10³" not in label, f"Unit string leaked into label: {label}"
        assert "cells/" not in label, f"Unit string leaked into label: {label}"
