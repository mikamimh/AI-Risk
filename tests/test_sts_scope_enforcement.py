"""Verify STS scope is enforced before web calculator query."""

from sts_calculator import classify_sts_eligibility


def test_classifies_heart_transplant_as_not_supported():
    row = {"Surgery": "Heart transplant", "Surgical Priority": "Elective"}
    status, reason = classify_sts_eligibility(row)
    assert status == "not_supported"


def test_classifies_bentall_as_not_supported():
    row = {"Surgery": "Bentall-de Bono procedure", "Surgical Priority": "Elective"}
    status, reason = classify_sts_eligibility(row)
    assert status == "not_supported"


def test_classifies_aortic_aneurism_repair_as_not_supported():
    row = {"Surgery": "Aortic Aneurism Repair", "Surgical Priority": "Elective"}
    status, reason = classify_sts_eligibility(row)
    assert status == "not_supported"


def test_classifies_isolated_cabg_as_supported():
    row = {"Surgery": "CABG", "Surgical Priority": "Elective"}
    status, reason = classify_sts_eligibility(row)
    assert status == "supported"


def test_classifies_avr_cabg_combo_as_supported():
    row = {"Surgery": "AVR, CABG", "Surgical Priority": "Elective"}
    status, reason = classify_sts_eligibility(row)
    assert status == "supported"


def test_classifies_aortic_aneurism_plus_cabg_as_not_supported():
    """Mixed-scope surgeries default to not_supported (aorta keyword wins)."""
    row = {"Surgery": "Aortic Aneurism Repair, CABG", "Surgical Priority": "Elective"}
    status, reason = classify_sts_eligibility(row)
    assert status == "not_supported"
