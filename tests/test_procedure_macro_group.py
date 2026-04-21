"""Tests for procedure_group() and audit_surgery_coverage() derivation."""
import pandas as pd
import pytest
from risk_data import audit_surgery_coverage, procedure_group


# ── Single-procedure mappings ─────────────────────────────────────────────────

@pytest.mark.parametrize("surgery,expected", [
    # CABG / coronary
    ("CABG", "CABG_OPCAB"),
    ("OPCAB", "CABG_OPCAB"),
    ("Surgical Treatment of Anomalous Aortic Origin of Coronary", "CABG_OPCAB"),
    ("Left Ventricular Aneurysmectomy", "CABG_OPCAB"),
    # Aortic valve
    ("AVR", "AORTIC_VALVE"),
    ("AV Repair", "AORTIC_VALVE"),
    ("ROSS", "AORTIC_VALVE"),
    ("Pulmonary homograft implantation", "AORTIC_VALVE"),
    # Mitral / tricuspid
    ("MVR", "MITRAL_TRICUSPID"),
    ("MV Repair", "MITRAL_TRICUSPID"),
    ("TVR", "MITRAL_TRICUSPID"),
    ("TV Repair", "MITRAL_TRICUSPID"),
    ("Myectomy", "MITRAL_TRICUSPID"),
    # Aortic root
    ("Bentall-de Bono procedure", "AORTA_ROOT"),
    ("Valve Sparing Aortic Root Replacement (David procedure)", "AORTA_ROOT"),
    # Aortic aneurysm / dissection
    ("Aortic Aneurism Repair", "AORTA_ANEURYSM"),
    ("Aortic Dissection Repair", "AORTA_ANEURYSM"),
    ("TEVAR", "AORTA_ANEURYSM"),
    # HF / transplant
    ("Heart transplant", "HF_TRANSPLANT"),
    # Congenital / structural
    ("ASD closure", "CONGENITAL_STRUCTURAL"),
    ("VSD Correction", "CONGENITAL_STRUCTURAL"),
    ("PFO Closure", "CONGENITAL_STRUCTURAL"),
    ("LAAO", "CONGENITAL_STRUCTURAL"),
    # Cardiac mass / thrombus
    ("Intracardiac tumor resection", "CARDIAC_MASS_THROMBUS"),
    ("Resection of intracardiac and/or pulmonary artery thrombus", "CARDIAC_MASS_THROMBUS"),
    ("Thrombus removal", "CARDIAC_MASS_THROMBUS"),
    # Other cardiac
    ("Pericardiectomy", "OTHER_CARDIAC"),
    ("Pacemaker implantation", "OTHER_CARDIAC"),
    ("Pacemaker electrode extraction", "OTHER_CARDIAC"),
])
def test_single_procedure_mapping(surgery, expected):
    assert procedure_group(surgery) == expected


# ── Blank / missing token → UNKNOWN ──────────────────────────────────────────

@pytest.mark.parametrize("surgery", [
    "",
    None,
    "nan",
    "unknown",
    "not informed",
    "n/a",
])
def test_absent_or_missing_token_returns_unknown(surgery):
    assert procedure_group(surgery) == "UNKNOWN"


# ── Present text not in taxonomy → OTHER ─────────────────────────────────────

@pytest.mark.parametrize("surgery", [
    "unspecified procedure xyz",
    "Some Future Technique",
    "Hybrid Coronary Revascularization",
])
def test_unrecognised_text_returns_other(surgery):
    assert procedure_group(surgery) == "OTHER"


# ── Priority resolution in combined surgeries ─────────────────────────────────

def test_aortic_valve_beats_cabg():
    assert procedure_group("AVR, CABG") == "AORTIC_VALVE"


def test_aorta_aneurysm_beats_aortic_valve():
    assert procedure_group("Aortic Aneurism Repair, AVR") == "AORTA_ANEURYSM"


def test_aorta_root_beats_aortic_valve():
    # Bentall involves both root and valve; Bentall should dominate
    assert procedure_group("Bentall-de Bono procedure, AVR") == "AORTA_ROOT"


def test_hf_transplant_beats_all():
    assert procedure_group("Heart transplant, CABG") == "HF_TRANSPLANT"
    assert procedure_group("Heart transplant, AVR") == "HF_TRANSPLANT"
    assert procedure_group("Heart transplant, Aortic Dissection Repair") == "HF_TRANSPLANT"


def test_cardiac_mass_thrombus_beats_aorta():
    assert procedure_group("Intracardiac tumor resection, Aortic Aneurism Repair") == "CARDIAC_MASS_THROMBUS"


def test_mitral_tricuspid_beats_cabg():
    assert procedure_group("MVR, CABG") == "MITRAL_TRICUSPID"


def test_aortic_valve_beats_mitral():
    # In combined AVR + MVR the aortic valve is higher priority
    assert procedure_group("AVR, MVR") == "AORTIC_VALVE"


def test_other_cardiac_loses_to_cabg():
    assert procedure_group("CABG, Pacemaker implantation") == "CABG_OPCAB"


def test_cabg_pacemaker_real_dataset_entry():
    assert procedure_group("CABG, Pacemaker implantation") == "CABG_OPCAB"


def test_aorta_dissection_and_tevar():
    assert procedure_group("Aortic Dissection Repair, TEVAR") == "AORTA_ANEURYSM"


def test_mv_repair_tv_repair():
    assert procedure_group("MV Repair , TV Repair") == "MITRAL_TRICUSPID"


def test_avr_mvr_tv_repair():
    assert procedure_group("AVR, MVR, TV Repair") == "AORTIC_VALVE"


def test_aorta_root_and_cabg():
    assert procedure_group("CABG, Bentall-de Bono procedure") == "AORTA_ROOT"


def test_ross_and_aortic_aneurysm():
    # ROSS → AORTIC_VALVE; Aortic Aneurism Repair → AORTA_ANEURYSM; AORTA_ANEURYSM wins
    assert procedure_group("ROSS, Aortic Aneurism Repair") == "AORTA_ANEURYSM"


def test_pulmonary_homograft_and_bentall():
    # Pulmonary homograft → AORTIC_VALVE; Bentall → AORTA_ROOT; AORTA_ROOT wins
    assert procedure_group("Pulmonary homograft implantation, Bentall-de Bono procedure") == "AORTA_ROOT"


def test_triple_aorta_avr_cabg_aneurysmectomy():
    # Aortic Aneurism Repair → AORTA_ANEURYSM wins over AORTIC_VALVE and CABG_OPCAB
    result = procedure_group("Aortic Aneurism Repair, AVR, CABG, Left Ventricular Aneurysmectomy")
    assert result == "AORTA_ANEURYSM"


# ── OTHER in combined: known beats OTHER ─────────────────────────────────────

def test_known_beats_other_in_combined():
    assert procedure_group("CABG, Some Unknown Technique") == "CABG_OPCAB"


def test_other_alone_is_other_not_unknown():
    assert procedure_group("Future Technique") == "OTHER"


# ── Separators ───────────────────────────────────────────────────────────────

def test_plus_separator():
    assert procedure_group("AVR + CABG") == "AORTIC_VALVE"


def test_semicolon_separator():
    assert procedure_group("AVR; CABG") == "AORTIC_VALVE"


def test_spaces_around_comma():
    assert procedure_group("MV Repair , TV Repair") == "MITRAL_TRICUSPID"


# ── audit_surgery_coverage ────────────────────────────────────────────────────

def test_coverage_all_known():
    series = pd.Series(["CABG", "AVR", "OPCAB", "Heart transplant", "MVR"])
    result = audit_surgery_coverage(series)
    assert result["total"] == 5
    assert result["n_mapped"] == 5
    assert result["n_unknown"] == 0
    assert result["n_other"] == 0
    assert result["coverage_rate"] == 1.0
    assert result["top_unrecognized"] == []


def test_coverage_with_blank_and_other():
    series = pd.Series(["CABG", "", None, "UnknownTech", "AVR"])
    result = audit_surgery_coverage(series)
    assert result["total"] == 5
    assert result["n_mapped"] == 2
    assert result["n_unknown"] == 2
    assert result["n_other"] == 1
    assert result["coverage_rate"] == pytest.approx(2 / 5)
    assert len(result["top_unrecognized"]) == 1
    assert result["top_unrecognized"][0][0] == "UnknownTech"


def test_coverage_empty_series():
    result = audit_surgery_coverage(pd.Series([], dtype=object))
    assert result["total"] == 0
    assert result["coverage_rate"] == 0.0


def test_coverage_missing_tokens_count_as_unknown():
    series = pd.Series(["nan", "unknown", "not informed"])
    result = audit_surgery_coverage(series)
    assert result["n_unknown"] == 3
    assert result["n_other"] == 0


def test_coverage_top_unrecognized_ordered_by_frequency():
    series = pd.Series(["FutureTech", "FutureTech", "OtherTech", "CABG"])
    result = audit_surgery_coverage(series)
    assert result["n_other"] == 3
    assert result["top_unrecognized"][0] == ("FutureTech", 2)
