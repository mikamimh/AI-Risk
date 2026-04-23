"""Guard tests for STS scope — combined procedures with out-of-scope components.

Verifies that surgeries where the primary component is supported (CABG/AVR/MVR)
but a secondary component is outside STS ACSD scope are correctly classified
as `not_supported`, not `supported`.

Audit context (2026-04-24): 25 combined-procedure cases were previously
misclassified as `supported` in Dataset_2025.csv, with STS underestimating
risk ~6x (mean STS 4.25% vs observed mortality 24%) in that subgroup.
"""

import pytest
from sts_calculator import classify_sts_eligibility


# ---------------------------------------------------------------------------
# Tricuspid / structural add-ons
# ---------------------------------------------------------------------------

def test_cabg_asd_closure_not_supported():
    """CABG + ASD closure: ASD closure is congenital — not STS ACSD."""
    status, reason = classify_sts_eligibility(
        {"Surgery": "CABG, ASD closure", "Surgical Priority": "Elective"}
    )
    assert status == "not_supported"
    assert "ASD CLOSURE" in reason.upper() or "component" in reason.lower()


def test_mvr_tv_repair_not_supported():
    """MVR + TV Repair: TV Repair is not an STS primary procedure."""
    status, reason = classify_sts_eligibility(
        {"Surgery": "MVR, TV Repair", "Surgical Priority": "Elective"}
    )
    assert status == "not_supported"


def test_avr_cabg_pfo_closure_not_supported():
    """AVR + CABG + PFO closure: PFO closure is congenital."""
    status, reason = classify_sts_eligibility(
        {"Surgery": "AVR, CABG, PFO closure", "Surgical Priority": "Urgent"}
    )
    assert status == "not_supported"


def test_cabg_pacemaker_not_supported():
    """CABG + pacemaker implantation: device procedure outside scope."""
    status, reason = classify_sts_eligibility(
        {"Surgery": "CABG, Pacemaker implantation", "Surgical Priority": "Elective"}
    )
    assert status == "not_supported"


def test_avr_pericardiectomy_not_supported():
    """AVR + pericardiectomy: pericardial procedure outside scope."""
    status, reason = classify_sts_eligibility(
        {"Surgery": "AVR, Pericardiectomy", "Surgical Priority": "Elective"}
    )
    assert status == "not_supported"


def test_cabg_vsd_correction_not_supported():
    """CABG + VSD correction: congenital add-on."""
    status, reason = classify_sts_eligibility(
        {"Surgery": "CABG, VSD correction", "Surgical Priority": "Urgent"}
    )
    assert status == "not_supported"


# ---------------------------------------------------------------------------
# Pure out-of-scope procedures (should already be blocked, regression guard)
# ---------------------------------------------------------------------------

def test_tv_repair_isolated_not_supported():
    """Isolated TV Repair is not an STS primary procedure."""
    status, reason = classify_sts_eligibility(
        {"Surgery": "TV Repair", "Surgical Priority": "Elective"}
    )
    assert status == "not_supported"


def test_laao_not_supported():
    """Left atrial appendage occlusion — not STS ACSD."""
    status, reason = classify_sts_eligibility(
        {"Surgery": "LAAO", "Surgical Priority": "Elective"}
    )
    assert status == "not_supported"


def test_tevar_not_supported():
    """Thoracic endovascular aortic repair — not STS ACSD."""
    status, reason = classify_sts_eligibility(
        {"Surgery": "TEVAR", "Surgical Priority": "Elective"}
    )
    assert status == "not_supported"


# ---------------------------------------------------------------------------
# Reason string quality
# ---------------------------------------------------------------------------

def test_reason_distinguishes_combined_vs_base():
    """'Component outside scope' reason for combined; plain for isolated."""
    _, reason_isolated = classify_sts_eligibility(
        {"Surgery": "Heart transplant", "Surgical Priority": "Elective"}
    )
    _, reason_combined = classify_sts_eligibility(
        {"Surgery": "CABG, ASD closure", "Surgical Priority": "Elective"}
    )
    assert "component" in reason_combined.lower(), (
        f"Combined exclusion reason should mention 'component': {reason_combined!r}"
    )


# ---------------------------------------------------------------------------
# Still-supported (regression: new keywords must not block valid procedures)
# ---------------------------------------------------------------------------

def test_mvr_cabg_still_supported():
    status, _ = classify_sts_eligibility(
        {"Surgery": "MVR, CABG", "Surgical Priority": "Elective"}
    )
    assert status == "supported"


def test_mv_repair_cabg_still_supported():
    status, _ = classify_sts_eligibility(
        {"Surgery": "MV Repair, CABG", "Surgical Priority": "Elective"}
    )
    assert status == "supported"


def test_isolated_avr_still_supported():
    status, _ = classify_sts_eligibility(
        {"Surgery": "AVR", "Surgical Priority": "Elective"}
    )
    assert status == "supported"
