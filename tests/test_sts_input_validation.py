"""Tests for validate_sts_input and its integration with calculate_sts_batch Phase A.

Covers 10 scenarios (V-1 … V-10):

  V-1  Valid payload passes with no errors.
  V-2  Missing age → error reported.
  V-3  Non-numeric age → error reported.
  V-4  Age out of range (0, 111) → error reported.
  V-5  Invalid procid → error reported.
  V-6  Invalid gender → error reported.
  V-7  Invalid status → error reported.
  V-8  Multiple errors returned simultaneously (age + gender bad).
  V-9  Phase A integration: invalid payload is logged as build_input failure.
  V-10 Phase A integration: valid payload is NOT rejected pre-flight.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from sts_calculator import validate_sts_input


# ---------------------------------------------------------------------------
# Minimal valid payload helper
# ---------------------------------------------------------------------------

def _valid() -> dict:
    """Minimal STS payload that should pass all four checks."""
    return {
        "procid": "1",
        "age": "65",
        "gender": "Male",
        "status": "Elective",
    }


# ---------------------------------------------------------------------------
# V-1: valid payload — no errors
# ---------------------------------------------------------------------------

def test_valid_payload_no_errors():
    """A well-formed payload must return an empty error list."""
    errors = validate_sts_input(_valid())
    assert errors == [], f"Expected no errors, got: {errors}"


# ---------------------------------------------------------------------------
# V-2: missing age
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("age_val", ["", None])
def test_missing_age_reported(age_val):
    """Empty or absent age must produce an age error."""
    d = _valid()
    if age_val is None:
        d.pop("age")
    else:
        d["age"] = age_val
    errors = validate_sts_input(d)
    assert any("age" in e for e in errors), f"Expected age error, got: {errors}"


# ---------------------------------------------------------------------------
# V-3: non-numeric age
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_age", ["abc", "N/A", "sixty-five", "--"])
def test_non_numeric_age_reported(bad_age):
    """Non-numeric age string must produce an age error."""
    d = _valid()
    d["age"] = bad_age
    errors = validate_sts_input(d)
    assert any("age" in e and "not numeric" in e for e in errors), (
        f"Expected 'age: not numeric' error for {bad_age!r}, got: {errors}"
    )


# ---------------------------------------------------------------------------
# V-4: age out of range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_age", ["0", "0.5", "111", "200", "-1"])
def test_age_out_of_range_reported(bad_age):
    """Age outside [1, 110] must produce an age range error."""
    d = _valid()
    d["age"] = bad_age
    errors = validate_sts_input(d)
    assert any("age" in e and "out of range" in e for e in errors), (
        f"Expected 'age: out of range' error for {bad_age!r}, got: {errors}"
    )


@pytest.mark.parametrize("good_age", ["1", "18", "65", "99", "110"])
def test_age_in_range_accepted(good_age):
    """Age within [1, 110] must not produce an age error."""
    d = _valid()
    d["age"] = good_age
    errors = validate_sts_input(d)
    assert not any("age" in e for e in errors), (
        f"Unexpected age error for {good_age!r}: {errors}"
    )


# ---------------------------------------------------------------------------
# V-5: invalid procid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_pid", ["0", "9", "10", "", "cabg", None])
def test_invalid_procid_reported(bad_pid):
    """procid outside '1'–'8' must produce a procid error."""
    d = _valid()
    if bad_pid is None:
        d.pop("procid")
    else:
        d["procid"] = bad_pid
    errors = validate_sts_input(d)
    assert any("procid" in e for e in errors), (
        f"Expected procid error for {bad_pid!r}, got: {errors}"
    )


@pytest.mark.parametrize("good_pid", ["1", "2", "3", "4", "5", "6", "7", "8"])
def test_valid_procid_accepted(good_pid):
    """Each valid procid must not produce an error."""
    d = _valid()
    d["procid"] = good_pid
    errors = validate_sts_input(d)
    assert not any("procid" in e for e in errors), (
        f"Unexpected procid error for {good_pid!r}: {errors}"
    )


# ---------------------------------------------------------------------------
# V-6: invalid gender
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_gender", ["male", "female", "M", "F", "", "Unknown", None])
def test_invalid_gender_reported(bad_gender):
    """gender not in {'Male', 'Female'} must produce a gender error."""
    d = _valid()
    if bad_gender is None:
        d.pop("gender")
    else:
        d["gender"] = bad_gender
    errors = validate_sts_input(d)
    assert any("gender" in e for e in errors), (
        f"Expected gender error for {bad_gender!r}, got: {errors}"
    )


@pytest.mark.parametrize("good_gender", ["Male", "Female"])
def test_valid_gender_accepted(good_gender):
    d = _valid()
    d["gender"] = good_gender
    errors = validate_sts_input(d)
    assert not any("gender" in e for e in errors), (
        f"Unexpected gender error for {good_gender!r}: {errors}"
    )


# ---------------------------------------------------------------------------
# V-7: invalid status
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_status", ["Elective ", "urgent", "EMERGENCY", "", "Salvage", None])
def test_invalid_status_reported(bad_status):
    """status not in the four STS urgency values must produce a status error."""
    d = _valid()
    if bad_status is None:
        d.pop("status")
    else:
        d["status"] = bad_status
    errors = validate_sts_input(d)
    assert any("status" in e for e in errors), (
        f"Expected status error for {bad_status!r}, got: {errors}"
    )


@pytest.mark.parametrize("good_status", ["Elective", "Urgent", "Emergent", "Emergent Salvage"])
def test_valid_status_accepted(good_status):
    d = _valid()
    d["status"] = good_status
    errors = validate_sts_input(d)
    assert not any("status" in e for e in errors), (
        f"Unexpected status error for {good_status!r}: {errors}"
    )


# ---------------------------------------------------------------------------
# V-8: multiple errors returned at once
# ---------------------------------------------------------------------------

def test_multiple_errors_returned_simultaneously():
    """All validation errors must be collected and returned together."""
    d = {
        "procid": "99",      # bad
        "age": "not_a_number",  # bad
        "gender": "X",       # bad
        "status": "Unknown",  # bad
    }
    errors = validate_sts_input(d)
    assert len(errors) == 4, f"Expected 4 errors, got {len(errors)}: {errors}"
    assert any("age" in e for e in errors)
    assert any("procid" in e for e in errors)
    assert any("gender" in e for e in errors)
    assert any("status" in e for e in errors)


# ---------------------------------------------------------------------------
# V-9: Phase A integration — invalid payload logged as build_input failure
# ---------------------------------------------------------------------------

def test_phase_a_invalid_payload_logged_as_build_input_failure():
    """A row that produces an invalid STS payload must appear in failure_log
    with stage='build_input' and reason starting with 'payload_invalid:'."""
    from sts_calculator import calculate_sts_batch

    # Row with age=0 (out of range) to trigger validation failure.
    # The row has a recognisable surgery so it doesn't fail at the mapping stage.
    bad_row = {
        "surgery_pre": "CABG",
        "surgical_priority": "Elective",
        "age_years": "0",           # invalid — below 1
        "sex": "M",
        "patient_id": "TEST_BAD",
    }

    calculate_sts_batch([bad_row], patient_ids=["TEST_BAD"])

    failure_log = calculate_sts_batch.failure_log
    assert len(failure_log) == 1, (
        f"Expected 1 failure entry, got {len(failure_log)}: {failure_log}"
    )
    fl = failure_log[0]
    assert fl["stage"] == "build_input", f"Expected stage='build_input', got {fl['stage']!r}"
    assert fl["reason"].startswith("payload_invalid:"), (
        f"Expected reason to start with 'payload_invalid:', got: {fl['reason']!r}"
    )
    assert "age" in fl["reason"], f"Expected 'age' in reason, got: {fl['reason']!r}"


# ---------------------------------------------------------------------------
# V-10: Phase A integration — valid payload is NOT rejected pre-flight
# ---------------------------------------------------------------------------

def test_phase_a_valid_payload_not_rejected():
    """A row with a valid payload must not produce a build_input failure record.
    (It may still fail at fetch-time, but the pre-flight check must pass.)"""
    from sts_calculator import calculate_sts_batch

    good_row = {
        "surgery_pre": "CABG",
        "surgical_priority": "Elective",
        "age_years": "65",
        "sex": "M",
        "patient_id": "TEST_GOOD",
    }

    # We DON'T make a real network call — if there's no cache entry the batch
    # will fail at fetch stage, but the important thing is it does NOT fail at
    # the build_input stage.
    calculate_sts_batch(
        [good_row],
        patient_ids=["TEST_GOOD"],
    )

    build_input_failures = [
        fl for fl in calculate_sts_batch.failure_log
        if fl.get("stage") == "build_input"
    ]
    assert build_input_failures == [], (
        f"Valid row should not produce build_input failure: {build_input_failures}"
    )
