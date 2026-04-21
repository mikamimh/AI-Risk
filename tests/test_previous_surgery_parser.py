"""Tests for parse_previous_surgery() and the derived audit columns."""
import pandas as pd
import pytest
from risk_data import parse_previous_surgery, _add_previous_surgery_audit_cols, MISSING_TOKENS


# ── parse_previous_surgery: absence / no-surgery cases ───────────────────────

@pytest.mark.parametrize("value", [None, float("nan"), "", "No", "no", "NO"])
def test_no_surgery(value):
    r = parse_previous_surgery(value)
    assert r["any"] is False
    assert r["count_est"] == 0
    assert r["has_combined"] is False
    assert r["has_repeat_marker"] is False
    assert r["has_year_marker"] is False


@pytest.mark.parametrize("token", ["-", "n/a", "unknown", "nan"])
def test_missing_tokens_treated_as_no_surgery(token):
    r = parse_previous_surgery(token)
    assert r["any"] is False


# ── parse_previous_surgery: simple redo cases ────────────────────────────────

def test_single_redo_avr():
    r = parse_previous_surgery("AVR")
    assert r["any"] is True
    assert r["count_est"] == 1
    assert r["has_combined"] is False
    assert r["has_repeat_marker"] is False
    assert r["has_year_marker"] is False


def test_single_redo_cabg():
    r = parse_previous_surgery("CABG")
    assert r["any"] is True
    assert r["count_est"] == 1


# ── parse_previous_surgery: semicolon-separated episodes ─────────────────────

def test_two_separate_episodes():
    r = parse_previous_surgery("AVR; CABG")
    assert r["any"] is True
    assert r["count_est"] == 2
    assert r["has_combined"] is False


def test_three_episodes():
    r = parse_previous_surgery("AVR; MVR; CABG")
    assert r["count_est"] == 3


# ── parse_previous_surgery: combined procedures (+) ──────────────────────────

def test_combined_plus():
    r = parse_previous_surgery("AVR + CABG")
    assert r["any"] is True
    assert r["count_est"] == 1  # one episode, two procedures
    assert r["has_combined"] is True


def test_combined_with_two_episodes():
    r = parse_previous_surgery("AVR + CABG; MVR")
    assert r["count_est"] == 2
    assert r["has_combined"] is True


# ── parse_previous_surgery: repeat marker (xN) ───────────────────────────────

def test_repeat_marker_x2():
    r = parse_previous_surgery("AVR (x2)")
    assert r["any"] is True
    assert r["count_est"] == 2
    assert r["has_repeat_marker"] is True


def test_repeat_marker_x3():
    r = parse_previous_surgery("CABG (x3)")
    assert r["count_est"] == 3
    assert r["has_repeat_marker"] is True


def test_repeat_marker_case_insensitive():
    r = parse_previous_surgery("MVR (X2)")
    assert r["has_repeat_marker"] is True
    assert r["count_est"] == 2


def test_repeat_marker_with_space():
    r = parse_previous_surgery("AVR (x 2)")
    assert r["has_repeat_marker"] is True
    assert r["count_est"] == 2


def test_mixed_episode_with_and_without_repeat():
    # "AVR (x2); CABG" → 2 from x2 + 1 from CABG = 3
    r = parse_previous_surgery("AVR (x2); CABG")
    assert r["count_est"] == 3
    assert r["has_repeat_marker"] is True


# ── parse_previous_surgery: year marker (YYYY) ───────────────────────────────

def test_year_marker():
    r = parse_previous_surgery("AVR (2018)")
    assert r["any"] is True
    assert r["has_year_marker"] is True
    assert r["count_est"] == 1  # (2018) does not inflate count


def test_year_marker_multiple():
    r = parse_previous_surgery("AVR (2015); CABG (2019)")
    assert r["has_year_marker"] is True
    assert r["count_est"] == 2


def test_year_marker_not_confused_with_repeat():
    r = parse_previous_surgery("AVR (2018)")
    assert r["has_repeat_marker"] is False


def test_repeat_marker_not_confused_with_year():
    r = parse_previous_surgery("AVR (x2)")
    assert r["has_year_marker"] is False


# ── parse_previous_surgery: combined year and repeat ─────────────────────────

def test_year_and_repeat_together():
    r = parse_previous_surgery("CABG (x2) (2012); MVR (2017)")
    assert r["has_repeat_marker"] is True
    assert r["has_year_marker"] is True
    assert r["count_est"] == 3  # x2 + 1


# ── _add_previous_surgery_audit_cols: DataFrame integration ──────────────────

def test_audit_cols_derived_correctly():
    df = pd.DataFrame({
        "Previous surgery": ["No", "AVR", "CABG (x2)", "MVR + TV Repair (2019)", None],
    })
    out = _add_previous_surgery_audit_cols(df)
    assert list(out["previous_surgery_any"]) == [False, True, True, True, False]
    assert list(out["previous_surgery_count_est"]) == [0, 1, 2, 1, 0]
    assert list(out["previous_surgery_has_combined"]) == [False, False, False, True, False]
    assert list(out["previous_surgery_has_repeat_marker"]) == [False, False, True, False, False]
    assert list(out["previous_surgery_has_year_marker"]) == [False, False, False, True, False]


def test_audit_cols_not_modifying_source_column():
    df = pd.DataFrame({"Previous surgery": ["AVR", "No"]})
    out = _add_previous_surgery_audit_cols(df)
    # Source column is unchanged
    assert list(out["Previous surgery"]) == ["AVR", "No"]


def test_audit_cols_skipped_if_no_column():
    df = pd.DataFrame({"Other": [1, 2, 3]})
    out = _add_previous_surgery_audit_cols(df)
    assert "previous_surgery_any" not in out.columns


def test_audit_cols_are_not_in_feature_columns():
    """Audit columns must never appear as model features (allowlist check)."""
    from risk_data import FLAT_PREOP_ALLOWED_COLUMNS, _PREV_SURG_AUDIT_COLS
    leaked = [c for c in _PREV_SURG_AUDIT_COLS if c in FLAT_PREOP_ALLOWED_COLUMNS]
    assert leaked == [], f"Audit cols leaked into allowlist: {leaked}"
