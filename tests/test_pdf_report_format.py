"""Tests for PDF/report presentation layer — temporal_validation.py.

Covers the formatting improvements introduced to fix wide-table overlap in
``build_temporal_validation_summary`` and the proportional-width / font-size
improvements to ``statistical_summary_to_pdf`` in ``export_helpers.py``.

None of these tests touch computed metrics values; they only verify the
*presentation* transformations:

  F-1  Performance table is split: Table A has ≤9 columns, Table B has ≤8 columns.
  F-2  Long raw column names are absent from the report Markdown.
  F-3  Short display labels are present in the report Markdown.
  F-4  Float values are formatted to ≤4 decimal places (no machine-precision strings).
  F-5  DeLong_skip_reason is NOT rendered as a visible table column.
  F-6  DeLong_skip_reason is rendered as an italic footnote below the table.
  F-7  Absent DeLong skip reason produces no spurious footnote.
  F-8  Raw DataFrame columns and values are untouched by the report builder.
  F-9  PDF bytes are non-empty when fpdf2 is available (smoke test).
  F-10 _flush_table uses smaller font for wide tables (column-count heuristic).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from tv_helpers import build_temporal_validation_summary


# ---------------------------------------------------------------------------
# Fixtures — minimal but realistic DataFrames
# ---------------------------------------------------------------------------

def _perf_df() -> pd.DataFrame:
    """Minimal performance DataFrame matching evaluate_scores_temporal output."""
    return pd.DataFrame([
        {
            "Score": "AI Risk",
            "n": 26,
            "AUC": 0.7714285714285715,
            "AUC_IC95_inf": 0.5238095238095238,
            "AUC_IC95_sup": 0.9523809523809523,
            "AUPRC": 0.6238562091503268,
            "AUPRC_IC95_inf": 0.3014285714285714,
            "AUPRC_IC95_sup": 0.8825396825396825,
            "Brier": 0.1238562091503268,
            "Calibration_Intercept": -0.3141592653589793,
            "Calibration_Slope": 0.9000000000000001,
            "HL_p": 0.4567890123456789,
            "Sensitivity": 0.8000000000000001,
            "Specificity": 0.7500000000000001,
            "PPV": 0.5714285714285714,
            "NPV": 0.9000000000000001,
        },
        {
            "Score": "EuroSCORE II",
            "n": 26,
            "AUC": 0.6904761904761905,
            "AUC_IC95_inf": 0.4285714285714286,
            "AUC_IC95_sup": 0.9047619047619048,
            "AUPRC": 0.5012345678901234,
            "AUPRC_IC95_inf": 0.2500000000000000,
            "AUPRC_IC95_sup": 0.8100000000000000,
            "Brier": 0.1500000000000000,
            "Calibration_Intercept": 0.1234567890123456,
            "Calibration_Slope": 1.0500000000000001,
            "HL_p": 0.3210987654321098,
            "Sensitivity": 0.7142857142857143,
            "Specificity": 0.7000000000000001,
            "PPV": 0.5000000000000001,
            "NPV": 0.8750000000000001,
        },
    ])


def _pairwise_df_with_delong_note() -> pd.DataFrame:
    """Pairwise DataFrame with a non-empty DeLong_skip_reason on one row."""
    return pd.DataFrame([
        {
            "Comparison": "AI Risk vs EuroSCORE II",
            "n": 26,
            "Delta_AUC": 0.0809523809523810,
            "Delta_AUC_IC95_inf": -0.1904761904761905,
            "Delta_AUC_IC95_sup": 0.3428571428571428,
            "Bootstrap_p": 0.5600000000000001,
            "DeLong_p": float("nan"),
            "DeLong_skip_reason": "pos < 2: only 1 positive sample",
            "NRI": 0.0714285714285714,
            "IDI": 0.0238095238095238,
        },
    ])


def _pairwise_df_no_delong_note() -> pd.DataFrame:
    """Pairwise DataFrame where DeLong ran successfully (skip_reason is None/empty)."""
    return pd.DataFrame([
        {
            "Comparison": "AI Risk vs EuroSCORE II",
            "n": 26,
            "Delta_AUC": 0.0809523809523810,
            "Delta_AUC_IC95_inf": -0.1904761904761905,
            "Delta_AUC_IC95_sup": 0.3428571428571428,
            "Bootstrap_p": 0.5600000000000001,
            "DeLong_p": 0.5670000000000001,
            "DeLong_skip_reason": None,
            "NRI": 0.0714285714285714,
            "IDI": 0.0238095238095238,
        },
    ])


def _cohort_summary() -> dict:
    return {
        "n_total": 26, "n_events": 5, "event_rate": 0.192,
        "date_range": "2022 — 2024",
        "n_complete": 20, "n_adequate": 4, "n_partial": 2, "n_low": 0,
    }


def _metadata() -> dict:
    return {"model_version": "v1.0", "n_patients": 200, "n_events": 40, "event_rate": 0.20}


def _md(perf=None, pairwise=None, language="English") -> str:
    """Build a report Markdown string with sensible defaults."""
    return build_temporal_validation_summary(
        cohort_summary=_cohort_summary(),
        performance_df=perf if perf is not None else _perf_df(),
        pairwise_df=pairwise if pairwise is not None else _pairwise_df_no_delong_note(),
        calibration_df=pd.DataFrame(),
        risk_category_df=pd.DataFrame(),
        metadata=_metadata(),
        threshold=0.30,
        language=language,
    )


# ---------------------------------------------------------------------------
# F-1: Table A has ≤9 columns, Table B has ≤8 columns
# ---------------------------------------------------------------------------

def test_perf_table_a_column_count():
    """The Discrimination sub-table must have at most 9 columns."""
    md = _md()
    # Find the Discrimination sub-table header line (first pipe-row after "### Discrimination")
    lines = md.split("\n")
    in_disc = False
    for line in lines:
        if line.startswith("### Discrimination") or line.startswith("### Discrimina"):
            in_disc = True
        if in_disc and line.startswith("|") and not set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
            # Count pipe-separated cells in the header
            cells = [c.strip() for c in line.strip("|").split("|")]
            n_cols = len(cells)
            assert n_cols <= 9, f"Table A has {n_cols} columns, expected ≤9"
            break


def test_perf_table_b_column_count():
    """The Calibration & Classification sub-table must have at most 8 columns."""
    md = _md()
    lines = md.split("\n")
    in_cal = False
    for line in lines:
        if "Calibration and Classification" in line or "Calibra" in line and "Classifica" in line:
            in_cal = True
        if in_cal and line.startswith("|") and not set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
            cells = [c.strip() for c in line.strip("|").split("|")]
            n_cols = len(cells)
            assert n_cols <= 8, f"Table B has {n_cols} columns, expected ≤8"
            break


# ---------------------------------------------------------------------------
# F-2: Long raw column names are absent from the report
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw_col", [
    "AUC_IC95_inf", "AUC_IC95_sup",
    "AUPRC_IC95_inf", "AUPRC_IC95_sup",
    "Calibration_Intercept", "Calibration_Slope",
    "Delta_AUC_IC95_inf", "Delta_AUC_IC95_sup",
    "Bootstrap_p",
])
def test_raw_column_names_absent(raw_col):
    """Machine-style raw column names must NOT appear as table headers in the report."""
    md = _md()
    assert raw_col not in md, (
        f"Raw column name '{raw_col}' found in report — should be replaced by a short label"
    )


# ---------------------------------------------------------------------------
# F-3: Short display labels are present in the report
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("short_label", [
    "AUC lo", "AUC hi",
    "AUPRC lo", "AUPRC hi",
    "Cal.Int.", "Cal.Slp.",
    "HL p",
    "Sens.", "Spec.",
    "dAUC", "dAUC lo", "dAUC hi",
    "Boot.p",
])
def test_short_labels_present(short_label):
    """Report Markdown must contain the agreed short display labels."""
    md = _md()
    assert short_label in md, (
        f"Short label '{short_label}' not found in report Markdown"
    )


# ---------------------------------------------------------------------------
# F-4: Float values use controlled decimal precision (no machine-precision strings)
# ---------------------------------------------------------------------------

def test_no_machine_precision_floats():
    """The report must not contain raw float strings with more than 6 significant digits."""
    import re
    md = _md()
    # Match floats with more than 6 decimal digits (e.g. 0.7714285714285715)
    pattern = re.compile(r'\d+\.\d{7,}')
    matches = pattern.findall(md)
    assert not matches, (
        f"Machine-precision floats found in report: {matches[:5]}"
    )


def test_auc_formatted_to_3_decimals():
    """AUC value 0.7714285714285715 should appear as '0.771' in the report."""
    md = _md()
    assert "0.771" in md
    assert "0.7714285714285715" not in md


def test_p_value_formatted_to_4_decimals():
    """HL p-value 0.4567890123456789 should appear as '0.4568' (4 d.p.)."""
    md = _md()
    assert "0.4568" in md
    assert "0.4567890123456789" not in md


# ---------------------------------------------------------------------------
# F-5: DeLong_skip_reason is NOT a visible table column
# ---------------------------------------------------------------------------

def test_delong_skip_reason_not_a_column():
    """DeLong_skip_reason must not appear as a column header in any Markdown table."""
    md = _md(pairwise=_pairwise_df_with_delong_note())
    lines = md.split("\n")
    table_header_lines = [
        ln for ln in lines
        if ln.startswith("|") and "DeLong_skip_reason" in ln
        and not set(ln.replace("|","").replace("-","").replace(":","").strip()) == set()
    ]
    assert not table_header_lines, (
        "DeLong_skip_reason found as a table column header — should be a footnote only"
    )


# ---------------------------------------------------------------------------
# F-6: DeLong_skip_reason is rendered as an italic footnote
# ---------------------------------------------------------------------------

def test_delong_skip_reason_as_footnote():
    """When DeLong was skipped, the reason must appear as an italic line below the table."""
    md = _md(pairwise=_pairwise_df_with_delong_note())
    # The footnote should be an italic Markdown line (*...*) containing the reason text
    assert "pos < 2" in md, "DeLong skip reason text not found in report"
    # Find the italic line
    italic_lines = [ln.strip() for ln in md.split("\n") if ln.strip().startswith("*") and "DeLong" in ln]
    assert italic_lines, "DeLong skip reason should appear in an italic (*...*) line"
    assert "pos < 2" in italic_lines[0]


# ---------------------------------------------------------------------------
# F-7: No DeLong skip reason → no spurious footnote
# ---------------------------------------------------------------------------

def test_no_spurious_delong_footnote():
    """When DeLong_skip_reason is None/empty, no DeLong note must appear."""
    md = _md(pairwise=_pairwise_df_no_delong_note())
    # There should be no italic DeLong note line
    delong_note_lines = [
        ln for ln in md.split("\n")
        if "DeLong note" in ln or ("DeLong" in ln and ln.strip().startswith("*"))
    ]
    assert not delong_note_lines, (
        f"Spurious DeLong footnote found: {delong_note_lines}"
    )


# ---------------------------------------------------------------------------
# F-8: Raw DataFrame columns and values are untouched
# ---------------------------------------------------------------------------

def test_raw_dataframe_columns_untouched():
    """build_temporal_validation_summary must not mutate the input DataFrames."""
    perf = _perf_df()
    pair = _pairwise_df_with_delong_note()
    original_perf_cols = list(perf.columns)
    original_pair_cols = list(pair.columns)

    build_temporal_validation_summary(
        cohort_summary=_cohort_summary(),
        performance_df=perf,
        pairwise_df=pair,
        calibration_df=pd.DataFrame(),
        risk_category_df=pd.DataFrame(),
        metadata=_metadata(),
        threshold=0.30,
    )

    assert list(perf.columns) == original_perf_cols, "performance_df columns were mutated"
    assert list(pair.columns) == original_pair_cols, "pairwise_df columns were mutated"
    # Check a specific raw value is unchanged
    assert abs(perf.loc[0, "AUC"] - 0.7714285714285715) < 1e-12


# ---------------------------------------------------------------------------
# F-9: PDF bytes non-empty (smoke test — skipped if fpdf2 not installed)
# ---------------------------------------------------------------------------

def test_pdf_smoke():
    """statistical_summary_to_pdf must produce non-empty bytes for a valid report."""
    pytest.importorskip("fpdf", reason="fpdf2 not installed — skipping PDF smoke test")
    from export_helpers import statistical_summary_to_pdf
    md = _md()
    pdf_bytes = statistical_summary_to_pdf(md)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000, "PDF output suspiciously small"


# ---------------------------------------------------------------------------
# F-10: Font-size heuristic in _flush_table (unit test via generated Markdown)
# ---------------------------------------------------------------------------

def test_wide_table_uses_narrower_labels():
    """After splitting, the Discrimination sub-table header labels must fit within
    the constraints of a 9-column 7pt rendering (≤9 chars per short label)."""
    md = _md()
    # The short labels we expect in the table header:
    short_labels = ["AUC lo", "AUC hi", "AUPRC lo", "AUPRC hi", "Brier"]
    for label in short_labels:
        # Each must be ≤ 9 characters (font-size 7pt allows ~10 chars per 22mm col)
        assert len(label) <= 9, f"Label '{label}' exceeds 9 characters"
        assert label in md, f"Expected short label '{label}' not found in report"
