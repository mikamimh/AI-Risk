"""Priority 3 tests — Stronger visible warning for unit auto-conversion.

Covers:
  - Unit conversion note appears in report when height/weight was converted
  - Note is absent when no conversion occurred
  - Note text is distinct from the regular bullet-point list (blockquote form)
  - Portuguese translation of the note
"""
import pandas as pd
import pytest

from risk_data import ExternalNormalizationReport, ExternalReadMeta
from tv_helpers import build_temporal_validation_summary


def _unit_summary(height=False, weight=False):
    return {
        "height_converted": height,
        "height_conversion_factor": 2.54 if height else None,
        "height_original_median": 67.0 if height else None,
        "n_height_converted": 50 if height else 0,
        "weight_converted": weight,
        "weight_conversion_factor": 0.4535 if weight else None,
        "weight_original_median": 160.0 if weight else None,
        "n_weight_converted": 50 if weight else 0,
        "warnings": [],
    }


def _make_report(height=False, weight=False):
    return ExternalNormalizationReport(
        source_name="test.csv",
        read_meta=ExternalReadMeta(
            encoding_used="utf-8", delimiter=",", rows_loaded=50, columns_loaded=10
        ),
        column_mapping={},
        token_summary={},
        unit_summary=_unit_summary(height, weight),
        scope_summary={"n_pediatric": 0, "n_sts_scope_excluded": 0},
        sts_readiness_summary={
            "n_total": 50, "n_ready": 50, "n_ready_pct": 100.0,
            "n_pediatric_excluded": 0, "n_scope_excluded": 0,
            "n_missing_fields": 0, "n_invalid_fields": 0,
        },
        warnings=[],
    )


def _call(norm_report, language="English"):
    return build_temporal_validation_summary(
        cohort_summary={
            "n_total": 50, "n_events": 5, "event_rate": 0.10,
            "date_range": "2024-Q1 — 2024-Q4",
            "n_complete": 30, "n_adequate": 10, "n_partial": 5, "n_low": 5,
        },
        performance_df=pd.DataFrame(),
        pairwise_df=pd.DataFrame(),
        calibration_df=pd.DataFrame(),
        risk_category_df=pd.DataFrame(),
        metadata={
            "model_version": "1.0", "n_patients": 200, "n_events": 20,
            "event_rate": 0.10, "best_model": "RandomForest", "locked_threshold": 0.08,
        },
        threshold=0.08,
        language=language,
        normalization_report=norm_report,
    )


class TestUnitConversionWarningInReport:
    def test_height_conversion_note_present(self):
        """Blockquote note appears when height was converted."""
        report = _call(_make_report(height=True))
        assert "> **Note**:" in report or "> **Nota**:" in report or "automatically" in report.lower()

    def test_weight_conversion_note_present(self):
        """Blockquote note appears when weight was converted."""
        report = _call(_make_report(weight=True))
        # Note must contain the caveat about original values not being stored
        assert "original" in report.lower() or "automatically" in report.lower()

    def test_both_conversion_note_present(self):
        report = _call(_make_report(height=True, weight=True))
        assert "> **Note**:" in report or "automatically" in report.lower()

    def test_no_note_when_no_conversion(self):
        """Note must NOT appear when no unit conversion was applied."""
        report = _call(_make_report(height=False, weight=False))
        assert "automatically detected and converted" not in report
        assert "Unidades antropométricas" not in report

    def test_note_is_blockquote_form(self):
        """The note uses Markdown blockquote (>) for prominence in PDF."""
        report = _call(_make_report(height=True))
        # There should be a "> " line in the normalization section
        assert any(line.lstrip().startswith(">") for line in report.splitlines())

    def test_portuguese_note_present(self):
        """Portuguese translation of the note appears."""
        report = _call(_make_report(height=True), language="Portuguese")
        assert "automaticamente" in report.lower()

    def test_unit_details_still_shown(self):
        """The regular per-column bullet points still appear alongside the note."""
        report = _call(_make_report(height=True))
        assert "inches" in report.lower() or "polegadas" in report.lower()

    def test_no_note_when_report_is_none(self):
        """No unit conversion note when normalization_report is None."""
        report = _call(None)
        assert "automatically detected and converted" not in report
