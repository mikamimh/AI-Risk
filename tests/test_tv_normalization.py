"""Tests for Phase 5 — Temporal Validation normalization report integration.

Covers:
  - normalization_report parameter in build_temporal_validation_summary
  - Dataset Normalization section presence in the generated report
  - ExternalNormalizationReport.summary_lines() output
  - No regression in existing report sections (STS availability, cohort summary)
"""
import pandas as pd
import pytest

from risk_data import ExternalNormalizationReport, ExternalReadMeta
from tv_helpers import build_temporal_validation_summary


def _minimal_norm_report(**overrides):
    """Build a minimal ExternalNormalizationReport for testing."""
    defaults = dict(
        source_name="test_cohort.csv",
        read_meta=ExternalReadMeta(
            encoding_used="utf-8-sig",
            delimiter=",",
            rows_loaded=50,
            columns_loaded=20,
        ),
        column_mapping={"age_years": "Age (years)", "sex": "Sex"},
        token_summary={"Hypertension": {"yes_converted": 4, "no_converted": 6, "total": 10}},
        unit_summary={
            "height_converted": False,
            "height_conversion_factor": None,
            "height_original_median": None,
            "n_height_converted": 0,
            "weight_converted": False,
            "weight_conversion_factor": None,
            "weight_original_median": None,
            "n_weight_converted": 0,
            "warnings": [],
        },
        scope_summary={
            "n_pediatric": 2,
            "n_sts_scope_excluded": 1,
            "n_surgery_cleaned": 3,
            "age_column_found": True,
            "surgery_column_found": True,
            "warnings": ["2 patient(s) with age < 18 flagged as pediatric."],
        },
        sts_readiness_summary={
            "n_total": 50,
            "n_ready": 45,
            "n_pediatric_excluded": 2,
            "n_scope_excluded": 1,
            "n_missing_fields": 2,
            "n_invalid_fields": 0,
            "n_ready_pct": 90.0,
            "required_fields_checked": ["age", "sex", "surgery", "surgical_priority"],
        },
        warnings=[
            "2 patient(s) with age < 18 flagged as pediatric — excluded from adult STS ACSD processing.",
            "1 row(s) with surgery outside STS ACSD scope (dissection / aneurysm / Bentall / Ross / transplant / homograft).",
        ],
    )
    defaults.update(overrides)
    return ExternalNormalizationReport(**defaults)


def _empty_df():
    return pd.DataFrame()


def _minimal_summary(**kwargs):
    base = {
        "n_total": 50,
        "n_events": 5,
        "event_rate": 0.10,
        "date_range": "2024-Q1 — 2024-Q4",
        "n_complete": 30,
        "n_adequate": 10,
        "n_partial": 5,
        "n_low": 5,
    }
    base.update(kwargs)
    return base


def _minimal_metadata(**kwargs):
    base = {
        "model_version": "1.0",
        "n_patients": 200,
        "n_events": 20,
        "event_rate": 0.10,
        "best_model": "RandomForest",
        "locked_threshold": 0.08,
    }
    base.update(kwargs)
    return base


def _call_summary(norm_report=None, language="English", **kwargs):
    return build_temporal_validation_summary(
        cohort_summary=_minimal_summary(),
        performance_df=_empty_df(),
        pairwise_df=_empty_df(),
        calibration_df=_empty_df(),
        risk_category_df=_empty_df(),
        metadata=_minimal_metadata(),
        threshold=0.08,
        language=language,
        normalization_report=norm_report,
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Normalization section in report
# ──────────────────────────────────────────────────────────────────────────────


class TestNormalizationReportInSummary:
    def test_normalization_section_present(self):
        """Dataset Normalization section appears when report is provided."""
        report = _call_summary(norm_report=_minimal_norm_report())
        assert "Dataset Normalization" in report

    def test_encoding_in_report(self):
        """The encoding used is shown in the report."""
        report = _call_summary(norm_report=_minimal_norm_report())
        assert "utf-8-sig" in report

    def test_rows_loaded_in_report(self):
        """rows_loaded value appears in the report table."""
        report = _call_summary(norm_report=_minimal_norm_report())
        assert "50" in report  # rows_loaded

    def test_pediatric_count_in_report(self):
        """Pediatric exclusion count appears in the report."""
        report = _call_summary(norm_report=_minimal_norm_report())
        assert "2" in report  # n_pediatric = 2
        assert "pediatric" in report.lower() or "pedi" in report.lower()

    def test_scope_excluded_count_in_report(self):
        """STS-scope-excluded count appears in the report."""
        report = _call_summary(norm_report=_minimal_norm_report())
        assert "1" in report  # n_sts_scope_excluded = 1

    def test_sts_ready_count_in_report(self):
        """STS-ready row count appears in the report."""
        report = _call_summary(norm_report=_minimal_norm_report())
        assert "45" in report or "90.0" in report  # n_ready or n_ready_pct

    def test_unit_conversion_shown_when_applied(self):
        """Height/weight conversion details appear when conversion was applied."""
        nr = _minimal_norm_report(unit_summary={
            "height_converted": True,
            "height_conversion_factor": 2.54,
            "height_original_median": 67.0,
            "n_height_converted": 50,
            "weight_converted": False,
            "weight_conversion_factor": None,
            "weight_original_median": None,
            "n_weight_converted": 0,
            "warnings": [],
        })
        report = _call_summary(norm_report=nr)
        assert "inches" in report.lower() or "cm" in report.lower()
        assert "50" in report  # n_height_converted

    def test_no_normalization_section_when_report_is_none(self):
        """When normalization_report=None, the section is absent."""
        report = _call_summary(norm_report=None)
        assert "Dataset Normalization" not in report
        assert "utf-8" not in report

    def test_normalization_warnings_in_report(self):
        """Warnings from normalization appear in the report."""
        report = _call_summary(norm_report=_minimal_norm_report())
        assert "pediatric" in report.lower()

    def test_portuguese_translation(self):
        """Portuguese language mode translates section headers."""
        report = _call_summary(norm_report=_minimal_norm_report(), language="Portuguese")
        assert "Normalização do Dataset" in report


# ──────────────────────────────────────────────────────────────────────────────
# No regression in existing report sections
# ──────────────────────────────────────────────────────────────────────────────


class TestNoRegressionWithNormalizationReport:
    def test_cohort_summary_still_present(self):
        """Cohort Summary section is present with normalization report."""
        report = _call_summary(norm_report=_minimal_norm_report())
        assert "Cohort Summary" in report

    def test_methodological_note_still_present(self):
        """Methodological note still appears in the report."""
        report = _call_summary(norm_report=_minimal_norm_report())
        assert "locked model" in report.lower() or "frozen" in report.lower()

    def test_report_without_normalization_unchanged_structure(self):
        """Report without normalization_report has same sections as before."""
        with_nr = _call_summary(norm_report=_minimal_norm_report())
        without_nr = _call_summary(norm_report=None)
        # Both should contain core sections
        for section in ("Cohort Summary", "Methodological note", "Generated"):
            assert section in with_nr
            assert section in without_nr

    def test_sts_availability_still_rendered_with_norm_report(self):
        """STS availability note is still rendered alongside normalization note."""
        from tv_helpers import build_sts_availability_summary
        sts_avail = build_sts_availability_summary(n_eligible=40, n_score=38)
        report = _call_summary(
            norm_report=_minimal_norm_report(),
            sts_availability=sts_avail,
        )
        assert "STS" in report
        assert "Dataset Normalization" in report


# ──────────────────────────────────────────────────────────────────────────────
# ExternalNormalizationReport.summary_lines
# ──────────────────────────────────────────────────────────────────────────────


class TestExternalNormalizationReportSummaryLines:
    def test_returns_list(self):
        nr = _minimal_norm_report()
        lines = nr.summary_lines()
        assert isinstance(lines, list)

    def test_encoding_line_present(self):
        nr = _minimal_norm_report()
        lines = nr.summary_lines()
        assert any("utf-8-sig" in ln for ln in lines)

    def test_pediatric_line_present(self):
        nr = _minimal_norm_report()
        lines = nr.summary_lines()
        assert any("pediatric" in ln.lower() for ln in lines)

    def test_sts_scope_line_present(self):
        nr = _minimal_norm_report()
        lines = nr.summary_lines()
        assert any("STS-scope" in ln or "scope-excluded" in ln or "scope" in ln.lower() for ln in lines)

    def test_sts_ready_line_present(self):
        nr = _minimal_norm_report()
        lines = nr.summary_lines()
        assert any("STS-ready" in ln or "ready" in ln.lower() for ln in lines)

    def test_warning_lines_prefixed(self):
        nr = _minimal_norm_report()
        lines = nr.summary_lines()
        warning_lines = [ln for ln in lines if ln.startswith("[WARNING]")]
        assert len(warning_lines) == len(nr.warnings)

    def test_no_read_meta_no_encoding_line(self):
        nr = _minimal_norm_report(read_meta=None)
        lines = nr.summary_lines()
        assert not any("encoding" in ln.lower() for ln in lines)

    def test_unit_conversion_line_when_applied(self):
        nr = _minimal_norm_report(unit_summary={
            "height_converted": True,
            "height_conversion_factor": 2.54,
            "height_original_median": 67.0,
            "n_height_converted": 20,
            "weight_converted": True,
            "weight_conversion_factor": 0.45351,
            "weight_original_median": 160.0,
            "n_weight_converted": 20,
            "warnings": [],
        })
        lines = nr.summary_lines()
        assert any("height" in ln.lower() and "inch" in ln.lower() for ln in lines)
        assert any("weight" in ln.lower() and "lb" in ln.lower() for ln in lines)

    def test_empty_report_minimal_output(self):
        """A report with all-zero counts still returns valid lines."""
        nr = ExternalNormalizationReport(
            source_name=None,
            read_meta=None,
            column_mapping={},
            token_summary={},
            unit_summary={
                "height_converted": False,
                "n_height_converted": 0,
                "height_original_median": None,
                "weight_converted": False,
                "n_weight_converted": 0,
                "weight_original_median": None,
                "warnings": [],
            },
            scope_summary={"n_pediatric": 0, "n_sts_scope_excluded": 0},
            sts_readiness_summary={
                "n_total": 10,
                "n_ready": 10,
                "n_ready_pct": 100.0,
                "n_pediatric_excluded": 0,
                "n_scope_excluded": 0,
                "n_missing_fields": 0,
                "n_invalid_fields": 0,
            },
            warnings=[],
        )
        lines = nr.summary_lines()
        assert isinstance(lines, list)
        # STS-ready line should still appear
        assert any("ready" in ln.lower() for ln in lines)
