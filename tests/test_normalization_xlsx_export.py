"""Priority 2 tests — ExternalNormalizationReport.to_export_rows / XLSX sheet.

Covers:
  - to_export_rows() returns a list of {"Field", "Value"} dicts
  - Expected structured fields are present
  - Warnings and summary_lines appear as individual rows
  - No read_meta → no encoding/delimiter/rows rows
  - ExcelWriter produces a Normalization_Summary sheet with expected content
  - Main cohort sheet is not mutated when normalization sheet is added
"""
from io import BytesIO

import pandas as pd
import pytest

from risk_data import ExternalNormalizationReport, ExternalReadMeta


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_report(**overrides):
    defaults = dict(
        source_name="export_test.csv",
        read_meta=ExternalReadMeta(
            encoding_used="utf-8",
            delimiter=",",
            rows_loaded=100,
            columns_loaded=15,
        ),
        column_mapping={"age_years": "Age (years)"},
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
        scope_summary={"n_pediatric": 3, "n_sts_scope_excluded": 2},
        sts_readiness_summary={
            "n_total": 100,
            "n_ready": 95,
            "n_ready_pct": 95.0,
            "n_pediatric_excluded": 3,
            "n_scope_excluded": 2,
            "n_missing_fields": 0,
            "n_invalid_fields": 0,
        },
        warnings=["3 patient(s) with age < 18 flagged as pediatric."],
    )
    defaults.update(overrides)
    return ExternalNormalizationReport(**defaults)


# ──────────────────────────────────────────────────────────────────────────────
# to_export_rows structure
# ──────────────────────────────────────────────────────────────────────────────

class TestToExportRows:
    def test_returns_list_of_dicts(self):
        rows = _make_report().to_export_rows()
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_field_and_value_keys_present(self):
        rows = _make_report().to_export_rows()
        for row in rows:
            assert "Field" in row
            assert "Value" in row

    def test_encoding_row_present_with_read_meta(self):
        rows = _make_report().to_export_rows()
        fields = {r["Field"]: r["Value"] for r in rows}
        assert "encoding_used" in fields
        assert fields["encoding_used"] == "utf-8"

    def test_rows_loaded_present(self):
        rows = _make_report().to_export_rows()
        fields = {r["Field"]: r["Value"] for r in rows}
        assert "rows_loaded" in fields
        assert fields["rows_loaded"] == 100

    def test_n_pediatric_row(self):
        rows = _make_report().to_export_rows()
        fields = {r["Field"]: r["Value"] for r in rows}
        assert "n_pediatric" in fields
        assert fields["n_pediatric"] == 3

    def test_n_sts_scope_excluded_row(self):
        rows = _make_report().to_export_rows()
        fields = {r["Field"]: r["Value"] for r in rows}
        assert "n_sts_scope_excluded" in fields
        assert fields["n_sts_scope_excluded"] == 2

    def test_n_sts_ready_row(self):
        rows = _make_report().to_export_rows()
        fields = {r["Field"]: r["Value"] for r in rows}
        assert "n_sts_ready" in fields
        assert fields["n_sts_ready"] == 95

    def test_warning_rows_present(self):
        rows = _make_report().to_export_rows()
        warning_rows = [r for r in rows if r["Field"].startswith("warning_")]
        assert len(warning_rows) == 1
        assert "pediatric" in warning_rows[0]["Value"].lower()

    def test_summary_line_rows_present(self):
        rows = _make_report().to_export_rows()
        summary_rows = [r for r in rows if r["Field"].startswith("summary_line_")]
        assert len(summary_rows) >= 1

    def test_no_encoding_row_without_read_meta(self):
        report = _make_report(read_meta=None)
        rows = report.to_export_rows()
        fields = [r["Field"] for r in rows]
        assert "encoding_used" not in fields
        assert "rows_loaded" not in fields

    def test_height_converted_row(self):
        report = _make_report(unit_summary={
            "height_converted": True,
            "n_height_converted": 50,
            "height_original_median": 67.0,
            "weight_converted": False,
            "n_weight_converted": 0,
            "weight_original_median": None,
            "warnings": [],
        })
        rows = report.to_export_rows()
        fields = {r["Field"]: r["Value"] for r in rows}
        assert fields["height_converted"] is True

    def test_multiple_warnings_numbered(self):
        report = _make_report(warnings=["Warning A", "Warning B", "Warning C"])
        rows = report.to_export_rows()
        warning_rows = {r["Field"]: r["Value"] for r in rows if r["Field"].startswith("warning_")}
        assert "warning_1" in warning_rows
        assert "warning_2" in warning_rows
        assert "warning_3" in warning_rows


# ──────────────────────────────────────────────────────────────────────────────
# XLSX round-trip
# ──────────────────────────────────────────────────────────────────────────────

class TestXlsxNormalizationSheet:
    def _write_xlsx(self, norm_report=None):
        buf = BytesIO()
        cohort_df = pd.DataFrame([{"Property": "n_total", "Value": 100}])
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            cohort_df.to_excel(writer, sheet_name="cohort_summary", index=False)
            if norm_report is not None:
                pd.DataFrame(norm_report.to_export_rows()).to_excel(
                    writer, sheet_name="Normalization_Summary", index=False
                )
        buf.seek(0)
        return buf

    def test_normalization_sheet_present_when_report_given(self):
        buf = self._write_xlsx(_make_report())
        xls = pd.ExcelFile(buf)
        assert "Normalization_Summary" in xls.sheet_names

    def test_normalization_sheet_absent_when_no_report(self):
        buf = self._write_xlsx(None)
        xls = pd.ExcelFile(buf)
        assert "Normalization_Summary" not in xls.sheet_names

    def test_normalization_sheet_has_field_value_columns(self):
        buf = self._write_xlsx(_make_report())
        df = pd.read_excel(buf, sheet_name="Normalization_Summary")
        assert "Field" in df.columns
        assert "Value" in df.columns

    def test_encoding_in_normalization_sheet(self):
        buf = self._write_xlsx(_make_report())
        df = pd.read_excel(buf, sheet_name="Normalization_Summary")
        assert any(df["Field"] == "encoding_used")
        row = df[df["Field"] == "encoding_used"]
        assert row["Value"].iloc[0] == "utf-8"

    def test_main_cohort_sheet_not_mutated(self):
        """cohort_summary sheet must be unchanged after adding normalization sheet."""
        buf = self._write_xlsx(_make_report())
        cohort_df = pd.read_excel(buf, sheet_name="cohort_summary")
        assert list(cohort_df.columns) == ["Property", "Value"]
        assert cohort_df.iloc[0]["Property"] == "n_total"

    def test_normalization_sheet_contains_warnings(self):
        buf = self._write_xlsx(_make_report())
        df = pd.read_excel(buf, sheet_name="Normalization_Summary")
        warning_rows = df[df["Field"].str.startswith("warning_")]
        assert len(warning_rows) >= 1

    def test_normalization_sheet_rows_loaded(self):
        buf = self._write_xlsx(_make_report())
        df = pd.read_excel(buf, sheet_name="Normalization_Summary")
        row = df[df["Field"] == "rows_loaded"]
        assert not row.empty
        assert int(row["Value"].iloc[0]) == 100
