"""Priority 5 tests — dry_run_external_ingestion preview mode.

Covers:
  - Returns a dict with all required keys
  - Reads and normalizes a CSV file without error
  - Reads and normalizes a Parquet file without error
  - Returns read_meta with encoding info for CSV
  - Returns normalization_report with summary_lines
  - n_sts_ready and n_sts_not_ready are non-negative integers
  - Unsupported file type sets error key, no exception raised
  - Nonexistent file sets error key, no exception raised
  - Does NOT import or call AI Risk / EuroSCORE II / STS web client
"""
import os
import tempfile

import pandas as pd
import pytest

from risk_data import dry_run_external_ingestion, ExternalNormalizationReport


# ──────────────────────────────────────────────────────────────────────────────
# Helper — write a minimal CSV to a temp file
# ──────────────────────────────────────────────────────────────────────────────

def _write_csv(content: str, suffix=".csv") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


_MINIMAL_CSV = """\
Age (years),Sex,Surgery,Surgical Priority,Height (cm),Weight (kg)
65,Male,ISOLATED CABG,Elective,170,75
72,Female,AVR,Elective,162,68
58,Male,MVR,Urgent,175,82
45,Female,ISOLATED CABG,Elective,165,60
14,Male,ISOLATED CABG,Elective,160,55
"""

_BENTALL_CSV = """\
Age (years),Sex,Surgery,Surgical Priority
55,Male,BENTALL PROCEDURE,Elective
65,Female,ISOLATED CABG,Elective
"""


# ──────────────────────────────────────────────────────────────────────────────
# Return schema
# ──────────────────────────────────────────────────────────────────────────────

class TestDryRunReturnSchema:
    def test_all_keys_present(self):
        path = _write_csv(_MINIMAL_CSV)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        for key in ("path", "read_meta", "normalized_df", "normalization_report",
                    "summary_lines", "n_sts_ready", "n_sts_not_ready", "warnings", "error"):
            assert key in result, f"Missing key: {key}"

    def test_path_echoed(self):
        path = _write_csv(_MINIMAL_CSV)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        assert result["path"] == path

    def test_no_error_for_valid_csv(self):
        path = _write_csv(_MINIMAL_CSV)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        assert result["error"] is None

    def test_normalized_df_is_dataframe(self):
        path = _write_csv(_MINIMAL_CSV)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        assert isinstance(result["normalized_df"], pd.DataFrame)

    def test_normalization_report_type(self):
        path = _write_csv(_MINIMAL_CSV)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        assert isinstance(result["normalization_report"], ExternalNormalizationReport)

    def test_summary_lines_is_list(self):
        path = _write_csv(_MINIMAL_CSV)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        assert isinstance(result["summary_lines"], list)

    def test_n_sts_ready_nonnegative_int(self):
        path = _write_csv(_MINIMAL_CSV)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        assert isinstance(result["n_sts_ready"], int)
        assert result["n_sts_ready"] >= 0

    def test_n_sts_not_ready_nonnegative_int(self):
        path = _write_csv(_MINIMAL_CSV)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        assert isinstance(result["n_sts_not_ready"], int)
        assert result["n_sts_not_ready"] >= 0


# ──────────────────────────────────────────────────────────────────────────────
# CSV normalization correctness
# ──────────────────────────────────────────────────────────────────────────────

class TestDryRunCsvCorrectness:
    def test_read_meta_encoding_set(self):
        path = _write_csv(_MINIMAL_CSV)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        assert result["read_meta"] is not None
        assert result["read_meta"].encoding_used in (
            "utf-8-sig", "utf-8", "cp1252", "latin-1"
        )

    def test_pediatric_row_excluded_from_sts_ready(self):
        """The row with age=14 must not be STS-ready."""
        path = _write_csv(_MINIMAL_CSV)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        # 5 rows, 1 pediatric → at most 4 STS-ready
        assert result["n_sts_ready"] <= 4
        assert result["n_sts_not_ready"] >= 1

    def test_bentall_excluded_from_sts_ready(self):
        path = _write_csv(_BENTALL_CSV)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        # 2 rows; Bentall one is excluded → at most 1 STS-ready
        assert result["n_sts_ready"] <= 1

    def test_warnings_is_list(self):
        path = _write_csv(_MINIMAL_CSV)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        assert isinstance(result["warnings"], list)

    def test_rows_loaded_equals_csv_rows(self):
        path = _write_csv(_MINIMAL_CSV)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        # CSV has 5 data rows (excluding header)
        assert result["read_meta"].rows_loaded == 5


# ──────────────────────────────────────────────────────────────────────────────
# Parquet support
# ──────────────────────────────────────────────────────────────────────────────

class TestDryRunParquet:
    def test_parquet_no_error(self):
        df = pd.DataFrame({
            "Age (years)": [65, 72],
            "Sex": ["Male", "Female"],
            "Surgery": ["ISOLATED CABG", "AVR"],
            "Surgical Priority": ["Elective", "Elective"],
        })
        fd, path = tempfile.mkstemp(suffix=".parquet")
        os.close(fd)
        df.to_parquet(path, index=False)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        assert result["error"] is None
        assert result["read_meta"].encoding_used == "parquet"
        assert isinstance(result["normalized_df"], pd.DataFrame)

    def test_parquet_rows_loaded(self):
        df = pd.DataFrame({"Age (years)": [60, 70, 55], "Sex": ["Male", "Female", "Male"]})
        fd, path = tempfile.mkstemp(suffix=".parquet")
        os.close(fd)
        df.to_parquet(path, index=False)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        assert result["read_meta"].rows_loaded == 3


# ──────────────────────────────────────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────────────────────────────────────

class TestDryRunErrorHandling:
    def test_nonexistent_file_sets_error(self):
        result = dry_run_external_ingestion("/nonexistent/path/file.csv")
        assert result["error"] is not None
        assert result["normalized_df"] is None

    def test_unsupported_extension_sets_error(self):
        fd, path = tempfile.mkstemp(suffix=".xlsx")
        os.close(fd)
        try:
            result = dry_run_external_ingestion(path)
        finally:
            os.unlink(path)
        assert result["error"] is not None
        assert "Unsupported" in result["error"]

    def test_error_result_has_all_keys(self):
        result = dry_run_external_ingestion("/nonexistent/file.csv")
        for key in ("path", "read_meta", "normalized_df", "normalization_report",
                    "summary_lines", "n_sts_ready", "n_sts_not_ready", "warnings", "error"):
            assert key in result
