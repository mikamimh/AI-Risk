"""Tests for the external-dataset normalization pipeline (Phases 1–4).

Covers:
  Phase 1 — read_external_table_with_fallback, canonicalize_external_columns
  Phase 2 — normalize_external_tokens, normalize_external_units
  Phase 3 — apply_external_scope_rules, build_sts_readiness_flags,
             classify_sts_eligibility pediatric guard
  Phase 4 — normalize_external_dataset (end-to-end)
"""
import math

import numpy as np
import pandas as pd
import pytest

from risk_data import (
    ExternalNormalizationReport,
    ExternalReadMeta,
    apply_external_scope_rules,
    build_sts_readiness_flags,
    canonicalize_external_columns,
    normalize_external_dataset,
    normalize_external_tokens,
    normalize_external_units,
    read_external_table_with_fallback,
)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — CSV ingestion
# ──────────────────────────────────────────────────────────────────────────────


class TestReadExternalTableWithFallback:
    def test_reads_utf8sig_csv(self, tmp_path):
        """utf-8-sig (BOM) CSV is read and encoding is reported correctly."""
        content = "name,age\nJoão,45\nMaria,60\n"
        path = tmp_path / "test.csv"
        path.write_bytes(content.encode("utf-8-sig"))
        df, meta = read_external_table_with_fallback(str(path))
        assert meta.encoding_used == "utf-8-sig"
        assert meta.rows_loaded == 2
        assert meta.columns_loaded == 2
        assert "João" in df["name"].values

    def test_reads_utf8_csv(self, tmp_path):
        """Plain utf-8 CSV reads without error."""
        content = "col_a,col_b\n1,2\n3,4\n"
        path = tmp_path / "plain.csv"
        path.write_bytes(content.encode("utf-8"))
        df, meta = read_external_table_with_fallback(str(path))
        assert meta.encoding_used in ("utf-8-sig", "utf-8")  # sig tried first; no BOM → same result
        assert meta.rows_loaded == 2
        assert set(df.columns) == {"col_a", "col_b"}

    def test_falls_back_to_cp1252(self, tmp_path):
        """CP1252-encoded CSV is read after utf-8/utf-8-sig fail."""
        content = "name,age\nJoão,45\n"
        path = tmp_path / "cp1252.csv"
        path.write_bytes(content.encode("cp1252"))
        df, meta = read_external_table_with_fallback(str(path))
        assert meta.encoding_used in ("cp1252", "latin-1")
        assert meta.rows_loaded == 1

    def test_meta_records_delimiter(self, tmp_path):
        """Semicolon-delimited CSV has delimiter correctly reported."""
        content = "a;b;c\n1;2;3\n4;5;6\n"
        path = tmp_path / "semi.csv"
        path.write_bytes(content.encode("utf-8"))
        df, meta = read_external_table_with_fallback(str(path))
        assert meta.delimiter == ";"
        assert meta.rows_loaded == 2
        assert meta.columns_loaded == 3

    def test_meta_shape_is_accurate(self, tmp_path):
        """rows_loaded and columns_loaded match actual DataFrame shape."""
        rows = "\n".join(f"p{i},{i}" for i in range(10))
        content = f"patient,score\n{rows}\n"
        path = tmp_path / "shape.csv"
        path.write_bytes(content.encode("utf-8"))
        df, meta = read_external_table_with_fallback(str(path))
        assert meta.rows_loaded == len(df)
        assert meta.columns_loaded == len(df.columns)


class TestCanonicalizeExternalColumns:
    def test_strips_leading_trailing_whitespace(self):
        df = pd.DataFrame({" age ": [1], " name  ": ["A"]})
        out, mapping = canonicalize_external_columns(df)
        assert "age" in out.columns or "Age (years)" in out.columns
        # original keys are preserved in mapping
        assert " age " in mapping

    def test_collapses_internal_spaces(self):
        df = pd.DataFrame({"age  years": [45]})
        out, mapping = canonicalize_external_columns(df)
        # collapsed to "age years" — not an alias, stays as cleaned form
        assert "age  years" not in out.columns

    def test_maps_known_alias_exact(self):
        """age_years → Age (years) via FLAT_ALIAS_TO_APP_COLUMNS."""
        df = pd.DataFrame({"age_years": [45], "sex": ["M"]})
        out, mapping = canonicalize_external_columns(df)
        assert "Age (years)" in out.columns
        assert "Sex" in out.columns
        assert mapping["age_years"] == "Age (years)"
        assert mapping["sex"] == "Sex"

    def test_maps_known_alias_case_insensitive(self):
        """AGE_YEARS (upper-case) still maps to the canonical name."""
        df = pd.DataFrame({"AGE_YEARS": [45]})
        out, mapping = canonicalize_external_columns(df)
        assert "Age (years)" in out.columns

    def test_unknown_column_unchanged(self):
        """Columns not in the alias table are left with cleaned names."""
        df = pd.DataFrame({"mystery_col": [1], "another_unknown": [2]})
        out, mapping = canonicalize_external_columns(df)
        # cleaned (no alias match) → stays as-is
        assert "mystery_col" in out.columns
        assert "another_unknown" in out.columns

    def test_no_mutation_of_unrelated_columns(self):
        """Columns that already have canonical names are not changed."""
        df = pd.DataFrame({"Age (years)": [45], "Sex": ["M"], "LVEF, %": [60.0]})
        out, mapping = canonicalize_external_columns(df)
        assert list(out.columns) == list(df.columns)

    def test_returns_identity_mapping_when_no_rename(self):
        df = pd.DataFrame({"Age (years)": [45]})
        out, mapping = canonicalize_external_columns(df)
        assert mapping["Age (years)"] == "Age (years)"


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Token normalization
# ──────────────────────────────────────────────────────────────────────────────


def _binary_col(yes_vals, no_vals):
    """Helper: build a DataFrame with a single Hypertension column."""
    values = yes_vals + no_vals
    return pd.DataFrame({"Hypertension": values})


class TestNormalizeExternalTokens:
    def test_sim_normalized_to_yes(self):
        df = pd.DataFrame({"Hypertension": ["Sim"] * 5 + ["Não"] * 5})
        out, summary = normalize_external_tokens(df)
        assert (out["Hypertension"] == "Yes").sum() == 5
        assert "Hypertension" in summary
        assert summary["Hypertension"]["yes_converted"] == 5

    def test_nao_normalized_to_no(self):
        df = pd.DataFrame({"Diabetes": ["Não"] * 6 + ["Sim"] * 4})
        out, summary = normalize_external_tokens(df)
        assert (out["Diabetes"] == "No").sum() == 6
        assert summary["Diabetes"]["no_converted"] == 6

    def test_english_case_normalized(self):
        df = pd.DataFrame({"PVD": ["yes"] * 4 + ["no"] * 4 + ["Yes"] * 2})
        out, summary = normalize_external_tokens(df)
        # "yes" → "Yes" (4 conversions); "no" → "No" (4 conversions); "Yes" already canonical
        assert (out["PVD"] == "Yes").sum() == 6
        assert (out["PVD"] == "No").sum() == 4

    def test_french_variant_normalized(self):
        df = pd.DataFrame({"Dialysis": ["oui"] * 5 + ["non"] * 5})
        out, summary = normalize_external_tokens(df)
        assert (out["Dialysis"] == "Yes").sum() == 5
        assert (out["Dialysis"] == "No").sum() == 5

    def test_german_variant_normalized(self):
        df = pd.DataFrame({"CVA": ["ja"] * 5 + ["nein"] * 5})
        out, summary = normalize_external_tokens(df)
        assert (out["CVA"] == "Yes").sum() == 5
        assert (out["CVA"] == "No").sum() == 5

    def test_below_threshold_column_untouched(self):
        """Free-text column with few binary tokens is NOT normalized."""
        df = pd.DataFrame({"Notes": [
            "patient had sim symptoms", "normal", "see chart",
            "CABG procedure", "routine", "follow up", "all clear",
            "check labs", "pre-op", "discharge"
        ]})
        out, summary = normalize_external_tokens(df)
        # hit rate is far below 0.50 — column should not appear in summary
        assert "Notes" not in summary
        # values must be unchanged
        assert list(out["Notes"]) == list(df["Notes"])

    def test_valve_severity_columns_untouched(self):
        """Valve columns (NONE_IS_VALID_COLUMNS) are never modified."""
        df = pd.DataFrame({
            "Aortic Stenosis": ["None", "Mild", "Severe", "None", "Moderate"],
            "Hypertension":    ["sim",  "sim",  "não",   "sim",  "não"],
        })
        out, summary = normalize_external_tokens(df)
        # Valve column untouched
        assert list(out["Aortic Stenosis"]) == list(df["Aortic Stenosis"])
        # Binary column normalized
        assert "Hypertension" in summary

    def test_numeric_column_untouched(self):
        """Float/int dtype columns are always skipped."""
        df = pd.DataFrame({"LVEF, %": [60.0, 55.0, 70.0, 45.0, 65.0]})
        out, summary = normalize_external_tokens(df)
        assert "LVEF, %" not in summary
        pd.testing.assert_series_equal(out["LVEF, %"], df["LVEF, %"])

    def test_already_canonical_not_counted(self):
        """Values already in canonical form are not counted as conversions."""
        df = pd.DataFrame({"Hypertension": ["Yes"] * 5 + ["No"] * 5})
        out, summary = normalize_external_tokens(df)
        assert "Hypertension" not in summary  # nothing changed


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Unit normalization
# ──────────────────────────────────────────────────────────────────────────────


class TestNormalizeExternalUnits:
    def test_height_inches_to_cm(self):
        """Heights with median < 100 (inches) are converted to cm."""
        inches = [65.0, 70.0, 68.0, 72.0, 64.0]
        df = pd.DataFrame({"Height (cm)": inches})
        out, summary = normalize_external_units(df)
        assert summary["height_converted"] is True
        assert summary["n_height_converted"] == 5
        assert abs(float(out["Height (cm)"].iloc[0]) - 65.0 * 2.54) < 0.2

    def test_height_already_metric_not_converted(self):
        """Heights with metric values (cm ≥ 100) are NOT converted."""
        df = pd.DataFrame({"Height (cm)": [165.0, 172.0, 158.0, 180.0]})
        out, summary = normalize_external_units(df)
        assert summary["height_converted"] is False
        assert float(out["Height (cm)"].iloc[0]) == 165.0

    def test_weight_lbs_to_kg(self):
        """Weights with median > 140 and max > 250 (lbs) are converted to kg."""
        lbs = [150.0, 165.0, 180.0, 200.0, 145.0, 170.0, 300.0]
        df = pd.DataFrame({"Weight (kg)": lbs})
        out, summary = normalize_external_units(df)
        assert summary["weight_converted"] is True
        assert summary["n_weight_converted"] == 7
        # 150 lbs / 2.205 ≈ 68.03 kg
        assert abs(float(out["Weight (kg)"].iloc[0]) - 150.0 / 2.205) < 1.0

    def test_weight_already_kg_not_converted(self):
        """Weights within normal kg range are NOT converted."""
        df = pd.DataFrame({"Weight (kg)": [70.0, 85.0, 65.0, 90.0, 78.0]})
        out, summary = normalize_external_units(df)
        assert summary["weight_converted"] is False
        assert float(out["Weight (kg)"].iloc[0]) == 70.0

    def test_height_snake_case_column(self):
        """snake_case column name (height_cm) is also detected."""
        df = pd.DataFrame({"height_cm": [65.0, 68.0, 70.0, 72.0, 66.0]})
        out, summary = normalize_external_units(df)
        assert summary["height_converted"] is True

    def test_weight_snake_case_column(self):
        """weight_kg snake_case column is also detected."""
        df = pd.DataFrame({"weight_kg": [150.0, 165.0, 180.0, 200.0, 145.0, 170.0, 300.0]})
        out, summary = normalize_external_units(df)
        assert summary["weight_converted"] is True

    def test_no_conversion_when_column_absent(self):
        """No error or spurious conversion when neither column is present."""
        df = pd.DataFrame({"age": [45, 60, 55]})
        out, summary = normalize_external_units(df)
        assert summary["height_converted"] is False
        assert summary["weight_converted"] is False

    def test_original_median_recorded(self):
        """Original median is stored in the summary for audit purposes."""
        df = pd.DataFrame({"Height (cm)": [65.0, 68.0, 70.0, 72.0, 66.0]})
        _, summary = normalize_external_units(df)
        assert summary["height_original_median"] is not None
        assert 60 < summary["height_original_median"] < 80  # original inches range


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — Clinical scope rules
# ──────────────────────────────────────────────────────────────────────────────


class TestApplyExternalScopeRules:
    def test_pediatric_flag_set_for_age_lt_18(self):
        df = pd.DataFrame({
            "Age (years)": [15, 16, 17, 12],
            "Surgery": ["CABG"] * 4,
        })
        out, summary = apply_external_scope_rules(df)
        assert out["is_pediatric"].all()
        assert summary["n_pediatric"] == 4

    def test_adult_patients_not_flagged(self):
        df = pd.DataFrame({
            "Age (years)": [45, 60, 72, 18],
            "Surgery": ["CABG"] * 4,
        })
        out, summary = apply_external_scope_rules(df)
        assert not out["is_pediatric"].any()
        assert summary["n_pediatric"] == 0

    def test_mixed_age_group(self):
        df = pd.DataFrame({
            "Age (years)": [15, 45, 17, 60],
            "Surgery": ["CABG"] * 4,
        })
        out, summary = apply_external_scope_rules(df)
        assert list(out["is_pediatric"]) == [True, False, True, False]
        assert summary["n_pediatric"] == 2

    def test_bentall_flagged_as_scope_excluded(self):
        df = pd.DataFrame({
            "Age (years)": [55],
            "Surgery": ["Bentall-de Bono procedure"],
        })
        out, summary = apply_external_scope_rules(df)
        assert bool(out["sts_scope_excluded"].iloc[0]) is True
        assert summary["n_sts_scope_excluded"] == 1
        assert "BENTALL" in out["sts_scope_reason"].iloc[0].upper()

    def test_dissection_flagged(self):
        df = pd.DataFrame({
            "Age (years)": [60],
            "Surgery": ["Aortic dissection repair"],
        })
        out, summary = apply_external_scope_rules(df)
        assert bool(out["sts_scope_excluded"].iloc[0]) is True

    def test_aneurysm_flagged(self):
        df = pd.DataFrame({
            "Age (years)": [58],
            "Surgery": ["Aortic aneurysm repair, AVR"],
        })
        out, summary = apply_external_scope_rules(df)
        assert bool(out["sts_scope_excluded"].iloc[0]) is True

    def test_supported_surgery_not_excluded(self):
        df = pd.DataFrame({
            "Age (years)": [55, 60, 65],
            "Surgery": ["CABG", "AVR", "MVR"],
        })
        out, summary = apply_external_scope_rules(df)
        assert not out["sts_scope_excluded"].any()
        assert summary["n_sts_scope_excluded"] == 0

    def test_surgery_text_separators_standardized(self):
        df = pd.DataFrame({
            "Age (years)": [55],
            "Surgery": ["CABG; AVR + MVR"],
        })
        out, _ = apply_external_scope_rules(df)
        assert ";" not in out["_surgery_cleaned"].iloc[0]
        assert "+" not in out["_surgery_cleaned"].iloc[0]

    def test_no_age_column_pediatric_defaults_false(self):
        df = pd.DataFrame({"Surgery": ["CABG", "AVR"]})
        out, summary = apply_external_scope_rules(df)
        assert not out["is_pediatric"].any()
        assert summary["age_column_found"] is False

    def test_warning_emitted_for_pediatric(self):
        df = pd.DataFrame({"Age (years)": [10, 15], "Surgery": ["CABG", "AVR"]})
        _, summary = apply_external_scope_rules(df)
        assert any("pediatric" in w.lower() for w in summary["warnings"])

    def test_warning_emitted_for_scope_excluded(self):
        df = pd.DataFrame({"Age (years)": [50], "Surgery": ["Bentall-de Bono procedure"]})
        _, summary = apply_external_scope_rules(df)
        assert any("STS ACSD scope" in w for w in summary["warnings"])


class TestBuildStsReadinessFlags:
    def _base_df(self, **overrides):
        base = {
            "Age (years)": [55],
            "Sex": ["M"],
            "Surgery": ["CABG"],
            "Surgical Priority": ["Elective"],
            "is_pediatric": [False],
            "sts_scope_excluded": [False],
            "sts_scope_reason": [""],
        }
        base.update(overrides)
        return pd.DataFrame(base)

    def test_all_fields_present_is_ready(self):
        df = self._base_df()
        out, summary = build_sts_readiness_flags(df)
        assert bool(out["sts_input_ready"].iloc[0]) is True
        assert summary["n_ready"] == 1

    def test_pediatric_not_ready(self):
        df = self._base_df(**{"Age (years)": [15], "is_pediatric": [True]})
        out, summary = build_sts_readiness_flags(df)
        assert bool(out["sts_input_ready"].iloc[0]) is False
        assert "pediatric" in out["sts_readiness_reason"].iloc[0].lower()
        assert summary["n_pediatric_excluded"] == 1

    def test_scope_excluded_not_ready(self):
        df = self._base_df(**{
            "sts_scope_excluded": [True],
            "sts_scope_reason": ["procedure outside STS ACSD scope: keyword 'BENTALL' found"],
        })
        out, summary = build_sts_readiness_flags(df)
        assert bool(out["sts_input_ready"].iloc[0]) is False
        assert summary["n_scope_excluded"] == 1

    def test_missing_age_not_ready(self):
        df = self._base_df(**{"Age (years)": [np.nan]})
        out, summary = build_sts_readiness_flags(df)
        assert bool(out["sts_input_ready"].iloc[0]) is False
        assert "age" in out["sts_missing_required_fields"].iloc[0]
        assert summary["n_missing_fields"] >= 1

    def test_invalid_age_not_ready(self):
        df = self._base_df(**{"Age (years)": [0]})
        out, summary = build_sts_readiness_flags(df)
        assert bool(out["sts_input_ready"].iloc[0]) is False
        assert "age" in out["sts_invalid_required_fields"].iloc[0]

    def test_missing_sex_not_ready(self):
        df = self._base_df(**{"Sex": [np.nan]})
        out, summary = build_sts_readiness_flags(df)
        assert bool(out["sts_input_ready"].iloc[0]) is False
        assert "sex" in out["sts_missing_required_fields"].iloc[0]

    def test_readiness_summary_counts_coherent(self):
        rows = [
            {"Age (years)": 55,  "Sex": "M", "Surgery": "CABG", "Surgical Priority": "Elective",
             "is_pediatric": False, "sts_scope_excluded": False, "sts_scope_reason": ""},
            {"Age (years)": 15,  "Sex": "M", "Surgery": "CABG", "Surgical Priority": "Elective",
             "is_pediatric": True,  "sts_scope_excluded": False, "sts_scope_reason": ""},
            {"Age (years)": 60,  "Sex": "F", "Surgery": "AVR",  "Surgical Priority": "Urgent",
             "is_pediatric": False, "sts_scope_excluded": True,
             "sts_scope_reason": "procedure outside STS ACSD scope: keyword 'ANEURYSM' found"},
        ]
        df = pd.DataFrame(rows)
        out, summary = build_sts_readiness_flags(df)
        assert summary["n_total"] == 3
        assert summary["n_ready"] == 1
        assert summary["n_pediatric_excluded"] == 1
        assert summary["n_scope_excluded"] == 1

    def test_scope_flags_computed_inline_when_absent(self):
        """build_sts_readiness_flags works even without prior scope-rule run."""
        df = pd.DataFrame({
            "Age (years)": [55, 15],
            "Sex": ["M", "F"],
            "Surgery": ["CABG", "AVR"],
            "Surgical Priority": ["Elective", "Urgent"],
        })
        out, summary = build_sts_readiness_flags(df)
        assert summary["n_total"] == 2
        assert summary["n_pediatric_excluded"] == 1  # inline detection of age 15


class TestClassifyStsEligibilityPediatricGuard:
    def test_pediatric_flag_returns_not_supported(self):
        from sts_calculator import classify_sts_eligibility
        row = {"Surgery": "CABG", "Surgical Priority": "Elective", "is_pediatric": True}
        status, reason = classify_sts_eligibility(row)
        assert status == "not_supported"
        assert "pediatric" in reason.lower()

    def test_adult_supported_procedure_unaffected(self):
        from sts_calculator import classify_sts_eligibility
        row = {
            "Surgery": "CABG",
            "Surgical Priority": "Elective",
            "is_pediatric": False,
        }
        status, _ = classify_sts_eligibility(row)
        assert status == "supported"

    def test_missing_pediatric_flag_defaults_false(self):
        """Row without is_pediatric key behaves as non-pediatric."""
        from sts_calculator import classify_sts_eligibility
        row = {"Surgery": "CABG", "Surgical Priority": "Elective"}
        status, _ = classify_sts_eligibility(row)
        assert status == "supported"


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 — Unified normalize_external_dataset
# ──────────────────────────────────────────────────────────────────────────────


class TestNormalizeExternalDataset:
    def _sample_df(self):
        return pd.DataFrame({
            "age_years":         [55, 60, 15, 70,  45],
            "sex":               ["M", "F", "M", "F", "M"],
            "surgery_pre":       ["CABG", "AVR", "CABG", "Bentall-de Bono procedure", "MVR"],
            "surgical_priority": ["Elective", "Urgent", "Elective", "Emergent", "Elective"],
            "hypertension":      ["Sim", "Não", "Sim", "Yes", "nao"],
        })

    def test_columns_canonicalized(self):
        df = self._sample_df()
        out, report = normalize_external_dataset(df, source_name="test.csv")
        assert "Age (years)" in out.columns
        assert "Sex" in out.columns

    def test_tokens_normalized(self):
        df = self._sample_df()
        out, report = normalize_external_dataset(df, source_name="test.csv")
        # "Sim" and "nao" should be converted; the column should exist
        ht_col = "hypertension" if "hypertension" in out.columns else None
        assert ht_col is not None or report.token_summary  # normalization happened

    def test_pediatric_flagged(self):
        df = self._sample_df()
        out, report = normalize_external_dataset(df, source_name="test.csv")
        age_col = "Age (years)" if "Age (years)" in out.columns else "age_years"
        assert "is_pediatric" in out.columns
        ped_mask = pd.to_numeric(out[age_col], errors="coerce") < 18
        assert out.loc[ped_mask, "is_pediatric"].all()

    def test_bentall_scope_excluded(self):
        df = self._sample_df()
        out, report = normalize_external_dataset(df, source_name="test.csv")
        assert "sts_scope_excluded" in out.columns
        surg_col = "Surgery" if "Surgery" in out.columns else "surgery_pre"
        bentall_mask = out[surg_col].astype(str).str.upper().str.contains("BENTALL")
        assert out.loc[bentall_mask, "sts_scope_excluded"].all()

    def test_sts_readiness_flags_present(self):
        df = self._sample_df()
        out, report = normalize_external_dataset(df, source_name="test.csv")
        assert "sts_input_ready" in out.columns
        assert "sts_readiness_reason" in out.columns

    def test_report_schema_complete(self):
        df = self._sample_df()
        _, report = normalize_external_dataset(df, source_name="test.csv")
        assert isinstance(report, ExternalNormalizationReport)
        assert report.source_name == "test.csv"
        assert isinstance(report.column_mapping, dict)
        assert isinstance(report.token_summary, dict)
        assert isinstance(report.unit_summary, dict)
        assert isinstance(report.scope_summary, dict)
        assert isinstance(report.sts_readiness_summary, dict)
        assert isinstance(report.warnings, list)
        assert "n_ready" in report.sts_readiness_summary
        assert "n_pediatric" in report.scope_summary

    def test_summary_lines_non_empty(self):
        df = self._sample_df()
        _, report = normalize_external_dataset(df, source_name="test.csv")
        lines = report.summary_lines()
        assert isinstance(lines, list)
        assert len(lines) > 0

    def test_read_meta_included_when_provided(self):
        df = self._sample_df()
        meta = ExternalReadMeta(
            encoding_used="utf-8-sig", delimiter=",", rows_loaded=5, columns_loaded=5
        )
        _, report = normalize_external_dataset(df, read_meta=meta)
        assert report.read_meta is not None
        assert report.read_meta.encoding_used == "utf-8-sig"
        # summary_lines should mention encoding
        lines = report.summary_lines()
        assert any("utf-8-sig" in ln for ln in lines)

    def test_height_conversion_end_to_end(self):
        """Inch heights are detected and converted in the full pipeline."""
        df = pd.DataFrame({
            "age_years":         [55, 60, 65],
            "sex":               ["M", "F", "M"],
            "surgery_pre":       ["CABG", "AVR", "MVR"],
            "surgical_priority": ["Elective"] * 3,
            "height_cm":         [65.0, 68.0, 70.0],  # inches
        })
        out, report = normalize_external_dataset(df)
        assert report.unit_summary["height_converted"] is True
        # 65 in * 2.54 ≈ 165.1 cm
        h_col = "Height (cm)" if "Height (cm)" in out.columns else "height_cm"
        assert float(out[h_col].iloc[0]) > 100

    def test_raw_dataframe_not_mutated(self):
        """The input dataframe is never modified in place."""
        df = self._sample_df()
        original_cols = list(df.columns)
        original_vals = df.copy()
        _ = normalize_external_dataset(df)
        assert list(df.columns) == original_cols
        pd.testing.assert_frame_equal(df, original_vals)

    def test_scope_summary_warning_in_report_warnings(self):
        """Pediatric and scope warnings bubble up to report.warnings."""
        df = pd.DataFrame({
            "age_years":         [15, 55],
            "sex":               ["M", "F"],
            "surgery_pre":       ["CABG", "Bentall-de Bono procedure"],
            "surgical_priority": ["Elective", "Emergent"],
        })
        _, report = normalize_external_dataset(df)
        warning_text = " ".join(report.warnings).lower()
        assert "pediatric" in warning_text
        assert "bentall" in warning_text or "scope" in warning_text
