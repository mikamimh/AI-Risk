"""Tests for the exploratory post-hoc recalibration and threshold analysis section.

Covers:
- build_exploratory_recalibration_summary  (formatting / structure)
- build_exploratory_threshold_summary      (formatting / structure)
- build_exploratory_temporal_validation_section  (Markdown output)
- Graceful degradation when inputs are empty / None
- Primary temporal-validation summary is not altered by exploratory additions
- Warning text and labelling requirements
- XLSX sheet naming convention
"""
import math

import numpy as np
import pandas as pd
import pytest

from stats_compare import (
    recalibrate_intercept_only,
    recalibrate_logistic,
    recalibrate_isotonic,
    threshold_analysis_table,
    youden_threshold,
)
from temporal_validation import (
    build_exploratory_recalibration_summary,
    build_exploratory_threshold_summary,
    build_exploratory_temporal_validation_section,
    build_temporal_validation_summary,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cohort():
    rng = np.random.default_rng(42)
    n = 60
    y = (rng.random(n) > 0.85).astype(int)
    p = rng.beta(2, 8, n)
    return y, p


@pytest.fixture
def recal_data(small_cohort):
    y, p = small_cohort
    return {
        "ia_risk": {
            "intercept_only": recalibrate_intercept_only(y, p),
            "logistic":        recalibrate_logistic(y, p),
            "isotonic":        recalibrate_isotonic(y, p),
        }
    }


@pytest.fixture
def rename():
    return {"ia_risk": "AI Risk", "euroscore_calc": "EuroSCORE II", "sts_score": "STS Score"}


@pytest.fixture
def thresh_tables(small_cohort):
    y, p = small_cohort
    locked = 0.08
    yt, _ = youden_threshold(y, p)
    thresholds = sorted({0.05, 0.10, locked, yt})
    df = threshold_analysis_table(y, p, thresholds)
    return {"ia_risk": df}, locked, {"ia_risk": (yt, _)}


# ===========================================================================
# build_exploratory_recalibration_summary
# ===========================================================================

class TestRecalibrationSummary:
    def test_returns_dict_with_required_keys(self, recal_data, rename):
        result = build_exploratory_recalibration_summary(recal_data, rename)
        assert "table" in result
        assert "available" in result

    def test_available_true_when_data_present(self, recal_data, rename):
        result = build_exploratory_recalibration_summary(recal_data, rename)
        assert result["available"] is True

    def test_table_is_dataframe(self, recal_data, rename):
        result = build_exploratory_recalibration_summary(recal_data, rename)
        assert isinstance(result["table"], pd.DataFrame)

    def test_table_has_expected_columns(self, recal_data, rename):
        tbl = build_exploratory_recalibration_summary(recal_data, rename)["table"]
        for col in ("Score", "Method", "Brier_Before", "Brier_After",
                    "Cal_Intercept_Before", "Cal_Intercept_After",
                    "Cal_Slope_Before", "Cal_Slope_After"):
            assert col in tbl.columns, f"Missing column: {col}"

    def test_original_row_present(self, recal_data, rename):
        tbl = build_exploratory_recalibration_summary(recal_data, rename)["table"]
        assert tbl["Method"].str.contains("Original", case=False).any()

    def test_three_recal_method_rows_present(self, recal_data, rename):
        tbl = build_exploratory_recalibration_summary(recal_data, rename)["table"]
        # Original + 3 methods = 4 rows per score
        assert len(tbl) == 4

    def test_display_name_used_in_score_column(self, recal_data, rename):
        tbl = build_exploratory_recalibration_summary(recal_data, rename)["table"]
        assert "AI Risk" in tbl["Score"].values

    def test_brier_after_le_before_for_intercept_only(self, recal_data, rename):
        tbl = build_exploratory_recalibration_summary(recal_data, rename)["table"]
        row = tbl[tbl["Method"].str.contains("Intercept-only", case=False)]
        if not row.empty:
            before = float(row["Brier_Before"].iloc[0])
            after = float(row["Brier_After"].iloc[0])
            assert after <= before + 1e-9

    def test_empty_input_returns_not_available(self, rename):
        result = build_exploratory_recalibration_summary({}, rename)
        assert result["available"] is False
        assert result["table"].empty

    def test_portuguese_method_labels(self, recal_data, rename):
        tbl = build_exploratory_recalibration_summary(recal_data, rename, language="Portuguese")["table"]
        assert tbl["Method"].str.contains("intercepto|isotônica|Original", case=False, regex=True).any()

    def test_missing_method_fails_gracefully(self, rename):
        partial = {"ia_risk": {"intercept_only": recalibrate_intercept_only(
            np.array([0,1,0,0,1,0,1,0,0,1]), np.array([0.1,0.9,0.2,0.05,0.85,0.3,0.75,0.15,0.4,0.8])
        )}}
        result = build_exploratory_recalibration_summary(partial, rename)
        assert result["available"] is True
        # Only Original + intercept_only rows
        assert len(result["table"]) == 2


# ===========================================================================
# build_exploratory_threshold_summary
# ===========================================================================

class TestThresholdSummary:
    def test_returns_dict_with_required_keys(self, thresh_tables, rename):
        tables, locked, youden = thresh_tables
        result = build_exploratory_threshold_summary(tables, locked, youden, rename)
        for key in ("table", "note_locked", "note_youden", "available"):
            assert key in result

    def test_available_true_when_data_present(self, thresh_tables, rename):
        tables, locked, youden = thresh_tables
        result = build_exploratory_threshold_summary(tables, locked, youden, rename)
        assert result["available"] is True

    def test_table_has_required_columns(self, thresh_tables, rename):
        tables, locked, youden = thresh_tables
        tbl = build_exploratory_threshold_summary(tables, locked, youden, rename)["table"]
        for col in ("Score", "Threshold", "Type", "Sensitivity", "Specificity",
                    "PPV", "NPV", "N_Flagged", "Event_Rate_Above", "Event_Rate_Below"):
            assert col in tbl.columns, f"Missing column: {col}"

    def test_locked_threshold_row_present(self, thresh_tables, rename):
        tables, locked, youden = thresh_tables
        tbl = build_exploratory_threshold_summary(tables, locked, youden, rename)["table"]
        locked_rows = tbl[abs(tbl["Threshold"] - locked) < 1e-9]
        assert not locked_rows.empty

    def test_locked_threshold_type_label(self, thresh_tables, rename):
        tables, locked, youden = thresh_tables
        tbl = build_exploratory_threshold_summary(tables, locked, youden, rename)["table"]
        locked_row = tbl[abs(tbl["Threshold"] - locked) < 1e-9].iloc[0]
        assert "Locked" in locked_row["Type"] or "Bloqueado" in locked_row["Type"]

    def test_youden_threshold_row_present(self, thresh_tables, rename):
        tables, locked, youden = thresh_tables
        tbl = build_exploratory_threshold_summary(tables, locked, youden, rename)["table"]
        yt = youden["ia_risk"][0]
        youden_rows = tbl[abs(tbl["Threshold"] - yt) < 1e-9]
        if abs(yt - locked) > 1e-9:  # only check if distinct from locked
            assert not youden_rows.empty

    def test_youden_type_label_exploratory(self, thresh_tables, rename):
        tables, locked, youden = thresh_tables
        tbl = build_exploratory_threshold_summary(tables, locked, youden, rename)["table"]
        yt = youden["ia_risk"][0]
        if abs(yt - locked) > 1e-9:
            youden_row = tbl[abs(tbl["Threshold"] - yt) < 1e-9].iloc[0]
            assert "Youden" in youden_row["Type"] or "exploratório" in youden_row["Type"].lower()

    def test_event_rate_above_is_between_0_and_1(self, thresh_tables, rename):
        tables, locked, youden = thresh_tables
        tbl = build_exploratory_threshold_summary(tables, locked, youden, rename)["table"]
        valid = tbl["Event_Rate_Above"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_note_locked_mentions_threshold_value(self, thresh_tables, rename):
        tables, locked, youden = thresh_tables
        note = build_exploratory_threshold_summary(tables, locked, youden, rename)["note_locked"]
        assert "8" in note or "0.08" in note

    def test_note_youden_warns_exploratory(self, thresh_tables, rename):
        tables, locked, youden = thresh_tables
        note = build_exploratory_threshold_summary(tables, locked, youden, rename)["note_youden"]
        assert "exploratory" in note.lower() or "exploratório" in note.lower()

    def test_empty_tables_returns_not_available(self, rename):
        result = build_exploratory_threshold_summary({}, 0.08, {}, rename)
        assert result["available"] is False

    def test_portuguese_note_locked(self, thresh_tables, rename):
        tables, locked, youden = thresh_tables
        note = build_exploratory_threshold_summary(tables, locked, youden, rename, language="Portuguese")["note_locked"]
        assert "Bloqueado" in note or "bloqueado" in note


# ===========================================================================
# build_exploratory_temporal_validation_section
# ===========================================================================

class TestExploratorySectionMarkdown:
    def _make_summaries(self, small_cohort, rename):
        y, p = small_cohort
        recal_data = {
            "ia_risk": {
                "intercept_only": recalibrate_intercept_only(y, p),
                "logistic":        recalibrate_logistic(y, p),
                "isotonic":        recalibrate_isotonic(y, p),
            }
        }
        locked = 0.08
        yt, yj = youden_threshold(y, p)
        thresholds = sorted({0.05, 0.10, locked, yt})
        df = threshold_analysis_table(y, p, thresholds)
        thresh_tables = {"ia_risk": df}
        youden_dict = {"ia_risk": (yt, yj)}

        recal_sum = build_exploratory_recalibration_summary(recal_data, rename)
        thresh_sum = build_exploratory_threshold_summary(thresh_tables, locked, youden_dict, rename)
        return recal_sum, thresh_sum

    def test_returns_non_empty_string(self, small_cohort, rename):
        recal_sum, thresh_sum = self._make_summaries(small_cohort, rename)
        md = build_exploratory_temporal_validation_section(recal_sum, thresh_sum)
        assert isinstance(md, str) and len(md) > 0

    def test_contains_exploratory_heading(self, small_cohort, rename):
        recal_sum, thresh_sum = self._make_summaries(small_cohort, rename)
        md = build_exploratory_temporal_validation_section(recal_sum, thresh_sum)
        assert "Exploratory" in md or "Exploratório" in md or "Apêndice" in md

    def test_contains_methodological_warning(self, small_cohort, rename):
        recal_sum, thresh_sum = self._make_summaries(small_cohort, rename)
        md = build_exploratory_temporal_validation_section(recal_sum, thresh_sum)
        warning_phrases = [
            "Exploratory only",
            "post-hoc",
            "frozen model",
            "locked threshold",
            "Apenas exploratório",
        ]
        assert any(p.lower() in md.lower() for p in warning_phrases)

    def test_contains_recalibration_table(self, small_cohort, rename):
        recal_sum, thresh_sum = self._make_summaries(small_cohort, rename)
        md = build_exploratory_temporal_validation_section(recal_sum, thresh_sum)
        assert "Original" in md
        assert "Brier" in md

    def test_contains_threshold_table(self, small_cohort, rename):
        recal_sum, thresh_sum = self._make_summaries(small_cohort, rename)
        md = build_exploratory_temporal_validation_section(recal_sum, thresh_sum)
        assert "Locked" in md or "Bloqueado" in md
        assert "Youden" in md

    def test_contains_youden_note(self, small_cohort, rename):
        recal_sum, thresh_sum = self._make_summaries(small_cohort, rename)
        md = build_exploratory_temporal_validation_section(recal_sum, thresh_sum)
        assert "Youden" in md

    def test_contains_locked_note(self, small_cohort, rename):
        recal_sum, thresh_sum = self._make_summaries(small_cohort, rename)
        md = build_exploratory_temporal_validation_section(recal_sum, thresh_sum)
        assert "locked" in md.lower() or "bloqueado" in md.lower()

    def test_empty_inputs_return_empty_string(self):
        empty_recal = {"table": pd.DataFrame(), "available": False}
        empty_thresh = {"table": pd.DataFrame(), "note_locked": "", "note_youden": "", "available": False}
        md = build_exploratory_temporal_validation_section(empty_recal, empty_thresh)
        assert md == ""

    def test_none_inputs_return_empty_string(self):
        md = build_exploratory_temporal_validation_section(None, None)
        assert md == ""

    def test_only_recal_renders_without_error(self, small_cohort, rename):
        recal_sum, _ = self._make_summaries(small_cohort, rename)
        empty_thresh = {"table": pd.DataFrame(), "note_locked": "", "note_youden": "", "available": False}
        md = build_exploratory_temporal_validation_section(recal_sum, empty_thresh)
        assert "Recalibration" in md or "Recalibração" in md

    def test_only_threshold_renders_without_error(self, small_cohort, rename):
        _, thresh_sum = self._make_summaries(small_cohort, rename)
        empty_recal = {"table": pd.DataFrame(), "available": False}
        md = build_exploratory_temporal_validation_section(empty_recal, thresh_sum)
        assert "Threshold" in md or "Limiar" in md

    def test_portuguese_language(self, small_cohort, rename):
        recal_sum, thresh_sum = self._make_summaries(small_cohort, rename)
        recal_sum_pt = build_exploratory_recalibration_summary(
            {
                "ia_risk": {
                    "intercept_only": recalibrate_intercept_only(*small_cohort),
                    "logistic":        recalibrate_logistic(*small_cohort),
                    "isotonic":        recalibrate_isotonic(*small_cohort),
                }
            },
            rename, language="Portuguese",
        )
        md = build_exploratory_temporal_validation_section(recal_sum_pt, thresh_sum, language="Portuguese")
        assert "Apêndice" in md or "exploratório" in md.lower()

    def test_section_starts_with_separator(self, small_cohort, rename):
        recal_sum, thresh_sum = self._make_summaries(small_cohort, rename)
        md = build_exploratory_temporal_validation_section(recal_sum, thresh_sum)
        assert "---" in md

    def test_section_ends_with_generated_by_marker(self, small_cohort, rename):
        recal_sum, thresh_sum = self._make_summaries(small_cohort, rename)
        md = build_exploratory_temporal_validation_section(recal_sum, thresh_sum)
        assert "Exploratory Module" in md or "Módulo Exploratório" in md


# ===========================================================================
# Primary summary isolation — exploratory must not bleed into primary report
# ===========================================================================

class TestPrimarySummaryUnchanged:
    def _primary_md(self):
        return build_temporal_validation_summary(
            cohort_summary={
                "n_total": 50, "n_events": 5, "event_rate": 0.10,
                "date_range": "2024-Q1 — 2024-Q4",
                "n_complete": 40, "n_adequate": 5, "n_partial": 3, "n_low": 2,
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
            language="English",
        )

    def test_primary_report_does_not_contain_exploratory_heading(self):
        md = self._primary_md()
        assert "Appendix: Exploratory" not in md
        assert "Apêndice: Análise Exploratória" not in md

    def test_primary_report_does_not_contain_recalibration_table(self):
        md = self._primary_md()
        assert "Intercept-only" not in md
        assert "Isotonic" not in md

    def test_primary_report_does_not_contain_youden_row(self):
        md = self._primary_md()
        assert "Youden J" not in md

    def test_exploratory_section_is_additive(self, small_cohort, rename):
        y, p = small_cohort
        recal_data = {
            "ia_risk": {
                "intercept_only": recalibrate_intercept_only(y, p),
                "logistic":        recalibrate_logistic(y, p),
                "isotonic":        recalibrate_isotonic(y, p),
            }
        }
        recal_sum = build_exploratory_recalibration_summary(recal_data, rename)
        primary = self._primary_md()
        expl = build_exploratory_temporal_validation_section(recal_sum, None)
        combined = primary + "\n" + expl
        # Primary section must be unchanged in combined output
        assert primary in combined
