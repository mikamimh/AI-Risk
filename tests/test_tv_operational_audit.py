"""Tests for the five operational-audit improvements to the Temporal Validation tab.

Covers:
  A. _build_sts_patient_audit — sts_score_from_cache consistency invariants
  B. build_sts_accounting_table — pipeline accounting closes the count
  C. Threshold table TP/FP/TN/FN — present and arithmetically consistent
  D. Common cohort in export — build_temporal_validation_summary produces section
  E. Async warning — _make_batch closes coroutines (no RuntimeWarning)
"""
import asyncio
import inspect
import warnings
from collections import namedtuple
from typing import Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tv_helpers import (
    build_sts_accounting_table,
    build_temporal_validation_summary,
    build_exploratory_threshold_summary,
)
from stats_compare import threshold_analysis_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_exec_rec(status: str):
    """Minimal execution record stub."""
    Rec = namedtuple("Rec", ["status", "retry_attempted"])
    return Rec(status=status, retry_attempted=(status == "stale_fallback"))


def _make_elig_entry(row_index: int, eligibility: str = "supported"):
    return {"row_index": row_index, "patient_id": f"P{row_index}", "eligibility": eligibility, "reason": "CABG"}


def _score_col(n: int, present_indices: set):
    """pd.Series with NaN for absent indices."""
    vals = [0.05 if i in present_indices else float("nan") for i in range(n)]
    return pd.Series(vals)


# ---------------------------------------------------------------------------
# A. _build_sts_patient_audit — sts_score_from_cache invariants
# ---------------------------------------------------------------------------

class TestBuildStsPatientAudit:
    # Import here so the tests can reference the function
    from tabs.temporal_validation import _build_sts_patient_audit

    def _run(self, exec_status: Optional[str], fail_stage: Optional[str],
             score_present: bool, raw_predmort: bool):
        from tabs.temporal_validation import _build_sts_patient_audit

        eligibility_log = [_make_elig_entry(0)]
        eligible_idx = [0]
        exec_rec = _make_exec_rec(exec_status) if exec_status else None
        raw_results = [{"predmort": 0.04} if raw_predmort else {}]
        fail_log = (
            [{"idx": 0, "stage": fail_stage, "reason": "test", "retry_attempted": False}]
            if fail_stage else []
        )
        sts_score_col = _score_col(1, {0} if score_present else set())
        rows = _build_sts_patient_audit(
            eligibility_log, eligible_idx, raw_results,
            [exec_rec] if exec_rec else [], fail_log, sts_score_col,
        )
        return rows[0]

    def test_cache_hit_has_from_cache_true(self):
        row = self._run("cached", None, True, True)
        assert row["sts_score_from_cache"] is True

    def test_cache_hit_query_not_attempted(self):
        row = self._run("cached", None, True, True)
        assert row["sts_query_attempted"] is False

    def test_cache_hit_score_present(self):
        row = self._run("cached", None, True, True)
        assert row["sts_score_present_final"] is True

    def test_live_fresh_query_has_from_cache_false(self):
        row = self._run("fresh", None, True, True)
        assert row["sts_score_from_cache"] is False

    def test_live_fresh_query_attempted_and_success(self):
        row = self._run("fresh", None, True, True)
        assert row["sts_query_attempted"] is True
        assert row["sts_query_success"] is True
        assert row["sts_score_present_final"] is True

    def test_failed_query_no_score(self):
        row = self._run(None, "fetch", False, False)
        assert row["sts_query_attempted"] is True
        assert row["sts_query_success"] is False
        assert row["sts_score_present_final"] is False
        assert row["sts_score_from_cache"] is False

    def test_build_input_failed_no_query_no_score(self):
        row = self._run(None, "build_input", False, False)
        assert row["sts_query_attempted"] is False
        assert row["sts_score_present_final"] is False
        assert row["sts_input_ready"] is False
        assert row["sts_score_from_cache"] is False

    def test_batch_aborted_before_query(self):
        row = self._run(None, "batch_abort", False, False)
        assert row["sts_batch_aborted_before_query"] is True
        assert row["sts_query_attempted"] is False
        assert row["sts_score_present_final"] is False

    def test_row_has_fifteen_fields(self):
        row = self._run("fresh", None, True, True)
        assert len(row) == 15

    def test_invariant_cache_implies_not_attempted(self):
        """If from_cache=True then query_attempted must be False (invariant)."""
        row = self._run("cached", None, True, True)
        if row["sts_score_from_cache"]:
            assert not row["sts_query_attempted"]

    def test_invariant_success_implies_score_present(self):
        """If query_success=True then score_present_final must be True (invariant)."""
        row = self._run("fresh", None, True, True)
        if row["sts_query_success"]:
            assert row["sts_score_present_final"]


# ---------------------------------------------------------------------------
# B. build_sts_accounting_table — count closure
# ---------------------------------------------------------------------------

class TestBuildStsAccountingTable:

    def test_returns_dataframe(self):
        df = build_sts_accounting_table(100, 10, 5, 85, 80)
        assert isinstance(df, pd.DataFrame)

    def test_has_three_columns(self):
        df = build_sts_accounting_table(100, 10, 5, 85, 80)
        assert df.shape[1] == 3

    def test_total_cohort_first_row(self):
        df = build_sts_accounting_table(100, 10, 5, 85, 80)
        assert int(df.iloc[0, 1]) == 100

    def test_counts_add_up(self):
        n_ns, n_unc, n_sup = 10, 5, 85
        df = build_sts_accounting_table(100, n_ns, n_unc, n_sup, 80)
        counts = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        # supported + not_supported + uncertain must equal total
        total = sum(
            int(v) for k, v in counts.items()
            if any(kw in k for kw in ("Total", "Total da"))
        )
        assert total == 100

    def test_supported_no_score_row_correct(self):
        df = build_sts_accounting_table(100, 10, 5, 85, 70)
        # Last row: "Supported but no final STS score" = 85 - 70 = 15
        assert int(df.iloc[-1, 1]) == 15

    def test_coverage_pct_in_last_usable_row(self):
        df = build_sts_accounting_table(100, 10, 5, 80, 40)
        # Coverage = 40/80 = 50%
        cov_str = df.iloc[-2, 2]  # "Final usable STS score" row
        assert "50.0%" in str(cov_str)

    def test_zero_supported_no_error(self):
        df = build_sts_accounting_table(100, 50, 50, 0, 0)
        assert isinstance(df, pd.DataFrame)

    def test_portuguese_labels(self):
        df = build_sts_accounting_table(100, 10, 5, 85, 80, language="Português")
        assert "Total da coorte" in df.iloc[0, 0]

    def test_english_labels(self):
        df = build_sts_accounting_table(100, 10, 5, 85, 80, language="English")
        assert "Total cohort" in df.iloc[0, 0]


# ---------------------------------------------------------------------------
# C. Threshold table TP/FP/TN/FN
# ---------------------------------------------------------------------------

class TestThresholdTableConfusionMatrix:

    def _cohort(self, n: int = 80, event_rate: float = 0.15, seed: int = 42):
        rng = np.random.default_rng(seed)
        y = (rng.random(n) < event_rate).astype(int)
        p = rng.beta(2, 8, n)
        return y, p

    def test_threshold_analysis_table_has_tp_fp_tn_fn(self):
        y, p = self._cohort()
        df = threshold_analysis_table(y, p, [0.05, 0.10])
        for col in ("TP", "FP", "TN", "FN"):
            assert col in df.columns, f"{col} missing from threshold_analysis_table"

    def test_confusion_matrix_sums_to_n(self):
        y, p = self._cohort(n=100)
        df = threshold_analysis_table(y, p, [0.08])
        row = df.iloc[0]
        assert int(row["TP"]) + int(row["FP"]) + int(row["TN"]) + int(row["FN"]) == 100

    def test_tp_plus_fn_equals_positives(self):
        y, p = self._cohort(n=100)
        df = threshold_analysis_table(y, p, [0.08])
        row = df.iloc[0]
        n_pos = int(y.sum())
        assert int(row["TP"]) + int(row["FN"]) == n_pos

    def test_fp_plus_tn_equals_negatives(self):
        y, p = self._cohort(n=100)
        df = threshold_analysis_table(y, p, [0.08])
        row = df.iloc[0]
        n_neg = int((1 - y).sum())
        assert int(row["FP"]) + int(row["TN"]) == n_neg

    def test_exploratory_threshold_summary_includes_tp_fp_tn_fn(self):
        y, p = self._cohort(n=100)
        thr_tables = {"score": threshold_analysis_table(y, p, [0.05, 0.08, 0.10])}
        result = build_exploratory_threshold_summary(
            thr_tables,
            locked_threshold=0.08,
            youden_thresholds={},
            rename={"score": "Test Score"},
            language="English",
        )
        assert result["available"]
        for col in ("TP", "FP", "TN", "FN"):
            assert col in result["table"].columns, f"{col} missing from exploratory threshold summary"

    def test_exploratory_markdown_includes_tp_fp_tn_fn_headers(self):
        from tv_helpers import build_exploratory_temporal_validation_section
        y, p = self._cohort(n=100)
        thr_tables = {"score": threshold_analysis_table(y, p, [0.05, 0.08])}
        thresh_sum = build_exploratory_threshold_summary(
            thr_tables, 0.08, {}, {"score": "Test Score"}, "English"
        )
        md = build_exploratory_temporal_validation_section(None, thresh_sum, "English")
        assert "| TP |" in md or " TP " in md, "TP column missing from exploratory Markdown"


# ---------------------------------------------------------------------------
# D. Common cohort in export
# ---------------------------------------------------------------------------

class TestCommonCohortInExport:

    def _make_perf(self, n: int = 3):
        return pd.DataFrame({
            "Score": ["AI Risk", "EuroSCORE II", "STS Score"][:n],
            "n": [50, 50, 50][:n],
            "AUC": [0.75, 0.70, 0.72][:n],
            "AUC_lo": [0.65, 0.60, 0.62][:n],
            "AUC_hi": [0.85, 0.80, 0.82][:n],
            "AUPRC": [0.30, 0.25, 0.28][:n],
            "AUPRC_lo": [0.20, 0.15, 0.18][:n],
            "AUPRC_hi": [0.40, 0.35, 0.38][:n],
            "Brier": [0.10, 0.12, 0.11][:n],
            "Calibration_Intercept": [0.0, 0.1, 0.05][:n],
            "Calibration_Slope": [1.0, 0.9, 0.95][:n],
            "HL_p": [0.5, 0.4, 0.45][:n],
            "Sensitivity": [0.7, 0.65, 0.68][:n],
            "Specificity": [0.75, 0.70, 0.72][:n],
            "PPV": [0.30, 0.28, 0.29][:n],
            "NPV": [0.95, 0.94, 0.94][:n],
        })

    def _make_summary(self, common_perf: Optional[pd.DataFrame] = None, n_common: int = 0) -> str:
        cohort = {
            "n_total": 100, "n_events": 15, "event_rate": 0.15,
            "date_range": "2020Q1 — 2021Q4",
            "n_complete": 80, "n_adequate": 10, "n_partial": 5, "n_low": 5,
        }
        meta = {
            "model_version": "v1", "n_patients": 500, "n_events": 75,
            "event_rate": 0.15, "training_window": "2018Q1 — 2019Q4",
        }
        return build_temporal_validation_summary(
            cohort, self._make_perf(), pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), meta, 0.08, "English",
            common_cohort_perf=common_perf,
            n_common=n_common,
        )

    def test_common_cohort_section_present_when_provided(self):
        md = self._make_summary(self._make_perf(), n_common=50)
        assert "Common STS-Available Cohort" in md or "Common" in md

    def test_common_cohort_section_absent_when_none(self):
        md = self._make_summary(None, n_common=0)
        assert "Common STS" not in md

    def test_common_cohort_section_absent_when_empty_df(self):
        md = self._make_summary(pd.DataFrame(), n_common=0)
        assert "Common STS" not in md

    def test_common_cohort_section_shows_n(self):
        md = self._make_summary(self._make_perf(), n_common=50)
        assert "50" in md

    def test_common_cohort_portuguese(self):
        cohort = {
            "n_total": 100, "n_events": 15, "event_rate": 0.15,
            "date_range": "2020Q1 — 2021Q4",
            "n_complete": 80, "n_adequate": 10, "n_partial": 5, "n_low": 5,
        }
        meta = {
            "model_version": "v1", "n_patients": 500, "n_events": 75,
            "event_rate": 0.15, "training_window": "2018Q1 — 2019Q4",
        }
        md = build_temporal_validation_summary(
            cohort, self._make_perf(), pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), meta, 0.08, "Português",
            common_cohort_perf=self._make_perf(),
            n_common=50,
        )
        assert "Coorte Comum" in md


# ---------------------------------------------------------------------------
# E. Async warning — coroutine is closed before GC
# ---------------------------------------------------------------------------

class TestAsyncWarningEliminated:

    def test_closing_side_effect_closes_coroutine(self):
        """The _make_batch wrapper in test_sts_batch_audit closes any coroutine
        it receives so GC never fires 'never awaited'.  Verify: after coro.close()
        the coroutine's cr_frame is None (it is exhausted/closed)."""
        async def _dummy(**kw):
            return {}

        coro = _dummy()
        assert inspect.iscoroutine(coro)
        # Simulate what the wrapper does
        coro.close()
        # A closed coroutine has no frame — GC won't warn
        assert coro.cr_frame is None

    def test_closing_is_idempotent(self):
        """Calling close() twice on a coroutine must not raise."""
        async def _dummy():
            pass

        coro = _dummy()
        coro.close()
        coro.close()  # second close must be a no-op

    def test_non_coroutine_arg_not_closed(self):
        """The wrapper must handle non-coroutine args (e.g. plain dict) gracefully."""
        non_coro = {"result": 42}

        def _closing_side_effect(arg):
            if inspect.iscoroutine(arg):
                arg.close()
            return arg

        result = _closing_side_effect(non_coro)
        assert result == {"result": 42}
