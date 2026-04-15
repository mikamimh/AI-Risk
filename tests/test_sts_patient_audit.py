"""Tests for the _build_sts_patient_audit helper in tabs/temporal_validation.py.

Covers 8 scenarios (A-1 … A-8):

  A-1  All-success batch (fresh queries) — every eligible patient has a score.
  A-2  build_input failure — patient is lost before a query is attempted.
  A-3  fetch failure with no stale fallback — query attempted but no score.
  A-4  batch_abort with no stale fallback — patient never queried.
  A-5  Cached result — no live query, score present (Phase A short-circuit).
  A-6  stale_fallback — live query failed, fallback used; score present.
  A-7  Consistency check: n_eligible > n_score_present → loss breakdown correct.
  A-8  Reconciliation: n_eligible == n_with_score + n_without_score (all paths).

The helper is a pure function with no Streamlit or network dependency, so no
mocking is needed beyond constructing the expected data structures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict

import pandas as pd
import pytest

from tabs.temporal_validation import _build_sts_patient_audit


# ---------------------------------------------------------------------------
# Minimal ExecutionRecord stand-in
# (mirrors the real sts_cache.ExecutionRecord dataclass signature)
# ---------------------------------------------------------------------------

@dataclass
class _ExecRec:
    status: str = "failed"
    stage: str = "done"
    reason: str = ""
    retry_attempted: bool = False
    used_previous_cache: bool = False
    result: Dict = field(default_factory=dict)
    patient_id: Optional[str] = None
    input_hash: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elig_entry(row_index: int, patient_id: str = "P?",
                eligibility: str = "supported", reason: str = "CABG") -> dict:
    return {"row_index": row_index, "patient_id": patient_id,
            "eligibility": eligibility, "reason": reason}


def _scores_series(values: list) -> pd.Series:
    """pd.Series whose .iloc[i] returns values[i]."""
    return pd.Series(values)


# ---------------------------------------------------------------------------
# A-1: all-success (fresh)
# ---------------------------------------------------------------------------

def test_all_success_fresh():
    """Every eligible patient queried successfully → all flags True/False as expected."""
    n = 3
    eligibility_log = [_elig_entry(i, f"P{i}") for i in range(n)]
    eligible_idx    = list(range(n))
    raw_results     = [{"predmort": 0.04, "predmm": 0.12}] * n
    exec_log        = [_ExecRec(status="fresh", stage="done") for _ in range(n)]
    fail_log        = []
    sts_score_col   = _scores_series([0.04, 0.04, 0.04])

    rows = _build_sts_patient_audit(
        eligibility_log, eligible_idx, raw_results, exec_log, fail_log, sts_score_col
    )

    assert len(rows) == n
    for r in rows:
        assert r["sts_eligibility_status"] == "supported"
        assert r["sts_input_ready"]               is True
        assert r["sts_query_attempted"]            is True
        assert r["sts_query_success"]              is True
        assert r["sts_parse_success"]              is True
        assert r["sts_score_present_final"]        is True
        assert r["sts_batch_aborted_before_query"] is False
        assert r["sts_failure_stage"]              == ""
        assert r["sts_failure_reason"]             == ""


# ---------------------------------------------------------------------------
# A-2: build_input failure
# ---------------------------------------------------------------------------

def test_build_input_failure():
    """Patient whose input build fails → lost before any query; score absent."""
    eligibility_log = [_elig_entry(0, "P0")]
    eligible_idx    = [0]
    raw_results     = [{}]
    exec_log        = [_ExecRec(status="failed", stage="build_input",
                                reason="mapping_failure: bad field")]
    fail_log        = [{
        "idx": 0, "patient_id": "P0", "status": "failed",
        "stage": "build_input", "reason": "mapping_failure: bad field",
        "retry_attempted": False, "used_previous_cache": False,
    }]
    sts_score_col = _scores_series([float("nan")])

    rows = _build_sts_patient_audit(
        eligibility_log, eligible_idx, raw_results, exec_log, fail_log, sts_score_col
    )

    assert len(rows) == 1
    r = rows[0]
    assert r["sts_input_ready"]               is False
    assert r["sts_query_attempted"]            is False
    assert r["sts_query_success"]              is False
    assert r["sts_parse_success"]              is False
    assert r["sts_score_present_final"]        is False
    assert r["sts_batch_aborted_before_query"] is False
    assert r["sts_failure_stage"]              == "build_input"
    assert "bad field" in r["sts_failure_reason"]


# ---------------------------------------------------------------------------
# A-3: fetch failure, no stale fallback
# ---------------------------------------------------------------------------

def test_fetch_failure_no_fallback():
    """Patient queried but endpoint unreachable; no stale fallback → score absent."""
    eligibility_log = [_elig_entry(0, "P0")]
    eligible_idx    = [0]
    raw_results     = [{}]
    exec_log        = [_ExecRec(status="failed", stage="fetch",
                                reason="fetch_failed; no fallback available",
                                retry_attempted=True)]
    fail_log        = [{
        "idx": 0, "patient_id": "P0", "status": "failed",
        "stage": "fetch", "reason": "fetch_failed; no fallback available",
        "retry_attempted": True, "used_previous_cache": False,
    }]
    sts_score_col = _scores_series([float("nan")])

    rows = _build_sts_patient_audit(
        eligibility_log, eligible_idx, raw_results, exec_log, fail_log, sts_score_col
    )

    r = rows[0]
    assert r["sts_input_ready"]               is True
    assert r["sts_query_attempted"]            is True    # fetch was attempted
    assert r["sts_query_success"]              is False
    assert r["sts_parse_success"]              is False
    assert r["sts_score_present_final"]        is False
    assert r["sts_batch_aborted_before_query"] is False
    assert r["sts_failure_stage"]              == "fetch"
    assert r["sts_retry_attempted"]            is True


# ---------------------------------------------------------------------------
# A-4: batch_abort, no stale fallback
# ---------------------------------------------------------------------------

def test_batch_abort_no_fallback():
    """Patient never queried because batch was aborted earlier."""
    eligibility_log = [_elig_entry(0, "P0")]
    eligible_idx    = [0]
    raw_results     = [{}]
    exec_log        = [_ExecRec(status="failed", stage="batch_abort",
                                reason="batch_aborted after 10 consecutive failures")]
    fail_log        = [{
        "idx": 0, "patient_id": "P0", "status": "failed",
        "stage": "batch_abort",
        "reason": "batch_aborted after 10 consecutive failures",
        "retry_attempted": False, "used_previous_cache": False,
    }]
    sts_score_col = _scores_series([float("nan")])

    rows = _build_sts_patient_audit(
        eligibility_log, eligible_idx, raw_results, exec_log, fail_log, sts_score_col
    )

    r = rows[0]
    assert r["sts_batch_aborted_before_query"] is True
    assert r["sts_query_attempted"]            is False
    assert r["sts_query_success"]              is False
    assert r["sts_score_present_final"]        is False
    assert r["sts_failure_stage"]              == "batch_abort"


# ---------------------------------------------------------------------------
# A-5: cached (Phase A short-circuit)
# ---------------------------------------------------------------------------

def test_cached_no_live_query():
    """Patient served from cache; no live network query, but score is present."""
    eligibility_log = [_elig_entry(0, "P0")]
    eligible_idx    = [0]
    # raw_results is populated from the cache; predmort will be there
    raw_results     = [{"predmort": 0.05, "predmm": 0.10}]
    exec_log        = [_ExecRec(status="cached", stage="done",
                                retry_attempted=False)]
    fail_log        = []  # cached → no failure
    sts_score_col   = _scores_series([0.05])

    rows = _build_sts_patient_audit(
        eligibility_log, eligible_idx, raw_results, exec_log, fail_log, sts_score_col
    )

    r = rows[0]
    assert r["sts_input_ready"]               is True
    assert r["sts_query_attempted"]            is False   # cache hit → no live query
    assert r["sts_query_success"]              is False   # no query was made
    assert r["sts_parse_success"]              is True    # predmort present in raw_res
    assert r["sts_score_present_final"]        is True
    assert r["sts_batch_aborted_before_query"] is False
    assert r["sts_failure_stage"]              == ""


# ---------------------------------------------------------------------------
# A-6: stale_fallback — live query failed, previous cache entry used
# ---------------------------------------------------------------------------

def test_stale_fallback():
    """Live query failed but a previous cache entry was used as fallback.
    The patient has a score, but the live query was attempted and failed."""
    eligibility_log = [_elig_entry(0, "P0")]
    eligible_idx    = [0]
    # stale fallback means results[0] has predmort from the fallback entry
    raw_results     = [{"predmort": 0.06, "predmm": 0.11}]
    # ExecutionRecord has status=stale_fallback, stage=fetch
    exec_log        = [_ExecRec(status="stale_fallback", stage="fetch",
                                reason="endpoint timeout; returned previous cache entry",
                                retry_attempted=True, used_previous_cache=True)]
    # stale_fallback entries DO appear in failure_log with stage=rec.stage ("fetch")
    fail_log        = [{
        "idx": 0, "patient_id": "P0", "status": "stale_fallback",
        "stage": "fetch",
        "reason": "endpoint timeout; returned previous cache entry",
        "retry_attempted": True, "used_previous_cache": True,
    }]
    sts_score_col = _scores_series([0.06])

    rows = _build_sts_patient_audit(
        eligibility_log, eligible_idx, raw_results, exec_log, fail_log, sts_score_col
    )

    r = rows[0]
    assert r["sts_input_ready"]               is True
    assert r["sts_query_attempted"]            is True    # fetch was attempted
    assert r["sts_query_success"]              is False   # live query failed
    assert r["sts_parse_success"]              is True    # fallback result has predmort
    assert r["sts_score_present_final"]        is True    # fallback score in output
    assert r["sts_batch_aborted_before_query"] is False
    assert r["sts_failure_stage"]              == "fetch"
    assert r["sts_retry_attempted"]            is True


# ---------------------------------------------------------------------------
# A-7: consistency check — n_eligible > n_score_present, correct breakdown
# ---------------------------------------------------------------------------

def test_consistency_breakdown():
    """Mixed batch: 1 success, 1 build_input failure, 1 fetch failure.
    Loss breakdown must correctly account for all missing scores."""
    # Patient 0: cohort row 0, fresh success
    # Patient 1: cohort row 1, build_input failure
    # Patient 2: cohort row 2, fetch failure
    eligibility_log = [_elig_entry(i, f"P{i}") for i in range(3)]
    eligible_idx    = [0, 1, 2]
    raw_results     = [{"predmort": 0.04}, {}, {}]
    exec_log        = [
        _ExecRec(status="fresh",  stage="done"),
        _ExecRec(status="failed", stage="build_input", reason="mapping error"),
        _ExecRec(status="failed", stage="fetch",       reason="fetch_failed"),
    ]
    fail_log        = [
        {"idx": 1, "patient_id": "P1", "status": "failed",
         "stage": "build_input", "reason": "mapping error",
         "retry_attempted": False, "used_previous_cache": False},
        {"idx": 2, "patient_id": "P2", "status": "failed",
         "stage": "fetch", "reason": "fetch_failed; no fallback available",
         "retry_attempted": True, "used_previous_cache": False},
    ]
    sts_score_col = _scores_series([0.04, float("nan"), float("nan")])

    rows = _build_sts_patient_audit(
        eligibility_log, eligible_idx, raw_results, exec_log, fail_log, sts_score_col
    )

    n_score   = sum(1 for r in rows if r["sts_score_present_final"])
    n_bi      = sum(1 for r in rows if r["sts_failure_stage"] == "build_input")
    n_fetch   = sum(1 for r in rows if r["sts_failure_stage"] == "fetch" and not r["sts_score_present_final"])
    n_abort   = sum(1 for r in rows if r["sts_batch_aborted_before_query"] and not r["sts_score_present_final"])

    assert n_score == 1
    assert n_bi    == 1
    assert n_fetch == 1
    assert n_abort == 0

    # Consistency: 3 eligible → 1 score, 2 lost
    assert n_score == 1
    assert len(rows) - n_score == 2


# ---------------------------------------------------------------------------
# A-8: reconciliation — all loss paths + successes sum to n_eligible
# ---------------------------------------------------------------------------

def test_reconciliation_all_paths():
    """Mixed batch with all four outcome types: fresh, cached, fetch-fail, batch_abort.
    The reconciliation identity must hold:
      n_eligible == n_with_score + n_build_input + n_fetch_no_score + n_abort_no_score
    """
    # 4 eligible patients, 4 different outcomes
    eligibility_log = [_elig_entry(i, f"P{i}") for i in range(4)]
    eligible_idx    = [0, 1, 2, 3]
    raw_results     = [
        {"predmort": 0.04},          # 0: fresh success
        {"predmort": 0.05},          # 1: cached
        {},                          # 2: fetch failure
        {},                          # 3: batch_abort
    ]
    exec_log = [
        _ExecRec(status="fresh",      stage="done"),
        _ExecRec(status="cached",     stage="done"),
        _ExecRec(status="failed",     stage="fetch"),
        _ExecRec(status="failed",     stage="batch_abort"),
    ]
    fail_log = [
        {"idx": 2, "patient_id": "P2", "status": "failed",
         "stage": "fetch", "reason": "fetch_failed; no fallback available",
         "retry_attempted": True, "used_previous_cache": False},
        {"idx": 3, "patient_id": "P3", "status": "failed",
         "stage": "batch_abort",
         "reason": "batch_aborted after 10 consecutive failures",
         "retry_attempted": False, "used_previous_cache": False},
    ]
    sts_score_col = _scores_series([0.04, 0.05, float("nan"), float("nan")])

    rows = _build_sts_patient_audit(
        eligibility_log, eligible_idx, raw_results, exec_log, fail_log, sts_score_col
    )

    n_eligible = len(rows)
    n_score    = sum(1 for r in rows if r["sts_score_present_final"])
    n_bi       = sum(1 for r in rows if r["sts_failure_stage"] == "build_input")
    n_fetch_ns = sum(1 for r in rows if r["sts_failure_stage"] == "fetch"
                     and not r["sts_score_present_final"])
    n_abort_ns = sum(1 for r in rows if r["sts_batch_aborted_before_query"]
                     and not r["sts_score_present_final"])

    # Reconciliation identity
    assert n_score + n_bi + n_fetch_ns + n_abort_ns == n_eligible, (
        f"Reconciliation failed: {n_score} + {n_bi} + {n_fetch_ns} + {n_abort_ns} "
        f"≠ {n_eligible}"
    )

    # Specific counts for this batch
    assert n_score    == 2   # fresh + cached
    assert n_bi       == 0
    assert n_fetch_ns == 1
    assert n_abort_ns == 1
