"""Regression tests for the sts_patient_audit correctness bug.

Root cause
----------
``calculate_sts_batch.last_execution_log`` was built by appending Phase-A
resolutions (cache hits, build failures) first, then Phase-C fetch results.
When cache hits appear at *non-contiguous* eligible positions, the appended
order diverges from eligible-position order, so ``exec_log[eli_pos]`` in
``_build_sts_patient_audit`` reads the wrong record for most patients.

A second independent bug: ``_tv_sts_eligibility_log`` was reassigned from
ctx *after* the audit call, so the audit was always built with an empty
eligibility log (wrong patient_id / eligibility_status / supported_class).

Observed contradiction (real run, 22 eligible patients)
---------------------------------------------------------
- Batch aborted after 10 consecutive endpoint failures.
- 6 patients had scores (from Phase-A cache hits scattered across eligible
  positions 3, 7, 15, 19, 20, 21).
- 6 patients never queried (batch_abort, positions 16-21 in pending order).
- The exported sts_patient_audit.csv showed:
  * sts_query_attempted = False for ALL 22 rows
  * sts_batch_aborted_before_query = False for ALL rows
  * sts_failure_stage / sts_failure_reason = "" for ALL rows
  * yet sts_score_present_final = True for 6 rows

Tests in this file
------------------
  B-1  execution_log is positionally indexed after the fix
       (scatter cache hits at non-contiguous positions; exec_log[i] must
        return the correct record for position i, not a record for a
        different patient).

  B-2  _build_sts_patient_audit with scattered cache hits: cache rows have
       query_attempted=False; fetch-failure rows have query_attempted=True.

  B-3  _build_sts_patient_audit end-to-end via calculate_sts_batch:
       scattered cache hits + endpoint failures → audit never shows the
       impossible combination (query_attempted=False for a fetch-failure row).

  B-4  batch_abort rows appear in audit with batch_aborted_before_query=True
       even when cache hits are scattered.

  B-5  Reconciliation identity holds for a mixed batch with scattered cache
       hits, fetch failures, and batch_abort rows.

  B-6  No row can simultaneously have sts_score_present_final=True and
       sts_batch_aborted_before_query=True without a stale fallback score.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

import sts_calculator
from sts_calculator import calculate_sts_batch
from tabs.temporal_validation import _build_sts_patient_audit


# ---------------------------------------------------------------------------
# Minimal ExecutionRecord stand-in (mirrors sts_cache.ExecutionRecord)
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
    return pd.Series(values)


def _row(suffix: str = "0") -> dict:
    age = 50 + int(suffix)
    return {
        "surgery_pre": "CABG",
        "surgical_priority": "Elective",
        "age_years": str(age),
        "sex": "M",
        "patient_id": f"P{suffix}",
    }


@contextmanager
def _isolated_batch():
    """Run calculate_sts_batch with all cache layers mocked out."""
    with (
        patch("sts_calculator._sts_cache.load_entry", return_value=None),
        patch("sts_calculator._sts_cache.persist_fresh_result"),
        patch("sts_calculator._sts_cache.remember_patient_hash"),
        patch("sts_calculator._sts_cache.find_stale_fallback", return_value=None),
        patch.object(sts_calculator, "STS_CONSECUTIVE_FAILURE_BACKOFF_BASE_S", 0),
        patch("asyncio.sleep", new=AsyncMock(return_value=None)),
    ):
        old_cache = dict(sts_calculator._sts_memory_cache)
        sts_calculator._sts_memory_cache.clear()
        try:
            yield
        finally:
            sts_calculator._sts_memory_cache.clear()
            sts_calculator._sts_memory_cache.update(old_cache)


def _good_result() -> dict:
    return {"predmort": 0.03, "predmm": 0.12}


# ---------------------------------------------------------------------------
# B-1: execution_log is positionally indexed (not appended sequentially)
# ---------------------------------------------------------------------------

def test_B1_execution_log_is_positionally_indexed():
    """After calculate_sts_batch, last_execution_log[i] must hold the
    execution record for eligible position i, regardless of which rows were
    cache hits.

    Scenario: 5 eligible rows; rows at positions 1 and 3 are in-memory cache
    hits (returned in Phase A); rows 0, 2, 4 go to the network and fail.
    Old (buggy) behaviour: exec_log = [cache_rec_1, cache_rec_3,
    fetch_fail_0, fetch_fail_2, fetch_fail_4].
    Fixed behaviour: exec_log[0] = fetch_fail_0, exec_log[1] = cache_rec_1,
    exec_log[2] = fetch_fail_2, exec_log[3] = cache_rec_3,
    exec_log[4] = fetch_fail_4.
    """
    rows = [_row(str(i)) for i in range(5)]
    pids = [f"P{i}" for i in range(5)]

    # Pre-populate in-memory cache for positions 1 and 3.
    # We need valid cache entries for those rows.
    from sts_calculator import build_sts_input_from_row, _sts_memory_cache
    import sts_cache as _sc

    def _make_mem_entry(row):
        sts_input = build_sts_input_from_row(row)
        h = _sc.compute_input_hash(sts_input)
        return h, {
            "input_hash": h,
            "integration_version": _sc.STS_SCORE_INTEGRATION_VERSION,
            "result": _good_result(),
            "created_ts": 1_000_000.0,
        }

    h1, entry1 = _make_mem_entry(rows[1])
    h3, entry3 = _make_mem_entry(rows[3])

    with (
        patch("sts_calculator._sts_cache.load_entry", return_value=None),
        patch("sts_calculator._sts_cache.persist_fresh_result"),
        patch("sts_calculator._sts_cache.remember_patient_hash"),
        patch("sts_calculator._sts_cache.find_stale_fallback", return_value=None),
        patch.object(sts_calculator, "STS_CONSECUTIVE_FAILURE_BACKOFF_BASE_S", 0),
        patch("asyncio.sleep", new=AsyncMock(return_value=None)),
        patch(
            "sts_calculator._query_sts_ws",
            side_effect=sts_calculator.StsConnectError("refused"),
        ),
    ):
        old_cache = dict(_sts_memory_cache)
        _sts_memory_cache.clear()
        _sts_memory_cache[h1] = entry1
        _sts_memory_cache[h3] = entry3
        try:
            calculate_sts_batch(rows, patient_ids=pids, chunk_size=1)
        finally:
            _sts_memory_cache.clear()
            _sts_memory_cache.update(old_cache)

    exec_log = calculate_sts_batch.last_execution_log

    assert len(exec_log) == 5, f"Expected 5 entries, got {len(exec_log)}"

    # Positions 1 and 3 must be cache hits.
    assert exec_log[1] is not None, "exec_log[1] should not be None"
    assert exec_log[3] is not None, "exec_log[3] should not be None"
    assert getattr(exec_log[1], "status", None) == "cached", (
        f"exec_log[1].status expected 'cached', got {getattr(exec_log[1], 'status', None)!r}"
    )
    assert getattr(exec_log[3], "status", None) == "cached", (
        f"exec_log[3].status expected 'cached', got {getattr(exec_log[3], 'status', None)!r}"
    )

    # Positions 0, 2, 4 must be fetch failures (status='failed', stage='fetch').
    for pos in (0, 2, 4):
        rec = exec_log[pos]
        assert rec is not None, f"exec_log[{pos}] should not be None"
        assert getattr(rec, "status", None) == "failed", (
            f"exec_log[{pos}].status expected 'failed', got {getattr(rec, 'status', None)!r}"
        )
        assert getattr(rec, "stage", None) == "fetch", (
            f"exec_log[{pos}].stage expected 'fetch', got {getattr(rec, 'stage', None)!r}"
        )


# ---------------------------------------------------------------------------
# B-2: _build_sts_patient_audit with pre-built scattered-cache exec_log
# ---------------------------------------------------------------------------

def test_B2_audit_correct_with_scattered_cache_hits():
    """_build_sts_patient_audit receives an exec_log indexed by eligible
    position.  Cache-hit rows → query_attempted=False; fetch-failure rows →
    query_attempted=True (via fail_log stage='fetch')."""
    # 5 eligible patients at cohort positions 10, 11, 12, 13, 14.
    # Eligible positions 1, 3 → cache hits; 0, 2, 4 → fetch failures.
    eligible_idx = [10, 11, 12, 13, 14]
    n = len(eligible_idx)
    eligibility_log = [_elig_entry(cohort_i, f"P{cohort_i}") for cohort_i in eligible_idx]

    # raw_results: cache hits have predmort; fetch failures are empty.
    raw_results = [
        {},                                    # pos 0: fetch failure
        {"predmort": 0.03, "predmm": 0.12},   # pos 1: cache hit
        {},                                    # pos 2: fetch failure
        {"predmort": 0.04, "predmm": 0.08},   # pos 3: cache hit
        {},                                    # pos 4: fetch failure
    ]

    # exec_log positionally indexed — this is what the fixed code produces.
    exec_log = [
        _ExecRec(status="failed", stage="fetch", reason="fetch_failed; no fallback"),
        _ExecRec(status="cached", stage="done",  reason="cache_hit"),
        _ExecRec(status="failed", stage="fetch", reason="fetch_failed; no fallback"),
        _ExecRec(status="cached", stage="done",  reason="cache_hit"),
        _ExecRec(status="failed", stage="fetch", reason="fetch_failed; no fallback"),
    ]

    # fail_log: only the fetch failures (cache hits have no fail_log entry).
    fail_log = [
        {"idx": 0, "patient_id": "P10", "status": "failed",
         "stage": "fetch", "reason": "fetch_failed; no fallback available",
         "retry_attempted": True, "used_previous_cache": False},
        {"idx": 2, "patient_id": "P12", "status": "failed",
         "stage": "fetch", "reason": "fetch_failed; no fallback available",
         "retry_attempted": True, "used_previous_cache": False},
        {"idx": 4, "patient_id": "P14", "status": "failed",
         "stage": "fetch", "reason": "fetch_failed; no fallback available",
         "retry_attempted": True, "used_previous_cache": False},
    ]

    scores = [float("nan"), 0.03, float("nan"), 0.04, float("nan")]
    sts_score_col = _scores_series([float("nan")] * 10 + scores + [float("nan")] * 5)

    rows = _build_sts_patient_audit(
        eligibility_log, eligible_idx, raw_results, exec_log, fail_log, sts_score_col
    )

    assert len(rows) == n

    # Cache-hit positions (1, 3) → query_attempted=False, score present.
    for eli_pos in (1, 3):
        r = rows[eli_pos]
        assert r["sts_query_attempted"] is False, (
            f"eli_pos={eli_pos}: cache hit should have query_attempted=False, got {r['sts_query_attempted']}"
        )
        assert r["sts_score_present_final"] is True, (
            f"eli_pos={eli_pos}: cache hit should have score_present=True"
        )
        assert r["sts_batch_aborted_before_query"] is False

    # Fetch-failure positions (0, 2, 4) → query_attempted=True, no score.
    for eli_pos in (0, 2, 4):
        r = rows[eli_pos]
        assert r["sts_query_attempted"] is True, (
            f"eli_pos={eli_pos}: fetch failure should have query_attempted=True, got {r['sts_query_attempted']}"
        )
        assert r["sts_score_present_final"] is False, (
            f"eli_pos={eli_pos}: fetch failure should have score_present=False"
        )
        assert r["sts_failure_stage"] == "fetch", (
            f"eli_pos={eli_pos}: expected failure_stage='fetch', got {r['sts_failure_stage']!r}"
        )


# ---------------------------------------------------------------------------
# B-3: end-to-end via calculate_sts_batch — the observed contradiction cannot
#       occur when exec_log is positionally indexed.
# ---------------------------------------------------------------------------

def test_B3_end_to_end_no_impossible_audit_combination():
    """After calculate_sts_batch with scattered cache hits + endpoint failures,
    the execution_log fed to _build_sts_patient_audit must NOT produce the
    observed contradiction (query_attempted=False for a fetch-failure row)."""
    rows = [_row(str(i)) for i in range(5)]
    pids = [f"P{i}" for i in range(5)]

    from sts_calculator import build_sts_input_from_row, _sts_memory_cache
    import sts_cache as _sc

    def _make_mem_entry(row):
        sts_input = build_sts_input_from_row(row)
        h = _sc.compute_input_hash(sts_input)
        return h, {
            "input_hash": h,
            "integration_version": _sc.STS_SCORE_INTEGRATION_VERSION,
            "result": _good_result(),
            "created_ts": 1_000_000.0,
        }

    # Scatter cache hits at positions 1 and 3.
    h1, entry1 = _make_mem_entry(rows[1])
    h3, entry3 = _make_mem_entry(rows[3])

    with (
        patch("sts_calculator._sts_cache.load_entry", return_value=None),
        patch("sts_calculator._sts_cache.persist_fresh_result"),
        patch("sts_calculator._sts_cache.remember_patient_hash"),
        patch("sts_calculator._sts_cache.find_stale_fallback", return_value=None),
        patch.object(sts_calculator, "STS_CONSECUTIVE_FAILURE_BACKOFF_BASE_S", 0),
        patch("asyncio.sleep", new=AsyncMock(return_value=None)),
        patch(
            "sts_calculator._query_sts_ws",
            side_effect=sts_calculator.StsConnectError("refused"),
        ),
    ):
        old_cache = dict(_sts_memory_cache)
        _sts_memory_cache.clear()
        _sts_memory_cache[h1] = entry1
        _sts_memory_cache[h3] = entry3
        try:
            raw_results = list(calculate_sts_batch(rows, patient_ids=pids, chunk_size=1))
        finally:
            _sts_memory_cache.clear()
            _sts_memory_cache.update(old_cache)

    exec_log = calculate_sts_batch.last_execution_log
    fail_log = calculate_sts_batch.failure_log

    # Build a minimal eligibility log and score column.
    eligibility_log = [_elig_entry(i, pids[i]) for i in range(5)]
    eligible_idx = list(range(5))
    sts_scores = [
        (raw_results[i] or {}).get("predmort", float("nan"))
        for i in range(5)
    ]
    sts_score_col = pd.Series(sts_scores)

    audit = _build_sts_patient_audit(
        eligibility_log, eligible_idx, raw_results, exec_log, fail_log, sts_score_col
    )

    assert len(audit) == 5

    # Fetch-failure rows (0, 2, 4) must NOT be marked query_attempted=False.
    for eli_pos in (0, 2, 4):
        r = audit[eli_pos]
        assert r["sts_query_attempted"] is True, (
            f"REGRESSION: eli_pos={eli_pos} (fetch failure) has "
            f"query_attempted=False — exec_log misindexing not fixed!\n"
            f"  exec_log[{eli_pos}].status = "
            f"{getattr(exec_log[eli_pos], 'status', 'MISSING')!r}\n"
            f"  fail_log entry = {next((f for f in fail_log if f['idx'] == eli_pos), None)}"
        )

    # Cache-hit rows (1, 3) must have query_attempted=False and score present.
    for eli_pos in (1, 3):
        r = audit[eli_pos]
        assert r["sts_query_attempted"] is False, (
            f"eli_pos={eli_pos} (cache hit) should have query_attempted=False"
        )
        assert r["sts_score_present_final"] is True, (
            f"eli_pos={eli_pos} (cache hit) should have score present"
        )


# ---------------------------------------------------------------------------
# B-4: batch_abort rows correctly identified when cache hits are scattered
# ---------------------------------------------------------------------------

def test_B4_batch_abort_rows_correctly_marked_with_scattered_cache_hits():
    """When cache hits scatter across eligible positions and a batch abort
    occurs, batch_abort rows must have batch_aborted_before_query=True and
    query_attempted=False."""
    # 6 eligible rows: positions 0, 1, 2, 3, 4, 5
    # Positions 1, 4 → cache hits; rest → pending → positions 0, 2, 3 fail (3 failures)
    # Positions 5 → batch_abort (only if we trigger abort, but with threshold=10 and
    # only 3 failures we won't abort naturally — so simulate via exec_log directly).
    n = 6
    eligibility_log = [_elig_entry(i, f"P{i}") for i in range(n)]
    eligible_idx = list(range(n))
    raw_results = [
        {},
        {"predmort": 0.03},
        {},
        {},
        {"predmort": 0.04},
        {},
    ]
    exec_log = [
        _ExecRec(status="failed", stage="fetch"),       # 0: fetch fail
        _ExecRec(status="cached", stage="done"),         # 1: cache hit
        _ExecRec(status="failed", stage="fetch"),       # 2: fetch fail
        _ExecRec(status="failed", stage="fetch"),       # 3: fetch fail
        _ExecRec(status="cached", stage="done"),         # 4: cache hit
        _ExecRec(status="failed", stage="batch_abort"), # 5: never queried
    ]
    fail_log = [
        {"idx": 0, "stage": "fetch",       "reason": "fetch_failed",    "retry_attempted": True,  "used_previous_cache": False},
        {"idx": 2, "stage": "fetch",       "reason": "fetch_failed",    "retry_attempted": True,  "used_previous_cache": False},
        {"idx": 3, "stage": "fetch",       "reason": "fetch_failed",    "retry_attempted": True,  "used_previous_cache": False},
        {"idx": 5, "stage": "batch_abort", "reason": "aborted",         "retry_attempted": False, "used_previous_cache": False},
    ]
    sts_score_col = _scores_series([float("nan"), 0.03, float("nan"), float("nan"), 0.04, float("nan")])

    audit = _build_sts_patient_audit(
        eligibility_log, eligible_idx, raw_results, exec_log, fail_log, sts_score_col
    )

    r5 = audit[5]
    assert r5["sts_batch_aborted_before_query"] is True, (
        f"Row 5 (batch_abort) must have batch_aborted_before_query=True; got {r5}"
    )
    assert r5["sts_query_attempted"] is False, (
        f"Row 5 (batch_abort) must have query_attempted=False; got {r5['sts_query_attempted']}"
    )
    assert r5["sts_failure_stage"] == "batch_abort"

    # Cache-hit rows still correct despite scattered positions.
    for pos in (1, 4):
        r = audit[pos]
        assert r["sts_query_attempted"] is False
        assert r["sts_score_present_final"] is True

    # Fetch-failure rows must have query_attempted=True.
    for pos in (0, 2, 3):
        r = audit[pos]
        assert r["sts_query_attempted"] is True, (
            f"Row {pos} (fetch fail) must have query_attempted=True; got {r['sts_query_attempted']}"
        )


# ---------------------------------------------------------------------------
# B-5: reconciliation identity holds for scattered-cache + abort scenario
# ---------------------------------------------------------------------------

def test_B5_reconciliation_holds_with_scattered_cache_and_abort():
    """n_eligible == n_score + n_bi_fail + n_fetch_no_score + n_abort_no_score.
    Tests the exact pattern from the real run: scattered cache hits, endpoint
    failures, and batch_abort rows."""
    # Mimic the reported run: 22 eligible, 6 cache hits scattered, 10 endpoint
    # failures, 6 batch_abort (no stale fallback, so no score for aborted rows).
    n_eligible = 22
    cache_hit_positions  = {2, 5, 9, 14, 17, 21}   # 6 scattered cache hits
    endpoint_fail_positions = set(range(22)) - cache_hit_positions  # 16 pending
    # Simulate 10 endpoint failures then 6 batch_aborts among the 16 pending.
    # The 16 pending rows by ascending position: 0,1,3,4,6,7,8,10,11,12,13,15,16,18,19,20
    pending_positions = sorted(set(range(22)) - cache_hit_positions)
    fetch_fail_positions = set(pending_positions[:10])   # first 10 pending fail
    batch_abort_positions = set(pending_positions[10:])  # last 6 are aborted

    eligibility_log = [_elig_entry(i, f"P{i}") for i in range(n_eligible)]
    eligible_idx = list(range(n_eligible))

    raw_results = []
    exec_log = []
    fail_log = []
    scores = []

    for pos in range(n_eligible):
        if pos in cache_hit_positions:
            raw_results.append({"predmort": 0.03})
            exec_log.append(_ExecRec(status="cached", stage="done"))
            scores.append(0.03)
        elif pos in fetch_fail_positions:
            raw_results.append({})
            exec_log.append(_ExecRec(status="failed", stage="fetch"))
            fail_log.append({
                "idx": pos, "stage": "fetch", "reason": "fetch_failed",
                "retry_attempted": True, "used_previous_cache": False,
            })
            scores.append(float("nan"))
        else:  # batch_abort
            raw_results.append({})
            exec_log.append(_ExecRec(status="failed", stage="batch_abort"))
            fail_log.append({
                "idx": pos, "stage": "batch_abort", "reason": "aborted",
                "retry_attempted": False, "used_previous_cache": False,
            })
            scores.append(float("nan"))

    sts_score_col = _scores_series(scores)

    audit = _build_sts_patient_audit(
        eligibility_log, eligible_idx, raw_results, exec_log, fail_log, sts_score_col
    )

    assert len(audit) == n_eligible

    n_score    = sum(1 for r in audit if r["sts_score_present_final"])
    n_bi       = sum(1 for r in audit if r["sts_failure_stage"] == "build_input")
    n_fetch_ns = sum(1 for r in audit if r["sts_failure_stage"] == "fetch" and not r["sts_score_present_final"])
    n_abort_ns = sum(1 for r in audit if r["sts_batch_aborted_before_query"] and not r["sts_score_present_final"])

    assert n_score + n_bi + n_fetch_ns + n_abort_ns == n_eligible, (
        f"Reconciliation failed: {n_score} + {n_bi} + {n_fetch_ns} + {n_abort_ns} ≠ {n_eligible}"
    )

    assert n_score == len(cache_hit_positions),           f"Expected {len(cache_hit_positions)} scores"
    assert n_fetch_ns == len(fetch_fail_positions),       f"Expected {len(fetch_fail_positions)} fetch failures"
    assert n_abort_ns == len(batch_abort_positions),      f"Expected {len(batch_abort_positions)} batch_abort"

    # Confirm query_attempted is correct for each category.
    for pos in cache_hit_positions:
        assert audit[pos]["sts_query_attempted"] is False, f"pos={pos} cache hit must have query_attempted=False"
    for pos in fetch_fail_positions:
        assert audit[pos]["sts_query_attempted"] is True, f"pos={pos} fetch fail must have query_attempted=True"
    for pos in batch_abort_positions:
        assert audit[pos]["sts_batch_aborted_before_query"] is True, f"pos={pos} must be batch_aborted"
        assert audit[pos]["sts_query_attempted"] is False, f"pos={pos} batch_abort must have query_attempted=False"


# ---------------------------------------------------------------------------
# B-6: no row has score=True AND batch_aborted=True without stale fallback
# ---------------------------------------------------------------------------

def test_B6_no_row_has_score_and_batch_aborted_without_stale_fallback():
    """A row that was never queried (batch_abort, no stale fallback) cannot
    have sts_score_present_final=True.  This guards against a second class
    of contradiction: score present but marked as unqueried and no fallback."""
    n = 4
    eligibility_log = [_elig_entry(i, f"P{i}") for i in range(n)]
    eligible_idx = list(range(n))
    raw_results = [
        {"predmort": 0.03},   # 0: cache hit
        {},                    # 1: fetch fail
        {},                    # 2: batch_abort, no stale fallback → no score
        {},                    # 3: batch_abort, no stale fallback → no score
    ]
    exec_log = [
        _ExecRec(status="cached",  stage="done"),
        _ExecRec(status="failed",  stage="fetch"),
        _ExecRec(status="failed",  stage="batch_abort"),
        _ExecRec(status="failed",  stage="batch_abort"),
    ]
    fail_log = [
        {"idx": 1, "stage": "fetch",       "reason": "fetch_failed",
         "retry_attempted": True,  "used_previous_cache": False},
        {"idx": 2, "stage": "batch_abort", "reason": "aborted",
         "retry_attempted": False, "used_previous_cache": False},
        {"idx": 3, "stage": "batch_abort", "reason": "aborted",
         "retry_attempted": False, "used_previous_cache": False},
    ]
    sts_score_col = _scores_series([0.03, float("nan"), float("nan"), float("nan")])

    audit = _build_sts_patient_audit(
        eligibility_log, eligible_idx, raw_results, exec_log, fail_log, sts_score_col
    )

    for r in audit:
        if r["sts_batch_aborted_before_query"] and not r.get("sts_used_previous_cache", False):
            assert r["sts_score_present_final"] is False, (
                f"IMPOSSIBLE: row {r['row_index']} is batch_aborted without stale fallback "
                f"but sts_score_present_final=True: {r}"
            )
