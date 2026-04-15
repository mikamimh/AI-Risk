"""Tests for STS batch pipeline robustness improvements.

Tests 10 scenarios (R-1 … R-10):

  R-1  Exception hierarchy — StsQueryError subclasses are importable and
       correctly ordered in the inheritance tree.

  R-2  Transient endpoint failure then recovery — the batch completes once
       the endpoint starts responding; abort does NOT fire.

  R-3  Repeated endpoint failures reach abort threshold — batch is aborted
       after STS_MAX_CONSECUTIVE_FAILURES endpoint-level chunks.

  R-4  Empty-response failures do NOT count toward abort — consecutive
       StsEmptyResponseError chunks reset the counter, not increment it.

  R-5  Parse failure (StsParseError) does not count toward abort and is
       correctly classified in chunk_log.

  R-6  Rows after abort are marked with stage='batch_abort' in failure_log.

  R-7  chunk_log entries contain failure_type, endpoint_failure_count, and
       counted_toward_abort for every chunk attempted.

  R-8  endpoint_health_summary is populated with correct counts after a
       mixed batch.

  R-9  Deterministic build_input failure does not count as endpoint failure.

  R-10 Per-patient timeout recorded as endpoint-level failure type.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sts_calculator
from sts_calculator import (
    StsEmptyResponseError,
    StsEndpointUnreachableError,
    StsParseError,
    StsQueryError,
    StsSessionTimeoutError,
    calculate_sts_batch,
)


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

def _row(suffix: str = "0") -> dict:
    """Minimal valid patient row with a unique age so sts_input['age'] is
    distinct per patient (age = 50 + int(suffix)).  This lets mocks key on
    ``sts_input['age']`` to determine per-patient behaviour without needing a
    shared call counter (which breaks when _calc_one retries)."""
    age = 50 + int(suffix)
    return {
        "surgery_pre": "CABG",
        "surgical_priority": "Elective",
        "age_years": str(age),
        "sex": "M",
        "patient_id": f"P{suffix}",
    }


def _rows(n: int) -> list:
    return [_row(str(i)) for i in range(n)]


def _patient_idx(sts_input: dict) -> int:
    """Recover the 0-based patient index from sts_input['age'] (= 50 + idx)."""
    return int(sts_input.get("age", "50")) - 50


def _pids(n: int) -> list:
    return [f"P{i}" for i in range(n)]


def _good_result() -> dict:
    return {
        "predmort": 0.04, "predmm": 0.12, "predstro": 0.01,
        "predrenf": 0.03, "predreop": 0.02, "predvent": 0.10,
        "preddeep": 0.005, "pred14d": 0.08, "pred6d": 0.50,
    }


@contextmanager
def _isolated_batch():
    """Context manager that prevents all cache hits and removes delays.

    Effect:
    * In-memory cache appears empty for the duration.
    * Disk cache load_entry returns None (no cached entry).
    * persist_fresh_result, remember_patient_hash, find_stale_fallback are no-ops.
    * STS_CONSECUTIVE_FAILURE_BACKOFF_BASE_S is 0 → no inter-chunk sleep.
    * asyncio.sleep is a no-op → no per-retry sleep inside _calc_one.

    All rows therefore reach Phase B, and no real I/O or sleeps occur.
    """
    with (
        patch("sts_calculator._sts_cache.load_entry", return_value=None),
        patch("sts_calculator._sts_cache.persist_fresh_result"),
        patch("sts_calculator._sts_cache.remember_patient_hash"),
        patch("sts_calculator._sts_cache.find_stale_fallback", return_value=None),
        patch.object(sts_calculator, "STS_CONSECUTIVE_FAILURE_BACKOFF_BASE_S", 0),
        patch("asyncio.sleep", new=AsyncMock(return_value=None)),
    ):
        # Also clear the in-memory cache dict for the duration.
        old_cache = dict(sts_calculator._sts_memory_cache)
        sts_calculator._sts_memory_cache.clear()
        try:
            yield
        finally:
            sts_calculator._sts_memory_cache.clear()
            sts_calculator._sts_memory_cache.update(old_cache)


# ---------------------------------------------------------------------------
# R-1: Exception hierarchy importable and correctly structured
# ---------------------------------------------------------------------------

def test_exception_hierarchy():
    """All STS exception classes must be importable and form the expected tree."""
    assert issubclass(StsEndpointUnreachableError, StsQueryError)
    assert issubclass(StsSessionTimeoutError, StsQueryError)
    assert issubclass(StsEmptyResponseError, StsQueryError)
    assert issubclass(StsParseError, StsQueryError)
    assert issubclass(StsQueryError, Exception)

    classes = [StsEndpointUnreachableError, StsSessionTimeoutError,
               StsEmptyResponseError, StsParseError]
    assert len(set(classes)) == 4
    for cls in classes:
        inst = cls("test message")
        assert isinstance(inst, StsQueryError)
        assert str(inst) == "test message"


# ---------------------------------------------------------------------------
# R-2: Transient endpoint failure then recovery — no abort
# ---------------------------------------------------------------------------

def test_transient_failure_then_recovery_no_abort():
    """Patients 0–2 always raise StsEndpointUnreachableError; patients 3–5 succeed.
    With MAX_CONSECUTIVE=10, the batch must NOT abort."""
    n = 6

    async def intermittent(sts_input):
        # Key on patient index (embedded in sts_input['age'] = 50 + idx).
        if _patient_idx(sts_input) < 3:
            raise StsEndpointUnreachableError("connection refused")
        return _good_result()

    with _isolated_batch(), \
         patch("sts_calculator._query_sts_ws", new=AsyncMock(side_effect=intermittent)):
        calculate_sts_batch(_rows(n), patient_ids=_pids(n), chunk_size=1)

    assert not calculate_sts_batch._batch_aborted, (
        "Batch must not abort: only 3 consecutive endpoint failures < threshold 10"
    )
    endpoint_chunks = [
        cl for cl in calculate_sts_batch.chunk_log
        if cl.get("endpoint_failure_count", 0) > 0
    ]
    assert len(endpoint_chunks) == 3, (
        f"Expected 3 endpoint-failure chunks, got {len(endpoint_chunks)}"
    )
    success_chunks = [
        cl for cl in calculate_sts_batch.chunk_log
        if cl.get("success_count", 0) > 0
    ]
    assert len(success_chunks) == 3, (
        f"Expected 3 success chunks, got {len(success_chunks)}"
    )


# ---------------------------------------------------------------------------
# R-3: 10 consecutive endpoint failures → abort
# ---------------------------------------------------------------------------

def test_consecutive_endpoint_failures_trigger_abort():
    """STS_MAX_CONSECUTIVE_FAILURES endpoint-level chunks must trigger abort."""
    n = 15  # more rows than needed; abort should leave some unqueried

    async def always_unreachable(sts_input):
        raise StsEndpointUnreachableError("endpoint down")

    with _isolated_batch(), \
         patch("sts_calculator._query_sts_ws", new=AsyncMock(side_effect=always_unreachable)):
        calculate_sts_batch(_rows(n), patient_ids=_pids(n), chunk_size=1)

    assert calculate_sts_batch._batch_aborted, "Batch should have been aborted"
    assert calculate_sts_batch._abort_before_query_count > 0

    ehs = calculate_sts_batch.endpoint_health_summary
    assert ehs.get("abort_reason") == "consecutive_failures"
    assert ehs.get("abort_endpoint_failures") == sts_calculator.STS_MAX_CONSECUTIVE_FAILURES

    counted = [cl for cl in calculate_sts_batch.chunk_log if cl.get("counted_toward_abort")]
    assert len(counted) == sts_calculator.STS_MAX_CONSECUTIVE_FAILURES, (
        f"Expected {sts_calculator.STS_MAX_CONSECUTIVE_FAILURES} chunks counted toward abort, "
        f"got {len(counted)}"
    )


# ---------------------------------------------------------------------------
# R-4: Empty-response failures do NOT count toward abort
# ---------------------------------------------------------------------------

def test_empty_response_failures_do_not_trigger_abort():
    """StsEmptyResponseError = endpoint reachable.
    More than STS_MAX_CONSECUTIVE_FAILURES empty-response failures must NOT abort."""
    n = sts_calculator.STS_MAX_CONSECUTIVE_FAILURES + 5

    async def always_empty(sts_input):
        raise StsEmptyResponseError("80 messages, no predmort")

    with _isolated_batch(), \
         patch("sts_calculator._query_sts_ws", new=AsyncMock(side_effect=always_empty)):
        calculate_sts_batch(_rows(n), patient_ids=_pids(n), chunk_size=1)

    assert not calculate_sts_batch._batch_aborted, (
        "Batch must NOT abort on empty-response failures — endpoint is reachable"
    )
    assert calculate_sts_batch._abort_before_query_count == 0

    counted = [cl for cl in calculate_sts_batch.chunk_log if cl.get("counted_toward_abort")]
    assert counted == [], "No chunks should count toward abort for empty-response failures"

    failure_types = {cl.get("failure_type") for cl in calculate_sts_batch.chunk_log}
    assert failure_types == {"empty_response"}, (
        f"Expected only 'empty_response' failure type, got: {failure_types}"
    )


# ---------------------------------------------------------------------------
# R-5: Parse failure classified correctly, does not count toward abort
# ---------------------------------------------------------------------------

def test_parse_failure_classified_and_not_counted():
    """StsParseError: endpoint reachable, must not count toward abort."""
    n = 5

    async def always_parse_error(sts_input):
        raise StsParseError("html_parse_failure: keys=['other_field']")

    with _isolated_batch(), \
         patch("sts_calculator._query_sts_ws", new=AsyncMock(side_effect=always_parse_error)):
        calculate_sts_batch(_rows(n), patient_ids=_pids(n), chunk_size=1)

    assert not calculate_sts_batch._batch_aborted

    for cl in calculate_sts_batch.chunk_log:
        assert cl.get("failure_type") == "parse_error", (
            f"Expected failure_type='parse_error', got {cl.get('failure_type')!r}"
        )
        assert not cl.get("counted_toward_abort"), "Parse errors must not count toward abort"
        assert cl.get("endpoint_failure_count", 0) == 0


# ---------------------------------------------------------------------------
# R-6: Rows after abort marked batch_abort in failure_log
# ---------------------------------------------------------------------------

def test_rows_after_abort_marked_batch_abort():
    """Rows not reached before abort must appear in failure_log stage='batch_abort'."""
    n = 20

    async def always_unreachable(sts_input):
        raise StsEndpointUnreachableError("connection refused")

    with _isolated_batch(), \
         patch("sts_calculator._query_sts_ws", new=AsyncMock(side_effect=always_unreachable)):
        calculate_sts_batch(_rows(n), patient_ids=_pids(n), chunk_size=1)

    assert calculate_sts_batch._batch_aborted

    batch_abort_entries = [
        fl for fl in calculate_sts_batch.failure_log
        if fl.get("stage") == "batch_abort"
    ]
    assert len(batch_abort_entries) > 0, (
        "At least some rows should be marked batch_abort after abort"
    )
    for fl in batch_abort_entries:
        reason = fl.get("reason", "").lower()
        assert "abort" in reason or "not queried" in reason, (
            f"batch_abort reason should mention abort: {fl['reason']!r}"
        )


# ---------------------------------------------------------------------------
# R-7: chunk_log entries contain all required fields
# ---------------------------------------------------------------------------

def test_chunk_log_contains_required_fields():
    """Every chunk_log entry must contain the fields added by this fix."""
    n = 4
    call_count = [0]

    async def alternating(sts_input):
        call_count[0] += 1
        if call_count[0] % 2 == 0:
            return _good_result()
        raise StsEndpointUnreachableError("intermittent failure")

    with _isolated_batch(), \
         patch("sts_calculator._query_sts_ws", new=AsyncMock(side_effect=alternating)):
        calculate_sts_batch(_rows(n), patient_ids=_pids(n), chunk_size=1)

    required_fields = {
        "chunk_index", "row_count", "patient_ids",
        "success_count", "failure_count",
        "failure_type", "endpoint_failure_count", "counted_toward_abort",
        "exception_type", "exception_message", "aborted_after_this_chunk",
    }
    for i, cl in enumerate(calculate_sts_batch.chunk_log):
        missing = required_fields - cl.keys()
        assert not missing, f"chunk_log[{i}] missing fields: {missing}"

    # Alternating: chunks 0,2 fail (endpoint), chunks 1,3 succeed.
    for i, cl in enumerate(calculate_sts_batch.chunk_log):
        if i % 2 == 0:  # 0-indexed: attempts 1,3 of call_count
            assert cl["failure_type"] == "endpoint" or cl["success_count"] > 0
        else:
            assert cl["success_count"] > 0 or cl["failure_type"] is not None


# ---------------------------------------------------------------------------
# R-8: endpoint_health_summary has correct counts
# ---------------------------------------------------------------------------

def test_endpoint_health_summary_correct_counts():
    """endpoint_health_summary must reflect actual batch execution accurately."""
    n = 5  # patients 0–4; patients 1 and 3 always fail

    async def mixed(sts_input):
        # Patient indices 1 and 3 always fail (keyed on age).
        if _patient_idx(sts_input) in (1, 3):
            raise StsEndpointUnreachableError("transient failure")
        return _good_result()

    with _isolated_batch(), \
         patch("sts_calculator._query_sts_ws", new=AsyncMock(side_effect=mixed)):
        calculate_sts_batch(_rows(n), patient_ids=_pids(n), chunk_size=1)

    ehs = calculate_sts_batch.endpoint_health_summary
    assert ehs["n_eligible_for_fetch"] == n, (
        f"Expected {n} eligible for fetch, got {ehs['n_eligible_for_fetch']}"
    )
    assert ehs["n_queried"] == n, (
        f"All rows should have been queried (no abort), got {ehs['n_queried']}"
    )
    assert ehs["n_queried_with_score"] == 3, (
        f"Expected 3 with score (patients 0, 2, 4), got {ehs['n_queried_with_score']}"
    )
    assert ehs["n_chunks_endpoint_failure"] == 2, (
        f"Expected 2 endpoint-failure chunks (patients 1, 3), got {ehs['n_chunks_endpoint_failure']}"
    )
    assert ehs["n_rows_unqueried"] == 0
    assert ehs["abort_reason"] is None


# ---------------------------------------------------------------------------
# R-9: build_input failure does not reach Phase B / endpoint counter
# ---------------------------------------------------------------------------

def test_build_input_failure_not_endpoint_failure():
    """A row failing Phase A (invalid age) never reaches Phase B.
    It must not appear in chunk_log endpoint_failure_count."""
    bad_row = {
        "surgery_pre": "CABG",
        "surgical_priority": "Elective",
        "age_years": "0",   # fails validate_sts_input (age < 1)
        "sex": "M",
    }

    # No cache isolation needed — row fails before hashing
    calculate_sts_batch([bad_row], patient_ids=["BAD"])

    assert len(calculate_sts_batch.failure_log) == 1
    assert calculate_sts_batch.failure_log[0]["stage"] == "build_input"

    total_endpoint = sum(
        cl.get("endpoint_failure_count", 0)
        for cl in calculate_sts_batch.chunk_log
    )
    assert total_endpoint == 0, (
        "build_input failure must not contribute to endpoint_failure_count"
    )
    assert not calculate_sts_batch._batch_aborted


# ---------------------------------------------------------------------------
# R-10: Per-patient timeout classified as endpoint-level failure
# ---------------------------------------------------------------------------

def test_per_patient_timeout_classified_as_endpoint():
    """asyncio.TimeoutError in _calc_one_bounded must yield endpoint-level
    classification in chunk_log (endpoint_failure_count > 0)."""
    n = 2

    async def slow_query(sts_input):
        # Real sleep — will be cancelled by the per-patient timeout.
        await asyncio.sleep(60)
        return _good_result()

    # Use per-patient timeout of 0.05 s (50 ms) to trigger _calc_one_bounded.
    # No retry-delay mock since _calc_one is cancelled before any retry.
    # Do NOT mock asyncio.sleep globally — the real sleep is needed so the
    # timeout fires against actual waiting.
    with (
        patch("sts_calculator._sts_cache.load_entry", return_value=None),
        patch("sts_calculator._sts_cache.persist_fresh_result"),
        patch("sts_calculator._sts_cache.remember_patient_hash"),
        patch("sts_calculator._sts_cache.find_stale_fallback", return_value=None),
        patch.object(sts_calculator, "STS_CONSECUTIVE_FAILURE_BACKOFF_BASE_S", 0),
        patch.object(sts_calculator, "STS_PER_PATIENT_TIMEOUT_S", 0.05),
        patch("sts_calculator._query_sts_ws", new=AsyncMock(side_effect=slow_query)),
    ):
        sts_calculator._sts_memory_cache.clear()
        calculate_sts_batch(_rows(n), patient_ids=_pids(n), chunk_size=1)
        sts_calculator._sts_memory_cache.clear()

    # Per-patient timeout is endpoint-level: endpoint_failure_count > 0
    endpoint_chunks = [
        cl for cl in calculate_sts_batch.chunk_log
        if cl.get("endpoint_failure_count", 0) > 0
    ]
    assert len(endpoint_chunks) == n, (
        f"Expected {n} endpoint-failure chunks (per-patient timeouts), "
        f"got {len(endpoint_chunks)}"
    )
    # They should be counted toward the abort counter
    counted = [cl for cl in calculate_sts_batch.chunk_log if cl.get("counted_toward_abort")]
    assert len(counted) == n
