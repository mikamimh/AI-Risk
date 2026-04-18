"""STS batch pipeline audit and robustness tests.

These tests cover the auditability and abort-handling improvements to
``calculate_sts_batch`` in ``sts_calculator.py``.  All network I/O is
mocked: ``_run_async`` is patched so the async WebSocket path never fires.

Scenarios tested (per task requirements H-1 … H-8):

  H-1  unsupported STS procedure → classify_sts_eligibility returns not_supported
  H-2  supported row, input build fails → build_input failure record
  H-3  supported row, query attempted, endpoint failure → fetch_failed record
  H-4  supported row, query succeeds, parse fails → response_validation_failure
  H-5  mixed batch: early success, later repeated failures, then batch abort
  H-6  rows after abort are marked batch_abort with explicit reason
  H-7  partial successful STS results survive despite later abort
  H-8  execution summary counts are consistent with patient-level failure log

Tests are independent of Streamlit and make no network calls.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
from unittest.mock import patch, MagicMock

import pytest

import sts_calculator as _sts_mod
from sts_calculator import (
    calculate_sts_batch,
    classify_sts_eligibility,
    STS_MAX_CONSECUTIVE_FAILURES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patch_run_async(side_effect):
    """Context manager that patches ``_run_async`` with *side_effect* while
    ensuring any coroutine object passed to the mock is closed immediately
    (preventing ``RuntimeWarning: coroutine … was never awaited`` from GC)."""
    def _closing_wrapper(coro):
        if inspect.iscoroutine(coro):
            coro.close()
        return side_effect(coro)

    with patch("sts_calculator._run_async", side_effect=_closing_wrapper):
        yield


def _row(name="Patient A", surgery="CABG"):
    """Minimal patient row for the STS pipeline."""
    return {
        "Name": name,
        "Surgery": surgery,
        "surgery_pre": surgery,
        "age_years": 65,
        "sex": "M",
        "surgical_priority": "Elective",
    }


def _make_batch(rows: list, *, run_async_side_effect):
    """Run calculate_sts_batch with ``_run_async`` patched to the given
    side_effect callable.  Returns the batch result list."""
    with _patch_run_async(run_async_side_effect):
        results = calculate_sts_batch(
            rows,
            patient_ids=[r["Name"] for r in rows],
            chunk_size=1,
        )
    return results


def _fresh_result():
    """A minimal valid STS result dict (passes ``is_valid_result``)."""
    return {"predmort": 0.04, "predmm": 0.12}


# ---------------------------------------------------------------------------
# Fixture: isolate cache state and eliminate backoff sleeps
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_caches():
    """Prevent cross-test cache pollution and suppress the progressive backoff
    sleep so test runs finish in seconds instead of 15+ minutes.

    Specifically:
      * clears ``_sts_memory_cache`` before and after each test so successful
        results from one test don't cause Phase A cache-hits in later tests;
      * patches ``sts_cache.load_entry`` → ``None``    (no disk reads);
      * patches ``sts_cache.persist_fresh_result``     (no disk writes);
      * patches ``sts_cache.remember_patient_hash``    (no disk writes);
      * patches ``sts_cache.find_stale_fallback``      → ``None``;
      * patches ``time.sleep``                         (no backoff delay).
    """
    _sts_mod._sts_memory_cache.clear()
    with (
        patch("sts_cache.load_entry", return_value=None),
        patch("sts_cache.persist_fresh_result"),
        patch("sts_cache.remember_patient_hash"),
        patch("sts_cache.find_stale_fallback", return_value=None),
        patch("time.sleep"),
    ):
        yield
    _sts_mod._sts_memory_cache.clear()


# ---------------------------------------------------------------------------
# H-1: unsupported STS procedure
# ---------------------------------------------------------------------------

def test_classify_sts_eligibility_unsupported_bentall():
    """Bentall / aortic-root procedures must be classified as not_supported."""
    status, reason = classify_sts_eligibility({"surgery_pre": "Bentall Procedure"})
    assert status == "not_supported"
    assert "STS ACSD scope" in reason


def test_classify_sts_eligibility_unsupported_dissection():
    status, reason = classify_sts_eligibility({"surgery_pre": "Aortic Dissection Repair"})
    assert status == "not_supported"


def test_classify_sts_eligibility_supported_cabg():
    status, _ = classify_sts_eligibility({"surgery_pre": "CABG"})
    assert status == "supported"


def test_classify_sts_eligibility_uncertain_empty():
    status, _ = classify_sts_eligibility({"surgery_pre": ""})
    assert status == "uncertain"


# ---------------------------------------------------------------------------
# H-2: supported row, STS input build fails
# ---------------------------------------------------------------------------

def test_batch_build_input_failure_recorded():
    """If ``build_sts_input_from_row`` raises, the failure log must capture it
    as a build_input stage failure, not a fetch failure."""
    rows = [_row()]

    with patch("sts_calculator.build_sts_input_from_row", side_effect=ValueError("bad field")):
        _ = calculate_sts_batch(rows, patient_ids=["P1"], chunk_size=1)

    fail_log = calculate_sts_batch.failure_log
    assert len(fail_log) == 1
    assert fail_log[0]["stage"] == "build_input"
    assert "bad field" in fail_log[0]["reason"]
    assert fail_log[0]["retry_attempted"] is False


# ---------------------------------------------------------------------------
# H-3: supported row, query attempted, endpoint failure
# ---------------------------------------------------------------------------

def test_batch_fetch_failure_recorded():
    """When _run_async returns an empty dict (endpoint unreachable), the row
    must appear in the failure_log as a fetch-stage failure with
    retry_attempted=True."""

    def _empty_result(coro):
        # Return an empty result dict (predmort absent) → fetch_failed path
        return {0: {}}

    results = _make_batch([_row()], run_async_side_effect=_empty_result)

    assert results == [{}]
    fail_log = calculate_sts_batch.failure_log
    assert len(fail_log) == 1
    assert fail_log[0]["stage"] == "fetch"
    assert "fetch_failed" in fail_log[0]["reason"]
    assert fail_log[0]["retry_attempted"] is True


# ---------------------------------------------------------------------------
# H-4: supported row, query succeeds, parse fails
# ---------------------------------------------------------------------------

def test_batch_response_validation_failure_recorded():
    """When _run_async returns a non-empty dict lacking 'predmort', the row is
    classified as response_validation_failure."""

    def _bad_parse(coro):
        # Has something but predmort is absent → is_valid_result returns False
        return {0: {"predmm": 0.12}}

    results = _make_batch([_row()], run_async_side_effect=_bad_parse)

    assert results == [{}]
    fail_log = calculate_sts_batch.failure_log
    assert len(fail_log) == 1
    assert "response_validation_failure" in fail_log[0]["reason"]


# ---------------------------------------------------------------------------
# H-5 + H-6: mixed batch — early success, later repeated failures → abort
# ---------------------------------------------------------------------------

def _make_mixed_run_async(n_rows: int, success_indices: set):
    """Factory for _run_async side effects that succeed only for the
    given 0-based chunk indices and fail the rest.

    With chunk_size=1 the N-th call processes local_i=N, so the returned
    dict must use ``idx`` (not 0) as the key — Phase C looks up results by
    their global ``local_i``.
    """
    _call_count = [-1]

    def _side_effect(coro):
        _call_count[0] += 1
        idx = _call_count[0]
        if idx in success_indices:
            return {idx: _fresh_result()}
        return {idx: {}}  # failure

    return _side_effect


def test_batch_abort_after_consecutive_failures():
    """After STS_MAX_CONSECUTIVE_FAILURES consecutive chunk failures the
    batch must abort with _batch_aborted=True."""
    n = STS_MAX_CONSECUTIVE_FAILURES + 3   # a few more rows than the abort threshold
    rows = [_row(name=f"P{i}") for i in range(n)]

    # No row ever succeeds → abort must trigger at chunk = STS_MAX_CONSECUTIVE_FAILURES
    def _all_fail(coro):
        return {0: {}}

    with _patch_run_async(_all_fail):
        results = calculate_sts_batch(
            rows,
            patient_ids=[r["Name"] for r in rows],
            chunk_size=1,
        )

    assert calculate_sts_batch._batch_aborted is True


def test_batch_abort_marks_unqueried_rows_as_batch_abort():
    """Rows that were never reached due to batch abort must have
    stage='batch_abort' in the failure log, NOT 'fetch'."""
    n = STS_MAX_CONSECUTIVE_FAILURES + 3
    rows = [_row(name=f"P{i}") for i in range(n)]

    def _all_fail(coro):
        return {0: {}}

    with _patch_run_async(_all_fail):
        calculate_sts_batch(rows, patient_ids=[r["Name"] for r in rows], chunk_size=1)

    abort_entries = [f for f in calculate_sts_batch.failure_log if f["stage"] == "batch_abort"]
    assert len(abort_entries) > 0, "No batch_abort entries found in failure_log"
    for entry in abort_entries:
        assert entry["retry_attempted"] is False
        assert "consecutive" in entry["reason"].lower() or "aborted" in entry["reason"].lower()


def test_batch_abort_count_exposed():
    """_abort_before_query_count must equal the number of rows that were
    never queried due to the abort."""
    n = STS_MAX_CONSECUTIVE_FAILURES + 3
    rows = [_row(name=f"P{i}") for i in range(n)]

    def _all_fail(coro):
        return {0: {}}

    with _patch_run_async(_all_fail):
        calculate_sts_batch(rows, patient_ids=[r["Name"] for r in rows], chunk_size=1)

    abort_count = calculate_sts_batch._abort_before_query_count
    abort_in_log = sum(1 for f in calculate_sts_batch.failure_log if f["stage"] == "batch_abort")
    assert abort_count == abort_in_log
    assert abort_count > 0


# ---------------------------------------------------------------------------
# H-7: partial successful STS results survive despite later abort
# ---------------------------------------------------------------------------

def test_partial_results_survive_abort():
    """Early successes must be preserved in the result list even when the
    batch aborts on later rows.  Only the unqueried tail should be empty."""
    n_success = 2
    n_total = STS_MAX_CONSECUTIVE_FAILURES + n_success + 1
    rows = [_row(name=f"P{i}") for i in range(n_total)]
    # First n_success rows succeed; the rest fail → eventually abort
    success_set = set(range(n_success))

    results = _make_batch(rows, run_async_side_effect=_make_mixed_run_async(n_total, success_set))

    # The first n_success rows must have a non-empty result with predmort
    for i in range(n_success):
        assert results[i].get("predmort") is not None, (
            f"Row {i} should have been a success but got {results[i]}"
        )
    # The batch must have aborted
    assert calculate_sts_batch._batch_aborted is True
    # At least some rows must be empty (aborted)
    n_empty = sum(1 for r in results if not r)
    assert n_empty > 0


# ---------------------------------------------------------------------------
# H-8: execution summary counts consistent with failure log
# ---------------------------------------------------------------------------

def test_execution_log_and_failure_log_are_consistent():
    """Total entries in execution_log must cover every row; entries marked
    'failed' in execution_log must match entries in failure_log (plus
    batch_abort rows are also in failure_log)."""
    n = STS_MAX_CONSECUTIVE_FAILURES + 2
    rows = [_row(name=f"P{i}") for i in range(n)]

    def _all_fail(coro):
        return {0: {}}

    with _patch_run_async(_all_fail):
        calculate_sts_batch(rows, patient_ids=[r["Name"] for r in rows], chunk_size=1)

    exec_log = calculate_sts_batch.last_execution_log
    fail_log = calculate_sts_batch.failure_log

    # Every row must appear in exactly one execution record.
    assert len(exec_log) == len(rows)

    # All execution_log entries with status='failed' must have a corresponding
    # entry in failure_log.
    failed_in_exec = sum(1 for r in exec_log if getattr(r, "status", None) == "failed")
    assert failed_in_exec == len(fail_log)

    # batch_abort entries must be a subset of failure_log.
    abort_in_fail = [f for f in fail_log if f.get("stage") == "batch_abort"]
    assert len(abort_in_fail) == calculate_sts_batch._abort_before_query_count


# ---------------------------------------------------------------------------
# Chunk log structure
# ---------------------------------------------------------------------------

def test_chunk_log_populated_after_batch():
    """chunk_log must contain one entry per chunk processed."""
    n = 3
    rows = [_row(name=f"P{i}") for i in range(n)]
    # All succeed
    def _all_succeed(coro):
        return {0: _fresh_result()}

    with _patch_run_async(_all_succeed):
        calculate_sts_batch(rows, patient_ids=[r["Name"] for r in rows], chunk_size=1)

    chunk_log = calculate_sts_batch.chunk_log
    assert len(chunk_log) == n  # chunk_size=1 → one chunk per row
    for entry in chunk_log:
        assert "chunk_index" in entry
        assert "success_count" in entry
        assert "failure_count" in entry
        assert "aborted_after_this_chunk" in entry


def test_chunk_log_marks_abort_chunk():
    """The chunk that triggers the abort must have aborted_after_this_chunk=True."""
    n = STS_MAX_CONSECUTIVE_FAILURES + 2
    rows = [_row(name=f"P{i}") for i in range(n)]

    def _all_fail(coro):
        return {0: {}}

    with _patch_run_async(_all_fail):
        calculate_sts_batch(rows, patient_ids=[r["Name"] for r in rows], chunk_size=1)

    abort_chunks = [c for c in calculate_sts_batch.chunk_log if c.get("aborted_after_this_chunk")]
    assert len(abort_chunks) == 1
    assert abort_chunks[0]["chunk_index"] == STS_MAX_CONSECUTIVE_FAILURES - 1


def test_attributes_reset_between_calls():
    """Calling calculate_sts_batch twice must produce independent state."""
    rows = [_row()]

    def _succeed(coro):
        return {0: _fresh_result()}

    def _fail(coro):
        return {0: {}}

    with _patch_run_async(_succeed):
        calculate_sts_batch(rows, patient_ids=["P1"], chunk_size=1)
    first_fail_count = len(calculate_sts_batch.failure_log)

    # The first call wrote the result to ``_sts_memory_cache``.  Clear it so
    # the second call is not served from cache and actually hits Phase B.
    _sts_mod._sts_memory_cache.clear()

    with _patch_run_async(_fail):
        calculate_sts_batch(rows, patient_ids=["P1"], chunk_size=1)
    second_fail_count = len(calculate_sts_batch.failure_log)

    # First call succeeded, second failed — must differ and must not accumulate.
    assert first_fail_count == 0
    assert second_fail_count == 1
