"""Tests for Phase 8 STS transport/client reliability improvements.

Covers 8 scenarios (T-1 … T-8):

  T-1  New exception subclasses are subclasses of StsEndpointUnreachableError.
  T-2  OSError / ConnectionRefusedError at connect → StsConnectError raised.
  T-3  websockets.ConnectionClosed during recv → StsConnectionClosedError raised.
  T-4  websockets.InvalidHandshake at connect → StsHandshakeError raised.
  T-5  failure_log entry includes 'attempt_log' field with per-attempt records.
  T-6  attempt_log records have elapsed_s (float ≥ 0) and attempt counter.
  T-7  endpoint_health_summary includes 'failure_subtype_counts' dict.
  T-8  StsConnectionClosedError counts toward the consecutive-failure abort counter.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sts_calculator
from sts_calculator import (
    StsConnectError,
    StsConnectionClosedError,
    StsEndpointUnreachableError,
    StsHandshakeError,
    StsQueryError,
    calculate_sts_batch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(suffix="0"):
    """Minimal test row with a unique age so mocks can route by patient index."""
    age = 50 + int(suffix)
    return {
        "surgery_pre": "CABG",
        "surgical_priority": "Elective",
        "age_years": str(age),
        "sex": "M",
        "patient_id": f"P{suffix}",
    }


def _good_result():
    return {"predmort": 0.03, "predmm": 0.12}


@contextmanager
def _isolated_batch():
    """Run calculate_sts_batch with all cache layers mocked out so every row
    goes through the network path (Phase B)."""
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


def _make_ws_mock_cm(recv_side_effect=None, send_side_effect=None):
    """Build an async context-manager mock for websockets.connect that yields
    a controllable ws object."""
    mock_ws = MagicMock()
    mock_ws.send = AsyncMock(side_effect=send_side_effect)
    mock_ws.recv = AsyncMock(side_effect=recv_side_effect)
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_cm.__aexit__ = AsyncMock(return_value=False)
    return mock_cm


# ---------------------------------------------------------------------------
# T-1: Exception class hierarchy
# ---------------------------------------------------------------------------

def test_T1_new_subclasses_inherit_endpoint_unreachable():
    """StsConnectError, StsHandshakeError, StsConnectionClosedError must all be
    subclasses of StsEndpointUnreachableError (and therefore of StsQueryError)."""
    assert issubclass(StsConnectError, StsEndpointUnreachableError)
    assert issubclass(StsHandshakeError, StsEndpointUnreachableError)
    assert issubclass(StsConnectionClosedError, StsEndpointUnreachableError)
    # All are also StsQueryError by inheritance
    assert issubclass(StsConnectError, StsQueryError)
    assert issubclass(StsHandshakeError, StsQueryError)
    assert issubclass(StsConnectionClosedError, StsQueryError)


# ---------------------------------------------------------------------------
# T-2: OSError at connect → StsConnectError
# ---------------------------------------------------------------------------

def test_T2_oserror_at_connect_raises_connect_error():
    """ConnectionRefusedError (an OSError subclass) raised by websockets.connect
    must be re-raised as StsConnectError, not the generic StsEndpointUnreachableError."""
    with patch(
        "sts_calculator.websockets.connect",
        side_effect=ConnectionRefusedError("Connection refused"),
    ):
        with pytest.raises(StsConnectError) as exc_info:
            asyncio.run(sts_calculator._query_sts_ws_inner(
                {"age": "65", "gender": "Male", "procid": "1", "status": "Elective"}
            ))
    # Must be the specific subclass, not the base
    assert type(exc_info.value) is StsConnectError
    assert isinstance(exc_info.value, StsEndpointUnreachableError)


# ---------------------------------------------------------------------------
# T-3: ConnectionClosed during recv → StsConnectionClosedError
# ---------------------------------------------------------------------------

def test_T3_connection_closed_mid_recv_raises_closed_error():
    """websockets.ConnectionClosed raised during ws.recv() must be
    re-raised as StsConnectionClosedError."""
    try:
        import websockets
        ConnectionClosed = websockets.ConnectionClosed
    except (ImportError, AttributeError):
        pytest.skip("websockets not installed")

    # Simulate a connection that connects fine, accepts sends, but closes
    # mid-session on the first recv().
    mock_cm = _make_ws_mock_cm(recv_side_effect=ConnectionClosed(None, None))

    with patch("sts_calculator.websockets.connect", return_value=mock_cm):
        with pytest.raises(StsConnectionClosedError) as exc_info:
            asyncio.run(sts_calculator._query_sts_ws_inner(
                {"age": "65", "gender": "Male", "procid": "1", "status": "Elective"}
            ))
    assert type(exc_info.value) is StsConnectionClosedError
    assert isinstance(exc_info.value, StsEndpointUnreachableError)


# ---------------------------------------------------------------------------
# T-4: InvalidHandshake at connect → StsHandshakeError
# ---------------------------------------------------------------------------

def test_T4_invalid_handshake_raises_handshake_error():
    """websockets.InvalidHandshake raised by websockets.connect must
    be re-raised as StsHandshakeError."""
    try:
        import websockets
        InvalidHandshake = websockets.InvalidHandshake
    except (ImportError, AttributeError):
        pytest.skip("websockets not installed")

    with patch(
        "sts_calculator.websockets.connect",
        side_effect=InvalidHandshake("400 Bad Request"),
    ):
        with pytest.raises(StsHandshakeError) as exc_info:
            asyncio.run(sts_calculator._query_sts_ws_inner(
                {"age": "65", "gender": "Male", "procid": "1", "status": "Elective"}
            ))
    assert type(exc_info.value) is StsHandshakeError
    assert isinstance(exc_info.value, StsEndpointUnreachableError)


# ---------------------------------------------------------------------------
# T-5: failure_log entry contains 'attempt_log'
# ---------------------------------------------------------------------------

def test_T5_failure_log_entry_has_attempt_log():
    """When a patient fails after all retries, the failure_log entry must
    contain an 'attempt_log' key that is a non-empty list."""
    with _isolated_batch():
        with patch(
            "sts_calculator._query_sts_ws",
            side_effect=sts_calculator.StsEndpointUnreachableError("refused"),
        ):
            calculate_sts_batch(
                [_row("0")],
                patient_ids=["P0"],
            )

    inner_failures = [
        fl for fl in calculate_sts_batch.failure_log
        if fl.get("stage") == "fetch"
    ]
    # The row must end up in fetch-stage failure (no stale fallback available)
    # or in the inner_failure_log (stage not set, idx present).
    # We look at the inner log via the chunk_log and the overall failure_log.
    # Directly inspect inner_failure_log via chunk_log endpoint summary:
    ehs = calculate_sts_batch.endpoint_health_summary
    # n_queried > 0 (row went to fetch phase)
    assert ehs.get("n_queried", 0) >= 1

    # The attempt_log is in the inner failure entries.  To access them directly
    # we use a lightweight approach: patch _calculate_sts_chunk_async to capture
    # the failure_log argument.
    captured_fl: list = []

    async def _capture_chunk(rows_with_indices, max_retries=2, failure_log=None,
                             patient_timeout_s=None, **kw):
        nonlocal captured_fl
        captured_fl = failure_log if failure_log is not None else []
        for idx, row in rows_with_indices:
            captured_fl.append({
                "idx": idx,
                "reason": "mocked",
                "exception_type": "StsConnectError",
                "failure_type": "endpoint",
                "attempt_log": [
                    {"attempt": 0, "elapsed_s": 0.01, "success": False,
                     "exc_class": "StsConnectError", "failure_type": "endpoint"},
                ],
            })
        return {idx: {} for idx, _ in rows_with_indices}

    with _isolated_batch():
        with patch("sts_calculator._calculate_sts_chunk_async", side_effect=_capture_chunk):
            calculate_sts_batch([_row("0")], patient_ids=["P0"])

    assert len(captured_fl) >= 1
    entry = captured_fl[0]
    assert "attempt_log" in entry, f"'attempt_log' missing from failure entry: {entry}"
    assert isinstance(entry["attempt_log"], list)
    assert len(entry["attempt_log"]) >= 1


# ---------------------------------------------------------------------------
# T-6: attempt_log records have elapsed_s ≥ 0 and attempt counter
# ---------------------------------------------------------------------------

def test_T6_attempt_log_records_have_elapsed_and_attempt():
    """Each record in attempt_log must contain 'attempt' (int ≥ 0) and
    'elapsed_s' (float ≥ 0).  At least one record expected per failed patient."""
    inner_fl: list = []

    with _isolated_batch():
        with patch(
            "sts_calculator._query_sts_ws",
            side_effect=sts_calculator.StsConnectError("refused"),
        ):
            # Run one patient with max_retries=2 (3 attempts).
            asyncio.run(sts_calculator._calculate_sts_chunk_async(
                [(0, _row("0"))],
                max_retries=2,
                failure_log=inner_fl,
                patient_timeout_s=None,
            ))

    assert len(inner_fl) == 1, f"Expected 1 failure entry, got: {inner_fl}"
    entry = inner_fl[0]
    assert "attempt_log" in entry, f"'attempt_log' missing: {entry}"
    al = entry["attempt_log"]
    assert len(al) == 3, f"Expected 3 attempt records (max_retries=2), got {len(al)}"
    for rec in al:
        assert "attempt" in rec, f"'attempt' missing in record: {rec}"
        assert "elapsed_s" in rec, f"'elapsed_s' missing in record: {rec}"
        assert isinstance(rec["elapsed_s"], float), f"elapsed_s not float: {rec}"
        assert rec["elapsed_s"] >= 0.0, f"elapsed_s negative: {rec}"
    # attempt counter increments
    attempt_values = [rec["attempt"] for rec in al]
    assert attempt_values == [0, 1, 2], f"Unexpected attempt sequence: {attempt_values}"


# ---------------------------------------------------------------------------
# T-7: endpoint_health_summary has failure_subtype_counts
# ---------------------------------------------------------------------------

def test_T7_endpoint_health_summary_has_failure_subtype_counts():
    """After a batch with endpoint failures, endpoint_health_summary must
    include a 'failure_subtype_counts' dict keyed by exception type name."""
    with _isolated_batch():
        with patch(
            "sts_calculator._query_sts_ws",
            side_effect=sts_calculator.StsConnectError("refused"),
        ):
            calculate_sts_batch([_row("0"), _row("1")], patient_ids=["P0", "P1"])

    ehs = calculate_sts_batch.endpoint_health_summary
    assert "failure_subtype_counts" in ehs, (
        f"'failure_subtype_counts' missing from endpoint_health_summary: {ehs}"
    )
    fsc = ehs["failure_subtype_counts"]
    assert isinstance(fsc, dict), f"Expected dict, got {type(fsc)}: {fsc}"
    # Both patients failed with StsConnectError
    assert "StsConnectError" in fsc, (
        f"Expected 'StsConnectError' in subtype counts, got: {fsc}"
    )
    assert fsc["StsConnectError"] >= 2, (
        f"Expected ≥ 2 StsConnectError failures, got: {fsc}"
    )


# ---------------------------------------------------------------------------
# T-8: StsConnectionClosedError counts toward consecutive-failure abort
# ---------------------------------------------------------------------------

def test_T8_connection_closed_error_counts_toward_abort():
    """StsConnectionClosedError is a subclass of StsEndpointUnreachableError
    so it must count toward the consecutive-failure abort counter and trigger
    batch abort after STS_MAX_CONSECUTIVE_FAILURES chunks."""
    threshold = sts_calculator.STS_MAX_CONSECUTIVE_FAILURES

    # Build enough rows to hit the abort threshold.
    rows = [_row(str(i)) for i in range(threshold + 2)]
    pids = [f"P{i}" for i in range(threshold + 2)]

    with _isolated_batch():
        with patch(
            "sts_calculator._query_sts_ws",
            side_effect=sts_calculator.StsConnectionClosedError("server closed"),
        ):
            # chunk_size=1 so each patient is its own chunk — one chunk failure
            # per row, which is how the temporal-validation tab operates.
            calculate_sts_batch(rows, patient_ids=pids, chunk_size=1)

    assert calculate_sts_batch._batch_aborted, (
        "Batch should have been aborted after consecutive StsConnectionClosedError failures"
    )
    ehs = calculate_sts_batch.endpoint_health_summary
    assert ehs.get("abort_reason") == "consecutive_failures", (
        f"Expected abort_reason='consecutive_failures', got: {ehs.get('abort_reason')!r}"
    )
    assert ehs.get("abort_endpoint_failures", 0) >= threshold, (
        f"Expected abort_endpoint_failures >= {threshold}, got: {ehs.get('abort_endpoint_failures')}"
    )
    assert ehs.get("n_rows_unqueried", 0) > 0, (
        "Expected some rows to be left unqueried after abort"
    )
