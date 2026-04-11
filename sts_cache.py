"""Persistent cache and revalidation policy for STS Score calculations.

Phase 2 scope: make every STS Score calculation auditable and explainable
without changing its clinical behavior.

Policy (defensible and observable only — we do NOT claim to detect any
server-side or database changes inside the STS Score website):

    A cached STS Score entry is returned unchanged if ALL hold:
      1. the clinically relevant STS Score input fields hash to the same
         value as the cached entry's input hash
      2. the cached entry's integration_version matches the current
         STS_SCORE_INTEGRATION_VERSION
      3. the cached result dict passes validation (has a non-NaN predmort)
      4. the entry is within the TTL window (default 14 days)

    Otherwise the STS Score web calculator is queried again.  On failure,
    the cache layer will:
      - retry (max_retries attempts)
      - fall back to a still-present (expired) same-hash entry
      - fall back to the patient's previous-hash entry via the patient
        index (cross-hash fallback)
      - otherwise return an explicit "failed" record

Statuses exposed per execution:
    fresh          - no valid prior entry; fetched successfully
    cached         - valid entry within TTL returned unchanged
    refreshed      - TTL expired or entry invalid; re-fetched successfully
    stale_fallback - fetch failed; returned a previous valid entry
    failed         - fetch failed and no fallback was available

Stages (where execution failed, if applicable):
    build_input, cache_lookup, fetch, validate, done
"""

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from config import AppConfig

log = logging.getLogger("sts_score.cache")

# ---------------------------------------------------------------------------
# Configuration (sourced from AppConfig so changes live in one place)
# ---------------------------------------------------------------------------

STS_SCORE_CACHE_DIR: Path = AppConfig.STS_SCORE_CACHE_DIR
STS_SCORE_CACHE_TTL_SECONDS: int = int(AppConfig.STS_SCORE_CACHE_TTL_DAYS) * 86400
STS_SCORE_INTEGRATION_VERSION: str = AppConfig.STS_SCORE_INTEGRATION_VERSION

# Minimum required keys for a result to be considered a valid STS Score
# response. We only require predmort because the sub-scores can legitimately
# be missing for some procedures; predmort is the primary endpoint.
REQUIRED_RESULT_KEYS = {"predmort"}


def _ensure_cache_dir() -> Path:
    try:
        STS_SCORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log.warning(
            "STS Score cache dir could not be created at %s: %s",
            STS_SCORE_CACHE_DIR, e,
        )
    return STS_SCORE_CACHE_DIR


# ---------------------------------------------------------------------------
# Execution record — structured status for one STS Score calculation
# ---------------------------------------------------------------------------

@dataclass
class ExecutionRecord:
    """Structured record describing one STS Score calculation attempt."""
    status: str = "failed"               # fresh|cached|refreshed|stale_fallback|failed
    result: Dict[str, float] = field(default_factory=dict)
    patient_id: Optional[str] = None
    input_hash: Optional[str] = None
    integration_version: str = STS_SCORE_INTEGRATION_VERSION
    stage: str = "done"                  # build_input|cache_lookup|fetch|validate|done
    reason: str = ""                     # human-readable reason / error
    retry_attempted: bool = False
    used_previous_cache: bool = False
    cache_age_days: Optional[float] = None
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Input hashing
# ---------------------------------------------------------------------------

def compute_input_hash(sts_input: Dict[str, Any]) -> str:
    """Compute a stable hash of clinically relevant STS Score input fields.

    ``sts_input`` is expected to be the dict returned by
    ``sts_calculator.build_sts_input_from_row`` — the canonical set of
    fields that gets sent to the STS Score website.  Hashing that dict is
    therefore equivalent to hashing "the patient's STS Score inputs".

    Empty strings, None, and NaN values are normalized out so that two
    inputs that are semantically identical from the STS Score website's
    point of view produce the same hash.  The current integration version
    is mixed into the hash so bumping STS_SCORE_INTEGRATION_VERSION
    automatically invalidates every prior entry.
    """
    canonical: Dict[str, Any] = {}
    for k, v in sts_input.items():
        if v is None:
            continue
        if isinstance(v, float):
            if v != v:  # NaN
                continue
            canonical[k] = round(float(v), 6)
            continue
        s = str(v).strip()
        if s == "" or s.lower() in ("nan", "none", "-"):
            continue
        canonical[k] = s
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    blob_full = f"{STS_SCORE_INTEGRATION_VERSION}|{blob}"
    return hashlib.sha256(blob_full.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Per-entry storage
# ---------------------------------------------------------------------------

def _entry_path(input_hash: str) -> Path:
    return _ensure_cache_dir() / f"{input_hash}.json"


def _patient_index_path() -> Path:
    return _ensure_cache_dir() / "_patient_index.json"


def _atomic_write_json(path: Path, obj: Any) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_sts_", dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def load_entry(input_hash: str) -> Optional[Dict[str, Any]]:
    p = _entry_path(input_hash)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("STS Score cache entry %s unreadable: %s", p.name, e)
        return None


def save_entry(entry: Dict[str, Any]) -> None:
    input_hash = entry.get("input_hash")
    if not input_hash:
        return
    p = _entry_path(input_hash)
    try:
        _atomic_write_json(p, entry)
    except Exception as e:
        log.warning("STS Score cache entry %s could not be saved: %s", p.name, e)


def is_expired(entry: Dict[str, Any], now: Optional[float] = None) -> bool:
    if now is None:
        now = time.time()
    created_ts = entry.get("created_ts")
    if not isinstance(created_ts, (int, float)):
        return True
    return (now - float(created_ts)) > STS_SCORE_CACHE_TTL_SECONDS


def cache_age_days(entry: Dict[str, Any], now: Optional[float] = None) -> Optional[float]:
    if now is None:
        now = time.time()
    created_ts = entry.get("created_ts")
    if not isinstance(created_ts, (int, float)):
        return None
    return (now - float(created_ts)) / 86400.0


def is_valid_result(result: Any) -> bool:
    """Check that an STS Score result dict has the minimum expected structure."""
    if not isinstance(result, dict):
        return False
    for k in REQUIRED_RESULT_KEYS:
        v = result.get(k)
        if v is None:
            return False
        if isinstance(v, float) and v != v:
            return False
    return True


# ---------------------------------------------------------------------------
# Patient index (enables cross-hash stale fallback)
# ---------------------------------------------------------------------------

def _load_patient_index() -> Dict[str, str]:
    p = _patient_index_path()
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        log.warning("STS Score patient index unreadable: %s", e)
        return {}


def _save_patient_index(ix: Dict[str, str]) -> None:
    p = _patient_index_path()
    try:
        _atomic_write_json(p, ix)
    except Exception as e:
        log.warning("STS Score patient index could not be saved: %s", e)


def get_previous_hash_for_patient(patient_id: Optional[str]) -> Optional[str]:
    if not patient_id:
        return None
    return _load_patient_index().get(str(patient_id))


def remember_patient_hash(patient_id: Optional[str], input_hash: Optional[str]) -> None:
    if not patient_id or not input_hash:
        return
    ix = _load_patient_index()
    if ix.get(str(patient_id)) == input_hash:
        return
    ix[str(patient_id)] = input_hash
    _save_patient_index(ix)


# ---------------------------------------------------------------------------
# Helpers to build execution records consistently
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_cache_hit_record(
    entry: Dict[str, Any],
    patient_id: Optional[str],
    input_hash: str,
) -> ExecutionRecord:
    return ExecutionRecord(
        status="cached",
        result=dict(entry.get("result") or {}),
        patient_id=patient_id,
        input_hash=input_hash,
        integration_version=STS_SCORE_INTEGRATION_VERSION,
        stage="done",
        reason="cache_hit",
        cache_age_days=cache_age_days(entry),
        timestamp=_now_iso(),
    )


def make_fresh_record(
    result: Dict[str, float],
    patient_id: Optional[str],
    input_hash: str,
    refreshed: bool,
    retry_attempted: bool,
) -> ExecutionRecord:
    return ExecutionRecord(
        status="refreshed" if refreshed else "fresh",
        result=dict(result),
        patient_id=patient_id,
        input_hash=input_hash,
        integration_version=STS_SCORE_INTEGRATION_VERSION,
        stage="done",
        reason="cache_refresh" if refreshed else "cache_miss_fetched_ok",
        retry_attempted=retry_attempted,
        cache_age_days=0.0,
        timestamp=_now_iso(),
    )


def make_stale_fallback_record(
    fallback_entry: Dict[str, Any],
    patient_id: Optional[str],
    input_hash: str,
    reason: str,
    retry_attempted: bool,
) -> ExecutionRecord:
    return ExecutionRecord(
        status="stale_fallback",
        result=dict(fallback_entry.get("result") or {}),
        patient_id=patient_id,
        input_hash=input_hash,
        integration_version=STS_SCORE_INTEGRATION_VERSION,
        stage="fetch",
        reason=f"{reason}; returned previous cache entry",
        retry_attempted=retry_attempted,
        used_previous_cache=True,
        cache_age_days=cache_age_days(fallback_entry),
        timestamp=_now_iso(),
    )


def make_failed_record(
    patient_id: Optional[str],
    input_hash: Optional[str],
    stage: str,
    reason: str,
    retry_attempted: bool = False,
) -> ExecutionRecord:
    return ExecutionRecord(
        status="failed",
        result={},
        patient_id=patient_id,
        input_hash=input_hash,
        integration_version=STS_SCORE_INTEGRATION_VERSION,
        stage=stage,
        reason=reason,
        retry_attempted=retry_attempted,
        timestamp=_now_iso(),
    )


# ---------------------------------------------------------------------------
# Primary entry point (single patient, with fetch callback)
# ---------------------------------------------------------------------------

def persist_fresh_result(
    sts_input: Dict[str, Any],
    result: Dict[str, float],
    input_hash: str,
    patient_id: Optional[str],
) -> None:
    """Persist a freshly fetched STS Score result to the cache."""
    entry = {
        "input_hash": input_hash,
        "integration_version": STS_SCORE_INTEGRATION_VERSION,
        "sts_input": sts_input,
        "result": result,
        "created_ts": time.time(),
        "created_iso": _now_iso(),
        "patient_id": patient_id,
    }
    save_entry(entry)
    if patient_id:
        remember_patient_hash(patient_id, input_hash)


def find_stale_fallback(
    patient_id: Optional[str],
    input_hash: str,
    same_hash_entry: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Locate the best stale cache entry to return on fetch failure.

    Preference order:
        1. Same-hash expired/invalid entry (TTL fallback)
        2. Patient's previous-hash entry via the patient index
    """
    if same_hash_entry is not None and is_valid_result(same_hash_entry.get("result")):
        return same_hash_entry
    if patient_id:
        prev_hash = get_previous_hash_for_patient(patient_id)
        if prev_hash and prev_hash != input_hash:
            prev = load_entry(prev_hash)
            if prev is not None and is_valid_result(prev.get("result")):
                return prev
    return None


def get_cached_or_fetch(
    sts_input: Dict[str, Any],
    patient_id: Optional[str],
    fetch_func: Callable[[Dict[str, Any]], Dict[str, float]],
    max_retries: int = 2,
) -> ExecutionRecord:
    """Resolve one STS Score calculation using the cache + revalidation policy.

    ``fetch_func`` is a callable taking the canonical sts_input and
    returning a result dict (empty on failure).  The cache layer handles
    retries, validation, persistence, and stale fallback.
    """
    patient_id_str = str(patient_id) if patient_id is not None else None

    # --- Stage: build_input (hashing) ---
    try:
        input_hash = compute_input_hash(sts_input)
    except Exception as e:
        log.error("STS Score input hash failed patient=%s: %s", patient_id_str, e)
        return make_failed_record(
            patient_id_str, None, "build_input", f"input_hash_error: {e}"
        )

    # --- Stage: cache_lookup ---
    entry = load_entry(input_hash)
    entry_version_ok = (
        entry is not None
        and entry.get("integration_version") == STS_SCORE_INTEGRATION_VERSION
    )
    entry_result_ok = entry is not None and is_valid_result(entry.get("result"))

    if entry is not None and entry_version_ok and entry_result_ok and not is_expired(entry):
        log.info(
            "STS Score cache_hit patient=%s hash=%s age_days=%.2f",
            patient_id_str, input_hash[:10], cache_age_days(entry) or 0.0,
        )
        if patient_id_str:
            remember_patient_hash(patient_id_str, input_hash)
        return make_cache_hit_record(entry, patient_id_str, input_hash)

    # Log the reason for not using the on-disk entry
    if entry is None:
        log.info("STS Score cache_miss patient=%s hash=%s", patient_id_str, input_hash[:10])
    elif not entry_version_ok:
        log.info(
            "STS Score cache_miss (integration_version changed) patient=%s hash=%s",
            patient_id_str, input_hash[:10],
        )
    elif not entry_result_ok:
        log.info(
            "STS Score cache_miss (invalid cached result) patient=%s hash=%s",
            patient_id_str, input_hash[:10],
        )
    else:
        log.info(
            "STS Score cache_refresh (expired) patient=%s hash=%s age_days=%.2f",
            patient_id_str, input_hash[:10], cache_age_days(entry) or 0.0,
        )

    need_refresh = entry is not None and entry_version_ok and entry_result_ok

    # --- Stage: fetch (with retries) ---
    fresh_result: Dict[str, float] = {}
    last_error = ""
    last_stage = "fetch"
    retry_attempted = False
    attempts = max(1, int(max_retries) + 1)
    for attempt in range(attempts):
        if attempt > 0:
            retry_attempted = True
            time.sleep(0.5 * attempt)
        try:
            candidate = fetch_func(sts_input) or {}
        except Exception as e:
            last_error = f"connection_failure: {type(e).__name__}: {e}"
            last_stage = "fetch"
            log.warning(
                "STS Score connection_failure patient=%s attempt=%d: %s",
                patient_id_str, attempt + 1, last_error,
            )
            continue

        if is_valid_result(candidate):
            fresh_result = candidate
            break

        last_stage = "validate"
        keys = (
            list(candidate.keys()) if isinstance(candidate, dict)
            else type(candidate).__name__
        )
        last_error = f"response_validation_failure (keys={keys})"
        log.warning(
            "STS Score response_validation_failure patient=%s attempt=%d: %s",
            patient_id_str, attempt + 1, last_error,
        )

    if is_valid_result(fresh_result):
        persist_fresh_result(sts_input, fresh_result, input_hash, patient_id_str)
        log.info(
            "STS Score %s patient=%s hash=%s",
            "refreshed" if need_refresh else "fresh",
            patient_id_str, input_hash[:10],
        )
        return make_fresh_record(
            fresh_result, patient_id_str, input_hash, need_refresh, retry_attempted
        )

    # --- Fetch failed: try stale fallback ---
    fallback = find_stale_fallback(patient_id_str, input_hash, entry)
    if fallback is not None:
        log.warning(
            "STS Score stale_fallback patient=%s requested_hash=%s reason=%s",
            patient_id_str, input_hash[:10], last_error,
        )
        return make_stale_fallback_record(
            fallback, patient_id_str, input_hash, last_error, retry_attempted
        )

    log.error(
        "STS Score failed patient=%s hash=%s reason=%s",
        patient_id_str, input_hash[:10], last_error,
    )
    return make_failed_record(
        patient_id_str, input_hash, last_stage, last_error, retry_attempted
    )


# ---------------------------------------------------------------------------
# Summary helper for UI
# ---------------------------------------------------------------------------

def summarise_execution_log(records) -> Dict[str, int]:
    """Return a {status: count} summary for a list of ExecutionRecord objects."""
    summary = {
        "fresh": 0,
        "cached": 0,
        "refreshed": 0,
        "stale_fallback": 0,
        "failed": 0,
    }
    for r in records or []:
        if isinstance(r, ExecutionRecord):
            summary[r.status] = summary.get(r.status, 0) + 1
        elif isinstance(r, dict):
            s = r.get("status", "failed")
            summary[s] = summary.get(s, 0) + 1
    return summary
