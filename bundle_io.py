"""Bundle I/O helpers for AI Risk.

Extracted from ``app.py`` during the Phase 4 conservative refactor.
These are pure data-transform helpers with **no** Streamlit dependency:

* :func:`bundle_signature` - build the cache-key signature for a training
  source based on its file stat and the current model version.
* :func:`serialize_bundle` - convert a training bundle (containing
  ``PreparedData`` and ``TrainedArtifacts`` dataclasses) into plain dicts
  suitable for ``joblib.dump``.  Streamlit reloads module objects between
  reruns, so pickling dataclass instances directly would tie the cache to
  specific class references; plain dicts sidestep that.
* :func:`deserialize_bundle` - reverse :func:`serialize_bundle`, rebuilding
  the dataclass instances via an ``importlib.reload`` of ``modeling`` and
  ``observability`` so the freshly loaded classes are used.

Both serialize/deserialize also handle the Phase 3 ``run_report`` field
(a :class:`observability.RunReport`) via its ``to_dict`` / ``from_dict``
methods.

No methodology is changed.  The cache file path itself
(``MODEL_CACHE_FILE``) is deliberately left in ``app.py`` because it is
owned by the cache policy layer (``load_train_bundle`` and
``load_cached_bundle_only``) which remain in the app for this phase.
"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Dict

from config import AppConfig
from risk_data import PreparedData

MODEL_VERSION = AppConfig.MODEL_VERSION

# ---------------------------------------------------------------------------
# Bundle schema versioning
# ---------------------------------------------------------------------------
# The persisted joblib payload is a dict with the shape:
#
#     {
#         "bundle_schema_version": <int>,  # since v1
#         "signature": {...},
#         "bundle": {"prepared": {...}, "artifacts": {...}, "run_report": {...}?},
#         "saved_at": "<iso-utc>",
#         "training_source": "<filename>",
#     }
#
# ``BUNDLE_SCHEMA_VERSION`` is the version every freshly written payload
# carries.  Legacy payloads written before versioning was introduced do NOT
# contain this key; ``read_payload_schema_version`` interprets their absence
# as ``LEGACY_BUNDLE_SCHEMA_VERSION`` (0) and ``normalize_payload`` upgrades
# them in-memory.
#
# Version history:
#   0 — implicit, pre-versioning.  Inner bundle may omit any of the optional
#       artifact fields (``oof_raw``, ``youden_thresholds``,
#       ``best_youden_threshold``, ``calibration_method``) and ``run_report``.
#       ``deserialize_bundle`` already tolerates these via ``.get()``.
#   1 — current.  Adds the explicit ``bundle_schema_version`` field itself;
#       no structural change to the inner bundle.  The inner optional-field
#       tolerance from v0 is preserved and now *contractual*.
BUNDLE_SCHEMA_VERSION = 1
LEGACY_BUNDLE_SCHEMA_VERSION = 0


class BundleSchemaError(ValueError):
    """Raised when a payload is missing a field that is required regardless
    of schema version (e.g. the inner ``bundle`` dict itself, or its
    ``prepared`` / ``artifacts`` sections).  Optional fields never raise."""


def read_payload_schema_version(payload: Dict[str, object]) -> int:
    """Return the schema version stored on a loaded payload.

    Missing / malformed → :data:`LEGACY_BUNDLE_SCHEMA_VERSION`.  The function
    never raises: it is safe to call on arbitrary ``joblib.load`` output.
    """
    v = payload.get("bundle_schema_version") if isinstance(payload, dict) else None
    if isinstance(v, bool):
        # bool is a subclass of int — reject explicitly so a stray True/False
        # cannot masquerade as a valid version.
        return LEGACY_BUNDLE_SCHEMA_VERSION
    if isinstance(v, int) and v >= 0:
        return v
    return LEGACY_BUNDLE_SCHEMA_VERSION


def validate_payload(payload: Dict[str, object]) -> None:
    """Raise :class:`BundleSchemaError` if the payload lacks truly-required
    fields.  Optional fields are deliberately not checked here — they are
    the responsibility of :func:`deserialize_bundle` and its ``.get()``
    fallbacks.

    Required:
      * ``bundle`` — a dict containing ``prepared`` and ``artifacts``.

    Everything else (signature, saved_at, training_source, run_report,
    optional artifact fields) is tolerated as missing.
    """
    if not isinstance(payload, dict):
        raise BundleSchemaError(f"payload must be a dict, got {type(payload).__name__}")
    bundle = payload.get("bundle")
    if not isinstance(bundle, dict):
        raise BundleSchemaError("payload missing required 'bundle' dict")
    for required in ("prepared", "artifacts"):
        if required not in bundle:
            raise BundleSchemaError(f"bundle missing required '{required}' section")


def normalize_payload(payload: Dict[str, object]) -> Dict[str, object]:
    """Return a shallow copy of ``payload`` upgraded to the current schema.

    The return value always has:
      * ``bundle_schema_version`` set to :data:`BUNDLE_SCHEMA_VERSION`
      * ``_loaded_schema_version`` — the version actually read from disk
        (useful for diagnostics / telemetry / user-visible compat notices).
        This key is NOT re-persisted; callers should drop it before save.

    No migration logic is currently required (v0 → v1 is purely the addition
    of the version field; the inner ``deserialize_bundle`` already tolerates
    every optional-field gap that existed in v0 payloads).  Future schema
    bumps should branch on ``source_version`` here before writing the
    upgraded fields onto ``out``.

    The inner ``bundle`` dict is deserialized eagerly so that callers can
    access ``payload["artifacts"]`` and ``payload["prepared"]`` directly,
    without a separate :func:`deserialize_bundle` call.  Callers that still
    pass ``payload["bundle"]`` through :func:`deserialize_bundle` continue
    to work: ``deserialize_bundle`` is idempotent when the values are
    already dataclass instances.
    """
    validate_payload(payload)
    out = dict(payload)
    source_version = read_payload_schema_version(out)

    # --- Migration ladder ---------------------------------------------------
    # Each ``if source_version < N`` block upgrades from N-1 → N.  Currently
    # the only step is 0 → 1, which is a no-op on the bytes (optional-field
    # tolerance was already the de-facto contract) and merely stamps the
    # explicit version.  Kept as an explicit branch so future migrations have
    # an obvious template.
    if source_version < 1:
        # v0 → v1: no structural change.  Intentionally empty.
        pass
    # ------------------------------------------------------------------------

    out["bundle_schema_version"] = BUNDLE_SCHEMA_VERSION
    out["_loaded_schema_version"] = source_version
    # Eagerly deserialize the inner bundle so payload["artifacts"] and
    # payload["prepared"] are accessible at the top level without a
    # separate deserialize_bundle call.  out["bundle"] is intentionally
    # kept as-is (same object reference) for callers that rely on it.
    _inner = deserialize_bundle(out["bundle"])
    out["artifacts"] = _inner.get("artifacts")
    out["prepared"] = _inner.get("prepared")
    return out


def bundle_signature(xlsx_path: str) -> Dict[str, object]:
    """Signature used as the cache key for a training source."""
    p = Path(xlsx_path)
    stt = p.stat()
    return {
        "xlsx_path": str(p.resolve()),
        "xlsx_mtime_ns": int(stt.st_mtime_ns),
        "xlsx_size": int(stt.st_size),
        "model_version": MODEL_VERSION,
    }


def bundle_metadata_from_payload(
    payload: Dict[str, object],
    fallback_training_source: str = "Unknown",
) -> Dict[str, object]:
    """Return the canonical ``bundle_info`` dict for a loaded payload.

    This is the **single source of truth** for export-time metadata.  Callers
    must always read ``model_version`` and ``active_model_name`` from this
    dict (not from ``AppConfig.MODEL_VERSION`` directly), so an export can
    never drift from the bundle that produced its data.

    Fields:
      * ``model_version``      — taken from the persisted ``signature``;
                                  falls back to the current ``MODEL_VERSION``
                                  only when no signature is present (legacy /
                                  in-memory payloads).
      * ``active_model_name``  — the bundle's ``best_model_name`` (the model
                                  that was actually trained and selected).
                                  ``None`` when the inner bundle is not
                                  deserialized — callers populate this from
                                  ``artifacts.best_model_name`` after deserialize.
      * ``saved_at``           — ISO timestamp from disk.
      * ``training_source``    — filename used at training time.
      * ``schema_version``     — current bundle schema version.
      * ``loaded_schema_version`` — what was actually read from disk (≤ current).
      * ``dataset_fingerprint`` — composite of (basename, path hash, mtime_ns, size)
                                  from the signature; ``None`` when absent.
      * ``bundle_fingerprint`` — short hash of the signature block, suitable
                                  for embedding in audit logs / manifests.
    """
    sig = payload.get("signature") if isinstance(payload, dict) else None
    sig = sig if isinstance(sig, dict) else {}

    inner = payload.get("bundle") if isinstance(payload, dict) else None
    inner_artifacts = (
        inner.get("artifacts") if isinstance(inner, dict) else None
    ) or {}
    active_model_name = None
    if isinstance(inner_artifacts, dict):
        active_model_name = inner_artifacts.get("best_model_name")

    # Dataset fingerprint MUST NOT carry the absolute on-disk path — that
    # leaks the local environment (e.g. ``C:\Users\<name>\...``) into every
    # exported manifest.  We keep the basename for human auditability and a
    # short SHA1 of the full path for uniqueness, but the full path itself
    # never leaves the process.
    dataset_fp = None
    if sig.get("xlsx_path") and "xlsx_mtime_ns" in sig and "xlsx_size" in sig:
        import hashlib
        _full_path = str(sig.get("xlsx_path"))
        _basename = PureWindowsPath(_full_path).name
        _path_hash = hashlib.sha1(_full_path.encode("utf-8")).hexdigest()[:8]
        dataset_fp = (
            f"name={_basename}|"
            f"path_sha1={_path_hash}|"
            f"mtime_ns={sig.get('xlsx_mtime_ns')}|"
            f"size={sig.get('xlsx_size')}"
        )

    # Bundle fingerprint hashes the path internally (not exposed) so two
    # bundles trained from different absolute paths but identical
    # content/version still produce different fingerprints when reloaded
    # from different locations.  The hash output does not leak the path.
    bundle_fp = None
    if sig:
        import hashlib
        bundle_fp = hashlib.sha1(
            "|".join(
                f"{k}={sig.get(k)}"
                for k in ("xlsx_path", "xlsx_mtime_ns", "xlsx_size", "model_version")
            ).encode("utf-8")
        ).hexdigest()[:12]

    return {
        "model_version": sig.get("model_version") or MODEL_VERSION,
        "active_model_name": active_model_name,
        "saved_at": payload.get("saved_at", "Unknown") if isinstance(payload, dict) else "Unknown",
        "training_source": payload.get("training_source", fallback_training_source) if isinstance(payload, dict) else fallback_training_source,
        "schema_version": payload.get("bundle_schema_version", BUNDLE_SCHEMA_VERSION) if isinstance(payload, dict) else BUNDLE_SCHEMA_VERSION,
        "loaded_schema_version": payload.get("_loaded_schema_version", LEGACY_BUNDLE_SCHEMA_VERSION) if isinstance(payload, dict) else LEGACY_BUNDLE_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_fp,
        "bundle_fingerprint": bundle_fp,
    }


class BundleVersionMismatch(RuntimeError):
    """Raised when an export's bundle metadata diverges from the current config.

    The export layer treats the loaded bundle as the source of truth: a
    divergence here means the cached bundle was loaded but the on-disk
    config was bumped without retraining.  Failing loudly is preferable to
    silently emitting a CSV/PDF whose ``model_version`` line lies about
    what model produced the numbers.
    """


def assert_bundle_metadata_consistency(
    bundle_info: Dict[str, object],
    artifacts_best_model_name: str | None = None,
) -> None:
    """Fail fast if ``bundle_info`` disagrees with the current config or with
    the deserialized artifacts.  Called by the export entry points so a
    divergence can never silently be persisted to disk.

    The check is intentionally narrow:

    * ``bundle_info["model_version"]`` must equal :data:`MODEL_VERSION`
      (the current ``AppConfig.MODEL_VERSION``).  If they differ, the
      cache contract that protects exports has been broken — bail.
    * If both ``bundle_info["active_model_name"]`` and the artifacts'
      ``best_model_name`` are present, they must match.  Divergence means
      the metadata block has been mutated independently of the model that
      actually scored the patients.

    No methodology, threshold, or scientific behavior is touched here.
    """
    bi_version = bundle_info.get("model_version") if isinstance(bundle_info, dict) else None
    if bi_version and bi_version != MODEL_VERSION:
        raise BundleVersionMismatch(
            f"Bundle version mismatch: loaded bundle is '{bi_version}' "
            f"but AppConfig.MODEL_VERSION is '{MODEL_VERSION}'. "
            f"Retrain the bundle (sidebar → 'Train/Retrain models') before exporting."
        )

    bi_model = bundle_info.get("active_model_name") if isinstance(bundle_info, dict) else None
    if bi_model and artifacts_best_model_name and bi_model != artifacts_best_model_name:
        raise BundleVersionMismatch(
            f"Active-model mismatch: bundle_info reports '{bi_model}' "
            f"but artifacts carry '{artifacts_best_model_name}'. "
            f"Refusing to export to avoid recording the wrong model."
        )


def serialize_bundle(bundle: Dict[str, object]) -> Dict[str, object]:
    """Convert dataclasses to plain dicts for pickle compatibility.

    Streamlit may reload modules between runs, creating new class objects.
    Pickle requires the exact same class reference, so we store plain dicts.
    """
    out = dict(bundle)
    prepared = out["prepared"]
    out["prepared"] = {
        "data": prepared.data,
        "feature_columns": prepared.feature_columns,
        "info": prepared.info,
    }
    artifacts = out["artifacts"]
    out["artifacts"] = {
        "model": artifacts.model,
        "leaderboard": artifacts.leaderboard,
        "oof_predictions": artifacts.oof_predictions,       # calibrated OOF (primary)
        "oof_raw": getattr(artifacts, "oof_raw", None),     # uncalibrated, audit only
        "feature_columns": artifacts.feature_columns,
        "fitted_models": artifacts.fitted_models,
        "best_model_name": artifacts.best_model_name,
        "calibration_method": getattr(artifacts, "calibration_method", "sigmoid"),
        "youden_thresholds": getattr(artifacts, "youden_thresholds", None),
        "best_youden_threshold": getattr(artifacts, "best_youden_threshold", None),
        "training_manifest": getattr(artifacts, "training_manifest", None),
    }
    # Phase 3: persist the run report as a plain dict so module reloads
    # don't break unpickling.
    run_report = out.get("run_report")
    if run_report is not None and hasattr(run_report, "to_dict"):
        out["run_report"] = run_report.to_dict()
    return out


def deserialize_bundle(bundle: Dict[str, object]) -> Dict[str, object]:
    """Reconstruct dataclasses from plain dicts."""
    import importlib
    import modeling as _mod
    importlib.reload(_mod)
    TrainedArtifacts = _mod.TrainedArtifacts
    out = dict(bundle)
    p = out["prepared"]
    if isinstance(p, dict):
        out["prepared"] = PreparedData(
            data=p["data"],
            feature_columns=p["feature_columns"],
            info=p["info"],
        )
    a = out["artifacts"]
    if isinstance(a, dict):
        out["artifacts"] = TrainedArtifacts(
            model=a["model"],
            leaderboard=a["leaderboard"],
            oof_predictions=a["oof_predictions"],           # calibrated OOF
            feature_columns=a["feature_columns"],
            fitted_models=a["fitted_models"],
            best_model_name=a["best_model_name"],
            calibration_method=a.get("calibration_method", "sigmoid"),
            oof_raw=a.get("oof_raw"),                       # may be None in old caches
            youden_thresholds=a.get("youden_thresholds"),
            best_youden_threshold=a.get("best_youden_threshold"),
            training_manifest=a.get("training_manifest"),
        )
    # Phase 3: rebuild RunReport from persisted dict (legacy bundles without
    # the key simply carry no report).
    rr = out.get("run_report")
    if isinstance(rr, dict):
        import observability as _obs
        importlib.reload(_obs)
        out["run_report"] = _obs.RunReport.from_dict(rr)
    return out
