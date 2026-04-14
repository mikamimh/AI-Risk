"""Bundle schema-version compatibility tests.

These tests cover the small schema-versioning layer introduced on top of the
pre-existing ``bundle_io`` serialization helpers.  The goals are narrow and
explicit:

1. Fresh payloads written by current code carry ``bundle_schema_version`` and
   round-trip cleanly through ``normalize_payload`` + ``deserialize_bundle``.
2. Legacy payloads (no ``bundle_schema_version`` key — what every
   ``ia_risk_bundle.joblib`` file on disk right now looks like) load with
   ``_loaded_schema_version == 0`` and produce a working bundle dict.
3. Optional inner-bundle fields (``oof_raw``, ``youden_thresholds``,
   ``best_youden_threshold``, ``calibration_method``, ``run_report``) may be
   missing without raising — the deserializer already tolerates this; this
   behavior is now covered by an explicit test so it cannot silently
   regress.
4. Truly required fields (the ``bundle`` dict; ``prepared`` / ``artifacts``
   sections) raise :class:`BundleSchemaError` with a clear message.

No Streamlit, no joblib file I/O, no ML training — the serialization helpers
are pure dict transforms and can be exercised directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bundle_io import (
    BUNDLE_SCHEMA_VERSION,
    LEGACY_BUNDLE_SCHEMA_VERSION,
    BundleSchemaError,
    deserialize_bundle,
    normalize_payload,
    read_payload_schema_version,
    serialize_bundle,
    validate_payload,
)


# ---------------------------------------------------------------------------
# Helpers — construct minimal bundle/payload shapes without touching the ML
# training pipeline.  These mirror the dict shape produced by
# ``serialize_bundle`` closely enough that ``deserialize_bundle`` rebuilds
# the dataclasses correctly.
# ---------------------------------------------------------------------------

def _minimal_serialized_bundle(
    *,
    with_optional_artifacts: bool = True,
    with_run_report: bool = False,
) -> dict:
    """Return a dict shaped like ``serialize_bundle``'s output — without
    actually training anything.  ``with_optional_artifacts=False`` omits the
    v0-era optional fields to exercise the legacy-tolerant deserialize path.
    """
    artifacts_dict: dict = {
        "model": None,                         # model slot — deserialize does not introspect
        "leaderboard": pd.DataFrame(),
        "oof_predictions": np.zeros(0),        # calibrated OOF
        "feature_columns": ["age", "bmi"],
        "fitted_models": {},
        "best_model_name": "stub_model",
    }
    if with_optional_artifacts:
        artifacts_dict.update(
            oof_raw=np.zeros(0),
            calibration_method="sigmoid",
            youden_thresholds={"stub_model": 0.08},
            best_youden_threshold=0.08,
        )

    bundle = {
        "prepared": {
            "data": pd.DataFrame({"age": [60, 70], "bmi": [25.0, 28.0]}),
            "feature_columns": ["age", "bmi"],
            "info": {"n_rows": 2},
        },
        "artifacts": artifacts_dict,
    }
    if with_run_report:
        bundle["run_report"] = {"phases": [], "events": []}
    return bundle


def _make_payload(
    *,
    include_schema_version: bool,
    **bundle_kwargs,
) -> dict:
    """Build a top-level payload dict as it would be written to disk."""
    payload = {
        "signature": {
            "xlsx_path": "/stub.xlsx",
            "xlsx_mtime_ns": 1,
            "xlsx_size": 1,
            "model_version": "stub-v0",
        },
        "bundle": _minimal_serialized_bundle(**bundle_kwargs),
        "saved_at": "2026-04-14T00:00:00+00:00",
        "training_source": "stub.xlsx",
    }
    if include_schema_version:
        payload["bundle_schema_version"] = BUNDLE_SCHEMA_VERSION
    return payload


# ---------------------------------------------------------------------------
# 1. Version constants & read helper
# ---------------------------------------------------------------------------

def test_schema_version_is_positive_int():
    """The current schema version is a concrete positive integer."""
    assert isinstance(BUNDLE_SCHEMA_VERSION, int)
    assert BUNDLE_SCHEMA_VERSION >= 1
    assert LEGACY_BUNDLE_SCHEMA_VERSION == 0


def test_read_payload_schema_version_defaults_to_legacy():
    """Payloads without the key are interpreted as the legacy version."""
    assert read_payload_schema_version({}) == LEGACY_BUNDLE_SCHEMA_VERSION
    assert read_payload_schema_version({"other": "stuff"}) == LEGACY_BUNDLE_SCHEMA_VERSION


def test_read_payload_schema_version_reads_explicit_value():
    assert read_payload_schema_version({"bundle_schema_version": 1}) == 1
    assert read_payload_schema_version({"bundle_schema_version": 42}) == 42


def test_read_payload_schema_version_rejects_malformed_values():
    """Strings, floats, negatives, and bools should all fall back to legacy."""
    for bad in ("1", 1.0, -1, True, False, None, object()):
        assert read_payload_schema_version({"bundle_schema_version": bad}) == LEGACY_BUNDLE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 2. validate_payload — essential fields only
# ---------------------------------------------------------------------------

def test_validate_payload_accepts_complete_payload():
    validate_payload(_make_payload(include_schema_version=True))


def test_validate_payload_accepts_legacy_payload():
    """Legacy payloads (no version key) still validate — the version field
    is not in the essential-fields set."""
    validate_payload(_make_payload(include_schema_version=False))


def test_validate_payload_raises_when_not_a_dict():
    with pytest.raises(BundleSchemaError):
        validate_payload(None)  # type: ignore[arg-type]
    with pytest.raises(BundleSchemaError):
        validate_payload("not a dict")  # type: ignore[arg-type]


def test_validate_payload_raises_when_bundle_missing():
    payload = _make_payload(include_schema_version=True)
    del payload["bundle"]
    with pytest.raises(BundleSchemaError, match="bundle"):
        validate_payload(payload)


def test_validate_payload_raises_when_prepared_missing():
    payload = _make_payload(include_schema_version=True)
    del payload["bundle"]["prepared"]
    with pytest.raises(BundleSchemaError, match="prepared"):
        validate_payload(payload)


def test_validate_payload_raises_when_artifacts_missing():
    payload = _make_payload(include_schema_version=True)
    del payload["bundle"]["artifacts"]
    with pytest.raises(BundleSchemaError, match="artifacts"):
        validate_payload(payload)


# ---------------------------------------------------------------------------
# 3. normalize_payload — upgrade in-memory and surface diagnostic
# ---------------------------------------------------------------------------

def test_normalize_new_payload_preserves_version_and_records_source():
    """A v1 payload round-trips through normalize with loaded_schema == current."""
    payload = _make_payload(include_schema_version=True)
    out = normalize_payload(payload)
    assert out["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION
    assert out["_loaded_schema_version"] == BUNDLE_SCHEMA_VERSION
    # Non-schema fields must survive unchanged.
    assert out["signature"] == payload["signature"]
    assert out["saved_at"] == payload["saved_at"]
    assert out["training_source"] == payload["training_source"]
    assert out["bundle"] is payload["bundle"]  # shallow copy — inner dict shared


def test_normalize_legacy_payload_upgrades_version_and_records_source_zero():
    """A legacy payload (no version key) is treated as v0 and upgraded to v1
    in-memory.  ``_loaded_schema_version`` carries the original 0 so the UI
    can render a compat notice if it chooses to."""
    payload = _make_payload(include_schema_version=False)
    out = normalize_payload(payload)
    assert out["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION
    assert out["_loaded_schema_version"] == LEGACY_BUNDLE_SCHEMA_VERSION
    # Essential content preserved through the upgrade.
    assert out["signature"] == payload["signature"]
    assert out["bundle"] is payload["bundle"]


def test_normalize_does_not_mutate_input():
    """``normalize_payload`` must return a shallow copy, not mutate in place."""
    payload = _make_payload(include_schema_version=False)
    before_keys = set(payload.keys())
    _ = normalize_payload(payload)
    assert set(payload.keys()) == before_keys
    assert "bundle_schema_version" not in payload
    assert "_loaded_schema_version" not in payload


def test_normalize_raises_on_invalid_payload():
    """normalize calls validate first — invalid payloads raise, not corrupt."""
    with pytest.raises(BundleSchemaError):
        normalize_payload({"bundle_schema_version": BUNDLE_SCHEMA_VERSION})


# ---------------------------------------------------------------------------
# 4. End-to-end: normalize + deserialize a legacy bundle
# ---------------------------------------------------------------------------

def test_legacy_bundle_without_optional_fields_deserializes_cleanly():
    """The v0 contract was: optional artifacts fields (``oof_raw``,
    ``calibration_method``, ``youden_thresholds``, ``best_youden_threshold``)
    may be missing.  Verify this still works end-to-end under the new
    versioning layer."""
    payload = _make_payload(
        include_schema_version=False,
        with_optional_artifacts=False,
    )
    normalized = normalize_payload(payload)
    assert normalized["_loaded_schema_version"] == LEGACY_BUNDLE_SCHEMA_VERSION

    reconstructed = deserialize_bundle(normalized["bundle"])

    # Essential dataclasses rebuilt.
    assert reconstructed["prepared"].feature_columns == ["age", "bmi"]
    assert reconstructed["artifacts"].best_model_name == "stub_model"
    # Missing optional fields surface as their defaults — not as attribute errors.
    assert reconstructed["artifacts"].oof_raw is None
    assert reconstructed["artifacts"].youden_thresholds is None
    assert reconstructed["artifacts"].best_youden_threshold is None
    assert reconstructed["artifacts"].calibration_method == "sigmoid"


def test_new_bundle_with_run_report_roundtrips():
    """A v1 payload carrying a ``run_report`` dict rebuilds the RunReport
    dataclass on deserialize (Phase-3 behavior still holds)."""
    payload = _make_payload(
        include_schema_version=True,
        with_optional_artifacts=True,
        with_run_report=True,
    )
    normalized = normalize_payload(payload)
    reconstructed = deserialize_bundle(normalized["bundle"])

    import observability
    assert isinstance(reconstructed["run_report"], observability.RunReport)


def test_serialize_then_normalize_produces_current_version():
    """Writing-side check: what ``serialize_bundle`` + the app wrapper
    produces is itself a valid current-version payload."""
    # Simulate what app.py's load_train_bundle does on a fresh write.
    inner = _minimal_serialized_bundle(with_optional_artifacts=True)
    payload = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "signature": {"xlsx_path": "/stub.xlsx", "xlsx_mtime_ns": 1,
                      "xlsx_size": 1, "model_version": "stub-v0"},
        "bundle": inner,
        "saved_at": "2026-04-14T00:00:00+00:00",
        "training_source": "stub.xlsx",
    }
    normalized = normalize_payload(payload)
    assert normalized["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION
    assert normalized["_loaded_schema_version"] == BUNDLE_SCHEMA_VERSION


def test_serialize_bundle_is_unchanged_by_versioning():
    """``serialize_bundle`` operates on the inner bundle only — the schema
    version lives on the outer payload wrapper and must not leak here."""
    from dataclasses import dataclass

    @dataclass
    class _StubArtifacts:
        model: object = None
        leaderboard: object = None
        oof_predictions: object = None
        feature_columns: object = None
        fitted_models: object = None
        best_model_name: str = "stub_model"
        calibration_method: str = "sigmoid"
        oof_raw: object = None
        youden_thresholds: object = None
        best_youden_threshold: object = None

    from risk_data import PreparedData

    inner = {
        "prepared": PreparedData(
            data=pd.DataFrame({"age": [60]}),
            feature_columns=["age"],
            info={},
        ),
        "artifacts": _StubArtifacts(),
    }
    out = serialize_bundle(inner)
    assert "bundle_schema_version" not in out
    assert isinstance(out["prepared"], dict)
    assert isinstance(out["artifacts"], dict)
