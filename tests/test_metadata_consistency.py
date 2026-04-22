"""Bundle metadata consistency & export manifest tests.

These tests cover the source-of-truth chain that every export depends on:

1.  ``bundle_metadata_from_payload`` reads ``model_version`` and
    ``active_model_name`` from the persisted bundle's signature/artifacts —
    NOT from ``AppConfig.MODEL_VERSION`` — so a stale config in process
    memory can never silently overwrite the value that the bundle was
    actually trained with.
2.  ``assert_bundle_metadata_consistency`` fails loudly when the loaded
    bundle's version disagrees with the current config, OR when the
    advertised active-model name disagrees with the artifacts.  The export
    layer is required to run this guard at boot — a failure here means an
    export would mislabel its numbers, and we prefer a clear error to
    silent corruption.
3.  ``build_export_manifest`` produces a flat, JSON-serialisable dict with
    all fields needed to answer "which bundle / model / threshold / dataset
    produced this export".
4.  The Comparison Full Package ZIP embeds ``manifest.json`` whose fields
    match what was passed in, so downstream consumers can audit the export
    without inspecting the application.

No methodology, threshold, or scientific behavior is exercised here — these
are governance / traceability checks only.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from bundle_io import (
    BUNDLE_SCHEMA_VERSION,
    LEGACY_BUNDLE_SCHEMA_VERSION,
    BundleVersionMismatch,
    MODEL_VERSION,
    assert_bundle_metadata_consistency,
    bundle_metadata_from_payload,
)
from export_helpers import (
    build_comparison_full_package,
    build_export_manifest,
    manifest_to_json_bytes,
    manifest_to_md_lines,
)


# ---------------------------------------------------------------------------
# Helpers — minimal payload shapes that mirror what disk would carry, without
# touching joblib I/O or training.
# ---------------------------------------------------------------------------

def _payload_with(
    *,
    model_version: str = "stub-v0",
    best_model_name: str = "stub_model",
    include_signature: bool = True,
    include_schema_version: bool = True,
) -> dict:
    sig = {
        "xlsx_path": "/stub.xlsx",
        "xlsx_mtime_ns": 12345,
        "xlsx_size": 999,
        "model_version": model_version,
    }
    payload: dict = {
        "bundle": {
            "prepared": {
                "data": pd.DataFrame({"age": [60]}),
                "feature_columns": ["age"],
                "info": {"n_rows": 1},
            },
            "artifacts": {
                "model": None,
                "leaderboard": pd.DataFrame(),
                "oof_predictions": np.zeros(0),
                "feature_columns": ["age"],
                "fitted_models": {},
                "best_model_name": best_model_name,
            },
        },
        "saved_at": "2026-04-21T12:00:00+00:00",
        "training_source": "stub.xlsx",
    }
    if include_signature:
        payload["signature"] = sig
    if include_schema_version:
        payload["bundle_schema_version"] = BUNDLE_SCHEMA_VERSION
    return payload


# ---------------------------------------------------------------------------
# 1. bundle_metadata_from_payload — model_version comes from the BUNDLE
# ---------------------------------------------------------------------------

def test_metadata_reads_model_version_from_signature_not_config():
    """The signature's model_version is the source of truth — even when it
    differs from the current ``AppConfig.MODEL_VERSION``.  Drift detection
    happens later in ``assert_bundle_metadata_consistency``; this function
    must report the truth, not a sanitized value."""
    payload = _payload_with(model_version="hand-crafted-version-xyz")
    info = bundle_metadata_from_payload(payload)
    assert info["model_version"] == "hand-crafted-version-xyz"
    assert info["model_version"] != MODEL_VERSION  # the whole point


def test_metadata_active_model_extracted_from_artifacts():
    """``active_model_name`` mirrors the trained ``best_model_name`` so an
    export can name the model that actually scored the patients."""
    payload = _payload_with(best_model_name="RandomForest")
    info = bundle_metadata_from_payload(payload)
    assert info["active_model_name"] == "RandomForest"


def test_metadata_active_model_none_when_artifacts_missing():
    """If the inner artifacts block is absent (defensive — should never
    happen in practice), the function reports ``None`` instead of raising."""
    payload = _payload_with()
    payload["bundle"].pop("artifacts")
    info = bundle_metadata_from_payload(payload)
    assert info["active_model_name"] is None


def test_metadata_falls_back_to_config_when_signature_missing():
    """Legacy / ad-hoc payloads without a signature fall back to the
    in-process ``MODEL_VERSION`` so callers still get a usable value."""
    payload = _payload_with(include_signature=False)
    info = bundle_metadata_from_payload(payload)
    assert info["model_version"] == MODEL_VERSION


def test_metadata_carries_dataset_and_bundle_fingerprints():
    """Both fingerprints must be present and stable across calls when the
    signature is intact — they form the audit chain in the manifest."""
    payload = _payload_with(model_version="vA")
    info_a = bundle_metadata_from_payload(payload)
    info_b = bundle_metadata_from_payload(payload)
    assert info_a["dataset_fingerprint"] is not None
    assert info_a["bundle_fingerprint"] is not None
    assert info_a["dataset_fingerprint"] == info_b["dataset_fingerprint"]
    assert info_a["bundle_fingerprint"] == info_b["bundle_fingerprint"]


def test_bundle_fingerprint_changes_when_version_changes():
    """The bundle fingerprint depends on signature contents — different
    model_version values must produce different fingerprints, otherwise
    downstream auditors couldn't tell two exports apart."""
    fp_a = bundle_metadata_from_payload(_payload_with(model_version="vA"))["bundle_fingerprint"]
    fp_b = bundle_metadata_from_payload(_payload_with(model_version="vB"))["bundle_fingerprint"]
    assert fp_a != fp_b


def test_metadata_handles_non_dict_payload_gracefully():
    """Accept arbitrary input without raising — defensive boundary check."""
    info = bundle_metadata_from_payload({})  # type: ignore[arg-type]
    assert info["model_version"] == MODEL_VERSION
    assert info["active_model_name"] is None
    assert info["dataset_fingerprint"] is None


def test_metadata_carries_schema_version_fields():
    payload = _payload_with()
    payload["_loaded_schema_version"] = LEGACY_BUNDLE_SCHEMA_VERSION
    info = bundle_metadata_from_payload(payload)
    assert info["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert info["loaded_schema_version"] == LEGACY_BUNDLE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 2. assert_bundle_metadata_consistency — fail-fast guard
# ---------------------------------------------------------------------------

def test_consistency_passes_when_versions_match():
    """A bundle whose version equals the current config and whose artifacts
    agree with bundle_info is silent — guard must not be noisy."""
    info = bundle_metadata_from_payload(_payload_with(
        model_version=MODEL_VERSION, best_model_name="RandomForest",
    ))
    assert_bundle_metadata_consistency(info, artifacts_best_model_name="RandomForest")


def test_consistency_passes_when_artifacts_arg_omitted():
    """Callers may not always have access to the inner artifacts (e.g. the
    cache-only loader path).  The guard must still pass on a matching
    bundle_info alone."""
    info = bundle_metadata_from_payload(_payload_with(model_version=MODEL_VERSION))
    assert_bundle_metadata_consistency(info)


def test_consistency_raises_on_model_version_mismatch():
    """The flagship invariant: cached bundle says vX, AppConfig says
    something else → BundleVersionMismatch.  An export that proceeds in this
    state would lie about which model produced its numbers."""
    info = bundle_metadata_from_payload(_payload_with(model_version="WRONG-VERSION"))
    with pytest.raises(BundleVersionMismatch, match="version mismatch"):
        assert_bundle_metadata_consistency(info)


def test_consistency_raises_when_active_model_disagrees_with_artifacts():
    """Even if model_version matches, an active_model_name that disagrees
    with the live artifacts is a tampering / drift signal."""
    info = bundle_metadata_from_payload(_payload_with(
        model_version=MODEL_VERSION, best_model_name="RandomForest",
    ))
    with pytest.raises(BundleVersionMismatch, match="Active-model"):
        assert_bundle_metadata_consistency(info, artifacts_best_model_name="LogReg")


def test_consistency_silent_when_one_active_model_side_is_none():
    """If either side is unknown (e.g. legacy bundle without best_model_name,
    or caller did not pass artifacts), only the version is enforced."""
    info = bundle_metadata_from_payload(_payload_with(model_version=MODEL_VERSION))
    info["active_model_name"] = None
    assert_bundle_metadata_consistency(info, artifacts_best_model_name="RandomForest")
    info["active_model_name"] = "RandomForest"
    assert_bundle_metadata_consistency(info, artifacts_best_model_name=None)


def test_consistency_error_message_names_both_versions():
    """The error must be actionable — engineer needs to see which version
    was loaded and which version the config currently advertises."""
    info = bundle_metadata_from_payload(_payload_with(model_version="loaded-vX"))
    with pytest.raises(BundleVersionMismatch) as excinfo:
        assert_bundle_metadata_consistency(info)
    msg = str(excinfo.value)
    assert "loaded-vX" in msg
    assert MODEL_VERSION in msg
    assert "Retrain" in msg or "retrain" in msg.lower()


# ---------------------------------------------------------------------------
# 3. build_export_manifest — single recipe, all fields present
# ---------------------------------------------------------------------------

def test_manifest_contains_all_required_fields():
    """The export manifest is the contract surface for downstream auditors —
    every required field must be present even when the value is None."""
    manifest = build_export_manifest(
        export_kind="comparison",
        model_version="2026-04-21-v13-none-semantics",
        active_model_name="RandomForest",
        threshold_mode="clinical_fixed",
        threshold_value=0.08,
    )
    required = {
        "export_kind",
        "generated_at",
        "model_version",
        "active_model_name",
        "threshold_mode",
        "threshold_value",
        "dataset_fingerprint",
        "bundle_fingerprint",
        "bundle_saved_at",
        "training_source",
        "current_analysis_file",
    }
    assert required.issubset(manifest.keys())
    assert manifest["model_version"] == "2026-04-21-v13-none-semantics"
    assert manifest["active_model_name"] == "RandomForest"
    assert manifest["threshold_mode"] == "clinical_fixed"
    assert manifest["threshold_value"] == 0.08


def test_manifest_threshold_value_coerced_to_float():
    """numpy floats and ints are common in this codebase — manifest must
    coerce them so JSON serialisation cannot fail downstream."""
    manifest = build_export_manifest(
        export_kind="comparison",
        model_version="vX",
        active_model_name="RF",
        threshold_mode="youden",
        threshold_value=np.float64(0.12),
    )
    assert isinstance(manifest["threshold_value"], float)
    assert manifest["threshold_value"] == pytest.approx(0.12)


def test_manifest_extras_are_preserved_under_extra_key():
    """Per-export auxiliary fields (n_triple, language, etc.) live under
    ``extra`` so the top-level keys remain a stable contract."""
    manifest = build_export_manifest(
        export_kind="comparison",
        model_version="vX",
        active_model_name="RF",
        threshold_mode="clinical_fixed",
        threshold_value=0.08,
        extra={"n_triple": 1234, "language": "English"},
    )
    assert manifest["extra"]["n_triple"] == 1234
    assert manifest["extra"]["language"] == "English"


def test_manifest_to_json_bytes_round_trips():
    """JSON output must be valid UTF-8 and round-trip cleanly."""
    manifest = build_export_manifest(
        export_kind="comparison",
        model_version="vX",
        active_model_name="RandomForest",
        threshold_mode="clinical_fixed",
        threshold_value=0.08,
        dataset_fingerprint="ds-abc",
        bundle_fingerprint="bf-123",
    )
    raw = manifest_to_json_bytes(manifest)
    decoded = json.loads(raw.decode("utf-8"))
    assert decoded["model_version"] == "vX"
    assert decoded["active_model_name"] == "RandomForest"
    assert decoded["dataset_fingerprint"] == "ds-abc"
    assert decoded["bundle_fingerprint"] == "bf-123"


def test_manifest_md_lines_localised_pt_and_en():
    """Markdown rendering must respect the language flag — both locales must
    surface model_version, active model, and threshold."""
    manifest = build_export_manifest(
        export_kind="comparison",
        model_version="vX",
        active_model_name="RandomForest",
        threshold_mode="clinical_fixed",
        threshold_value=0.08,
    )
    en_lines = manifest_to_md_lines(manifest, language="English")
    pt_lines = manifest_to_md_lines(manifest, language="Português")
    assert any("Model version" in l and "vX" in l for l in en_lines)
    assert any("RandomForest" in l for l in en_lines)
    assert any("Versão do modelo" in l and "vX" in l for l in pt_lines)


# ---------------------------------------------------------------------------
# 4. Comparison Full Package — manifest is embedded in the ZIP
# ---------------------------------------------------------------------------

def _empty_df(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: [] for c in cols})


def test_comparison_full_package_embeds_manifest_json():
    """The ZIP returned by ``build_comparison_full_package`` must contain a
    top-level ``manifest.json`` whose fields match what the caller passed
    in.  This is the end-to-end traceability check: a reviewer can extract
    the ZIP and answer 'which bundle produced this' without running any
    code."""
    manifest = build_export_manifest(
        export_kind="comparison",
        model_version="2026-04-21-v13-none-semantics",
        active_model_name="RandomForest",
        threshold_mode="clinical_fixed",
        threshold_value=0.08,
        dataset_fingerprint="ds-fp-xyz",
        bundle_fingerprint="bf-fp-789",
        bundle_saved_at="2026-04-21T12:00:00+00:00",
        training_source="ai_risk_dataset.xlsx",
        current_analysis_file="ai_risk_dataset.xlsx",
        extra={"n_triple": 0, "language": "English"},
    )

    zip_bytes = build_comparison_full_package(
        triple_ci=_empty_df(["Score", "n", "AUC", "AUPRC", "Brier"]),
        calib_df=_empty_df(["Score", "Calibration intercept", "Calibration slope", "HL chi-square", "HL p-value", "Brier"]),
        formal_df=pd.DataFrame(),
        delong_df=pd.DataFrame(),
        reclass_df=pd.DataFrame(),
        threshold_metrics=_empty_df(["Score", "Sensitivity", "Specificity", "PPV", "NPV"]),
        threshold=0.08,
        n_triple=0,
        model_version="2026-04-21-v13-none-semantics",
        language="English",
        manifest=manifest,
    )

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        assert "manifest.json" in names, f"manifest.json missing from ZIP — got {names!r}"
        embedded = json.loads(zf.read("manifest.json").decode("utf-8"))

    assert embedded["model_version"] == "2026-04-21-v13-none-semantics"
    assert embedded["active_model_name"] == "RandomForest"
    assert embedded["threshold_mode"] == "clinical_fixed"
    assert embedded["threshold_value"] == 0.08
    assert embedded["dataset_fingerprint"] == "ds-fp-xyz"
    assert embedded["bundle_fingerprint"] == "bf-fp-789"
    assert embedded["training_source"] == "ai_risk_dataset.xlsx"


def test_comparison_full_package_omits_manifest_when_none_passed():
    """Backwards-compat: legacy callers that don't pass a manifest should
    still get a working ZIP (just without the manifest entry).  This guards
    against breakage during the rollout."""
    zip_bytes = build_comparison_full_package(
        triple_ci=_empty_df(["Score", "n", "AUC", "AUPRC", "Brier"]),
        calib_df=_empty_df(["Score", "Calibration intercept", "Calibration slope", "HL chi-square", "HL p-value", "Brier"]),
        formal_df=pd.DataFrame(),
        delong_df=pd.DataFrame(),
        reclass_df=pd.DataFrame(),
        threshold_metrics=_empty_df(["Score", "Sensitivity", "Specificity", "PPV", "NPV"]),
        threshold=0.08,
        n_triple=0,
        model_version=MODEL_VERSION,
        language="English",
        manifest=None,
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        assert "manifest.json" not in zf.namelist()


# ---------------------------------------------------------------------------
# 5. End-to-end source-of-truth chain — bundle → bundle_info → manifest
# ---------------------------------------------------------------------------

def test_chain_bundle_info_feeds_manifest_unchanged():
    """The whole point of the refactor: bundle_info fields flow into the
    manifest verbatim, with no opportunity for ``AppConfig.MODEL_VERSION``
    or another stale source to overwrite them along the way."""
    payload = _payload_with(model_version=MODEL_VERSION, best_model_name="RandomForest")
    info = bundle_metadata_from_payload(payload)
    manifest = build_export_manifest(
        export_kind="comparison",
        model_version=info["model_version"],
        active_model_name=info["active_model_name"],
        threshold_mode="clinical_fixed",
        threshold_value=0.08,
        dataset_fingerprint=info["dataset_fingerprint"],
        bundle_fingerprint=info["bundle_fingerprint"],
        bundle_saved_at=info["saved_at"],
        training_source=info["training_source"],
    )
    assert manifest["model_version"] == info["model_version"]
    assert manifest["active_model_name"] == info["active_model_name"]
    assert manifest["dataset_fingerprint"] == info["dataset_fingerprint"]
    assert manifest["bundle_fingerprint"] == info["bundle_fingerprint"]
    assert manifest["bundle_saved_at"] == info["saved_at"]
    assert manifest["training_source"] == info["training_source"]
