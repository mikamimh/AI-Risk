"""Residual governance/traceability tests after the metadata correction.

These cover four narrow invariants added in the residual-fix pass on top of
the main metadata-consistency work:

1. ``dataset_fingerprint`` must NOT contain an absolute on-disk path
   (e.g. ``C:\\Users\\name\\...``) — only the basename + a short SHA1 of
   the full path is exposed.  Stable when the same path is hashed twice.
2. ``_drop_unnamed_index_columns`` strips pandas ghost-index columns
   (``Unnamed: 0``, ``Unnamed: 1``...) at the read boundary so they
   never leak into downstream exports.  Real columns are untouched.
3. The Temporal Validation XLSX builder structure embeds a canonical
   manifest sheet via :func:`build_export_manifest`, with the
   ``model_version`` taken from the bundle (not from the current config).
4. The batch XLSX writer structure embeds the same manifest, and the
   batch CSV reader applies the unnamed-column drop.

For (3) and (4) we exercise the manifest path directly (via
``build_export_manifest``) plus the XLSX writer pattern used inside the
tab — running the full Streamlit tab is out of scope here.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import pytest

from bundle_io import bundle_metadata_from_payload
from export_helpers import build_export_manifest
from risk_data import _drop_unnamed_index_columns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _payload_with_path(path: str, *, mtime_ns: int = 12345, size: int = 999) -> dict:
    return {
        "signature": {
            "xlsx_path": path,
            "xlsx_mtime_ns": mtime_ns,
            "xlsx_size": size,
            "model_version": "stub-v0",
        },
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
                "best_model_name": "stub_model",
            },
        },
        "saved_at": "2026-04-21T12:00:00+00:00",
        "training_source": "stub.xlsx",
    }


# ---------------------------------------------------------------------------
# 1. dataset_fingerprint must not leak local absolute paths
# ---------------------------------------------------------------------------

def test_dataset_fingerprint_does_not_contain_windows_absolute_path():
    """A Windows absolute path must never appear verbatim in the
    fingerprint — the manifest is shipped to other users / auditors and
    must not advertise the local environment."""
    payload = _payload_with_path(r"C:\Users\hikar\AI Risk\local_data\Dataset_2025.csv")
    fp = bundle_metadata_from_payload(payload)["dataset_fingerprint"]
    assert fp is not None
    assert "C:\\" not in fp
    assert "C:/" not in fp
    assert "Users" not in fp
    assert "hikar" not in fp
    assert "local_data" not in fp


def test_dataset_fingerprint_does_not_contain_posix_absolute_path():
    """Same protection for POSIX-style paths — no ``/home/<user>/`` leak."""
    payload = _payload_with_path("/home/user/private/data/cohort.csv")
    fp = bundle_metadata_from_payload(payload)["dataset_fingerprint"]
    assert fp is not None
    assert "/home" not in fp
    assert "/private" not in fp
    assert "user" not in fp


def test_dataset_fingerprint_keeps_basename_for_human_readability():
    """The basename is still useful for an auditor and is safe to expose
    (no directory leakage); confirm it survives the sanitisation."""
    payload = _payload_with_path(r"C:\Users\hikar\AI Risk\Dataset_2025.csv")
    fp = bundle_metadata_from_payload(payload)["dataset_fingerprint"]
    assert "Dataset_2025.csv" in fp


def test_dataset_fingerprint_includes_path_hash_for_uniqueness():
    """Two bundles trained from files with the same basename but different
    directories must still produce different fingerprints — otherwise the
    auditor cannot distinguish them after the path is stripped."""
    fp_a = bundle_metadata_from_payload(
        _payload_with_path(r"C:\Code\dirA\Dataset.csv")
    )["dataset_fingerprint"]
    fp_b = bundle_metadata_from_payload(
        _payload_with_path(r"C:\Code\dirB\Dataset.csv")
    )["dataset_fingerprint"]
    assert fp_a != fp_b
    # Both should contain a short hash token.
    assert re.search(r"path_sha1=[0-9a-f]{6,}", fp_a)
    assert re.search(r"path_sha1=[0-9a-f]{6,}", fp_b)


def test_dataset_fingerprint_stable_for_same_inputs():
    """Two reads of the same payload produce identical fingerprints —
    deterministic by construction."""
    payload = _payload_with_path(r"C:\Code\AI Risk\Dataset.csv")
    fp_a = bundle_metadata_from_payload(payload)["dataset_fingerprint"]
    fp_b = bundle_metadata_from_payload(payload)["dataset_fingerprint"]
    assert fp_a == fp_b


def test_dataset_fingerprint_includes_mtime_and_size_tokens():
    """Mtime/size are still present as part of the auditability chain."""
    payload = _payload_with_path(r"C:\Code\AI Risk\Dataset.csv", mtime_ns=987654, size=42)
    fp = bundle_metadata_from_payload(payload)["dataset_fingerprint"]
    assert "mtime_ns=987654" in fp
    assert "size=42" in fp


# ---------------------------------------------------------------------------
# 2. _drop_unnamed_index_columns — read-boundary cleanup
# ---------------------------------------------------------------------------

def test_drop_unnamed_strips_ghost_index_column():
    """The classic case: a CSV that was written with ``index=True`` is
    re-read and produces an ``Unnamed: 0`` column.  It must be dropped
    silently before any downstream code touches the frame."""
    df = pd.DataFrame({
        "Unnamed: 0": [0, 1, 2],
        "age": [60, 70, 80],
        "outcome": [0, 1, 0],
    })
    cleaned = _drop_unnamed_index_columns(df)
    assert "Unnamed: 0" not in cleaned.columns
    assert list(cleaned.columns) == ["age", "outcome"]
    assert len(cleaned) == 3


def test_drop_unnamed_strips_multiple_ghost_columns():
    """A nested write/read can produce ``Unnamed: 0`` AND ``Unnamed: 1``.
    Both must be removed."""
    df = pd.DataFrame({
        "Unnamed: 0": [0, 1],
        "Unnamed: 1": [0, 1],
        "age": [60, 70],
    })
    cleaned = _drop_unnamed_index_columns(df)
    assert list(cleaned.columns) == ["age"]


def test_drop_unnamed_does_not_touch_real_columns_with_similar_names():
    """``Unnamed`` (no colon) and ``unnamed_thing`` are real columns —
    the regex must be strict to avoid accidental data loss."""
    df = pd.DataFrame({
        "Unnamed": [1, 2],          # no colon — real
        "unnamed_thing": [3, 4],    # lowercase, no colon — real
        "Unnamed: 0": [5, 6],       # ghost — drop
        "age": [60, 70],
    })
    cleaned = _drop_unnamed_index_columns(df)
    assert "Unnamed" in cleaned.columns
    assert "unnamed_thing" in cleaned.columns
    assert "age" in cleaned.columns
    assert "Unnamed: 0" not in cleaned.columns


def test_drop_unnamed_no_op_on_clean_dataframe():
    """No mutation when there are no ghost columns — common case must be
    cheap and side-effect free."""
    df = pd.DataFrame({"age": [60], "outcome": [1]})
    cleaned = _drop_unnamed_index_columns(df)
    assert list(cleaned.columns) == ["age", "outcome"]


def test_drop_unnamed_handles_empty_dataframe():
    """Defensive boundary: never raise on empty input."""
    df = pd.DataFrame()
    cleaned = _drop_unnamed_index_columns(df)
    assert cleaned is not None
    assert cleaned.empty


def test_drop_unnamed_round_trip_csv_with_default_index():
    """Integration-style: simulate the exact bug (write CSV with default
    index=True, re-read, observe ghost column, then drop it)."""
    original = pd.DataFrame({"age": [60, 70], "outcome": [0, 1]})
    buf = io.StringIO()
    original.to_csv(buf)  # index=True by default — this is the bug source
    buf.seek(0)
    reread = pd.read_csv(buf)
    assert "Unnamed: 0" in reread.columns  # confirm we reproduced the bug
    cleaned = _drop_unnamed_index_columns(reread)
    assert "Unnamed: 0" not in cleaned.columns
    assert list(cleaned.columns) == ["age", "outcome"]


# ---------------------------------------------------------------------------
# 3. TV/batch XLSX manifest structure check
# ---------------------------------------------------------------------------

def _manifest_to_property_value_rows(manifest: dict) -> pd.DataFrame:
    """Mirror the layout the tabs use for the ``manifest`` sheet."""
    return pd.DataFrame(
        [{"Property": k, "Value": v} for k, v in manifest.items() if k != "extra"]
        + [{"Property": f"extra.{k}", "Value": v} for k, v in (manifest.get("extra") or {}).items()]
    )


def test_tv_manifest_structure_contains_required_governance_fields():
    """The manifest dict written into the TV xlsx ``manifest`` sheet must
    expose, at minimum, model_version / active_model_name / threshold mode
    and value / fingerprints — the four governance fields demanded by
    the audit contract."""
    payload = _payload_with_path(r"C:\Code\AI Risk\Dataset.csv")
    info = bundle_metadata_from_payload(payload)
    manifest = build_export_manifest(
        export_kind="temporal_validation",
        model_version=info["model_version"],
        active_model_name=info["active_model_name"],
        threshold_mode="clinical_fixed",
        threshold_value=0.08,
        dataset_fingerprint=info["dataset_fingerprint"],
        bundle_fingerprint=info["bundle_fingerprint"],
        bundle_saved_at=info["saved_at"],
        training_source=info["training_source"],
        current_analysis_file="cohort_2026.csv",
        extra={"n_total": 1500, "language": "English"},
    )
    rows = _manifest_to_property_value_rows(manifest)
    props = set(rows["Property"].tolist())
    for required in (
        "export_kind",
        "model_version",
        "active_model_name",
        "threshold_mode",
        "threshold_value",
        "dataset_fingerprint",
        "bundle_fingerprint",
        "generated_at",
    ):
        assert required in props, f"missing property {required!r}"
    # Extras flatten to ``extra.<key>`` rows, also visible in the sheet.
    assert "extra.n_total" in props
    assert "extra.language" in props


def test_tv_manifest_xlsx_writeable_with_openpyxl():
    """The TV manifest sheet must round-trip through openpyxl — guards
    against accidentally putting a non-serialisable Python object in the
    Value column (e.g. a Plotly figure handle)."""
    manifest = build_export_manifest(
        export_kind="temporal_validation",
        model_version="vX",
        active_model_name="RandomForest",
        threshold_mode="clinical_fixed",
        threshold_value=0.08,
        dataset_fingerprint="ds-fp",
        bundle_fingerprint="bf-fp",
        extra={"n_total": 1500},
    )
    rows = _manifest_to_property_value_rows(manifest)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        rows.to_excel(writer, sheet_name="manifest", index=False)
        pd.DataFrame({"age": [60, 70]}).to_excel(writer, sheet_name="data", index=False)
    buf.seek(0)
    sheets = pd.read_excel(buf, sheet_name=None)
    assert "manifest" in sheets
    assert "data" in sheets
    # Round-trip values for the governance fields must survive.
    manifest_df = sheets["manifest"]
    pv = dict(zip(manifest_df["Property"].astype(str), manifest_df["Value"]))
    assert pv["model_version"] == "vX"
    assert pv["active_model_name"] == "RandomForest"
    assert pv["threshold_mode"] == "clinical_fixed"
    assert float(pv["threshold_value"]) == pytest.approx(0.08)


def test_batch_manifest_xlsx_carries_predictions_and_manifest_sheets():
    """The batch xlsx layout pairs a ``manifest`` sheet with a
    ``predictions`` sheet — the auditor reading either sheet alone has
    enough context to know which model produced the numbers."""
    manifest = build_export_manifest(
        export_kind="batch_prediction",
        model_version="vX",
        active_model_name="RandomForest",
        threshold_mode="clinical_fixed",
        threshold_value=0.08,
    )
    predictions = pd.DataFrame({
        "patient_id": [1, 2, 3],
        "ai_risk": [0.05, 0.12, 0.22],
    })
    rows = _manifest_to_property_value_rows(manifest)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        rows.to_excel(writer, sheet_name="manifest", index=False)
        predictions.to_excel(writer, sheet_name="predictions", index=False)
    buf.seek(0)
    sheets = pd.read_excel(buf, sheet_name=None)
    assert {"manifest", "predictions"}.issubset(set(sheets))
    pv = dict(zip(sheets["manifest"]["Property"].astype(str), sheets["manifest"]["Value"]))
    assert pv["export_kind"] == "batch_prediction"
    assert pv["model_version"] == "vX"
    assert "Unnamed: 0" not in sheets["predictions"].columns


# ---------------------------------------------------------------------------
# 4. Source-of-truth — UI metric must read from bundle_info, not config
# ---------------------------------------------------------------------------

def test_bundle_info_carries_model_version_for_ui_display():
    """The Model Snapshot UI metric reads ``bundle_info.get("model_version")``.
    Verify the field is present and equals the signature value, even when
    that disagrees with the config (drift detection runs separately)."""
    payload = _payload_with_path(r"C:\Code\AI Risk\Dataset.csv")
    payload["signature"]["model_version"] = "loaded-vX"
    info = bundle_metadata_from_payload(payload)
    assert info.get("model_version") == "loaded-vX"


# ---------------------------------------------------------------------------
# 5. tabs/temporal_validation.py must not export module-level MODEL_VERSION
# ---------------------------------------------------------------------------

def test_temporal_validation_module_does_not_define_module_level_MODEL_VERSION():
    """Removing the module-level binding closes the residual risk that some
    future code outside ``render()`` accidentally references it.  This test
    locks in the absence — any future re-introduction will trip it."""
    import tabs.temporal_validation as tv_mod
    assert not hasattr(tv_mod, "MODEL_VERSION"), (
        "tabs.temporal_validation must not expose a module-level MODEL_VERSION; "
        "the canonical version lives on TabContext.model_version."
    )
