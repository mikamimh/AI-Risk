"""Smoke tests for the extracted tab modules.

These tests verify that:
1. The tab modules can be imported without error.
2. Each module exposes a ``render`` callable.
3. The ``TabContext`` dataclass can be instantiated with minimal stub values.

No Streamlit runtime is required — these are pure import and structural checks.
"""

import types

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. Import checks
# ---------------------------------------------------------------------------

def test_import_tabs_package():
    """tabs package imports without error."""
    import tabs
    assert hasattr(tabs, "TabContext")


def test_import_comparison_module():
    """tabs.comparison imports and exposes render()."""
    from tabs import comparison
    assert callable(getattr(comparison, "render", None))


def test_import_batch_export_module():
    """tabs.batch_export imports and exposes render()."""
    from tabs import batch_export
    assert callable(getattr(batch_export, "render", None))


def test_import_temporal_validation_module():
    """tabs.temporal_validation imports and exposes render()."""
    from tabs import temporal_validation
    assert callable(getattr(temporal_validation, "render", None))
    # Helper moved out of app.py — verify it travelled with the tab.
    assert callable(getattr(temporal_validation, "_sts_score_status_caption", None))


# ---------------------------------------------------------------------------
# 1b. Chronological-state labels — the four canonical states are exposed.
# ---------------------------------------------------------------------------

def test_chronological_state_labels_are_distinct_and_cover_all_states():
    """All four chronological states produce a distinct, non-empty label."""
    from temporal_validation import (
        CHRONO_STATE_NO_OVERLAP,
        CHRONO_STATE_OVERLAP,
        CHRONO_STATE_RETROGRADE,
        CHRONO_STATE_UNKNOWN,
        CHRONO_STATES,
        chronological_state_label,
    )
    assert set(CHRONO_STATES) == {
        CHRONO_STATE_NO_OVERLAP,
        CHRONO_STATE_OVERLAP,
        CHRONO_STATE_RETROGRADE,
        CHRONO_STATE_UNKNOWN,
    }
    for lang in ("English", "Portuguese"):
        labels = {s: chronological_state_label(s, lang) for s in CHRONO_STATES}
        # All four labels must be non-empty and unique within a language.
        assert all(labels.values())
        assert len(set(labels.values())) == len(CHRONO_STATES)
    # English and Portuguese differ for the same state (i18n actually wired).
    assert (
        chronological_state_label(CHRONO_STATE_RETROGRADE, "English")
        != chronological_state_label(CHRONO_STATE_RETROGRADE, "Portuguese")
    )
    # Unknown/garbage input falls back to the "unknown" label (never raises).
    fallback = chronological_state_label("not_a_real_state", "English")
    assert fallback == chronological_state_label(CHRONO_STATE_UNKNOWN, "English")


def test_check_temporal_overlap_status_belongs_to_canonical_set():
    """check_temporal_overlap must only emit one of the four canonical states."""
    from temporal_validation import check_temporal_overlap, CHRONO_STATES
    cases = [
        # (train_start, train_end, val_start, val_end, expected_in_set)
        ("2020-Q1", "2021-Q4", "2022-Q1", "2023-Q4"),   # no_overlap
        ("2020-Q1", "2022-Q4", "2022-Q1", "2023-Q4"),   # overlap
        ("2022-Q1", "2023-Q4", "2020-Q1", "2021-Q4"),   # retrograde
        ("Unknown",  "Unknown",  "2022-Q1", "2023-Q4"), # unknown
    ]
    for ts, te, vs, ve in cases:
        r = check_temporal_overlap(ts, te, vs, ve)
        assert r["status"] in CHRONO_STATES, f"Bad status: {r['status']}"


def test_model_metadata_reexports_chronological_helpers():
    """Back-compat: model_metadata continues to re-export the new helpers."""
    import model_metadata as mm
    assert callable(getattr(mm, "chronological_state_label", None))
    assert hasattr(mm, "CHRONO_STATE_NO_OVERLAP")
    assert hasattr(mm, "CHRONO_STATE_OVERLAP")
    assert hasattr(mm, "CHRONO_STATE_RETROGRADE")
    assert hasattr(mm, "CHRONO_STATE_UNKNOWN")


# ---------------------------------------------------------------------------
# 2. TabContext instantiation
# ---------------------------------------------------------------------------

def _noop(*args, **kwargs):
    """No-op stub for callable fields."""
    pass


def _make_minimal_ctx():
    """Build a TabContext with stub values — no Streamlit required."""
    from tabs import TabContext

    return TabContext(
        tr=lambda en, pt: en,
        hp=lambda en, pt: en,
        language="English",
        prepared=None,
        artifacts=None,
        df=pd.DataFrame(),
        forced_model="stub_model",
        best_model_name="stub_model",
        bundle_info={},
        xlsx_path="/tmp/stub.xlsx",
        default_threshold=0.08,
        model_version="test-v0",
        has_sts=False,
        csv_download_btn=_noop,
        txt_download_btn=_noop,
        bytes_download_btn=_noop,
        update_phase=_noop,
        sts_score_patient_ids=_noop,
        general_table_column_config=_noop,
        stats_table_column_config=_noop,
        format_ppv_npv=_noop,
        to_csv_bytes=_noop,
        safe_prob=_noop,
        plot_roc=_noop,
        plot_calibration=_noop,
        plot_boxplots=_noop,
        plot_ia_model_boxplots=_noop,
        plot_dca=_noop,
        build_methods_text=_noop,
        build_results_text=_noop,
    )


def test_tab_context_instantiation():
    """TabContext can be created with minimal stubs."""
    ctx = _make_minimal_ctx()
    assert ctx.language == "English"
    assert ctx.default_threshold == 0.08
    assert ctx.forced_model == "stub_model"


def test_tab_context_fields_complete():
    """All TabContext fields are present and have expected types."""
    import dataclasses
    from tabs import TabContext

    ctx = _make_minimal_ctx()
    field_names = {f.name for f in dataclasses.fields(TabContext)}

    # Verify all fields were set (no missing kwargs)
    for name in field_names:
        assert hasattr(ctx, name), f"TabContext missing field: {name}"

    # Verify callable fields are actually callable
    callable_fields = [
        "tr", "hp", "csv_download_btn", "txt_download_btn",
        "bytes_download_btn", "update_phase", "sts_score_patient_ids",
        "general_table_column_config", "stats_table_column_config",
        "format_ppv_npv", "to_csv_bytes", "safe_prob",
        "plot_roc", "plot_calibration", "plot_boxplots",
        "plot_ia_model_boxplots", "plot_dca",
        "build_methods_text", "build_results_text",
    ]
    for name in callable_fields:
        assert callable(getattr(ctx, name)), f"TabContext.{name} should be callable"
