"""Regression tests: threshold-policy labeling contract.

Ensures that 8% is never labelled as the primary threshold when a
threshold_policy is present in the bundle, legacy bundles get an
explicit fallback label, and the README does not contain stale
"8% locked" or "8% primary" phrasing.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pandas as pd
import pytest

from stats_compare import (
    THRESHOLD_ROLE_EXPLORATORY,
    THRESHOLD_ROLE_HISTORICAL_COMPARATOR,
    THRESHOLD_ROLE_PRIMARY,
    threshold_analysis_table,
)
from tabs.comparison import _build_threshold_comparison_export_df as _build_threshold_comparison_table


# ── helpers ───────────────────────────────────────────────────────────────────

class _FakeArtifacts:
    """Minimal stand-in for TrainedArtifacts."""

    def __init__(self, threshold_policy=None, youden_thresholds=None, best_youden_threshold=None):
        self.threshold_policy = threshold_policy
        self.youden_thresholds = youden_thresholds or {}
        self.best_youden_threshold = best_youden_threshold


def _make_df(n: int = 100, prevalence: float = 0.15, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    p = np.clip(y * 0.5 + rng.random(n) * 0.4, 0.01, 0.99)
    return pd.DataFrame({"ia_risk_oof": p, "morte_30d": y})


_POLICY_OK = {
    "status": "ok",
    "selected_threshold": 0.085,
    "target_sensitivity": 0.90,
    "sensitivity": 0.912,
    "specificity": 0.350,
}


# ── 8% is HISTORICAL_COMPARATOR when threshold_policy exists ──────────────────

def test_8pct_role_is_historical_comparator_when_policy_present():
    """8% must carry HISTORICAL_COMPARATOR role, never PRIMARY, when threshold_policy is set."""
    df = _make_df()
    artifacts = _FakeArtifacts(threshold_policy=_POLICY_OK)
    result = _build_threshold_comparison_table(df, artifacts, forced_model="RandomForest")
    row_8 = result[result["Threshold"].apply(lambda v: math.isfinite(v) and abs(v - 0.08) < 1e-9)]
    assert not row_8.empty, "8% row must always appear in the comparison table"
    roles = row_8["Threshold Role"].tolist()
    assert all(r == THRESHOLD_ROLE_HISTORICAL_COMPARATOR for r in roles), (
        f"8% row role(s) must be HISTORICAL_COMPARATOR; got {roles}"
    )


def test_8pct_role_is_not_primary_when_policy_present():
    """8% must not receive the PRIMARY role when threshold_policy is present."""
    df = _make_df()
    artifacts = _FakeArtifacts(threshold_policy=_POLICY_OK)
    result = _build_threshold_comparison_table(df, artifacts, forced_model="RandomForest")
    row_8 = result[result["Threshold"].apply(lambda v: math.isfinite(v) and abs(v - 0.08) < 1e-9)]
    assert not row_8.empty
    for role in row_8["Threshold Role"].tolist():
        assert role != THRESHOLD_ROLE_PRIMARY, (
            "8% must not be labelled Primary when threshold_policy is present"
        )


# ── primary row uses policy threshold when threshold_policy is present ─────────

def test_primary_row_threshold_matches_policy_when_policy_present():
    """The PRIMARY row must use threshold_policy['selected_threshold'], not 0.08."""
    df = _make_df()
    artifacts = _FakeArtifacts(threshold_policy=_POLICY_OK)
    result = _build_threshold_comparison_table(df, artifacts, forced_model="RandomForest")
    primary_rows = result[result["Threshold Role"] == THRESHOLD_ROLE_PRIMARY]
    assert not primary_rows.empty, "A PRIMARY row must exist when threshold_policy is set"
    primary_thr = primary_rows["Threshold"].iloc[0]
    assert math.isfinite(primary_thr), "PRIMARY row threshold must be finite"
    assert abs(primary_thr - 0.085) < 1e-9, (
        f"PRIMARY threshold must match policy selected_threshold=0.085; got {primary_thr}"
    )


def test_primary_row_threshold_is_not_0_08_when_policy_differs():
    """If policy selects a threshold other than 0.08, primary row is NOT 0.08."""
    df = _make_df()
    policy = {**_POLICY_OK, "selected_threshold": 0.095}
    artifacts = _FakeArtifacts(threshold_policy=policy)
    result = _build_threshold_comparison_table(df, artifacts, forced_model="RandomForest")
    primary_rows = result[result["Threshold Role"] == THRESHOLD_ROLE_PRIMARY]
    if not primary_rows.empty:
        primary_thr = primary_rows["Threshold"].iloc[0]
        if math.isfinite(primary_thr):
            assert abs(primary_thr - 0.08) > 1e-9, (
                "Primary threshold must not equal 0.08 when policy selects a different value"
            )


# ── legacy bundles (no threshold_policy) show NaN / absent primary row ─────────

def test_legacy_bundle_primary_row_is_nan():
    """Legacy bundles (threshold_policy=None) produce no finite PRIMARY threshold."""
    df = _make_df()
    artifacts = _FakeArtifacts(threshold_policy=None)
    result = _build_threshold_comparison_table(df, artifacts, forced_model="RandomForest")
    primary_rows = result[result["Threshold Role"] == THRESHOLD_ROLE_PRIMARY]
    if not primary_rows.empty:
        for thr in primary_rows["Threshold"].tolist():
            assert not math.isfinite(thr), (
                "Legacy bundle PRIMARY row threshold must be NaN (no threshold_policy)"
            )


def test_legacy_bundle_8pct_still_appears_as_historical_comparator():
    """Even for legacy bundles, 8% must appear as HISTORICAL_COMPARATOR, never PRIMARY."""
    df = _make_df()
    artifacts = _FakeArtifacts(threshold_policy=None)
    result = _build_threshold_comparison_table(df, artifacts, forced_model="RandomForest")
    row_8 = result[result["Threshold"].apply(lambda v: math.isfinite(v) and abs(v - 0.08) < 1e-9)]
    assert not row_8.empty, "8% row must appear even in legacy bundles"
    for role in row_8["Threshold Role"].tolist():
        assert role == THRESHOLD_ROLE_HISTORICAL_COMPARATOR, (
            f"Legacy bundle: 8% must still be HISTORICAL_COMPARATOR; got {role}"
        )


def test_policy_with_status_not_ok_gives_nan_primary():
    """A threshold_policy dict with status != 'ok' must not produce a finite primary threshold."""
    df = _make_df()
    policy = {**_POLICY_OK, "status": "not_available"}
    artifacts = _FakeArtifacts(threshold_policy=policy)
    result = _build_threshold_comparison_table(df, artifacts, forced_model="RandomForest")
    primary_rows = result[result["Threshold Role"] == THRESHOLD_ROLE_PRIMARY]
    if not primary_rows.empty:
        for thr in primary_rows["Threshold"].tolist():
            assert not math.isfinite(thr), (
                "threshold_policy with status != 'ok' must not produce a finite PRIMARY threshold"
            )


# ── README must not contain stale phrasing ────────────────────────────────────

_README_PATH = pathlib.Path(__file__).parent.parent / "README.md"


def test_readme_does_not_contain_8pct_locked():
    """README must not contain the phrase '8% locked' (stale primary framing)."""
    text = _README_PATH.read_text(encoding="utf-8")
    assert "8% locked" not in text.lower(), (
        "README contains stale '8% locked' phrasing; replace with bundle-stored "
        "primary threshold policy language"
    )


def test_readme_does_not_contain_8pct_primary():
    """README must not describe 8% as 'primary' without qualification."""
    text = _README_PATH.read_text(encoding="utf-8")
    # "8% primary" as a direct label is disallowed; "former primary" is fine
    assert "8% primary" not in text.lower(), (
        "README must not label 8% as 'primary'; it is a historical/sensitivity comparator"
    )


def test_readme_threshold_policy_section_present():
    """README must document the sensitivity-constrained 90% threshold policy."""
    text = _README_PATH.read_text(encoding="utf-8")
    assert "sensitivity-constrained" in text.lower() or "threshold_policy" in text, (
        "README must document the sensitivity-constrained threshold policy"
    )
