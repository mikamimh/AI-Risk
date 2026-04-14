"""Guardrail tests for DeLong ROC comparison on sparse validation cohorts.

Background: on a temporal validation cohort with ``n=24`` and a single
event, ``stats_compare.delong_roc_test`` used to emit three numpy
``RuntimeWarning`` messages at the covariance step (``Degrees of freedom <=
0``, ``divide by zero encountered in divide``, ``invalid value encountered
in multiply``) and return ``p = NaN``.  The math is simply undefined for
fewer than two observations per class — the fix is to short-circuit before
``np.cov`` runs, not to try to silence the warnings.

These tests pin the guardrail: sparse cohorts must skip DeLong cleanly and
surface a human-readable reason; adequate cohorts must still compute it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from stats_compare import (
    delong_roc_test,
    pairwise_score_comparison,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_no_runtime_warnings(record):
    """Fail if numpy emitted the cov-collapse warnings during the call."""
    bad = [w for w in record if issubclass(w.category, RuntimeWarning)]
    assert not bad, f"Unexpected RuntimeWarnings: {[str(w.message) for w in bad]}"


# ---------------------------------------------------------------------------
# 1. delong_roc_test — direct guardrail
# ---------------------------------------------------------------------------

def test_delong_skipped_when_single_positive():
    """1 event / many non-events — classic temporal-cohort scenario."""
    rng = np.random.default_rng(0)
    n = 24
    y = np.zeros(n, dtype=int)
    y[0] = 1  # exactly one positive
    p1 = rng.uniform(0, 1, n)
    p2 = rng.uniform(0, 1, n)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = delong_roc_test(y, p1, p2)

    _assert_no_runtime_warnings(w)
    assert np.isnan(result["p"])
    assert np.isnan(result["z"])
    assert isinstance(result["reason"], str) and result["reason"]
    assert "2 events" in result["reason"] or "2 non-events" in result["reason"]


def test_delong_skipped_when_single_negative():
    """1 non-event / many events — symmetric sparse case."""
    rng = np.random.default_rng(1)
    n = 24
    y = np.ones(n, dtype=int)
    y[0] = 0  # exactly one negative
    p1 = rng.uniform(0, 1, n)
    p2 = rng.uniform(0, 1, n)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = delong_roc_test(y, p1, p2)

    _assert_no_runtime_warnings(w)
    assert np.isnan(result["p"])
    assert np.isnan(result["z"])
    assert isinstance(result["reason"], str) and result["reason"]


def test_delong_skipped_when_all_same_class():
    """Pre-existing degenerate case — no positives at all."""
    y = np.zeros(30, dtype=int)
    p1 = np.linspace(0.1, 0.9, 30)
    p2 = np.linspace(0.2, 0.8, 30)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = delong_roc_test(y, p1, p2)

    _assert_no_runtime_warnings(w)
    assert np.isnan(result["p"])
    assert isinstance(result["reason"], str) and result["reason"]


def test_delong_runs_with_adequate_class_counts():
    """Two events and two non-events is the hard minimum — above it the
    test must still compute a numeric p-value and set ``reason = None``."""
    rng = np.random.default_rng(42)
    n = 100
    # Balanced-ish: ~20% events, well above the 2-per-class floor.
    y = (rng.uniform(0, 1, n) < 0.2).astype(int)
    # Ensure at least 2 per class even on unlucky RNGs.
    y[0] = 1
    y[1] = 1
    y[2] = 0
    y[3] = 0
    p1 = rng.uniform(0, 1, n)
    p2 = rng.uniform(0, 1, n)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = delong_roc_test(y, p1, p2)

    _assert_no_runtime_warnings(w)
    assert not np.isnan(result["p"])
    assert 0.0 <= result["p"] <= 1.0
    assert not np.isnan(result["z"])
    assert result["reason"] is None


def test_delong_runs_with_exactly_two_per_class():
    """The floor is 2 positives / 2 negatives — must not be skipped."""
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    p1 = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    p2 = np.array([0.85, 0.7, 0.8, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = delong_roc_test(y, p1, p2)

    _assert_no_runtime_warnings(w)
    # Either a real p-value or, at worst, the degenerate-variance branch —
    # never the sparse-cohort skip.
    assert result["reason"] is None or "variance" in result["reason"].lower()


# ---------------------------------------------------------------------------
# 2. pairwise_score_comparison — reason propagates to the DataFrame
# ---------------------------------------------------------------------------

def test_pairwise_surfaces_delong_skip_reason_on_sparse_cohort():
    """The exact shape of the bug report: n=24, 1 event.  The pairwise
    DataFrame must include the comparison row with ``DeLong_p = NaN`` and
    a non-empty ``DeLong_skip_reason`` string, and no RuntimeWarnings."""
    rng = np.random.default_rng(7)
    n = 24
    y = np.zeros(n, dtype=int)
    y[0] = 1
    df = pd.DataFrame({
        "morte_30d": y,
        "ia_risk": rng.uniform(0, 1, n),
        "euroscore_calc": rng.uniform(0, 1, n),
    })

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = pairwise_score_comparison(
            df, "morte_30d", [("ia_risk", "euroscore_calc")],
            n_boot=200, seed=0,
        )

    _assert_no_runtime_warnings(w)
    assert not out.empty
    assert "DeLong_skip_reason" in out.columns
    row = out.iloc[0]
    assert np.isnan(row["DeLong_p"])
    assert isinstance(row["DeLong_skip_reason"], str) and row["DeLong_skip_reason"]
    # Bootstrap is preserved — its CI bounds must still be finite.
    assert pd.notna(row["Delta_AUC_IC95_inf"])
    assert pd.notna(row["Delta_AUC_IC95_sup"])


def test_pairwise_leaves_delong_reason_null_on_adequate_cohort():
    """When class counts support DeLong, the reason column must be null
    for that row (so the UI renders only a p-value, no footnote)."""
    rng = np.random.default_rng(3)
    n = 150
    y = (rng.uniform(0, 1, n) < 0.25).astype(int)
    y[0], y[1], y[2], y[3] = 1, 1, 0, 0  # guarantee ≥2 per class
    df = pd.DataFrame({
        "morte_30d": y,
        "ia_risk": rng.uniform(0, 1, n),
        "euroscore_calc": rng.uniform(0, 1, n),
    })

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = pairwise_score_comparison(
            df, "morte_30d", [("ia_risk", "euroscore_calc")],
            n_boot=200, seed=0,
        )

    _assert_no_runtime_warnings(w)
    assert not out.empty
    row = out.iloc[0]
    assert pd.notna(row["DeLong_p"])
    assert row["DeLong_skip_reason"] is None or (
        isinstance(row["DeLong_skip_reason"], float) and np.isnan(row["DeLong_skip_reason"])
    )
