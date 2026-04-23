"""Tests for AUPRC_baseline (prevalence) in evaluate_* functions."""

import numpy as np
import pandas as pd
import pytest
from stats_compare import (
    evaluate_scores,
    evaluate_scores_with_threshold,
    evaluate_scores_with_ci,
    evaluate_scores_temporal,
)


def _make_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "outcome": y,
        "score_a": np.clip(rng.random(n), 0.01, 0.99),
        "score_b": np.clip(rng.random(n), 0.01, 0.99),
    })
    return df


def test_evaluate_scores_has_auprc_baseline():
    df = _make_df()
    result = evaluate_scores(df, "outcome", ["score_a"])
    assert "AUPRC_baseline" in result.columns


def test_evaluate_scores_auprc_baseline_equals_prevalence():
    df = _make_df()
    result = evaluate_scores(df, "outcome", ["score_a"])
    expected = float(df["outcome"].mean())
    assert result["AUPRC_baseline"].iloc[0] == pytest.approx(expected, abs=1e-8)


def test_evaluate_scores_with_threshold_has_auprc_baseline():
    df = _make_df()
    result = evaluate_scores_with_threshold(df, "outcome", ["score_a"], threshold=0.08)
    assert "AUPRC_baseline" in result.columns


def test_evaluate_scores_with_ci_has_auprc_baseline():
    df = _make_df(n=60)
    result = evaluate_scores_with_ci(df, "outcome", ["score_a"], n_boot=100, seed=0)
    assert "AUPRC_baseline" in result.columns


def test_evaluate_scores_temporal_has_auprc_baseline():
    df = _make_df()
    result = evaluate_scores_temporal(
        df, "outcome", ["score_a"], threshold=0.08, n_boot=100, seed=0
    )
    assert "AUPRC_baseline" in result.columns
    assert result["AUPRC_baseline"].iloc[0] > 0
