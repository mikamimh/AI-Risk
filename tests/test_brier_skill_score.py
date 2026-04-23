"""Sanity tests for Brier Skill Score."""

import numpy as np
import pytest
from stats_compare import brier_skill_score


def test_perfect_model_bss_one():
    """BSS = 1 for a perfectly calibrated model (predicts exact outcome)."""
    y = np.array([0, 0, 1, 1, 0, 1])
    p = y.astype(float)  # p[i] = y[i] exactly
    assert brier_skill_score(y, p) == pytest.approx(1.0)


def test_constant_model_bss_zero():
    """BSS = 0 when model always predicts the prevalence."""
    y = np.array([0, 0, 1, 1, 0, 1, 0])
    p = np.full(len(y), y.mean())
    assert brier_skill_score(y, p) == pytest.approx(0.0, abs=1e-9)


def test_worse_than_constant_bss_negative():
    """BSS < 0 when model is worse than predicting prevalence."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Predict the opposite of the correct label → maximally wrong
    p = 1.0 - y.astype(float)
    bss = brier_skill_score(y, p)
    assert bss < 0.0, f"Expected BSS < 0, got {bss:.4f}"


def test_bss_nan_for_constant_outcome():
    """BSS is NaN when all outcomes are identical (no variance)."""
    y = np.zeros(10, dtype=int)
    p = np.full(10, 0.1)
    assert np.isnan(brier_skill_score(y, p))


def test_bss_in_evaluate_scores():
    """BSS column appears in evaluate_scores output."""
    import pandas as pd
    from stats_compare import evaluate_scores

    rng = np.random.default_rng(0)
    n = 100
    y = (rng.random(n) > 0.85).astype(int)
    df = pd.DataFrame({
        "morte_30d": y,
        "score_a": np.clip(rng.random(n), 0.01, 0.99),
    })
    result = evaluate_scores(df, "morte_30d", ["score_a"])
    assert "BSS" in result.columns


def test_bss_in_evaluate_scores_temporal():
    """BSS column appears in evaluate_scores_temporal output."""
    import pandas as pd
    from stats_compare import evaluate_scores_temporal

    rng = np.random.default_rng(1)
    n = 100
    y = (rng.random(n) > 0.85).astype(int)
    df = pd.DataFrame({
        "morte_30d": y,
        "score_a": np.clip(rng.random(n), 0.01, 0.99),
    })
    result = evaluate_scores_temporal(df, "morte_30d", ["score_a"], threshold=0.08, n_boot=100, seed=0)
    assert "BSS" in result.columns
