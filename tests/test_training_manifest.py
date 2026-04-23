"""Tests for training_manifest in TrainedArtifacts."""

import numpy as np
import pandas as pd
import pytest
from modeling import train_and_select_model


def _make_minimal_df(seed: int = 0, n: int = 80) -> tuple:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "_patient_key": [f"P{i}" for i in range(n)],
        "morte_30d": (rng.random(n) > 0.85).astype(int),
        "Age (years)": rng.uniform(40, 80, n),
        "Pré-LVEF, %": rng.uniform(30, 70, n),
        "Creatinine (mg/dL)": rng.uniform(0.5, 3.0, n),
    })
    feature_columns = ["Age (years)", "Pré-LVEF, %", "Creatinine (mg/dL)"]
    return df, feature_columns


def test_training_manifest_not_none():
    df, feature_columns = _make_minimal_df()
    artifacts = train_and_select_model(df, feature_columns)
    assert artifacts.training_manifest is not None


def test_training_manifest_has_expected_keys():
    df, feature_columns = _make_minimal_df()
    artifacts = train_and_select_model(df, feature_columns)
    manifest = artifacts.training_manifest
    required_keys = {
        "generated_at", "n_rows", "n_events", "prevalence",
        "n_features", "feature_columns", "cv_strategy", "cv_splits",
        "seed", "best_model", "calibration_method",
        "oof_auc", "oof_auprc", "oof_brier", "model_version",
    }
    missing = required_keys - set(manifest.keys())
    assert not missing, f"Manifest missing keys: {missing}"


def test_training_manifest_n_rows_matches():
    df, feature_columns = _make_minimal_df()
    artifacts = train_and_select_model(df, feature_columns)
    assert artifacts.training_manifest["n_rows"] == len(df)


def test_training_manifest_n_events_matches():
    df, feature_columns = _make_minimal_df()
    artifacts = train_and_select_model(df, feature_columns)
    assert artifacts.training_manifest["n_events"] == int(df["morte_30d"].sum())


def test_training_manifest_prevalence_correct():
    df, feature_columns = _make_minimal_df()
    artifacts = train_and_select_model(df, feature_columns)
    expected = float(df["morte_30d"].mean())
    assert artifacts.training_manifest["prevalence"] == pytest.approx(expected)


def test_training_manifest_best_model_matches_artifact():
    df, feature_columns = _make_minimal_df()
    artifacts = train_and_select_model(df, feature_columns)
    assert artifacts.training_manifest["best_model"] == artifacts.best_model_name


def test_training_manifest_oof_auc_finite():
    df, feature_columns = _make_minimal_df()
    artifacts = train_and_select_model(df, feature_columns)
    assert np.isfinite(artifacts.training_manifest["oof_auc"])


def test_training_manifest_dataset_hash_present():
    df, feature_columns = _make_minimal_df()
    artifacts = train_and_select_model(df, feature_columns)
    dh = artifacts.training_manifest.get("dataset_hash")
    assert dh is not None
    assert isinstance(dh, str)
    assert len(dh) == 24
