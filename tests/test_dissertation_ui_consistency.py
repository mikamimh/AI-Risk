"""The UI (Models / Comparison tabs) and the dissertation ZIP must show the
same numbers: both go through the compute blocks of ``dissertation_exports``.

These tests call the tabs' own cached wrappers and compare their output with
the CSVs written to the ZIP from the same inputs.
"""

import numpy as np
import pandas as pd
import pytest

import dissertation_exports as de

shap = pytest.importorskip("shap")

N_BOOT = 40
SEED = 42
THR = 0.085


@pytest.fixture(scope="module")
def setup():
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(3)
    n = 300
    X = pd.DataFrame({"risk_up": rng.normal(size=n), "noise_a": rng.normal(size=n), "noise_b": rng.normal(size=n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.0 + 2.0 * X["risk_up"])))).astype(int)
    pipe = Pipeline([
        ("prep", ColumnTransformer([("num", StandardScaler(), list(X.columns))])),
        ("model", RandomForestClassifier(n_estimators=60, min_samples_leaf=5, random_state=0)),
    ]).fit(X, y)

    p = np.clip(pipe.predict_proba(X)[:, 1] * rng.uniform(0.6, 1.2, n), 0.001, 0.999)
    euro = np.clip(p * rng.uniform(0.3, 1.1, n), 0.001, 0.99)
    sts = np.clip(p * rng.uniform(0.4, 1.3, n), 0.001, 0.99)
    sts[rng.random(n) < 0.3] = np.nan
    df = pd.DataFrame({"morte_30d": y, "ia_risk_oof": p, "euroscore_calc": euro, "sts_score": sts})

    files = de.build_dissertation_exports(df, THR, pipe, X, n_boot=N_BOOT, seed=SEED)
    return {"df": df, "pipe": pipe, "X": X, "files": files}


def _ui_policy(df):
    from tabs.comparison import _cached_threshold_policy

    return _cached_threshold_policy("test-bundle", THR, df.reindex(columns=de.TRIPLE_COLUMNS))


@pytest.mark.parametrize("key,fname", [
    ("B1", "B1_sensitivity_targets.csv"),
    ("B3", "B3_risk_bands_triple.csv"),
    ("B4", "B4_ruleout_primary_threshold.csv"),
])
def test_threshold_policy_tables_match_zip(setup, key, fname):
    ui = _ui_policy(setup["df"])[key].reset_index(drop=True)
    csv = de.read_export_csv(setup["files"][fname])
    pd.testing.assert_frame_equal(ui, csv, check_exact=True)


def test_b2_displayed_summary_matches_zip(setup):
    from tabs.comparison import _cached_bootstrap_block

    df = setup["df"]
    y, p = de.development_oof_arrays(df)
    blk = _cached_bootstrap_block("test-bundle", de.oof_hash(y, p), N_BOOT, SEED, THR, y, p)
    shown = de.b2_display_summary(blk["summary"])
    csv = de.b2_display_summary(de.read_export_csv(setup["files"]["B2_threshold_bootstrap_summary.csv"]))
    assert list(shown["metrica"]) == list(csv["metrica"])
    assert len(shown) == len(de.B2_DISPLAY_METRICS)
    np.testing.assert_array_equal(shown["valor"].astype(float).to_numpy(), csv["valor"].astype(float).to_numpy())
    # same draws as the ZIP and same figure bytes are what the UI shows
    pd.testing.assert_frame_equal(blk["draws"], de.read_export_csv(setup["files"]["B2_threshold_bootstrap_draws.csv"]))


def test_a1_displayed_table_matches_zip(setup):
    from tabs.models import _cached_shap_block

    blk = _cached_shap_block("x.xlsx", "RF", "vT", setup["pipe"], setup["X"])
    shown = de.shap_direction_display(blk["a1"])
    csv = de.read_export_csv(setup["files"]["A1_shap_global_top15.csv"])
    assert list(shown["feature"]) == list(csv["feature"])
    np.testing.assert_array_equal(shown["mean_abs_shap"].to_numpy(), csv["mean_abs_shap"].to_numpy())
    assert all(d[0] in "↑↓·–" for d in shown["direction"])


def test_oof_hash_changes_with_predictions(setup):
    y, p = de.development_oof_arrays(setup["df"])
    assert de.oof_hash(y, p) == de.oof_hash(y.copy(), p.copy())
    p2 = p.copy()
    p2[0] += 1e-6
    assert de.oof_hash(y, p) != de.oof_hash(y, p2)


def test_policy_without_sts_is_empty_triple(setup):
    df = setup["df"].copy()
    df["sts_score"] = np.nan
    pol = de.compute_threshold_policy(df.reindex(columns=de.TRIPLE_COLUMNS), THR)
    assert pol["n_triple"] == 0
    assert (pol["B3"]["n"] == 0).all()
    assert len(pol["B4"]) == 2
