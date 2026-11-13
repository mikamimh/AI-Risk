"""Tests for dissertation_exports (SHAP summary + threshold-policy analyses)."""

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

import dissertation_exports as de

shap = pytest.importorskip("shap")


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_pipe():
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({
        "risk_up": rng.normal(size=n),
        "noise_a": rng.normal(size=n),
        "noise_b": rng.normal(size=n),
    })
    logit = -2.0 + 2.5 * X["risk_up"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    prep = ColumnTransformer([("num", StandardScaler(), list(X.columns))])
    pipe = Pipeline([
        ("prep", prep),
        ("model", RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=0)),
    ])
    pipe.fit(X, y)
    return pipe, X


def _cohort_df(n=300, seed=1, missing_sts=0.25):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(1.2, 12, size=n), 0.001, 0.999)
    y = (rng.random(n) < p * 1.4).astype(int)
    euro = np.clip(p * rng.uniform(0.4, 1.2, n), 0.001, 0.99)
    sts = np.clip(p * rng.uniform(0.5, 1.5, n), 0.001, 0.99)
    sts[rng.random(n) < missing_sts] = np.nan
    return pd.DataFrame({"morte_30d": y, "ia_risk_oof": p, "euroscore_calc": euro, "sts_score": sts})


# ── Block A ─────────────────────────────────────────────────────────────────

def test_shap_summary_direction_and_order(synthetic_pipe):
    from explainability import shap_global_summary

    pipe, X = synthetic_pipe
    res = shap_global_summary(pipe, X, top_n=3)
    tbl = res["table"]
    assert res["explained_output"] in ("uncalibrated_rf_probability", "calibrated_probability")
    assert list(tbl["mean_abs_shap"]) == sorted(tbl["mean_abs_shap"], reverse=True)
    assert list(tbl["rank"]) == [1, 2, 3]
    row = tbl.loc[tbl["feature"] == "risk_up"].iloc[0]
    assert row["spearman_value_vs_shap"] > 0.5
    assert res["shap_values"].shape == res["X_proc"].shape == (len(X), 3)
    # uncalibrated sklearn Pipeline: no calibrator on top
    assert res["final_output_calibrated"] is False


def test_shap_summary_detects_calibrated_wrapper(synthetic_pipe):
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV

    from explainability import shap_global_summary
    from modeling import ClippedPipeline

    pipe, X = synthetic_pipe
    y = (X["risk_up"] > 0.8).astype(int)
    cal = CalibratedClassifierCV(clone(pipe), method="sigmoid", cv=3, ensemble=False).fit(X, y)
    res = shap_global_summary(ClippedPipeline(cal), X, top_n=3)
    assert res["explained_output"] == "uncalibrated_rf_probability"
    assert res["final_output_calibrated"] is True
    assert res["calibration_method"] == "sigmoid"


# ── Block B ─────────────────────────────────────────────────────────────────

def test_b1_threshold_non_increasing_with_target():
    df = _cohort_df()
    y, p = de.development_oof_arrays(df)
    tri = de.build_triple_cohort(df)
    out = de.sensitivity_targets_table(
        y, p, tri["morte_30d"].to_numpy(), tri["ia_risk_oof"].to_numpy(),
    )
    dev = out[out["coorte"] == de.COHORT_DEV].sort_values("alvo_sensibilidade")
    thr = dev["limiar"].to_numpy()
    assert thr[0] >= thr[1] >= thr[2]
    assert set(out["coorte"]) == {de.COHORT_DEV, de.COHORT_TRIPLE}
    # locked threshold: same value applied in both cohorts
    for tgt, g in out.groupby("alvo_sensibilidade"):
        assert g["limiar"].nunique() == 1


def test_b2_bootstrap_reproducible_and_consistent():
    df = _cohort_df()
    y, p = de.development_oof_arrays(df)
    s1, d1, i1 = de.threshold_bootstrap(y, p, n_boot=150, seed=42, stored_threshold=0.085)
    s2, d2, i2 = de.threshold_bootstrap(y, p, n_boot=150, seed=42, stored_threshold=0.085)
    pd.testing.assert_frame_equal(s1, s2)
    assert len(d1) == 150
    assert i1["t_b_p2_5"] <= i1["t_b_mediana"] <= i1["t_b_p97_5"]
    assert i1["flag_rate_boot_p2_5"] <= i1["flag_rate_boot_mediana"] <= i1["flag_rate_boot_p97_5"]
    assert i1["sensibilidade_corrigida_media_teste"] <= i1["sensibilidade_aparente"] + 0.01
    # defining death: threshold equals its predicted probability, FN within budget
    assert i1["obito_definidor_coincide_com_limiar"]
    assert i1["fn_observado"] <= i1["fn_max_permitido"]
    assert i1["fn_max_permitido"] == int(np.floor(0.10 * y.sum() + 1e-9))


def test_b3_band_counts_sum_to_cohort():
    df = _cohort_df()
    tri = de.build_triple_cohort(df)
    bands = de.risk_bands_table(tri)
    assert set(bands["metodo"]) == {"AI Risk", "EuroSCORE II", "STS"}
    for _, g in bands.groupby("metodo"):
        assert len(g) == 3
        assert g["n"].sum() == len(tri)
        assert g["obitos"].sum() == tri["morte_30d"].sum()


def test_b4_ruleout_partitions_cohort():
    df = _cohort_df()
    y, p = de.development_oof_arrays(df)
    tri = de.build_triple_cohort(df)
    out = de.ruleout_table(y, p, tri["morte_30d"].to_numpy(), tri["ia_risk_oof"].to_numpy(), 0.085)
    for _, r in out.iterrows():
        assert r["n_abaixo"] + r["n_acima"] == r["n"]
        assert r["mortalidade_abaixo_ic95_inf"] <= r["mortalidade_abaixo"] <= r["mortalidade_abaixo_ic95_sup"]


def test_wilson_ci_known_value():
    lo, hi = de.wilson_ci(6, 157)
    assert lo == pytest.approx(0.01763, abs=1e-4)
    assert hi == pytest.approx(0.08086, abs=1e-4)


def test_triple_cohort_drops_missing_scores():
    df = _cohort_df()
    tri = de.build_triple_cohort(df)
    assert len(tri) == int(df["sts_score"].notna().sum())
    assert not tri.isna().any().any()


# ── Orchestration ───────────────────────────────────────────────────────────

def test_zip_contains_all_files(synthetic_pipe):
    pipe, X = synthetic_pipe
    df = _cohort_df(n=len(X))
    data = de.build_dissertation_export_zip(df, 0.085, pipe, X, model_name="RF", model_version="vT", n_boot=20)
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = set(zf.namelist())
    expected = {
        "A1_shap_global_top15.csv", "A2_fig_shap_bar.png", "A2_fig_shap_bar.svg",
        "A3_fig_shap_beeswarm.png", "A3_fig_shap_beeswarm.svg",
        "B1_sensitivity_targets.csv", "B2_threshold_bootstrap_summary.csv",
        "B2_threshold_bootstrap_draws.csv", "B2_fig_threshold_bootstrap.png",
        "B2_fig_threshold_bootstrap.svg", "B3_risk_bands_triple.csv",
        "B4_ruleout_primary_threshold.csv", "metadata.json",
    }
    assert expected <= names
    meta = json.loads(zf.read("metadata.json"))
    assert meta["coorte_desenvolvimento"]["n"] == len(df)
    assert meta["seeds"]["bootstrap"] == 42
    assert meta["shap_version"] == shap.__version__
    assert meta["shap"]["explained_output"] == "uncalibrated_rf_probability"
