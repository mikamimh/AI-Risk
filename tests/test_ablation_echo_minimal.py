"""Unit tests for ablation_echo_minimal helpers.

All tests are isolated from the production pipeline — no data files, sklearn
models, or the official bundle are required.  Only pure helper functions that
accept plain Python structures are exercised here.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
import pytest

from ablation_echo_minimal import (
    ECHO_PRESERVE_CONTINUOUS,
    ECHO_QUANTITATIVE_VALVULAR,
    OPERATIONAL_THRESHOLD,
    VALVE_SEVERITY_CATEGORICAL,
    _bootstrap_diff_auprc,
    _bootstrap_diff_brier,
    _threshold_metrics,
    build_echo_minimal_features,
    build_features_dataframe,
    compute_arm_metrics,
    compute_missingness_summary,
    features_in_both,
    features_removed_from_baseline,
    interpret_result,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_probas(n: int = 300, prevalence: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    p = np.clip(y * 0.45 + rng.random(n) * 0.35, 0.01, 0.99)
    return y, p


def _sample_baseline_features() -> List[str]:
    """Synthetic baseline feature list that includes all echo categories."""
    return [
        "Age (years)",
        "Sex",
        "Pré-LVEF, %",
        "PSAP",
        "TAPSE",
        "Aortic Stenosis",
        "Aortic Regurgitation",
        "Mitral Stenosis",
        "Mitral Regurgitation",
        "Tricuspid Regurgitation",
        "AVA (cm²)",
        "MVA (cm²)",
        "Aortic Mean gradient (mmHg)",
        "Mitral Mean gradient (mmHg)",
        "PHT Aortic",
        "PHT Mitral",
        "Vena contracta",
        "Vena contracta (mm)",
        "Creatinine (mg/dL)",
        "Hematocrit (%)",
    ]


# ── build_echo_minimal_features ───────────────────────────────────────────────


class TestBuildEchoMinimalFeatures:
    def test_quantitative_valvular_features_removed(self):
        baseline = _sample_baseline_features()
        minimal = build_echo_minimal_features(baseline)
        for feat in ECHO_QUANTITATIVE_VALVULAR:
            assert feat not in minimal, f"{feat!r} should be removed but is present"

    def test_psap_preserved(self):
        baseline = _sample_baseline_features()
        minimal = build_echo_minimal_features(baseline)
        assert "PSAP" in minimal, "PSAP must be preserved in echo-minimal"

    def test_tapse_preserved(self):
        baseline = _sample_baseline_features()
        minimal = build_echo_minimal_features(baseline)
        assert "TAPSE" in minimal, "TAPSE must be preserved in echo-minimal"

    def test_lvef_preserved(self):
        baseline = _sample_baseline_features()
        minimal = build_echo_minimal_features(baseline)
        assert "Pré-LVEF, %" in minimal, "Pré-LVEF, % must be preserved in echo-minimal"

    def test_valve_severity_categorical_preserved(self):
        baseline = _sample_baseline_features()
        minimal = build_echo_minimal_features(baseline)
        for feat in VALVE_SEVERITY_CATEGORICAL:
            if feat in baseline:
                assert feat in minimal, f"{feat!r} (valve severity) must be preserved"

    def test_non_echo_features_preserved(self):
        baseline = _sample_baseline_features()
        minimal = build_echo_minimal_features(baseline)
        for feat in ["Age (years)", "Sex", "Creatinine (mg/dL)", "Hematocrit (%)"]:
            assert feat in minimal, f"Non-echo feature {feat!r} should not be removed"

    def test_baseline_input_not_modified(self):
        baseline = _sample_baseline_features()
        original_len = len(baseline)
        original_first = baseline[0]
        _ = build_echo_minimal_features(baseline)
        assert len(baseline) == original_len, "Input list must not be modified"
        assert baseline[0] == original_first, "Input list must not be modified"

    def test_returns_list(self):
        baseline = _sample_baseline_features()
        result = build_echo_minimal_features(baseline)
        assert isinstance(result, list)

    def test_empty_baseline_returns_empty(self):
        assert build_echo_minimal_features([]) == []

    def test_no_echo_features_in_baseline_unchanged(self):
        baseline = ["Age (years)", "Creatinine (mg/dL)", "Sex"]
        minimal = build_echo_minimal_features(baseline)
        assert minimal == baseline

    def test_only_quant_echo_in_baseline_returns_empty(self):
        baseline = ["AVA (cm²)", "MVA (cm²)", "PHT Aortic"]
        minimal = build_echo_minimal_features(baseline)
        assert minimal == []

    def test_size_reduced(self):
        baseline = _sample_baseline_features()
        minimal = build_echo_minimal_features(baseline)
        removed = [f for f in baseline if f in ECHO_QUANTITATIVE_VALVULAR]
        assert len(minimal) == len(baseline) - len(removed)

    def test_preserves_continuous_echo_constants(self):
        for feat in ECHO_PRESERVE_CONTINUOUS:
            baseline = [feat, "AVA (cm²)", "Age (years)"]
            minimal = build_echo_minimal_features(baseline)
            assert feat in minimal


# ── features_removed_from_baseline ───────────────────────────────────────────


class TestFeaturesRemovedFromBaseline:
    def test_identifies_removed_features(self):
        baseline = ["A", "B", "C", "D"]
        minimal = ["A", "C"]
        removed = features_removed_from_baseline(baseline, minimal)
        assert set(removed) == {"B", "D"}

    def test_empty_when_identical(self):
        baseline = ["A", "B"]
        removed = features_removed_from_baseline(baseline, baseline)
        assert removed == []

    def test_preserves_order(self):
        baseline = ["Z", "A", "M"]
        minimal = ["A"]
        removed = features_removed_from_baseline(baseline, minimal)
        assert removed == ["Z", "M"]


# ── features_in_both ─────────────────────────────────────────────────────────


class TestFeaturesInBoth:
    def test_returns_intersection(self):
        baseline = ["A", "B", "C"]
        minimal = ["B", "C", "D"]
        common = features_in_both(baseline, minimal)
        assert set(common) == {"B", "C"}

    def test_empty_when_no_overlap(self):
        common = features_in_both(["A"], ["B"])
        assert common == []


# ── build_features_dataframe ─────────────────────────────────────────────────


class TestBuildFeaturesDataframe:
    def test_columns_present(self):
        df = build_features_dataframe(["A", "B"], ["A"])
        assert "feature" in df.columns
        assert "in_baseline" in df.columns
        assert "in_echo_minimal" in df.columns
        assert "membership" in df.columns

    def test_membership_labels(self):
        baseline = ["A", "B"]
        minimal = ["A", "C"]
        df = build_features_dataframe(baseline, minimal)
        row_a = df[df["feature"] == "A"].iloc[0]
        row_b = df[df["feature"] == "B"].iloc[0]
        row_c = df[df["feature"] == "C"].iloc[0]
        assert row_a["membership"] == "both"
        assert row_b["membership"] == "baseline_only"
        assert row_c["membership"] == "minimal_only"

    def test_echo_quant_flagged(self):
        df = build_features_dataframe(["AVA (cm²)", "Age (years)"], ["Age (years)"])
        ava_row = df[df["feature"] == "AVA (cm²)"].iloc[0]
        assert ava_row["is_echo_quantitative_valvular"]

    def test_preserve_continuous_flagged(self):
        df = build_features_dataframe(["PSAP", "TAPSE"], ["PSAP", "TAPSE"])
        for feat in ["PSAP", "TAPSE"]:
            row = df[df["feature"] == feat].iloc[0]
            assert row["is_preserved_continuous_echo"]

    def test_valve_severity_flagged(self):
        df = build_features_dataframe(["Aortic Stenosis"], ["Aortic Stenosis"])
        row = df[df["feature"] == "Aortic Stenosis"].iloc[0]
        assert row["is_valve_severity_categorical"]


# ── compute_missingness_summary ───────────────────────────────────────────────


class TestComputeMissingnessSummary:
    def _make_df(self):
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "AVA (cm²)": rng.choice([np.nan, 1.5], n, p=[0.6, 0.4]),
            "PSAP": rng.choice([np.nan, 30.0], n, p=[0.2, 0.8]),
            "TAPSE": rng.uniform(10, 25, n),
        })
        return df

    def test_pct_missing_in_range(self):
        df = self._make_df()
        summary = compute_missingness_summary(df, ["AVA (cm²)"], ["PSAP"])
        for _, row in summary.iterrows():
            assert 0.0 <= row["pct_missing"] <= 1.0

    def test_groups_correctly_assigned(self):
        df = self._make_df()
        summary = compute_missingness_summary(df, ["AVA (cm²)"], ["PSAP"])
        ava_group = summary.loc[summary["feature"] == "AVA (cm²)", "group"].values[0]
        psap_group = summary.loc[summary["feature"] == "PSAP", "group"].values[0]
        assert ava_group == "removed_echo_quantitative"
        assert psap_group == "preserved_echo"

    def test_absent_column_handled(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        summary = compute_missingness_summary(df, ["AVA (cm²)"], [])
        row = summary[summary["feature"] == "AVA (cm²)"].iloc[0]
        assert row["pct_missing"] == 1.0
        assert not row["present_in_data"]


# ── _threshold_metrics ────────────────────────────────────────────────────────


class TestThresholdMetrics:
    def test_perfect_separation(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.9, 0.8, 0.2, 0.1])
        m = _threshold_metrics(y, p, 0.5)
        assert m["TP"] == 2
        assert m["TN"] == 2
        assert m["FP"] == 0
        assert m["FN"] == 0
        assert math.isclose(m["sensitivity"], 1.0)
        assert math.isclose(m["specificity"], 1.0)

    def test_flag_rate(self):
        y = np.array([1, 0, 0, 0])
        p = np.array([0.9, 0.9, 0.1, 0.1])
        m = _threshold_metrics(y, p, 0.5)
        assert math.isclose(m["flag_rate"], 0.5)


# ── compute_arm_metrics ───────────────────────────────────────────────────────


class TestComputeArmMetrics:
    def test_returns_expected_keys(self):
        y, p = _make_probas()
        row = compute_arm_metrics(y, p, float(y.mean()), "baseline", n_features=20)
        for key in ["AUC", "AUPRC", "Brier", "Brier_skill_score",
                    "calibration_intercept", "calibration_slope",
                    "CIL", "ICI", "sensitivity", "specificity",
                    "PPV", "NPV", "TP", "FP", "TN", "FN", "flag_rate"]:
            assert key in row, f"Missing key: {key}"

    def test_arm_name_stored(self):
        y, p = _make_probas()
        row = compute_arm_metrics(y, p, float(y.mean()), "echo_minimal", n_features=10)
        assert row["arm"] == "echo_minimal"

    def test_n_features_stored(self):
        y, p = _make_probas()
        row = compute_arm_metrics(y, p, float(y.mean()), "baseline", n_features=42)
        assert row["n_features"] == 42

    def test_nan_proba_handled(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([float("nan")] * 4)
        row = compute_arm_metrics(y, p, 0.5, "arm", n_features=5)
        assert math.isnan(row["AUC"])

    def test_auc_between_0_and_1(self):
        y, p = _make_probas()
        row = compute_arm_metrics(y, p, float(y.mean()), "baseline", n_features=20)
        assert 0.0 <= row["AUC"] <= 1.0


# ── Bootstrap diff helpers ────────────────────────────────────────────────────


class TestBootstrapDiffBrier:
    def test_returns_expected_keys(self):
        y, p = _make_probas()
        result = _bootstrap_diff_brier(y, p, p * 0.9, n_boot=50, seed=0)
        for k in ["delta_brier", "ci_low", "ci_high", "p"]:
            assert k in result

    def test_zero_diff_when_same_proba(self):
        y, p = _make_probas()
        result = _bootstrap_diff_brier(y, p, p, n_boot=100, seed=0)
        assert math.isclose(result["delta_brier"], 0.0, abs_tol=1e-9)

    def test_p_value_in_range(self):
        y, p = _make_probas()
        result = _bootstrap_diff_brier(y, p, p * 0.9, n_boot=100, seed=0)
        assert 0.0 <= result["p"] <= 1.0


class TestBootstrapDiffAuprc:
    def test_returns_expected_keys(self):
        y, p = _make_probas()
        result = _bootstrap_diff_auprc(y, p, p * 0.9, n_boot=50, seed=0)
        for k in ["delta_auprc", "ci_low", "ci_high", "p"]:
            assert k in result

    def test_zero_diff_when_same_proba(self):
        y, p = _make_probas()
        result = _bootstrap_diff_auprc(y, p, p, n_boot=100, seed=0)
        assert math.isclose(result["delta_auprc"], 0.0, abs_tol=1e-9)


# ── interpret_result ──────────────────────────────────────────────────────────


class TestInterpretResult:
    def _make_miss_df(self, pct: float = 0.50) -> pd.DataFrame:
        return pd.DataFrame([{
            "feature": "AVA (cm²)",
            "group": "removed_echo_quantitative",
            "pct_missing": pct,
        }])

    def test_worse_large_auc_drop(self):
        base = {"AUC": 0.75, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.90}
        minimal = {"AUC": 0.72, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.90}
        boot = {"delta_auc": 0.03, "ci_low": 0.01, "ci_high": 0.05, "p": 0.01}
        result = interpret_result(base, minimal, boot, self._make_miss_df())
        assert result == "worse"

    def test_worse_brier_deterioration(self):
        base = {"AUC": 0.75, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.90}
        minimal = {"AUC": 0.75, "Brier": 0.115, "ICI": 0.06, "sensitivity": 0.90}
        boot = {"delta_auc": 0.0, "ci_low": -0.01, "ci_high": 0.01, "p": 0.5}
        result = interpret_result(base, minimal, boot, self._make_miss_df())
        assert result == "worse"

    def test_worse_sensitivity_loss(self):
        base = {"AUC": 0.75, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.90}
        minimal = {"AUC": 0.75, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.86}
        boot = {"delta_auc": 0.0, "ci_low": -0.005, "ci_high": 0.005, "p": 0.5}
        result = interpret_result(base, minimal, boot, self._make_miss_df())
        assert result == "worse"

    def test_non_inferior_parsimonious(self):
        base = {"AUC": 0.75, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.90}
        minimal = {"AUC": 0.745, "Brier": 0.101, "ICI": 0.05, "sensitivity": 0.895}
        boot = {"delta_auc": 0.005, "ci_low": -0.005, "ci_high": 0.015, "p": 0.3}
        result = interpret_result(base, minimal, boot, self._make_miss_df(pct=0.50))
        assert result == "non_inferior_parsimonious"

    def test_superior(self):
        # boot from bootstrap_auc_diff(y, p_base, p_min) → delta = baseline − echo_minimal
        # ci_high < 0 means the entire CI is negative → echo_minimal consistently better
        base = {"AUC": 0.75, "Brier": 0.10, "ICI": 0.06, "sensitivity": 0.90}
        minimal = {"AUC": 0.758, "Brier": 0.098, "ICI": 0.05, "sensitivity": 0.900}
        boot = {"delta_auc": -0.008, "ci_low": -0.015, "ci_high": -0.002, "p": 0.02}
        result = interpret_result(base, minimal, boot, self._make_miss_df(pct=0.50))
        assert result == "superior"

    def test_superior_not_triggered_if_ci_spans_zero(self):
        # ci_high > 0 → CI spans zero → not "superior"
        base = {"AUC": 0.75, "Brier": 0.10, "ICI": 0.06, "sensitivity": 0.90}
        minimal = {"AUC": 0.758, "Brier": 0.098, "ICI": 0.05, "sensitivity": 0.900}
        boot = {"delta_auc": -0.008, "ci_low": -0.020, "ci_high": 0.005, "p": 0.15}
        result = interpret_result(base, minimal, boot, self._make_miss_df(pct=0.50))
        assert result != "superior"

    def test_inconclusive_nan_values(self):
        base = {"AUC": float("nan"), "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.90}
        minimal = {"AUC": 0.74, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.90}
        boot = {"delta_auc": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p": float("nan")}
        result = interpret_result(base, minimal, boot, self._make_miss_df())
        assert result == "inconclusive"

    def test_non_inferior_no_miss_reduction_is_inconclusive(self):
        base = {"AUC": 0.75, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.90}
        minimal = {"AUC": 0.745, "Brier": 0.101, "ICI": 0.05, "sensitivity": 0.895}
        boot = {"delta_auc": 0.005, "ci_low": -0.005, "ci_high": 0.015, "p": 0.3}
        # pct_missing=0.05 → not "meaningful" (< 10 %)
        miss = self._make_miss_df(pct=0.05)
        result = interpret_result(base, minimal, boot, miss)
        assert result == "inconclusive"


# ── Constants sanity checks ───────────────────────────────────────────────────


class TestConstants:
    def test_operational_threshold_positive(self):
        assert 0.0 < OPERATIONAL_THRESHOLD < 1.0

    def test_echo_quantitative_list_non_empty(self):
        assert len(ECHO_QUANTITATIVE_VALVULAR) > 0

    def test_preserve_continuous_contains_psap_tapse(self):
        assert "PSAP" in ECHO_PRESERVE_CONTINUOUS
        assert "TAPSE" in ECHO_PRESERVE_CONTINUOUS

    def test_valve_severity_contains_five_valves(self):
        expected = {
            "Aortic Stenosis", "Aortic Regurgitation",
            "Mitral Stenosis", "Mitral Regurgitation",
            "Tricuspid Regurgitation",
        }
        assert expected.issubset(set(VALVE_SEVERITY_CATEGORICAL))

    def test_no_overlap_between_quant_and_preserve(self):
        quant_set = set(ECHO_QUANTITATIVE_VALVULAR)
        preserve_set = set(ECHO_PRESERVE_CONTINUOUS) | set(VALVE_SEVERITY_CATEGORICAL)
        overlap = quant_set & preserve_set
        assert overlap == set(), f"Overlap between removed and preserved: {overlap}"

    def test_ava_mva_in_quantitative_list(self):
        assert "AVA (cm²)" in ECHO_QUANTITATIVE_VALVULAR
        assert "MVA (cm²)" in ECHO_QUANTITATIVE_VALVULAR

    def test_gradients_in_quantitative_list(self):
        assert "Aortic Mean gradient (mmHg)" in ECHO_QUANTITATIVE_VALVULAR
        assert "Mitral Mean gradient (mmHg)" in ECHO_QUANTITATIVE_VALVULAR

    def test_pht_in_quantitative_list(self):
        assert "PHT Aortic" in ECHO_QUANTITATIVE_VALVULAR

    def test_vena_contracta_in_quantitative_list(self):
        assert "Vena contracta" in ECHO_QUANTITATIVE_VALVULAR
        assert "Vena contracta (mm)" in ECHO_QUANTITATIVE_VALVULAR

    def test_lvef_in_preserve_continuous(self):
        assert "Pré-LVEF, %" in ECHO_PRESERVE_CONTINUOUS
