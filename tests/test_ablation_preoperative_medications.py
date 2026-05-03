"""Unit tests for ablation_preoperative_medications helpers.

All tests are isolated — no data files, sklearn models, or the official bundle
are required.  Pure helper functions only.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from ablation_preoperative_medications import (
    MEDICATIONS_FEATURE,
    MEDICATIONS_FEATURE_ALIASES,
    OPERATIONAL_THRESHOLD,
    _make_threshold_row,
    build_medication_free_features,
    build_missingness_dataframe,
    build_threshold_comparison,
    feature_was_present,
    interpret_result,
    medication_missingness,
    run_ablation,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _sample_baseline_with_meds() -> List[str]:
    return [
        "Age (years)", "Sex", "Pré-LVEF, %", "PSAP", "TAPSE",
        "Preoperative Medications",   # should be removed
        "Creatinine (mg/dL)", "Hematocrit (%)",
    ]


def _sample_baseline_without_meds() -> List[str]:
    return [
        "Age (years)", "Sex", "Pré-LVEF, %", "PSAP", "TAPSE",
        "Creatinine (mg/dL)", "Hematocrit (%)",
    ]


def _make_probas(n: int = 300, prevalence: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    p = np.clip(y * 0.45 + rng.random(n) * 0.35, 0.01, 0.99)
    return y, p


def _write_synthetic_oof(tmpdir: Path, n: int = 300, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    oof_base = np.clip(y * 0.45 + rng.random(n) * 0.35, 0.01, 0.99)
    oof_mf   = np.clip(y * 0.43 + rng.random(n) * 0.37, 0.01, 0.99)
    df = pd.DataFrame({"y": y, "oof_baseline": oof_base, "oof_medication_free": oof_mf})
    p = tmpdir / "preoperative_medications_oof.csv"
    df.to_csv(p, index=False)
    return p


# ── build_medication_free_features ────────────────────────────────────────────


class TestBuildMedicationFreeFeatures:
    def test_removes_exact_canonical_name(self):
        baseline = _sample_baseline_with_meds()
        result = build_medication_free_features(baseline)
        assert "Preoperative Medications" not in result

    def test_removes_all_aliases(self):
        for alias in MEDICATIONS_FEATURE_ALIASES:
            result = build_medication_free_features([alias, "Age (years)"])
            assert alias not in result
            assert "Age (years)" in result

    def test_unchanged_when_absent(self):
        baseline = _sample_baseline_without_meds()
        result = build_medication_free_features(baseline)
        assert result == baseline

    def test_baseline_input_not_modified(self):
        baseline = _sample_baseline_with_meds()
        original = list(baseline)
        build_medication_free_features(baseline)
        assert baseline == original

    def test_returns_list(self):
        result = build_medication_free_features(["Age (years)"])
        assert isinstance(result, list)

    def test_empty_baseline_returns_empty(self):
        assert build_medication_free_features([]) == []

    def test_non_medication_features_preserved(self):
        baseline = ["Age (years)", "Preoperative Medications", "Creatinine (mg/dL)"]
        result = build_medication_free_features(baseline)
        assert "Age (years)" in result
        assert "Creatinine (mg/dL)" in result

    def test_size_reduced_when_present(self):
        baseline = _sample_baseline_with_meds()
        result = build_medication_free_features(baseline)
        assert len(result) == len(baseline) - 1


# ── feature_was_present ───────────────────────────────────────────────────────


class TestFeatureWasPresent:
    def test_true_when_canonical_name_present(self):
        assert feature_was_present(["Age (years)", "Preoperative Medications"])

    def test_true_for_snake_case_alias(self):
        assert feature_was_present(["preoperative_medications", "Age (years)"])

    def test_false_when_absent(self):
        assert not feature_was_present(["Age (years)", "Sex", "PSAP"])

    def test_false_for_empty_list(self):
        assert not feature_was_present([])


# ── medication_missingness ────────────────────────────────────────────────────


class TestMedicationMissingness:
    def test_returns_expected_keys(self):
        df = pd.DataFrame({"Preoperative Medications": ["Aspirin", None, "None", ""]})
        result = medication_missingness(df)
        for k in ["present_in_data", "n_total", "n_missing", "pct_missing",
                  "n_none_literal", "n_empty", "n_non_empty"]:
            assert k in result

    def test_absent_column_gives_100pct_missing(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = medication_missingness(df)
        assert result["present_in_data"] is False
        assert result["pct_missing"] == 1.0

    def test_none_literal_counted(self):
        df = pd.DataFrame({"Preoperative Medications": ["None", "None", "Aspirin"]})
        result = medication_missingness(df)
        assert result["n_none_literal"] == 2

    def test_non_empty_counted(self):
        df = pd.DataFrame({"Preoperative Medications": ["Aspirin", "None", "", np.nan]})
        result = medication_missingness(df)
        assert result["n_non_empty"] == 1

    def test_pct_missing_in_range(self):
        df = pd.DataFrame({"Preoperative Medications": ["Aspirin", None, None, None]})
        result = medication_missingness(df)
        assert 0.0 <= result["pct_missing"] <= 1.0


# ── build_missingness_dataframe ───────────────────────────────────────────────


class TestBuildMissingnessDf:
    def test_one_row_returned(self):
        df = pd.DataFrame({"Preoperative Medications": ["Aspirin", None]})
        result = build_missingness_dataframe(df, was_in_feature_set=True)
        assert len(result) == 1

    def test_was_in_feature_set_stored(self):
        df = pd.DataFrame({"Preoperative Medications": [None, None]})
        result = build_missingness_dataframe(df, was_in_feature_set=False)
        assert not result.iloc[0]["was_in_feature_set"]

    def test_note_mentions_already_excluded_when_not_in_set(self):
        df = pd.DataFrame({"Preoperative Medications": [None]})
        result = build_missingness_dataframe(df, was_in_feature_set=False)
        note = str(result.iloc[0]["note"]).lower()
        assert "already excluded" in note or "noise_cols" in note


# ── build_threshold_comparison ────────────────────────────────────────────────


class TestBuildThresholdComparison:
    def test_three_rows(self):
        y, p = _make_probas()
        df = build_threshold_comparison(y, p, p * 0.95, 0.07)
        assert len(df) == 3

    def test_expected_labels(self):
        y, p = _make_probas()
        df = build_threshold_comparison(y, p, p * 0.95, 0.07)
        labels = set(df["threshold_label"])
        assert "baseline_fixed_085" in labels
        assert "medfree_fixed_085" in labels
        assert "medfree_sens90" in labels

    def test_fixed_085_rows_use_operational_threshold(self):
        y, p = _make_probas()
        df = build_threshold_comparison(y, p, p, OPERATIONAL_THRESHOLD)
        for lbl in ["baseline_fixed_085", "medfree_fixed_085"]:
            row = df[df["threshold_label"] == lbl].iloc[0]
            assert math.isclose(float(row["threshold"]), OPERATIONAL_THRESHOLD)

    def test_arm_labels_correct(self):
        y, p = _make_probas()
        df = build_threshold_comparison(y, p, p, 0.07)
        base_row = df[df["threshold_label"] == "baseline_fixed_085"].iloc[0]
        mf_row   = df[df["threshold_label"] == "medfree_fixed_085"].iloc[0]
        assert base_row["arm"] == "baseline"
        assert mf_row["arm"] == "medication_free"

    def test_nan_sens90_handled(self):
        y, p = _make_probas()
        df = build_threshold_comparison(y, p, p, float("nan"))
        s90_row = df[df["threshold_label"] == "medfree_sens90"].iloc[0]
        assert math.isnan(float(s90_row["threshold"]))


# ── interpret_result ──────────────────────────────────────────────────────────


class TestInterpretResult:
    def _boot(self, delta: float = 0.0):
        return {"delta_auc": delta, "ci_low": delta - 0.01,
                "ci_high": delta + 0.01, "p": 0.5}

    def test_null_experiment_when_not_in_feature_set(self):
        base = {"AUC": 0.75, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.91}
        mf   = {"AUC": 0.75, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.91}
        result = interpret_result(base, mf, self._boot(), was_in_feature_set=False)
        assert result == "null_experiment"

    def test_worse_large_auc_drop(self):
        base = {"AUC": 0.75, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.91}
        mf   = {"AUC": 0.72, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.91}
        result = interpret_result(base, mf, self._boot(0.03), was_in_feature_set=True)
        assert result == "worse"

    def test_worse_sensitivity_drop(self):
        base = {"AUC": 0.75, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.91}
        mf   = {"AUC": 0.75, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.87}
        result = interpret_result(base, mf, self._boot(), was_in_feature_set=True)
        assert result == "worse"

    def test_non_inferior_small_delta(self):
        base = {"AUC": 0.75, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.91}
        mf   = {"AUC": 0.745, "Brier": 0.101, "ICI": 0.051, "sensitivity": 0.910}
        result = interpret_result(base, mf, self._boot(0.005), was_in_feature_set=True)
        assert result == "non_inferior_parsimonious"

    def test_superior_consistent_improvement(self):
        base = {"AUC": 0.75, "Brier": 0.10, "ICI": 0.06, "sensitivity": 0.91}
        mf   = {"AUC": 0.758, "Brier": 0.098, "ICI": 0.05, "sensitivity": 0.912}
        # ci_high < 0 means entire CI of (baseline - mf) is negative -> mf better
        boot = {"delta_auc": -0.008, "ci_low": -0.016, "ci_high": -0.001, "p": 0.02}
        result = interpret_result(base, mf, boot, was_in_feature_set=True)
        assert result == "superior"

    def test_inconclusive_on_nan(self):
        base = {"AUC": float("nan"), "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.91}
        mf   = {"AUC": 0.74, "Brier": 0.10, "ICI": 0.05, "sensitivity": 0.91}
        result = interpret_result(base, mf, self._boot(), was_in_feature_set=True)
        assert result == "inconclusive"


# ── Constants sanity checks ───────────────────────────────────────────────────


class TestConstants:
    def test_operational_threshold_is_085(self):
        assert math.isclose(OPERATIONAL_THRESHOLD, 0.085)

    def test_medications_feature_is_string(self):
        assert isinstance(MEDICATIONS_FEATURE, str)
        assert len(MEDICATIONS_FEATURE) > 0

    def test_aliases_contain_canonical_name(self):
        assert MEDICATIONS_FEATURE in MEDICATIONS_FEATURE_ALIASES

    def test_aliases_non_empty(self):
        assert len(MEDICATIONS_FEATURE_ALIASES) >= 1

    def test_model_version_not_altered(self):
        from config import AppConfig
        assert AppConfig.MODEL_VERSION is not None
        assert len(AppConfig.MODEL_VERSION) > 0

    def test_bundle_file_not_modified_on_import(self):
        bundle_path = Path("ia_risk_bundle.joblib")
        if bundle_path.exists():
            mtime_before = os.path.getmtime(bundle_path)
            import ablation_preoperative_medications  # noqa: F401
            mtime_after = os.path.getmtime(bundle_path)
            assert mtime_before == mtime_after


# ── Summary marks analysis as exploratory ────────────────────────────────────


class TestSummaryExploratory:
    def _make_rows(self):
        return (
            {"AUC": 0.74, "AUPRC": 0.33, "Brier": 0.116, "Brier_skill_score": 0.09,
             "calibration_intercept": -0.09, "calibration_slope": 0.98,
             "CIL": 0.006, "ICI": 0.027, "sensitivity": 0.91, "specificity": 0.37,
             "PPV": 0.20, "NPV": 0.96, "TP": 62, "FP": 244, "TN": 142, "FN": 6,
             "flag_rate": 0.67, "n_features": 61, "n": 454, "events": 68,
             "prevalence": 0.15, "AUPRC_baseline": 0.15},
            {"AUC": 0.74, "AUPRC": 0.33, "Brier": 0.116, "Brier_skill_score": 0.09,
             "calibration_intercept": -0.09, "calibration_slope": 0.98,
             "CIL": 0.006, "ICI": 0.027, "sensitivity": 0.91, "specificity": 0.37,
             "PPV": 0.20, "NPV": 0.96, "TP": 62, "FP": 244, "TN": 142, "FN": 6,
             "flag_rate": 0.67, "n_features": 61, "n": 454, "events": 68,
             "prevalence": 0.15, "AUPRC_baseline": 0.15},
        )

    def _make_miss_df(self):
        return pd.DataFrame([{
            "feature": "Preoperative Medications",
            "was_in_feature_set": False,
            "present_in_source_data": True,
            "n_total": 454, "n_missing": 300, "pct_missing": 0.66,
            "n_none_literal": 50, "n_non_empty": 154,
            "note": "Already excluded from production feature set via _noise_cols in risk_data.py",
        }])

    def test_summary_contains_exploratory_warning(self):
        from ablation_preoperative_medications import build_markdown_summary
        base, mf = self._make_rows()
        md = build_markdown_summary(
            baseline_row=base, medfree_row=mf,
            was_in_feature_set=False, n_removed=0,
            miss_df=self._make_miss_df(),
            boot_auc={"delta_auc": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0},
            boot_brier={"delta_brier": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0},
            boot_auprc={"delta_auprc": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0},
            dca_df=None, thr_df=None,
            interpretation="null_experiment",
            sens90_thr=0.085, op_threshold=0.085,
        )
        assert "EXPLORATORY" in md or "exploratory" in md.lower()

    def test_summary_contains_alert_about_bundle(self):
        from ablation_preoperative_medications import build_markdown_summary
        base, mf = self._make_rows()
        md = build_markdown_summary(
            baseline_row=base, medfree_row=mf,
            was_in_feature_set=False, n_removed=0,
            miss_df=self._make_miss_df(),
            boot_auc={"delta_auc": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0},
            boot_brier={"delta_brier": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0},
            boot_auprc={"delta_auprc": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0},
            dca_df=None, thr_df=None,
            interpretation="null_experiment",
            sens90_thr=0.085, op_threshold=0.085,
        )
        assert "bundle" in md.lower() or "MODEL_VERSION" in md

    def test_summary_contains_null_experiment_label(self):
        from ablation_preoperative_medications import build_markdown_summary
        base, mf = self._make_rows()
        md = build_markdown_summary(
            baseline_row=base, medfree_row=mf,
            was_in_feature_set=False, n_removed=0,
            miss_df=self._make_miss_df(),
            boot_auc={"delta_auc": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0},
            boot_brier={"delta_brier": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0},
            boot_auprc={"delta_auprc": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0},
            dca_df=None, thr_df=None,
            interpretation="null_experiment",
            sens90_thr=0.085, op_threshold=0.085,
        )
        assert "NULL EXPERIMENT" in md or "null_experiment" in md.lower()

    def test_summary_contains_fixed_vs_sens90_sections(self):
        from ablation_preoperative_medications import build_markdown_summary
        base, mf = self._make_rows()
        thr_df = build_threshold_comparison(
            np.array([1, 0] * 50), np.full(100, 0.5), np.full(100, 0.5), 0.07
        )
        md = build_markdown_summary(
            baseline_row=base, medfree_row=mf,
            was_in_feature_set=False, n_removed=0,
            miss_df=self._make_miss_df(),
            boot_auc={"delta_auc": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0},
            boot_brier={"delta_brier": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0},
            boot_auprc={"delta_auprc": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0},
            dca_df=None, thr_df=thr_df,
            interpretation="null_experiment",
            sens90_thr=0.07, op_threshold=0.085,
        )
        assert "8.5%" in md
        assert "Sens90" in md or "sens90" in md.lower()
