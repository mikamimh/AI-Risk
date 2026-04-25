"""Unit tests for ablation_calibration_threshold_policy helpers.

Tests are intentionally isolated from the production pipeline — no data files
or sklearn models are required.  Every function under test is a pure helper
that accepts plain numpy arrays.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ablation_calibration_threshold_policy import (
    apply_threshold_policy,
    compute_brier_skill_score,
    compute_distribution_diagnostics,
    compute_threshold_metrics,
    find_npv_constrained_threshold,
    find_sensitivity_constrained_threshold,
    find_youden_threshold,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_data(n: int = 200, prevalence: float = 0.15, seed: int = 42):
    """Synthetic y_true and y_prob arrays with realistic signal."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    p = np.clip(y * 0.4 + rng.random(n) * 0.4, 0.01, 0.99)
    return y, p


# ── compute_threshold_metrics ─────────────────────────────────────────────────


class TestComputeThresholdMetrics:
    def test_perfect_classification(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.9, 0.8, 0.2, 0.1])
        m = compute_threshold_metrics(y, p, threshold=0.5)
        assert m["TP"] == 2
        assert m["TN"] == 2
        assert m["FP"] == 0
        assert m["FN"] == 0
        assert math.isclose(m["sensitivity"], 1.0)
        assert math.isclose(m["specificity"], 1.0)
        assert math.isclose(m["PPV"], 1.0)
        assert math.isclose(m["NPV"], 1.0)
        assert m["status"] == "ok"

    def test_all_flagged(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.9, 0.8, 0.7, 0.6])
        m = compute_threshold_metrics(y, p, threshold=0.5)
        assert m["TP"] == 2
        assert m["FP"] == 2
        assert m["TN"] == 0
        assert m["FN"] == 0
        assert m["n_flagged"] == 4
        assert math.isclose(m["flag_rate"], 1.0)
        assert math.isclose(m["sensitivity"], 1.0)
        assert math.isclose(m["specificity"], 0.0)

    def test_none_flagged(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.1, 0.2, 0.3, 0.4])
        m = compute_threshold_metrics(y, p, threshold=1.0)
        assert m["TP"] == 0
        assert m["FP"] == 0
        assert m["TN"] == 2
        assert m["FN"] == 2
        assert m["n_flagged"] == 0
        assert math.isclose(m["flag_rate"], 0.0)
        assert math.isnan(m["PPV"])
        assert math.isnan(m["event_rate_above"])

    def test_threshold_uses_probability_scale(self):
        """Threshold is in [0,1] — 8% = 0.08, not 8."""
        y = np.array([0, 0, 1, 1])
        p = np.array([0.03, 0.05, 0.10, 0.20])
        m = compute_threshold_metrics(y, p, threshold=0.08)
        # 0.10 and 0.20 are >= 0.08 → flagged
        assert m["n_flagged"] == 2
        assert m["TP"] == 2
        assert m["FP"] == 0

    def test_event_rates(self):
        y = np.array([1, 0, 1, 0, 0, 0])
        p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
        m = compute_threshold_metrics(y, p, threshold=0.5)
        # Flagged: indices 0,1,2 → TP=2, FP=1
        assert m["TP"] == 2
        assert m["FP"] == 1
        assert m["TN"] == 3
        assert m["FN"] == 0
        assert math.isclose(m["event_rate_above"], 2 / 3)
        assert math.isclose(m["event_rate_below"], 0.0)

    def test_flag_rate(self):
        y = np.zeros(10, dtype=int)
        p = np.array([0.9] * 3 + [0.1] * 7)
        m = compute_threshold_metrics(y, p, threshold=0.5)
        assert math.isclose(m["flag_rate"], 0.3)

    def test_ppv_npv(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.1])
        m = compute_threshold_metrics(y, p, threshold=0.5)
        # flagged: 0,1,2 → TP=2, FP=1
        assert m["TP"] == 2
        assert m["FP"] == 1
        assert m["TN"] == 2
        assert m["FN"] == 1
        assert math.isclose(m["PPV"], 2 / 3)
        assert math.isclose(m["NPV"], 2 / 3)


# ── find_youden_threshold ─────────────────────────────────────────────────────


class TestFindYoudenThreshold:
    def test_returns_float_in_unit_interval(self):
        y, p = _make_data()
        thr = find_youden_threshold(y, p)
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0

    def test_maximises_youden_j_over_fixed_8(self):
        rng = np.random.default_rng(0)
        y = (rng.random(100) > 0.80).astype(int)
        p = np.clip(y * 0.5 + rng.random(100) * 0.5, 0.01, 0.99)
        thr = find_youden_threshold(y, p)
        m_opt = compute_threshold_metrics(y, p, thr)
        j_opt = m_opt["sensitivity"] + m_opt["specificity"] - 1.0
        m_8   = compute_threshold_metrics(y, p, 0.08)
        j_8   = m_8["sensitivity"] + m_8["specificity"] - 1.0
        assert j_opt >= j_8 - 1e-6


# ── find_sensitivity_constrained_threshold ────────────────────────────────────


class TestFindSensitivityConstrainedThreshold:
    def test_achieves_target_sensitivity(self):
        y, p = _make_data(n=300, prevalence=0.20)
        thr = find_sensitivity_constrained_threshold(y, p, min_sensitivity=0.90)
        if not math.isnan(thr):
            m = compute_threshold_metrics(y, p, thr)
            assert m["sensitivity"] >= 0.90 - 1e-6

    def test_returns_nan_when_impossible(self):
        """Require sensitivity > 1.0 → impossible."""
        y, p = _make_data()
        thr = find_sensitivity_constrained_threshold(y, p, min_sensitivity=1.5)
        assert math.isnan(thr)

    def test_lower_target_allows_higher_threshold(self):
        """Lower sensitivity bound → can use a stricter (higher) threshold."""
        y, p = _make_data(n=300, prevalence=0.20)
        thr_90 = find_sensitivity_constrained_threshold(y, p, 0.90)
        thr_75 = find_sensitivity_constrained_threshold(y, p, 0.75)
        if not math.isnan(thr_90) and not math.isnan(thr_75):
            # Lower target → threshold can be >= threshold for higher target
            assert thr_75 >= thr_90 - 1e-6

    def test_no_positive_class_returns_nan(self):
        """No events → sensitivity undefined → NaN returned."""
        y = np.zeros(20, dtype=int)
        p = np.linspace(0.1, 0.9, 20)
        thr = find_sensitivity_constrained_threshold(y, p, min_sensitivity=0.90)
        # The grid won't find any t where sensitivity >= 0.90 because there are no events
        assert math.isnan(thr)

    def test_multiple_targets_consistency(self):
        rng = np.random.default_rng(42)
        y = (rng.random(200) > 0.80).astype(int)
        p = np.clip(y * 0.5 + rng.random(200) * 0.5, 0.01, 0.99)
        for target in (0.85, 0.90, 0.95):
            thr = find_sensitivity_constrained_threshold(y, p, target)
            if not math.isnan(thr):
                m = compute_threshold_metrics(y, p, thr)
                assert m["sensitivity"] >= target - 1e-6, (
                    f"target={target}, sens={m['sensitivity']:.4f}, thr={thr:.4f}"
                )


# ── find_npv_constrained_threshold ────────────────────────────────────────────


class TestFindNPVConstrainedThreshold:
    def test_achieves_target_npv(self):
        y, p = _make_data(n=300, prevalence=0.10, seed=1)
        thr = find_npv_constrained_threshold(y, p, min_npv=0.90)
        if not math.isnan(thr):
            m = compute_threshold_metrics(y, p, thr)
            assert m["NPV"] >= 0.90 - 0.01  # small tolerance for grid resolution

    def test_returns_nan_when_impossible(self):
        """NPV target > 1 → impossible."""
        y, p = _make_data()
        thr = find_npv_constrained_threshold(y, p, min_npv=1.5)
        assert math.isnan(thr)

    def test_all_events_extreme_npv_impossible(self):
        """All outcomes are events → NPV = 0 at any threshold → NaN for high NPV target."""
        y = np.ones(10, dtype=int)
        p = np.linspace(0.1, 0.9, 10)
        thr = find_npv_constrained_threshold(y, p, min_npv=0.99)
        assert math.isnan(thr)


# ── compute_brier_skill_score ─────────────────────────────────────────────────


class TestComputeBrierSkillScore:
    def test_perfect_predictor_bss_is_1(self):
        y = np.array([1, 0, 1, 0, 1])
        p = y.astype(float)
        assert math.isclose(compute_brier_skill_score(y, p), 1.0, abs_tol=1e-9)

    def test_prevalence_baseline_bss_is_0(self):
        y = np.array([1, 0, 1, 0, 0])
        p = np.full(len(y), y.mean())
        assert math.isclose(compute_brier_skill_score(y, p), 0.0, abs_tol=1e-9)

    def test_inverted_predictor_bss_negative(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = 1.0 - y.astype(float)
        assert compute_brier_skill_score(y, p) < 0.0

    def test_nan_for_constant_outcome(self):
        y = np.zeros(10, dtype=int)
        p = np.full(10, 0.15)
        assert math.isnan(compute_brier_skill_score(y, p))


# ── compute_distribution_diagnostics ─────────────────────────────────────────


class TestComputeDistributionDiagnostics:
    def test_returns_all_expected_keys(self):
        _, p = _make_data()
        d = compute_distribution_diagnostics(p)
        for key in [
            "prob_p01", "prob_p05", "prob_p25", "prob_p50",
            "prob_p75", "prob_p95", "prob_p99",
            "pct_below_2", "pct_below_5", "pct_below_8",
            "pct_above_15", "pct_above_30",
            "n_unique_probabilities", "n_prob_exact_0", "n_prob_exact_1",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_pct_below_8_is_fraction_not_percentage(self):
        """pct_below_8 must be in [0, 1], not 0-100."""
        p = np.array([0.05, 0.07, 0.10, 0.20])
        d = compute_distribution_diagnostics(p)
        assert 0.0 <= d["pct_below_8"] <= 1.0
        assert math.isclose(d["pct_below_8"], 0.5)

    def test_exact_zero_and_one_detection(self):
        p = np.array([0.0, 0.0, 0.5, 1.0])
        d = compute_distribution_diagnostics(p)
        assert d["n_prob_exact_0"] == 2
        assert d["n_prob_exact_1"] == 1

    def test_empty_input_returns_nans(self):
        d = compute_distribution_diagnostics(np.array([]))
        for v in d.values():
            assert math.isnan(float(v))

    def test_all_nan_input_returns_nans(self):
        d = compute_distribution_diagnostics(np.full(5, float("nan")))
        for v in d.values():
            assert math.isnan(float(v))

    def test_degenerate_all_zeros(self):
        p = np.zeros(50)
        d = compute_distribution_diagnostics(p)
        assert d["n_prob_exact_0"] == 50
        assert d["n_unique_probabilities"] == 1
        assert math.isclose(d["pct_below_8"], 1.0)

    def test_percentiles_are_ordered(self):
        rng = np.random.default_rng(99)
        p = rng.random(200)
        d = compute_distribution_diagnostics(p)
        assert d["prob_p01"] <= d["prob_p05"] <= d["prob_p25"]
        assert d["prob_p25"] <= d["prob_p50"] <= d["prob_p75"]
        assert d["prob_p75"] <= d["prob_p95"] <= d["prob_p99"]

    def test_pct_values_consistent(self):
        p = np.array([0.01, 0.03, 0.06, 0.09, 0.20, 0.40])
        d = compute_distribution_diagnostics(p)
        # 3/6 below 8%
        assert math.isclose(d["pct_below_8"], 0.5)
        # 2/6 above 15%
        assert math.isclose(d["pct_above_15"], 2 / 6)

    def test_unique_count(self):
        p = np.array([0.1, 0.1, 0.2, 0.3, 0.3])
        d = compute_distribution_diagnostics(p)
        assert d["n_unique_probabilities"] == 3


# ── apply_threshold_policy (string-based public dispatcher) ──────────────────


class TestApplyThresholdPolicy:
    def test_fixed_8_uses_008_scale(self):
        """fixed_8 must apply threshold = 0.08, not 8."""
        y, p = _make_data(n=200, prevalence=0.15)
        m_direct = compute_threshold_metrics(y, p, 0.08)
        m_policy = apply_threshold_policy(y, p, "fixed_8")
        assert math.isclose(m_policy["selected_threshold"], 0.08)
        assert m_policy["TP"] == m_direct["TP"]
        assert m_policy["TN"] == m_direct["TN"]

    @pytest.mark.parametrize("policy,thr", [
        ("fixed_2", 0.02), ("fixed_5", 0.05),
        ("fixed_10", 0.10), ("fixed_15", 0.15),
    ])
    def test_fixed_policies_correct_scale(self, policy: str, thr: float):
        y, p = _make_data(n=200)
        m = apply_threshold_policy(y, p, policy)
        assert math.isclose(m["selected_threshold"], thr), (
            f"{policy}: expected {thr}, got {m['selected_threshold']}"
        )

    def test_youden_status_ok(self):
        y, p = _make_data(n=200, prevalence=0.15)
        m = apply_threshold_policy(y, p, "youden")
        assert m["status"] == "ok"
        assert not math.isnan(m["selected_threshold"])

    def test_sensitivity_constrained_90_meets_target(self):
        y, p = _make_data(n=400, prevalence=0.20, seed=7)
        m = apply_threshold_policy(y, p, "sensitivity_constrained_90")
        if m["status"] == "ok":
            assert m["sensitivity"] >= 0.90 - 1e-6

    def test_sensitivity_constrained_85_meets_target(self):
        y, p = _make_data(n=400, prevalence=0.20, seed=7)
        m = apply_threshold_policy(y, p, "sensitivity_constrained_85")
        if m["status"] == "ok":
            assert m["sensitivity"] >= 0.85 - 1e-6

    def test_sensitivity_constrained_95_meets_target(self):
        y, p = _make_data(n=400, prevalence=0.20, seed=7)
        m = apply_threshold_policy(y, p, "sensitivity_constrained_95")
        if m["status"] == "ok":
            assert m["sensitivity"] >= 0.95 - 1e-6

    def test_npv_constrained_95(self):
        y, p = _make_data(n=400, prevalence=0.10, seed=9)
        m = apply_threshold_policy(y, p, "npv_constrained_95")
        if m["status"] == "ok":
            assert m["NPV"] >= 0.95 - 0.01

    def test_npv_constrained_97(self):
        y, p = _make_data(n=400, prevalence=0.10, seed=9)
        m = apply_threshold_policy(y, p, "npv_constrained_97")
        if m["status"] == "ok":
            assert m["NPV"] >= 0.97 - 0.01

    def test_unknown_policy_returns_unknown_status(self):
        y, p = _make_data()
        m = apply_threshold_policy(y, p, "nonexistent_policy")
        assert "unknown" in m["status"]

    def test_not_available_when_no_positive_class(self):
        """No events → sensitivity_constrained_90 cannot be satisfied."""
        y = np.zeros(20, dtype=int)
        p = np.linspace(0.1, 0.9, 20)
        m = apply_threshold_policy(y, p, "sensitivity_constrained_90")
        assert m["status"] in ("not_available", "failed")
