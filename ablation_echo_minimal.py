"""Ablation: Echo-Minimal Feature Set

Compares the current (baseline) feature set against an echo-minimal set that
removes quantitative valvular echocardiographic variables while preserving PSAP,
TAPSE, LVEF, and categorical valve severity scores.

Hypothesis
----------
Quantitative valvular echo metrics (valve areas, gradients, PHT, vena contracta)
may introduce noise and instability due to heterogeneous report filling.  Removing
them while preserving hemodynamically informative summary scores (PSAP, TAPSE) and
categorical valve severity could reduce missingness without degrading discrimination.

IMPORTANT: This script is purely exploratory.  It does NOT alter the official
trained bundle, MODEL_VERSION, or any threshold policy.  Promoting the echo-minimal
feature set to a new baseline requires an explicit human review decision.

Usage
-----
    python ablation_echo_minimal.py \\
        --data local_data/Dataset_2025.xlsx \\
        --outdir reports \\
        --seed 42 \\
        --n-splits 5 \\
        --n-boot 2000

Outputs (written to --outdir)
------------------------------
    echo_minimal_ablation_metrics.csv    — per-arm discrimination/calibration table
    echo_minimal_ablation_features.csv  — feature membership per arm
    echo_minimal_ablation_missingness.csv — missingness summary
    echo_minimal_ablation_summary.md    — human-readable summary with interpretation
"""

from __future__ import annotations

import argparse
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from config import AppConfig, get_model_params
from modeling import build_preprocessor, clean_features
from risk_data import prepare_master_dataset
from stats_compare import (
    bootstrap_auc_diff,
    brier_skill_score as _bss_fn,
    calibration_intercept_slope,
    decision_curve,
    integrated_calibration_index,
)

# ── Feature-set constants ─────────────────────────────────────────────────────

# Quantitative valvular echo features to REMOVE in the echo-minimal arm.
# These are report-dependent measurements with high heterogeneity of filling.
ECHO_QUANTITATIVE_VALVULAR: List[str] = [
    "AVA (cm²)",                    # Aortic valve area
    "MVA (cm²)",                    # Mitral valve area
    "Aortic Mean gradient (mmHg)",  # Mean transvalvular aortic gradient
    "Mitral Mean gradient (mmHg)",  # Mean transmitral gradient
    "PHT Aortic",                   # Aortic pressure half-time
    "PHT Mitral",                   # Mitral pressure half-time (already not in baseline)
    "Vena contracta",               # Vena contracta – aortic
    "Vena contracta (mm)",          # Vena contracta – mitral
]

# Features that MUST be present in the echo-minimal arm when available.
ECHO_PRESERVE_CONTINUOUS: List[str] = [
    "PSAP",        # Pulmonary artery systolic pressure
    "TAPSE",       # Tricuspid annular plane systolic excursion
    "Pré-LVEF, %", # Left ventricular ejection fraction
]

# Categorical valve severity ordinals — must be preserved in the echo-minimal arm.
VALVE_SEVERITY_CATEGORICAL: List[str] = [
    "Aortic Stenosis",
    "Aortic Regurgitation",
    "Mitral Stenosis",
    "Mitral Regurgitation",
    "Tricuspid Regurgitation",
]

# Operational threshold from the official v17 sens_constrained_90 policy (8.5 %).
OPERATIONAL_THRESHOLD: float = 0.085

# DCA evaluation points (fractions, not percentages).
DCA_THRESHOLDS: List[float] = [0.02, 0.05, 0.08, 0.085, 0.10, 0.15, 0.20]

_BASE_SEED: int = 42


# ── Feature-set helpers (pure, testable) ──────────────────────────────────────


def build_echo_minimal_features(feature_columns: Sequence[str]) -> List[str]:
    """Return the echo-minimal feature list derived from *feature_columns*.

    Removes all columns listed in ECHO_QUANTITATIVE_VALVULAR (case-sensitive).
    All other features — including PSAP, TAPSE, LVEF, and categorical valve
    severities — are preserved.

    The input list is NOT modified.
    """
    remove_set = set(ECHO_QUANTITATIVE_VALVULAR)
    return [c for c in feature_columns if c not in remove_set]


def features_removed_from_baseline(
    baseline: Sequence[str],
    minimal: Sequence[str],
) -> List[str]:
    """Return features present in baseline but absent from minimal."""
    minimal_set = set(minimal)
    return [c for c in baseline if c not in minimal_set]


def features_in_both(
    baseline: Sequence[str],
    minimal: Sequence[str],
) -> List[str]:
    """Return features common to both arms."""
    minimal_set = set(minimal)
    return [c for c in baseline if c in minimal_set]


def build_features_dataframe(
    baseline: Sequence[str],
    minimal: Sequence[str],
) -> pd.DataFrame:
    """Return a tidy DataFrame describing feature membership per arm."""
    all_features = list(dict.fromkeys(list(baseline) + list(minimal)))
    baseline_set = set(baseline)
    minimal_set = set(minimal)

    rows = []
    for feat in all_features:
        in_bl = feat in baseline_set
        in_mn = feat in minimal_set
        if in_bl and not in_mn:
            membership = "baseline_only"
        elif in_bl and in_mn:
            membership = "both"
        else:
            membership = "minimal_only"

        is_echo_quant = feat in ECHO_QUANTITATIVE_VALVULAR
        is_preserve_cont = feat in ECHO_PRESERVE_CONTINUOUS
        is_valve_severity = feat in VALVE_SEVERITY_CATEGORICAL

        rows.append({
            "feature": feat,
            "in_baseline": in_bl,
            "in_echo_minimal": in_mn,
            "membership": membership,
            "is_echo_quantitative_valvular": is_echo_quant,
            "is_preserved_continuous_echo": is_preserve_cont,
            "is_valve_severity_categorical": is_valve_severity,
        })

    return pd.DataFrame(rows)


# ── Missingness helpers ────────────────────────────────────────────────────────


def compute_missingness_summary(
    df: pd.DataFrame,
    removed_cols: Sequence[str],
    preserved_echo_cols: Sequence[str],
) -> pd.DataFrame:
    """Return a per-column missingness summary for removed and preserved echo columns."""
    n = len(df)
    rows = []

    for col_group, cols in [
        ("removed_echo_quantitative", removed_cols),
        ("preserved_echo", preserved_echo_cols),
    ]:
        for col in cols:
            if col not in df.columns:
                rows.append({
                    "feature": col,
                    "group": col_group,
                    "n_total": n,
                    "n_missing": n,
                    "pct_missing": 1.0,
                    "present_in_data": False,
                })
                continue
            n_miss = int(df[col].isna().sum())
            rows.append({
                "feature": col,
                "group": col_group,
                "n_total": n,
                "n_missing": n_miss,
                "pct_missing": n_miss / n if n > 0 else float("nan"),
                "present_in_data": True,
            })

    return pd.DataFrame(rows)


# ── OOF cross-validation ──────────────────────────────────────────────────────


def _run_oof_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    estimator: object,
    cal_method: Optional[str],
    cal_cv: Optional[int],
    seed: int,
    n_splits: int,
) -> Tuple[np.ndarray, str]:
    """Stratified-(group-)K-fold OOF with optional in-fold calibration.

    The calibrator is fitted INSIDE each outer fold so calibrated OOF
    probabilities are honest (no leakage from test fold into calibrator).

    Returns (oof_probs, error_message).  error_message is empty on success.
    """
    try:
        pos = int(y.sum())
        neg = int(len(y) - pos)

        if groups is not None:
            unique_grps = pd.Series(groups).nunique()
            k = min(n_splits, int(unique_grps), pos, neg)
            cv_obj = StratifiedGroupKFold(
                n_splits=k, shuffle=True, random_state=seed
            )
        else:
            k = min(n_splits, pos, neg, len(X))
            cv_obj = StratifiedKFold(
                n_splits=k, shuffle=True, random_state=seed
            )

        if k < 2:
            return (
                np.full(len(y), float("nan")),
                f"Not enough samples for CV: k={k}, pos={pos}, neg={neg}",
            )

        oof = np.full(len(y), float("nan"))

        for train_idx, test_idx in cv_obj.split(X, y, groups):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr = y[train_idx]

            prep = build_preprocessor(X_tr)
            est_copy = clone(estimator)
            fold_pipe = Pipeline([("prep", prep), ("clf", est_copy)])

            if cal_method is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fold_pipe.fit(X_tr, y_tr)
                oof[test_idx] = fold_pipe.predict_proba(X_te)[:, 1]
            else:
                fold_pos = int(y_tr.sum())
                fold_neg = int(len(y_tr) - fold_pos)
                eff_cv = min(cal_cv, fold_pos, fold_neg)
                if eff_cv < 2:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fold_pipe.fit(X_tr, y_tr)
                    oof[test_idx] = fold_pipe.predict_proba(X_te)[:, 1]
                else:
                    cal = CalibratedClassifierCV(
                        fold_pipe,
                        method=cal_method,
                        cv=eff_cv,
                        ensemble=False,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cal.fit(X_tr, y_tr)
                    oof[test_idx] = cal.predict_proba(X_te)[:, 1]

        return oof, ""
    except Exception:
        return np.full(len(y), float("nan")), traceback.format_exc(limit=3)


# ── Per-arm metric computation ────────────────────────────────────────────────


def compute_arm_metrics(
    y: np.ndarray,
    p: np.ndarray,
    auprc_baseline: float,
    arm_name: str,
    n_features: int,
    op_threshold: float = OPERATIONAL_THRESHOLD,
) -> Dict:
    """Compute the full metric suite for one OOF arm.

    Returns a flat dict ready for pd.DataFrame row construction.
    """
    valid = ~np.isnan(p)
    yv, pv = y[valid], p[valid]

    row: Dict = {
        "arm": arm_name,
        "n": int(len(yv)),
        "events": int(yv.sum()),
        "prevalence": float(yv.mean()) if len(yv) > 0 else float("nan"),
        "n_features": n_features,
        "AUPRC_baseline": auprc_baseline,
    }

    if len(yv) == 0 or int(yv.sum()) == 0 or int((1 - yv).sum()) == 0:
        for k in [
            "AUC", "AUPRC", "Brier", "Brier_skill_score",
            "calibration_intercept", "calibration_slope", "CIL", "ICI",
            "sensitivity", "specificity", "PPV", "NPV",
            "TP", "FP", "TN", "FN", "flag_rate", "threshold_used",
        ]:
            row[k] = float("nan")
        return row

    try:
        row["AUC"] = float(roc_auc_score(yv, pv))
    except Exception:
        row["AUC"] = float("nan")

    try:
        row["AUPRC"] = float(average_precision_score(yv, pv))
    except Exception:
        row["AUPRC"] = float("nan")

    try:
        row["Brier"] = float(brier_score_loss(yv, pv))
        row["Brier_skill_score"] = float(_bss_fn(yv, pv))
    except Exception:
        row["Brier"] = float("nan")
        row["Brier_skill_score"] = float("nan")

    try:
        cal = calibration_intercept_slope(yv, pv)
        row["calibration_intercept"] = cal.get("Calibration intercept", float("nan"))
        row["calibration_slope"] = cal.get("Calibration slope", float("nan"))
    except Exception:
        row["calibration_intercept"] = float("nan")
        row["calibration_slope"] = float("nan")

    row["CIL"] = float(np.mean(pv) - np.mean(yv))

    try:
        row["ICI"] = float(integrated_calibration_index(yv, pv))
    except Exception:
        row["ICI"] = float("nan")

    # Operational threshold metrics
    _thr_m = _threshold_metrics(yv, pv, op_threshold)
    row["threshold_used"] = op_threshold
    row["sensitivity"] = _thr_m["sensitivity"]
    row["specificity"] = _thr_m["specificity"]
    row["PPV"] = _thr_m["PPV"]
    row["NPV"] = _thr_m["NPV"]
    row["TP"] = _thr_m["TP"]
    row["FP"] = _thr_m["FP"]
    row["TN"] = _thr_m["TN"]
    row["FN"] = _thr_m["FN"]
    row["flag_rate"] = _thr_m["flag_rate"]

    return row


def _threshold_metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
) -> Dict:
    """Classification metrics at a fixed threshold."""
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y)
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "PPV": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else float("nan"),
        "flag_rate": (tp + fp) / n if n > 0 else float("nan"),
    }


# ── Bootstrap comparison ──────────────────────────────────────────────────────


def _bootstrap_diff_brier(
    y: np.ndarray,
    p_base: np.ndarray,
    p_min: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap for Brier score difference (baseline − echo_minimal).

    Positive value means baseline has higher Brier (worse calibration error).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            d = brier_score_loss(y[idx], p_base[idx]) - brier_score_loss(y[idx], p_min[idx])
            deltas.append(d)
        except Exception:
            continue
    if not deltas:
        return {"delta_brier": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p": float("nan")}
    arr = np.array(deltas)
    delta = float(brier_score_loss(y, p_base) - brier_score_loss(y, p_min))
    return {
        "delta_brier": delta,
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "p": float(2 * min((arr <= 0).mean(), (arr >= 0).mean())),
    }


def _bootstrap_diff_auprc(
    y: np.ndarray,
    p_base: np.ndarray,
    p_min: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap for AUPRC difference (baseline − echo_minimal).

    Positive value means baseline has higher AUPRC (better discrimination).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        try:
            d = (average_precision_score(ys, p_base[idx])
                 - average_precision_score(ys, p_min[idx]))
            deltas.append(d)
        except Exception:
            continue
    if not deltas:
        return {"delta_auprc": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p": float("nan")}
    arr = np.array(deltas)
    delta = float(
        average_precision_score(y, p_base) - average_precision_score(y, p_min)
    )
    return {
        "delta_auprc": delta,
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "p": float(2 * min((arr <= 0).mean(), (arr >= 0).mean())),
    }


# ── Interpretation ────────────────────────────────────────────────────────────


def interpret_result(
    baseline_row: Dict,
    minimal_row: Dict,
    boot_auc: Dict,
    miss_summary: pd.DataFrame,
    delta_auc_noni_tol: float = 0.015,
    delta_brier_tol: float = 0.005,
) -> str:
    """Conservative automatic interpretation of the echo-minimal ablation.

    Returns one of: "superior", "non_inferior_parsimonious", "inconclusive", "worse".

    Criteria (applied conservatively):
    - "worse"  : delta_AUC < -delta_auc_noni_tol OR Brier materially worsens
                 OR sensitivity drops more than 3 pp.
    - "superior": echo-minimal has strictly better AUC AND better/equal Brier AND
                  better/equal ICI AND better/equal sensitivity AND bootstrap
                  CI(delta_AUC) entirely above 0.
    - "non_inferior_parsimonious": AUC drop ≤ delta_auc_noni_tol AND
                  Brier does not worsen materially AND sensitivity not lost >3 pp
                  AND meaningful missingness reduction.
    - "inconclusive": everything else.
    """
    def _get(d: Dict, key: str) -> float:
        v = d.get(key, float("nan"))
        return float(v) if v is not None else float("nan")

    base_auc = _get(baseline_row, "AUC")
    min_auc  = _get(minimal_row,  "AUC")
    base_brier = _get(baseline_row, "Brier")
    min_brier  = _get(minimal_row,  "Brier")
    base_ici = _get(baseline_row, "ICI")
    min_ici  = _get(minimal_row,  "ICI")
    base_sens = _get(baseline_row, "sensitivity")
    min_sens  = _get(minimal_row,  "sensitivity")

    delta_auc   = min_auc - base_auc
    delta_brier = min_brier - base_brier
    delta_sens  = min_sens - base_sens

    # Check for missingness reduction in removed features
    removed_miss = miss_summary[miss_summary["group"] == "removed_echo_quantitative"]
    has_meaningful_miss_reduction = (
        removed_miss["pct_missing"].mean() > 0.10
        if len(removed_miss) > 0 else False
    )

    auc_ci_low = boot_auc.get("ci_low", float("nan"))

    # Guard: any NaN in key metrics → inconclusive
    if any(np.isnan(v) for v in [base_auc, min_auc, base_brier, min_brier,
                                  base_sens, min_sens]):
        return "inconclusive"

    # Worse
    if delta_auc < -delta_auc_noni_tol:
        return "worse"
    if delta_brier > delta_brier_tol:
        return "worse"
    if delta_sens < -0.03:
        return "worse"

    # Superior: echo-minimal consistently better — entire CI of (baseline − echo_minimal)
    # must be negative, meaning echo_minimal is better at every bootstrap draw.
    auc_ci_high = boot_auc.get("ci_high", float("nan"))
    if (
        delta_auc > 0.005
        and delta_brier <= 0
        and (np.isnan(min_ici) or np.isnan(base_ici) or min_ici <= base_ici)
        and delta_sens >= -0.005
        and not np.isnan(auc_ci_high)
        and auc_ci_high < 0
    ):
        return "superior"

    # Non-inferior / parsimonious
    if (
        delta_auc >= -delta_auc_noni_tol
        and delta_brier <= delta_brier_tol
        and delta_sens >= -0.03
        and has_meaningful_miss_reduction
    ):
        return "non_inferior_parsimonious"

    return "inconclusive"


# ── Markdown report builder ───────────────────────────────────────────────────


def build_markdown_summary(
    baseline_row: Dict,
    minimal_row: Dict,
    removed_features: List[str],
    preserved_features: List[str],
    boot_auc: Dict,
    boot_brier: Dict,
    boot_auprc: Dict,
    dca_df: pd.DataFrame,
    miss_df: pd.DataFrame,
    interpretation: str,
    op_threshold: float,
) -> str:
    """Produce the echo_minimal_ablation_summary.md content."""

    def _fmt(v, decimals: int = 4) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v:.{decimals}f}"

    def _pct(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v * 100:.1f}%"

    interp_labels = {
        "superior": "SUPERIOR — echo-minimal consistently outperforms baseline",
        "non_inferior_parsimonious": "NON-INFERIOR / PARSIMONIOUS — comparable performance with fewer features",
        "inconclusive": "INCONCLUSIVE — results do not support a clear conclusion",
        "worse": "WORSE — echo-minimal degrades performance vs. baseline",
    }
    interp_label = interp_labels.get(interpretation, interpretation.upper())

    bl = baseline_row
    mn = minimal_row

    removed_list = "\n".join(f"- `{f}`" for f in removed_features) if removed_features else "- (none)"
    preserved_echo = [f for f in (ECHO_PRESERVE_CONTINUOUS + VALVE_SEVERITY_CATEGORICAL)
                      if f in preserved_features]
    preserved_list = "\n".join(f"- `{f}`" for f in preserved_echo) if preserved_echo else "- (none)"

    delta_auc_str = _fmt(
        (mn.get("AUC") or float("nan")) - (bl.get("AUC") or float("nan"))
    )
    auc_ci = (
        f"[{_fmt(boot_auc.get('ci_low'), 4)}, {_fmt(boot_auc.get('ci_high'), 4)}]"
        if not np.isnan(boot_auc.get("ci_low", float("nan"))) else "N/A"
    )
    auc_p = _fmt(boot_auc.get("p", float("nan")), 3)

    dca_rows = ""
    if dca_df is not None and len(dca_df) > 0:
        for thr in DCA_THRESHOLDS:
            bl_nb = dca_df.loc[
                (dca_df["Threshold"] == thr) & (dca_df["Strategy"] == "baseline"), "Net Benefit"
            ]
            mn_nb = dca_df.loc[
                (dca_df["Threshold"] == thr) & (dca_df["Strategy"] == "echo_minimal"), "Net Benefit"
            ]
            bl_val = _fmt(bl_nb.values[0] if len(bl_nb) > 0 else float("nan"), 4)
            mn_val = _fmt(mn_nb.values[0] if len(mn_nb) > 0 else float("nan"), 4)
            dca_rows += f"| {_pct(thr)} | {bl_val} | {mn_val} |\n"
    else:
        dca_rows = "| (no DCA data) | | |\n"

    n_removed = len(removed_features)
    n_baseline = bl.get("n_features", "N/A")
    n_minimal = mn.get("n_features", "N/A")

    miss_removed = miss_df[miss_df["group"] == "removed_echo_quantitative"]
    miss_pct_mean = (
        f"{miss_removed['pct_missing'].mean() * 100:.1f}%"
        if len(miss_removed) > 0 else "N/A"
    )

    md = f"""# Echo-Minimal Feature Ablation — Summary

> **EXPLORATORY ANALYSIS ONLY.**  This ablation does not alter the official AI Risk
> trained bundle, MODEL_VERSION, or threshold policy.  Promoting the echo-minimal
> feature set to a new baseline requires explicit human review.

---

## Hypothesis

Quantitative valvular echocardiographic variables (valve areas, gradients, PHT,
vena contracta) are highly dependent on structured report completion and may
introduce noise and missingness heterogeneity.  Removing them while preserving PSAP,
TAPSE, LVEF, and categorical valve severity scores may reduce data sparsity without
materially degrading discrimination or calibration.

---

## Feature Changes

**Features removed in echo-minimal ({n_removed} features, mean missingness {miss_pct_mean}):**

{removed_list}

**Key echo features preserved:**

{preserved_list}

**Feature counts:**
- Baseline: {n_baseline} features
- Echo-minimal: {n_minimal} features
- Removed: {n_removed} features

---

## Comparative Metrics

| Metric | Baseline | Echo-Minimal | Delta |
|---|---|---|---|
| n | {bl.get('n', 'N/A')} | {mn.get('n', 'N/A')} | — |
| Events | {bl.get('events', 'N/A')} | {mn.get('events', 'N/A')} | — |
| Prevalence | {_pct(bl.get('prevalence'))} | {_pct(mn.get('prevalence'))} | — |
| Features | {n_baseline} | {n_minimal} | −{n_removed} |
| AUC | {_fmt(bl.get('AUC'))} | {_fmt(mn.get('AUC'))} | {delta_auc_str} |
| AUPRC | {_fmt(bl.get('AUPRC'))} | {_fmt(mn.get('AUPRC'))} | {_fmt((mn.get('AUPRC') or float('nan')) - (bl.get('AUPRC') or float('nan')))} |
| AUPRC baseline | {_fmt(bl.get('AUPRC_baseline'))} | {_fmt(mn.get('AUPRC_baseline'))} | — |
| Brier | {_fmt(bl.get('Brier'))} | {_fmt(mn.get('Brier'))} | {_fmt((mn.get('Brier') or float('nan')) - (bl.get('Brier') or float('nan')))} |
| Brier Skill Score | {_fmt(bl.get('Brier_skill_score'))} | {_fmt(mn.get('Brier_skill_score'))} | — |
| Cal. intercept | {_fmt(bl.get('calibration_intercept'))} | {_fmt(mn.get('calibration_intercept'))} | — |
| Cal. slope | {_fmt(bl.get('calibration_slope'))} | {_fmt(mn.get('calibration_slope'))} | — |
| CIL | {_fmt(bl.get('CIL'))} | {_fmt(mn.get('CIL'))} | — |
| ICI | {_fmt(bl.get('ICI'))} | {_fmt(mn.get('ICI'))} | — |
| Threshold used | {_fmt(op_threshold)} | {_fmt(op_threshold)} | — |
| Sensitivity | {_pct(bl.get('sensitivity'))} | {_pct(mn.get('sensitivity'))} | {_fmt((mn.get('sensitivity') or float('nan')) - (bl.get('sensitivity') or float('nan')))} |
| Specificity | {_pct(bl.get('specificity'))} | {_pct(mn.get('specificity'))} | — |
| PPV | {_pct(bl.get('PPV'))} | {_pct(mn.get('PPV'))} | — |
| NPV | {_pct(bl.get('NPV'))} | {_pct(mn.get('NPV'))} | — |
| Flag rate | {_pct(bl.get('flag_rate'))} | {_pct(mn.get('flag_rate'))} | — |
| TP | {bl.get('TP', 'N/A')} | {mn.get('TP', 'N/A')} | — |
| FP | {bl.get('FP', 'N/A')} | {mn.get('FP', 'N/A')} | — |
| TN | {bl.get('TN', 'N/A')} | {mn.get('TN', 'N/A')} | — |
| FN | {bl.get('FN', 'N/A')} | {mn.get('FN', 'N/A')} | — |

### Bootstrap Comparison (paired OOF, n_boot=2000)

| Metric | Delta (baseline − echo_minimal) | 95% CI | p-value |
|---|---|---|---|
| AUC | {_fmt(-(boot_auc.get('delta_auc') or float('nan')))} | {auc_ci} | {auc_p} |
| AUPRC | {_fmt(-(boot_auprc.get('delta_auprc') or float('nan')))} | [{_fmt(boot_auprc.get('ci_low'), 4)}, {_fmt(boot_auprc.get('ci_high'), 4)}] | {_fmt(boot_auprc.get('p', float('nan')), 3)} |
| Brier | {_fmt(-(boot_brier.get('delta_brier') or float('nan')))} | [{_fmt(boot_brier.get('ci_low'), 4)}, {_fmt(boot_brier.get('ci_high'), 4)}] | {_fmt(boot_brier.get('p', float('nan')), 3)} |

*Note: delta = echo_minimal − baseline; negative delta_AUC means echo-minimal is lower.*

---

## Decision Curve Analysis (Net Benefit)

| Threshold | Baseline NB | Echo-Minimal NB |
|---|---|---|
{dca_rows}
---

## Interpretation

**{interp_label}**

### Criteria applied

- Non-inferior if: ΔAUC ≥ −0.015, ΔBrier ≤ +0.005, ΔSensitivity ≥ −3 pp, meaningful missingness reduction
- Superior if: ΔAUC > +0.005 with bootstrap CI entirely above 0, ΔBrier ≤ 0, ΔICI ≤ 0, ΔSens ≥ −0.5 pp
- Worse if: ΔAUC < −0.015, or ΔBrier > +0.005, or ΔSensitivity < −3 pp

---

## Alert

> **This ablation does NOT automatically update the production model.**
> Any change to the official feature set requires:
> 1. Human review of these results
> 2. Explicit regeneration of the bundle via `regenerate_bundle.py`
> 3. Version bump in `AppConfig.MODEL_VERSION`
> 4. Full regression test suite passage
"""
    return md


# ── Main ablation runner ──────────────────────────────────────────────────────


def run_ablation(
    data_path: str,
    outdir: Path,
    seed: int = _BASE_SEED,
    n_splits: int = 5,
    n_boot: int = 2000,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Run the echo-minimal ablation and write all outputs.

    Returns (metrics_df, features_df, missingness_df, markdown_text).
    """
    outdir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading data: {data_path}")

    prepared = prepare_master_dataset(data_path)
    df = prepared.data
    baseline_features = prepared.feature_columns

    y_col = "morte_30d"
    g_col = AppConfig.GROUP_KEY_COLUMN
    y = df[y_col].values.astype(int)
    groups = df[g_col].values if g_col in df.columns else None

    n_total = len(y)
    n_events = int(y.sum())
    auprc_bl = float(y.mean())

    echo_minimal_features = build_echo_minimal_features(baseline_features)
    removed_features = features_removed_from_baseline(baseline_features, echo_minimal_features)

    if verbose:
        print(
            f"  n={n_total}  events={n_events}  prevalence={auprc_bl:.1%}"
        )
        print(f"  Baseline features : {len(baseline_features)}")
        print(f"  Echo-minimal features: {len(echo_minimal_features)}")
        print(f"  Removed features  : {len(removed_features)} → {removed_features}")

    # Build the estimator (RandomForest with sigmoid calibration — matches production)
    original_seed = AppConfig.RANDOM_SEED
    AppConfig.RANDOM_SEED = seed
    try:
        estimator = RandomForestClassifier(**get_model_params("RandomForest"))
        cal_method: Optional[str] = "sigmoid"
        cal_cv: Optional[int] = 3
    finally:
        AppConfig.RANDOM_SEED = original_seed

    # Clean features for both arms
    AppConfig.RANDOM_SEED = seed
    try:
        X_base_raw = clean_features(df[baseline_features])
        non_empty_base = [c for c in X_base_raw.columns if not X_base_raw[c].isna().all()]
        X_base = X_base_raw[non_empty_base].copy()

        minimal_feat_available = [
            c for c in echo_minimal_features if c in df.columns
        ]
        X_min_raw = clean_features(df[minimal_feat_available])
        non_empty_min = [c for c in X_min_raw.columns if not X_min_raw[c].isna().all()]
        X_min = X_min_raw[non_empty_min].copy()
    finally:
        AppConfig.RANDOM_SEED = original_seed

    if verbose:
        print("\nRunning OOF CV — Baseline arm ...")
    oof_base, err_base = _run_oof_cv(
        X_base, y, groups,
        estimator=clone(estimator),
        cal_method=cal_method,
        cal_cv=cal_cv,
        seed=seed,
        n_splits=n_splits,
    )
    if err_base and verbose:
        print(f"  [WARNING] Baseline OOF error: {err_base.splitlines()[-1]}")

    if verbose:
        print("Running OOF CV — Echo-minimal arm ...")
    oof_min, err_min = _run_oof_cv(
        X_min, y, groups,
        estimator=clone(estimator),
        cal_method=cal_method,
        cal_cv=cal_cv,
        seed=seed,
        n_splits=n_splits,
    )
    if err_min and verbose:
        print(f"  [WARNING] Echo-minimal OOF error: {err_min.splitlines()[-1]}")

    # Compute per-arm metrics
    base_row = compute_arm_metrics(
        y, oof_base, auprc_bl, "baseline",
        n_features=len(non_empty_base),
        op_threshold=OPERATIONAL_THRESHOLD,
    )
    min_row = compute_arm_metrics(
        y, oof_min, auprc_bl, "echo_minimal",
        n_features=len(non_empty_min),
        op_threshold=OPERATIONAL_THRESHOLD,
    )

    if verbose:
        print(
            f"\n  Baseline   AUC={base_row.get('AUC', float('nan')):.4f}  "
            f"Brier={base_row.get('Brier', float('nan')):.4f}  "
            f"Sens@{OPERATIONAL_THRESHOLD:.1%}={base_row.get('sensitivity', float('nan')):.3f}"
        )
        print(
            f"  Echo-min   AUC={min_row.get('AUC', float('nan')):.4f}  "
            f"Brier={min_row.get('Brier', float('nan')):.4f}  "
            f"Sens@{OPERATIONAL_THRESHOLD:.1%}={min_row.get('sensitivity', float('nan')):.3f}"
        )

    # Bootstrap comparisons (paired on the same observations)
    valid_both = ~np.isnan(oof_base) & ~np.isnan(oof_min)
    yv_both = y[valid_both]
    p_base_both = oof_base[valid_both]
    p_min_both = oof_min[valid_both]

    boot_auc: Dict = {}
    boot_brier: Dict = {}
    boot_auprc: Dict = {}

    if n_boot > 0 and len(yv_both) > 30 and len(np.unique(yv_both)) == 2:
        if verbose:
            print(f"  Bootstrap comparisons (n_boot={n_boot}) ...")
        # bootstrap_auc_diff returns delta = p1 - p2
        # We pass (baseline, echo_minimal) → positive = baseline better
        boot_auc = bootstrap_auc_diff(yv_both, p_base_both, p_min_both,
                                      n_boot=n_boot, seed=seed)
        boot_brier = _bootstrap_diff_brier(yv_both, p_base_both, p_min_both,
                                            n_boot=n_boot, seed=seed)
        boot_auprc = _bootstrap_diff_auprc(yv_both, p_base_both, p_min_both,
                                            n_boot=n_boot, seed=seed)
        if verbose:
            print(
                f"    ΔAUC(base−min)={boot_auc.get('delta_auc', float('nan')):.4f} "
                f"[{boot_auc.get('ci_low', float('nan')):.4f}, "
                f"{boot_auc.get('ci_high', float('nan')):.4f}]  "
                f"p={boot_auc.get('p', float('nan')):.3f}"
            )
    else:
        if verbose:
            print("  [INFO] Skipping bootstrap (n_boot=0 or insufficient paired OOF)")
        for d in [boot_auc, boot_brier, boot_auprc]:
            d.update({"delta_auc": float("nan"), "ci_low": float("nan"),
                      "ci_high": float("nan"), "p": float("nan")})
        boot_brier.update({"delta_brier": float("nan")})
        boot_auprc.update({"delta_auprc": float("nan")})

    # DCA
    dca_df: Optional[pd.DataFrame] = None
    if len(yv_both) > 0 and len(np.unique(yv_both)) == 2:
        thr_arr = np.array(DCA_THRESHOLDS, dtype=float)
        dca_df = decision_curve(
            yv_both,
            {"baseline": p_base_both, "echo_minimal": p_min_both},
            thr_arr,
        )

    # Missingness summary
    preserved_echo_cols = [
        c for c in ECHO_PRESERVE_CONTINUOUS + VALVE_SEVERITY_CATEGORICAL
        if c in baseline_features
    ]
    miss_df = compute_missingness_summary(df, removed_features, preserved_echo_cols)

    # Build outputs
    metrics_df = pd.DataFrame([base_row, min_row])
    features_df = build_features_dataframe(baseline_features, echo_minimal_features)

    # Add bootstrap comparison rows to metrics_df
    boot_row = {
        "arm": "comparison_bootstrap",
        "delta_auc_base_minus_min": boot_auc.get("delta_auc", float("nan")),
        "delta_auc_ci_low": boot_auc.get("ci_low", float("nan")),
        "delta_auc_ci_high": boot_auc.get("ci_high", float("nan")),
        "delta_auc_p": boot_auc.get("p", float("nan")),
        "delta_brier_base_minus_min": boot_brier.get("delta_brier", float("nan")),
        "delta_brier_ci_low": boot_brier.get("ci_low", float("nan")),
        "delta_brier_ci_high": boot_brier.get("ci_high", float("nan")),
        "delta_brier_p": boot_brier.get("p", float("nan")),
        "delta_auprc_base_minus_min": boot_auprc.get("delta_auprc", float("nan")),
        "delta_auprc_ci_low": boot_auprc.get("ci_low", float("nan")),
        "delta_auprc_ci_high": boot_auprc.get("ci_high", float("nan")),
        "delta_auprc_p": boot_auprc.get("p", float("nan")),
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([boot_row])], ignore_index=True)

    # Interpretation
    interpretation = interpret_result(
        base_row, min_row, boot_auc, miss_df
    )
    if verbose:
        print(f"\n  Interpretation: {interpretation.upper()}")

    # Markdown summary
    md_text = build_markdown_summary(
        baseline_row=base_row,
        minimal_row=min_row,
        removed_features=removed_features,
        preserved_features=echo_minimal_features,
        boot_auc=boot_auc,
        boot_brier=boot_brier,
        boot_auprc=boot_auprc,
        dca_df=dca_df if dca_df is not None else pd.DataFrame(),
        miss_df=miss_df,
        interpretation=interpretation,
        op_threshold=OPERATIONAL_THRESHOLD,
    )

    # Write outputs
    metrics_path = outdir / "echo_minimal_ablation_metrics.csv"
    features_path = outdir / "echo_minimal_ablation_features.csv"
    miss_path = outdir / "echo_minimal_ablation_missingness.csv"
    summary_path = outdir / "echo_minimal_ablation_summary.md"

    metrics_df.to_csv(metrics_path, index=False)
    features_df.to_csv(features_path, index=False)
    miss_df.to_csv(miss_path, index=False)
    summary_path.write_text(md_text, encoding="utf-8")

    if dca_df is not None:
        dca_path = outdir / "echo_minimal_ablation_dca.csv"
        dca_df.to_csv(dca_path, index=False)

    if verbose:
        print(f"\nOutputs written to {outdir}:")
        print(f"  {metrics_path.name}")
        print(f"  {features_path.name}")
        print(f"  {miss_path.name}")
        print(f"  {summary_path.name}")

    return metrics_df, features_df, miss_df, md_text


# ── CLI ────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Echo-Minimal Feature Ablation (exploratory — does not alter bundle)"
    )
    p.add_argument(
        "--data",
        default="local_data/Dataset_2025.xlsx",
        help="Path to the source data file",
    )
    p.add_argument(
        "--outdir",
        default="reports",
        help="Output directory for CSV and markdown files",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=_BASE_SEED,
        help="Random seed",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=AppConfig.CV_SPLITS,
        dest="n_splits",
        help="Number of CV splits",
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        dest="n_boot",
        help="Bootstrap resamples for comparison (0 = skip)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_ablation(
        data_path=args.data,
        outdir=Path(args.outdir),
        seed=args.seed,
        n_splits=args.n_splits,
        n_boot=args.n_boot,
        verbose=not args.quiet,
    )
