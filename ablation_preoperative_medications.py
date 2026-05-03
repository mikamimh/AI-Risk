"""Ablation: Preoperative Medications Field

Tests whether removing `preoperative_medications` from the feature set
preserves or improves model performance, calibration, and stability.

Hypothesis
----------
`Preoperative Medications` is a retrospective free-text field with heterogeneous
filling that may introduce noise.  Its clinical signal may already be captured
by more direct variables (arrhythmia, HF, diabetes, coronary disease, urgency,
LVEF, PSAP, surgery type).  Removing it may reduce documentation-driven variance.

IMPORTANT FINDING: At the time of this analysis, "Preoperative Medications" is
already excluded from the production feature set via the `_noise_cols` filter
in `risk_data.py`.  This experiment therefore constitutes a null-experiment /
control that:
  (1) confirms the field is absent from the active feature set,
  (2) documents its missingness and structural patterns,
  (3) verifies that both arms produce identical OOF predictions and metrics.

This script does NOT alter the official trained bundle, MODEL_VERSION, threshold
policy, EuroSCORE II, STS integration, or production pipeline.

Usage
-----
    python ablation_preoperative_medications.py \\
        --data local_data/Dataset_2025.xlsx \\
        --outdir reports \\
        --seed 42 \\
        --n-splits 5 \\
        --n-boot 2000

Outputs (written to --outdir)
------------------------------
    preoperative_medications_ablation_metrics.csv
    preoperative_medications_ablation_features.csv
    preoperative_medications_ablation_missingness.csv
    preoperative_medications_ablation_dca.csv
    preoperative_medications_threshold_sens90.csv
    preoperative_medications_ablation_summary.md
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

import traceback

from config import AppConfig, get_model_params
from modeling import build_preprocessor, clean_features
from risk_data import prepare_master_dataset
from stats_compare import (
    bootstrap_auc_diff,
    brier_skill_score as _bss_fn,
    calibration_intercept_slope,
    decision_curve,
    integrated_calibration_index,
    sensitivity_constrained_threshold as _sens_constrained_thr,
)

# ── Constants ─────────────────────────────────────────────────────────────────

# Canonical column name as it appears in the source data and risk_data.py
MEDICATIONS_FEATURE: str = "Preoperative Medications"

# All aliases that may appear in a feature list (defensive)
MEDICATIONS_FEATURE_ALIASES: List[str] = [
    "Preoperative Medications",
    "preoperative_medications",
    "Preoperative_Medications",
]

# Operational threshold from the official v17 sens_constrained_90 policy (8.5 %)
OPERATIONAL_THRESHOLD: float = 0.085

# DCA evaluation thresholds
DCA_THRESHOLDS: List[float] = [0.02, 0.05, 0.08, 0.085, 0.10, 0.15, 0.20]

_BASE_SEED: int = 42

# ── Self-contained pipeline helpers ──────────────────────────────────────────


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
    """Stratified-(group-)K-fold OOF with optional in-fold calibration."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import Pipeline

    try:
        pos = int(y.sum())
        neg = int(len(y) - pos)
        if groups is not None:
            k = min(n_splits, pd.Series(groups).nunique(), pos, neg)
            cv_obj = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        else:
            k = min(n_splits, pos, neg, len(X))
            cv_obj = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

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
            from sklearn.base import clone
            fold_pipe = Pipeline([("prep", prep), ("clf", clone(estimator))])
            if cal_method is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fold_pipe.fit(X_tr, y_tr)
                oof[test_idx] = fold_pipe.predict_proba(X_te)[:, 1]
            else:
                eff_cv = min(cal_cv, int(y_tr.sum()), int(len(y_tr) - y_tr.sum()))
                if eff_cv < 2:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fold_pipe.fit(X_tr, y_tr)
                    oof[test_idx] = fold_pipe.predict_proba(X_te)[:, 1]
                else:
                    cal = CalibratedClassifierCV(fold_pipe, method=cal_method,
                                                 cv=eff_cv, ensemble=False)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cal.fit(X_tr, y_tr)
                    oof[test_idx] = cal.predict_proba(X_te)[:, 1]
        return oof, ""
    except Exception:
        return np.full(len(y), float("nan")), traceback.format_exc(limit=3)


def _threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict:
    """Classification metrics at a fixed threshold, including observed mortality."""
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n  = len(y)
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity":  tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "specificity":  tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "PPV":          tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        "NPV":          tn / (tn + fn) if (tn + fn) > 0 else float("nan"),
        "flag_rate":    (tp + fp) / n  if n > 0       else float("nan"),
        "obs_mortality_above_thr": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        "obs_mortality_below_thr": fn / (fn + tn) if (fn + tn) > 0 else float("nan"),
    }


def _make_threshold_row(
    arm: str, label: str, thr: float, y: np.ndarray, p: np.ndarray
) -> Dict:
    """Build one row for the threshold comparison table."""
    if np.isnan(thr):
        row: Dict = {
            "arm": arm, "threshold_label": label, "threshold": float("nan"),
            "sensitivity": float("nan"), "specificity": float("nan"),
            "PPV": float("nan"), "NPV": float("nan"),
            "TP": float("nan"), "FP": float("nan"),
            "TN": float("nan"), "FN": float("nan"),
            "flag_rate": float("nan"),
            "obs_mortality_above_thr": float("nan"),
            "obs_mortality_below_thr": float("nan"),
        }
        return row
    m = _threshold_metrics(y, p, thr)
    return {"arm": arm, "threshold_label": label, "threshold": thr, **m}


def compute_arm_metrics(
    y: np.ndarray,
    p: np.ndarray,
    auprc_baseline: float,
    arm_name: str,
    n_features: int,
    op_threshold: float = 0.085,
) -> Dict:
    """Full metric suite for one OOF arm."""
    valid = ~np.isnan(p)
    yv, pv = y[valid], p[valid]

    row: Dict = {
        "arm": arm_name, "n": int(len(yv)), "events": int(yv.sum()),
        "prevalence": float(yv.mean()) if len(yv) > 0 else float("nan"),
        "n_features": n_features, "AUPRC_baseline": auprc_baseline,
    }
    if len(yv) == 0 or int(yv.sum()) == 0 or int((1 - yv).sum()) == 0:
        for k in ["AUC", "AUPRC", "Brier", "Brier_skill_score",
                  "calibration_intercept", "calibration_slope", "CIL", "ICI",
                  "sensitivity", "specificity", "PPV", "NPV",
                  "TP", "FP", "TN", "FN", "flag_rate", "threshold_used"]:
            row[k] = float("nan")
        return row

    try:
        row["AUC"]   = float(roc_auc_score(yv, pv))
    except Exception:
        row["AUC"]   = float("nan")
    try:
        row["AUPRC"] = float(average_precision_score(yv, pv))
    except Exception:
        row["AUPRC"] = float("nan")
    try:
        row["Brier"] = float(brier_score_loss(yv, pv))
        row["Brier_skill_score"] = float(_bss_fn(yv, pv))
    except Exception:
        row["Brier"] = row["Brier_skill_score"] = float("nan")
    try:
        cal = calibration_intercept_slope(yv, pv)
        row["calibration_intercept"] = cal.get("Calibration intercept", float("nan"))
        row["calibration_slope"]     = cal.get("Calibration slope",     float("nan"))
    except Exception:
        row["calibration_intercept"] = row["calibration_slope"] = float("nan")
    row["CIL"] = float(np.mean(pv) - np.mean(yv))
    try:
        row["ICI"] = float(integrated_calibration_index(yv, pv))
    except Exception:
        row["ICI"] = float("nan")

    thr_m = _threshold_metrics(yv, pv, op_threshold)
    row["threshold_used"] = op_threshold
    for k in ["sensitivity", "specificity", "PPV", "NPV",
              "TP", "FP", "TN", "FN", "flag_rate"]:
        row[k] = thr_m[k]
    return row


def _bootstrap_diff_brier(
    y: np.ndarray, p_base: np.ndarray, p_other: np.ndarray,
    n_boot: int = 2000, seed: int = 42,
) -> Dict:
    """Paired bootstrap for Brier difference (baseline - other)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            deltas.append(
                brier_score_loss(y[idx], p_base[idx]) - brier_score_loss(y[idx], p_other[idx])
            )
        except Exception:
            continue
    if not deltas:
        return {"delta_brier": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p": float("nan")}
    arr = np.array(deltas)
    return {
        "delta_brier": float(brier_score_loss(y, p_base) - brier_score_loss(y, p_other)),
        "ci_low":  float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "p": float(2 * min((arr <= 0).mean(), (arr >= 0).mean())),
    }


def _bootstrap_diff_auprc(
    y: np.ndarray, p_base: np.ndarray, p_other: np.ndarray,
    n_boot: int = 2000, seed: int = 42,
) -> Dict:
    """Paired bootstrap for AUPRC difference (baseline - other)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        try:
            deltas.append(
                average_precision_score(ys, p_base[idx]) - average_precision_score(ys, p_other[idx])
            )
        except Exception:
            continue
    if not deltas:
        return {"delta_auprc": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p": float("nan")}
    arr = np.array(deltas)
    return {
        "delta_auprc": float(average_precision_score(y, p_base) - average_precision_score(y, p_other)),
        "ci_low":  float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "p": float(2 * min((arr <= 0).mean(), (arr >= 0).mean())),
    }


# ── Feature-set helpers (pure, testable) ──────────────────────────────────────


def build_medication_free_features(feature_columns: Sequence[str]) -> List[str]:
    """Return the medication-free feature list derived from *feature_columns*.

    Removes any column matching MEDICATIONS_FEATURE_ALIASES (case-sensitive).
    If none match, the list is returned unchanged — this is expected when the
    field is already excluded by the pipeline (the current production state).

    The input list is NOT modified.
    """
    remove_set = set(MEDICATIONS_FEATURE_ALIASES)
    return [c for c in feature_columns if c not in remove_set]


def feature_was_present(feature_columns: Sequence[str]) -> bool:
    """Return True if any medications alias is found in *feature_columns*."""
    remove_set = set(MEDICATIONS_FEATURE_ALIASES)
    return any(c in remove_set for c in feature_columns)


def medication_missingness(df: pd.DataFrame) -> Dict:
    """Compute basic missingness statistics for the medications column.

    Returns a dict with: present_in_data, n_total, n_missing, pct_missing,
    n_none_literal, n_empty, n_non_empty.
    """
    n = len(df)
    col = MEDICATIONS_FEATURE
    if col not in df.columns:
        return {
            "present_in_data": False,
            "n_total": n,
            "n_missing": n,
            "pct_missing": 1.0,
            "n_none_literal": 0,
            "n_empty": n,
            "n_non_empty": 0,
        }

    s = df[col].fillna("").astype(str)
    missing_tokens = {"", "nan", "none", "na", "n/a", "-", "--", "null",
                      "not applicable", "unknown", "not informed", "nao informado"}
    is_empty = s.str.strip().str.lower().isin(missing_tokens)
    is_none_literal = s.str.strip().str.lower() == "none"

    return {
        "present_in_data": True,
        "n_total": n,
        "n_missing": int(is_empty.sum()),
        "pct_missing": float(is_empty.mean()),
        "n_none_literal": int(is_none_literal.sum()),
        "n_empty": int(is_empty.sum()),
        "n_non_empty": int((~is_empty).sum()),
    }


def build_missingness_dataframe(
    df: pd.DataFrame,
    was_in_feature_set: bool,
) -> pd.DataFrame:
    """Return a one-row DataFrame summarising medications missingness."""
    m = medication_missingness(df)
    return pd.DataFrame([{
        "feature": MEDICATIONS_FEATURE,
        "was_in_feature_set": was_in_feature_set,
        "present_in_source_data": m["present_in_data"],
        "n_total": m["n_total"],
        "n_missing": m["n_missing"],
        "pct_missing": m["pct_missing"],
        "n_none_literal": m["n_none_literal"],
        "n_non_empty": m["n_non_empty"],
        "note": (
            "Already excluded from production feature set via _noise_cols in risk_data.py"
            if not was_in_feature_set else
            "Was present in feature set and has been removed in this ablation"
        ),
    }])


# ── Threshold sensitivity helpers ─────────────────────────────────────────────

_THR_ANALYSIS_COLS = [
    "arm", "threshold_label", "threshold",
    "sensitivity", "specificity", "PPV", "NPV",
    "TP", "FP", "TN", "FN",
    "flag_rate", "obs_mortality_above_thr", "obs_mortality_below_thr",
]


def build_threshold_comparison(
    y: np.ndarray,
    p_base: np.ndarray,
    p_med_free: np.ndarray,
    sens90_thr: float,
) -> pd.DataFrame:
    """Return the three-row operating-point comparison table."""
    rows = [
        _make_threshold_row("baseline",         "baseline_fixed_085",    OPERATIONAL_THRESHOLD, y, p_base),
        _make_threshold_row("medication_free",  "medfree_fixed_085",     OPERATIONAL_THRESHOLD, y, p_med_free),
        _make_threshold_row("medication_free",  "medfree_sens90",        sens90_thr,             y, p_med_free),
    ]
    return pd.DataFrame(rows)


# ── Markdown builders ─────────────────────────────────────────────────────────


def _fmt(v, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(float(v))):
        return "N/A"
    return f"{float(v):.{decimals}f}"


def _pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(float(v))):
        return "N/A"
    return f"{float(v) * 100:.1f}%"


def build_markdown_summary(
    baseline_row: Dict,
    medfree_row: Dict,
    was_in_feature_set: bool,
    n_removed: int,
    miss_df: pd.DataFrame,
    boot_auc: Dict,
    boot_brier: Dict,
    boot_auprc: Dict,
    dca_df: Optional[pd.DataFrame],
    thr_df: Optional[pd.DataFrame],
    interpretation: str,
    sens90_thr: float,
    op_threshold: float,
) -> str:
    """Produce the preoperative_medications_ablation_summary.md content."""
    interp_labels = {
        "superior":                 "SUPERIOR — medication-free consistently outperforms baseline",
        "non_inferior_parsimonious":"NON-INFERIOR / PARSIMONIOUS — comparable performance with reduced complexity",
        "inconclusive":             "INCONCLUSIVE — results do not support a clear conclusion",
        "worse":                    "WORSE — medication-free degrades performance vs baseline",
        "null_experiment":          "NULL EXPERIMENT — feature was already excluded; both arms are identical",
    }
    interp_label = interp_labels.get(interpretation, interpretation.upper())

    bl = baseline_row
    mf = medfree_row

    already_excluded_note = ""
    if not was_in_feature_set:
        already_excluded_note = """
> **NOTE: `Preoperative Medications` was NOT in the production feature set.**
> It is excluded via `_noise_cols` in `risk_data.py` (line ~3441).  Both arms
> use the identical feature set, so all metrics are numerically equal.
> This experiment serves as a control / documentation run.

"""

    def _thr_row(df: Optional[pd.DataFrame], label: str) -> Dict:
        if df is None or len(df) == 0:
            return {}
        rows = df[df["threshold_label"] == label]
        return rows.iloc[0].to_dict() if len(rows) > 0 else {}

    thr_base = _thr_row(thr_df, "baseline_fixed_085")
    thr_mf085 = _thr_row(thr_df, "medfree_fixed_085")
    thr_mfs90 = _thr_row(thr_df, "medfree_sens90")

    dca_rows = ""
    if dca_df is not None and len(dca_df) > 0:
        for t in DCA_THRESHOLDS:
            bl_nb = dca_df.loc[
                (dca_df["Threshold"] == t) & (dca_df["Strategy"] == "baseline"), "Net Benefit"
            ]
            mf_nb = dca_df.loc[
                (dca_df["Threshold"] == t) & (dca_df["Strategy"] == "medication_free"), "Net Benefit"
            ]
            bl_val = _fmt(bl_nb.values[0] if len(bl_nb) > 0 else float("nan"), 4)
            mf_val = _fmt(mf_nb.values[0] if len(mf_nb) > 0 else float("nan"), 4)
            dca_rows += f"| {_pct(t)} | {bl_val} | {mf_val} |\n"
    else:
        dca_rows = "| (no DCA data) | | |\n"

    miss_row = miss_df.iloc[0].to_dict() if len(miss_df) > 0 else {}

    return f"""# Preoperative Medications Ablation — Summary

> **EXPLORATORY ANALYSIS ONLY.**  This ablation does not alter the official
> AI Risk trained bundle, MODEL_VERSION, or threshold policy.  Promoting any
> change to the production feature set requires explicit human review.

---

## Hypothesis

`Preoperative Medications` is a retrospective free-text field with heterogeneous
filling.  Its clinical signal may already be captured by more direct variables
(arrhythmia, HF, diabetes, coronary disease, urgency, LVEF, PSAP, surgery type).
Removing it may reduce documentation-driven noise without degrading discrimination.

---

## Feature Status

{already_excluded_note}- **Feature targeted:** `{MEDICATIONS_FEATURE}`
- **Was in production feature set:** {was_in_feature_set}
- **Features removed in medication-free arm:** {n_removed}
- **Baseline features:** {bl.get('n_features', 'N/A')}
- **Medication-free features:** {mf.get('n_features', 'N/A')}

---

## Missingness of `Preoperative Medications`

| Field | Value |
|---|---|
| Present in source data | {miss_row.get('present_in_source_data', 'N/A')} |
| n_total | {miss_row.get('n_total', 'N/A')} |
| n_missing | {miss_row.get('n_missing', 'N/A')} |
| pct_missing | {_pct(miss_row.get('pct_missing'))} |
| n_none_literal | {miss_row.get('n_none_literal', 'N/A')} |
| n_non_empty | {miss_row.get('n_non_empty', 'N/A')} |
| Note | {miss_row.get('note', '')} |

---

## Comparative Metrics (OOF)

| Metric | Baseline | Medication-Free | Delta |
|---|---|---|---|
| n | {bl.get('n', 'N/A')} | {mf.get('n', 'N/A')} | — |
| Events | {bl.get('events', 'N/A')} | {mf.get('events', 'N/A')} | — |
| Prevalence | {_pct(bl.get('prevalence'))} | {_pct(mf.get('prevalence'))} | — |
| Features | {bl.get('n_features', 'N/A')} | {mf.get('n_features', 'N/A')} | {int(mf.get('n_features', 0) or 0) - int(bl.get('n_features', 0) or 0):+d} |
| AUC | {_fmt(bl.get('AUC'))} | {_fmt(mf.get('AUC'))} | {_fmt((mf.get('AUC') or float('nan')) - (bl.get('AUC') or float('nan')))} |
| AUPRC | {_fmt(bl.get('AUPRC'))} | {_fmt(mf.get('AUPRC'))} | {_fmt((mf.get('AUPRC') or float('nan')) - (bl.get('AUPRC') or float('nan')))} |
| AUPRC baseline | {_fmt(bl.get('AUPRC_baseline'))} | {_fmt(mf.get('AUPRC_baseline'))} | — |
| Brier | {_fmt(bl.get('Brier'))} | {_fmt(mf.get('Brier'))} | {_fmt((mf.get('Brier') or float('nan')) - (bl.get('Brier') or float('nan')))} |
| Brier Skill Score | {_fmt(bl.get('Brier_skill_score'))} | {_fmt(mf.get('Brier_skill_score'))} | — |
| Cal. intercept | {_fmt(bl.get('calibration_intercept'))} | {_fmt(mf.get('calibration_intercept'))} | — |
| Cal. slope | {_fmt(bl.get('calibration_slope'))} | {_fmt(mf.get('calibration_slope'))} | — |
| CIL | {_fmt(bl.get('CIL'))} | {_fmt(mf.get('CIL'))} | — |
| ICI | {_fmt(bl.get('ICI'))} | {_fmt(mf.get('ICI'))} | — |

### Bootstrap comparisons (paired OOF, n_boot=2000)

| Metric | delta(base-medfree) | 95% CI | p |
|---|---|---|---|
| AUC | {_fmt(boot_auc.get('delta_auc', float('nan')))} | [{_fmt(boot_auc.get('ci_low'), 4)}, {_fmt(boot_auc.get('ci_high'), 4)}] | {_fmt(boot_auc.get('p', float('nan')), 3)} |
| AUPRC | {_fmt(boot_auprc.get('delta_auprc', float('nan')))} | [{_fmt(boot_auprc.get('ci_low'), 4)}, {_fmt(boot_auprc.get('ci_high'), 4)}] | {_fmt(boot_auprc.get('p', float('nan')), 3)} |
| Brier | {_fmt(boot_brier.get('delta_brier', float('nan')))} | [{_fmt(boot_brier.get('ci_low'), 4)}, {_fmt(boot_brier.get('ci_high'), 4)}] | {_fmt(boot_brier.get('p', float('nan')), 3)} |

*delta = baseline - medication_free; negative = medication-free is better.*

---

## Operating Point Comparison

### At fixed threshold 8.5%

| Metric | Baseline @8.5% | Med-free @8.5% | Delta |
|---|---|---|---|
| Sensitivity | {_pct(thr_base.get('sensitivity'))} | {_pct(thr_mf085.get('sensitivity'))} | {_fmt((thr_mf085.get('sensitivity') or float('nan')) - (thr_base.get('sensitivity') or float('nan')))} |
| Specificity | {_pct(thr_base.get('specificity'))} | {_pct(thr_mf085.get('specificity'))} | — |
| PPV | {_pct(thr_base.get('PPV'))} | {_pct(thr_mf085.get('PPV'))} | — |
| NPV | {_pct(thr_base.get('NPV'))} | {_pct(thr_mf085.get('NPV'))} | — |
| Flag rate | {_pct(thr_base.get('flag_rate'))} | {_pct(thr_mf085.get('flag_rate'))} | — |
| TP | {thr_base.get('TP', 'N/A')} | {thr_mf085.get('TP', 'N/A')} | — |
| FP | {thr_base.get('FP', 'N/A')} | {thr_mf085.get('FP', 'N/A')} | — |
| TN | {thr_base.get('TN', 'N/A')} | {thr_mf085.get('TN', 'N/A')} | — |
| FN | {thr_base.get('FN', 'N/A')} | {thr_mf085.get('FN', 'N/A')} | — |
| Obs. mortality above thr | {_pct(thr_base.get('obs_mortality_above_thr'))} | {_pct(thr_mf085.get('obs_mortality_above_thr'))} | — |
| Obs. mortality below thr | {_pct(thr_base.get('obs_mortality_below_thr'))} | {_pct(thr_mf085.get('obs_mortality_below_thr'))} | — |

### Baseline @8.5% vs Med-free @Sens90 ({_pct(sens90_thr)})

| Metric | Baseline @8.5% | Med-free @{_pct(sens90_thr)} | Delta |
|---|---|---|---|
| Threshold | {_pct(op_threshold)} | {_pct(thr_mfs90.get('threshold'))} | — |
| Sensitivity | {_pct(thr_base.get('sensitivity'))} | {_pct(thr_mfs90.get('sensitivity'))} | {_fmt((thr_mfs90.get('sensitivity') or float('nan')) - (thr_base.get('sensitivity') or float('nan')))} |
| Specificity | {_pct(thr_base.get('specificity'))} | {_pct(thr_mfs90.get('specificity'))} | — |
| PPV | {_pct(thr_base.get('PPV'))} | {_pct(thr_mfs90.get('PPV'))} | — |
| NPV | {_pct(thr_base.get('NPV'))} | {_pct(thr_mfs90.get('NPV'))} | — |
| Flag rate | {_pct(thr_base.get('flag_rate'))} | {_pct(thr_mfs90.get('flag_rate'))} | — |
| TP | {thr_base.get('TP', 'N/A')} | {thr_mfs90.get('TP', 'N/A')} | — |
| FP | {thr_base.get('FP', 'N/A')} | {thr_mfs90.get('FP', 'N/A')} | — |
| TN | {thr_base.get('TN', 'N/A')} | {thr_mfs90.get('TN', 'N/A')} | — |
| FN | {thr_base.get('FN', 'N/A')} | {thr_mfs90.get('FN', 'N/A')} | — |
| Obs. mortality above thr | {_pct(thr_base.get('obs_mortality_above_thr'))} | {_pct(thr_mfs90.get('obs_mortality_above_thr'))} | — |
| Obs. mortality below thr | {_pct(thr_base.get('obs_mortality_below_thr'))} | {_pct(thr_mfs90.get('obs_mortality_below_thr'))} | — |

---

## Decision Curve Analysis (Net Benefit)

| Threshold | Baseline | Med-free |
|---|---|---|
{dca_rows}
---

## Interpretation

**{interp_label}**

### Criteria applied

- Non-inferior if: dAUC >= -0.015, dBrier <= 0.005, dSens >= -3pp, any complexity reduction
- Superior if: consistent improvement in AUC, Brier, ICI, DCA with bootstrap CI above 0
- Worse if: dAUC < -0.015, dBrier > 0.005, or dSens < -3pp
- Null experiment if: feature was already excluded (both arms identical)

---

## Alert

> **This ablation does NOT automatically update the production model.**
> Any change to the official feature set requires:
> 1. Human review of these results
> 2. Explicit regeneration of the bundle via `regenerate_bundle.py`
> 3. Version bump in `AppConfig.MODEL_VERSION`
> 4. Full regression test suite passage
"""


# ── Interpretation ─────────────────────────────────────────────────────────────


def interpret_result(
    baseline_row: Dict,
    medfree_row: Dict,
    boot_auc: Dict,
    was_in_feature_set: bool,
    delta_auc_noni_tol: float = 0.015,
    delta_brier_tol: float = 0.005,
) -> str:
    """Conservative automatic interpretation.

    Returns one of: "null_experiment", "superior", "non_inferior_parsimonious",
    "inconclusive", "worse".
    """
    if not was_in_feature_set:
        return "null_experiment"

    def _get(d: Dict, key: str) -> float:
        v = d.get(key, float("nan"))
        return float(v) if v is not None else float("nan")

    base_auc   = _get(baseline_row, "AUC")
    mf_auc     = _get(medfree_row,  "AUC")
    base_brier = _get(baseline_row, "Brier")
    mf_brier   = _get(medfree_row,  "Brier")
    base_ici   = _get(baseline_row, "ICI")
    mf_ici     = _get(medfree_row,  "ICI")
    base_sens  = _get(baseline_row, "sensitivity")
    mf_sens    = _get(medfree_row,  "sensitivity")

    if any(np.isnan(v) for v in [base_auc, mf_auc, base_brier, mf_brier,
                                  base_sens, mf_sens]):
        return "inconclusive"

    delta_auc   = mf_auc   - base_auc
    delta_brier = mf_brier - base_brier
    delta_sens  = mf_sens  - base_sens

    if delta_auc < -delta_auc_noni_tol:
        return "worse"
    if delta_brier > delta_brier_tol:
        return "worse"
    if delta_sens < -0.03:
        return "worse"

    auc_ci_high = boot_auc.get("ci_high", float("nan"))
    if (
        delta_auc > 0.005
        and delta_brier <= 0
        and (np.isnan(mf_ici) or np.isnan(base_ici) or mf_ici <= base_ici)
        and delta_sens >= -0.005
        and not np.isnan(auc_ci_high)
        and auc_ci_high < 0
    ):
        return "superior"

    if delta_auc >= -delta_auc_noni_tol and delta_brier <= delta_brier_tol and delta_sens >= -0.03:
        return "non_inferior_parsimonious"

    return "inconclusive"


# ── Main ablation runner ───────────────────────────────────────────────────────


def run_ablation(
    data_path: str,
    outdir: Path,
    seed: int = _BASE_SEED,
    n_splits: int = 5,
    n_boot: int = 2000,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Run the preoperative medications ablation and write all outputs.

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

    n_total  = len(y)
    n_events = int(y.sum())
    auprc_bl = float(y.mean())

    medfree_features = build_medication_free_features(baseline_features)
    was_present = feature_was_present(baseline_features)
    n_removed = len(baseline_features) - len(medfree_features)

    if verbose:
        print(f"  n={n_total}  events={n_events}  prevalence={auprc_bl:.1%}")
        print(f"  Baseline features   : {len(baseline_features)}")
        print(f"  Medication-free     : {len(medfree_features)}")
        print(f"  Removed             : {n_removed}")
        if was_present:
            print(f"  -> '{MEDICATIONS_FEATURE}' WAS in the feature set and has been removed.")
        else:
            print(
                f"  -> '{MEDICATIONS_FEATURE}' was NOT in the feature set "
                f"(already excluded via _noise_cols in risk_data.py)."
            )
            print("  -> This is a null experiment: both arms are identical.")

    # Build estimator (RandomForest + sigmoid — matches production)
    original_seed = AppConfig.RANDOM_SEED
    AppConfig.RANDOM_SEED = seed
    try:
        estimator = RandomForestClassifier(**get_model_params("RandomForest"))
        cal_method: Optional[str] = "sigmoid"
        cal_cv: Optional[int] = 3
    finally:
        AppConfig.RANDOM_SEED = original_seed

    AppConfig.RANDOM_SEED = seed
    try:
        X_base_raw = clean_features(df[baseline_features])
        non_empty_base = [c for c in X_base_raw.columns if not X_base_raw[c].isna().all()]
        X_base = X_base_raw[non_empty_base].copy()

        mf_avail = [c for c in medfree_features if c in df.columns]
        X_mf_raw = clean_features(df[mf_avail])
        non_empty_mf = [c for c in X_mf_raw.columns if not X_mf_raw[c].isna().all()]
        X_mf = X_mf_raw[non_empty_mf].copy()
    finally:
        AppConfig.RANDOM_SEED = original_seed

    if verbose:
        print("\nRunning OOF CV -- Baseline arm ...")
    oof_base, err_base = _run_oof_cv(
        X_base, y, groups,
        estimator=clone(estimator),
        cal_method=cal_method, cal_cv=cal_cv,
        seed=seed, n_splits=n_splits,
    )
    if err_base and verbose:
        print(f"  [WARNING] Baseline OOF error: {err_base.splitlines()[-1]}")

    if verbose:
        print("Running OOF CV -- Medication-free arm ...")
    oof_mf, err_mf = _run_oof_cv(
        X_mf, y, groups,
        estimator=clone(estimator),
        cal_method=cal_method, cal_cv=cal_cv,
        seed=seed, n_splits=n_splits,
    )
    if err_mf and verbose:
        print(f"  [WARNING] Medication-free OOF error: {err_mf.splitlines()[-1]}")

    base_row = compute_arm_metrics(
        y, oof_base, auprc_bl, "baseline",
        n_features=len(non_empty_base),
        op_threshold=OPERATIONAL_THRESHOLD,
    )
    mf_row = compute_arm_metrics(
        y, oof_mf, auprc_bl, "medication_free",
        n_features=len(non_empty_mf),
        op_threshold=OPERATIONAL_THRESHOLD,
    )

    if verbose:
        print(
            f"\n  Baseline   AUC={base_row.get('AUC', float('nan')):.4f}  "
            f"Brier={base_row.get('Brier', float('nan')):.4f}  "
            f"Sens@8.5%={base_row.get('sensitivity', float('nan')):.3f}"
        )
        print(
            f"  Med-free   AUC={mf_row.get('AUC', float('nan')):.4f}  "
            f"Brier={mf_row.get('Brier', float('nan')):.4f}  "
            f"Sens@8.5%={mf_row.get('sensitivity', float('nan')):.3f}"
        )

    # Bootstrap comparisons (paired)
    valid_both = ~np.isnan(oof_base) & ~np.isnan(oof_mf)
    yv_both   = y[valid_both]
    p_base_v  = oof_base[valid_both]
    p_mf_v    = oof_mf[valid_both]

    boot_auc: Dict = {}
    boot_brier: Dict = {}
    boot_auprc: Dict = {}

    if n_boot > 0 and len(yv_both) > 30 and len(np.unique(yv_both)) == 2:
        if verbose:
            print(f"  Bootstrap comparisons (n_boot={n_boot}) ...")
        boot_auc   = bootstrap_auc_diff(yv_both, p_base_v, p_mf_v, n_boot=n_boot, seed=seed)
        boot_brier = _bootstrap_diff_brier(yv_both, p_base_v, p_mf_v, n_boot=n_boot, seed=seed)
        boot_auprc = _bootstrap_diff_auprc(yv_both, p_base_v, p_mf_v, n_boot=n_boot, seed=seed)
        if verbose:
            print(
                f"    dAUC(base-mf)={boot_auc.get('delta_auc', float('nan')):.4f} "
                f"[{boot_auc.get('ci_low', float('nan')):.4f}, "
                f"{boot_auc.get('ci_high', float('nan')):.4f}]  "
                f"p={boot_auc.get('p', float('nan')):.3f}"
            )
    else:
        for d in [boot_auc, boot_brier, boot_auprc]:
            d.update({"delta_auc": float("nan"), "ci_low": float("nan"),
                      "ci_high": float("nan"), "p": float("nan")})
        boot_brier["delta_brier"] = float("nan")
        boot_auprc["delta_auprc"] = float("nan")

    # DCA
    dca_df: Optional[pd.DataFrame] = None
    if len(yv_both) > 0 and len(np.unique(yv_both)) == 2:
        dca_df = decision_curve(
            yv_both,
            {"baseline": p_base_v, "medication_free": p_mf_v},
            np.array(DCA_THRESHOLDS, dtype=float),
        )

    # Threshold sensitivity analysis (Sens90 for medication-free)
    s90 = _sens_constrained_thr(yv_both, p_mf_v, min_sensitivity=0.90)
    sens90_thr = float(s90.get("threshold", float("nan")))
    if verbose and not np.isnan(sens90_thr):
        print(
            f"  Med-free Sens90 threshold: {sens90_thr:.4f} ({sens90_thr * 100:.2f}%)  "
            f"sens={s90.get('sensitivity', float('nan')):.3f}  "
            f"spec={s90.get('specificity', float('nan')):.3f}"
        )

    thr_df = build_threshold_comparison(yv_both, p_base_v, p_mf_v, sens90_thr)

    # Missingness summary
    miss_df = build_missingness_dataframe(df, was_present)

    # Feature membership
    all_feats = list(dict.fromkeys(baseline_features + medfree_features))
    base_set = set(baseline_features)
    mf_set   = set(medfree_features)
    feat_rows = []
    for f in all_feats:
        in_bl = f in base_set
        in_mf = f in mf_set
        feat_rows.append({
            "feature": f,
            "in_baseline": in_bl,
            "in_medication_free": in_mf,
            "membership": "baseline_only" if (in_bl and not in_mf) else
                          "both" if (in_bl and in_mf) else "medfree_only",
            "is_medications_field": f in MEDICATIONS_FEATURE_ALIASES,
        })
    features_df = pd.DataFrame(feat_rows)

    # Interpretation
    interpretation = interpret_result(base_row, mf_row, boot_auc, was_present)
    if verbose:
        print(f"\n  Interpretation: {interpretation.upper()}")

    # Build metrics DataFrame
    boot_row = {
        "arm": "comparison_bootstrap",
        "delta_auc_base_minus_mf":   boot_auc.get("delta_auc",   float("nan")),
        "delta_auc_ci_low":           boot_auc.get("ci_low",      float("nan")),
        "delta_auc_ci_high":          boot_auc.get("ci_high",     float("nan")),
        "delta_auc_p":                boot_auc.get("p",           float("nan")),
        "delta_brier_base_minus_mf":  boot_brier.get("delta_brier", float("nan")),
        "delta_brier_ci_low":         boot_brier.get("ci_low",    float("nan")),
        "delta_brier_ci_high":        boot_brier.get("ci_high",   float("nan")),
        "delta_brier_p":              boot_brier.get("p",         float("nan")),
        "delta_auprc_base_minus_mf":  boot_auprc.get("delta_auprc", float("nan")),
        "delta_auprc_ci_low":         boot_auprc.get("ci_low",   float("nan")),
        "delta_auprc_ci_high":        boot_auprc.get("ci_high",  float("nan")),
        "delta_auprc_p":              boot_auprc.get("p",        float("nan")),
    }
    metrics_df = pd.concat(
        [pd.DataFrame([base_row, mf_row]), pd.DataFrame([boot_row])],
        ignore_index=True,
    )

    # Markdown
    md_text = build_markdown_summary(
        baseline_row=base_row, medfree_row=mf_row,
        was_in_feature_set=was_present, n_removed=n_removed,
        miss_df=miss_df,
        boot_auc=boot_auc, boot_brier=boot_brier, boot_auprc=boot_auprc,
        dca_df=dca_df, thr_df=thr_df,
        interpretation=interpretation,
        sens90_thr=sens90_thr, op_threshold=OPERATIONAL_THRESHOLD,
    )

    # Write outputs
    metrics_df.to_csv(outdir / "preoperative_medications_ablation_metrics.csv", index=False)
    features_df.to_csv(outdir / "preoperative_medications_ablation_features.csv", index=False)
    miss_df.to_csv(outdir / "preoperative_medications_ablation_missingness.csv", index=False)
    thr_df.to_csv(outdir / "preoperative_medications_threshold_sens90.csv", index=False)
    (outdir / "preoperative_medications_ablation_summary.md").write_text(md_text, encoding="utf-8")
    if dca_df is not None:
        dca_df.to_csv(outdir / "preoperative_medications_ablation_dca.csv", index=False)

    if verbose:
        print(f"\nOutputs written to {outdir}:")
        for fname in [
            "preoperative_medications_ablation_metrics.csv",
            "preoperative_medications_ablation_features.csv",
            "preoperative_medications_ablation_missingness.csv",
            "preoperative_medications_threshold_sens90.csv",
            "preoperative_medications_ablation_dca.csv",
            "preoperative_medications_ablation_summary.md",
        ]:
            print(f"  {fname}")

    return metrics_df, features_df, miss_df, md_text


# ── CLI ────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preoperative Medications Ablation (exploratory — does not alter bundle)"
    )
    p.add_argument("--data", default="local_data/Dataset_2025.xlsx")
    p.add_argument("--outdir", default="reports")
    p.add_argument("--seed", type=int, default=_BASE_SEED)
    p.add_argument("--n-splits", type=int, default=AppConfig.CV_SPLITS, dest="n_splits")
    p.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    p.add_argument("--quiet", action="store_true")
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
