"""Ablation: Calibration × Threshold Policy

Evaluates combinations of model × calibrator × threshold_policy on honest OOF
predictions.  Does NOT alter the production model, bundle, or MODEL_VERSION.

Why this exists
---------------
The app uses a fixed 8 % clinical threshold and a single calibrator per model.
This ablation asks:

1. Which calibrator best preserves discrimination while improving calibration?
2. Which threshold policy achieves the best operating point given clinical
   constraints (sensitivity >= 90 %, or NPV >= 97 %)?

Results are exportable; any change to production calibrator or threshold
requires an explicit review decision, not an automatic swap.

Usage
-----
    python ablation_calibration_threshold_policy.py \\
        --data local_data/Dataset_2025.xlsx \\
        --seeds 20 \\
        --outdir ablation_outputs/calibration_threshold_policy
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from config import AppConfig, get_model_params
from modeling import build_preprocessor, clean_features
from risk_data import prepare_master_dataset
from stats_compare import (
    brier_skill_score as _bss_fn,
    calibration_intercept_slope,
    integrated_calibration_index,
)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

# ── Constants ─────────────────────────────────────────────────────────────────

_BASE_SEED = 42

CALIBRATOR_CONFIGS: List[Dict] = [
    {"name": "raw",          "method": None,        "cv": None},
    {"name": "sigmoid_cv3",  "method": "sigmoid",   "cv": 3},
    {"name": "sigmoid_cv5",  "method": "sigmoid",   "cv": 5},
    {"name": "isotonic_cv3", "method": "isotonic",  "cv": 3},
    {"name": "isotonic_cv5", "method": "isotonic",  "cv": 5},
]

# ── Pure helpers (testable in isolation) ──────────────────────────────────────


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict:
    """Classification metrics at a single threshold.

    threshold must be in [0, 1].  All rates returned as fractions in [0, 1].
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n  = len(y)
    n_flagged   = tp + fp
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv_val     = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    flag_rate   = n_flagged / n  if n > 0        else float("nan")
    # event_rate_above = fraction of flagged patients who are true events
    era_above   = tp / n_flagged  if n_flagged > 0 else float("nan")
    # event_rate_below = fraction of unflagged patients who are true events
    era_below   = fn / (fn + tn)  if (fn + tn) > 0 else float("nan")
    return {
        "selected_threshold": float(threshold),
        "sensitivity":        float(sensitivity),
        "specificity":        float(specificity),
        "PPV":                float(ppv),
        "NPV":                float(npv_val),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "n_flagged":          n_flagged,
        "flag_rate":          float(flag_rate),
        "event_rate_above":   float(era_above),
        "event_rate_below":   float(era_below),
        "status":             "ok",
        "error_message":      "",
    }


def find_youden_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Return the threshold maximising Youden's J = sensitivity + specificity − 1."""
    fpr, tpr, thresholds = roc_curve(
        np.asarray(y_true, dtype=int),
        np.asarray(y_prob, dtype=float),
    )
    j = tpr + (1.0 - fpr) - 1.0
    idx = int(np.argmax(j))
    return float(thresholds[idx])


def find_sensitivity_constrained_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_sensitivity: float,
) -> float:
    """Return the largest threshold that keeps sensitivity >= min_sensitivity.

    Among all thresholds on the ROC curve that achieve sensitivity >= target,
    this selects the LARGEST threshold (highest specificity, lowest flag rate).

    Returns NaN when the constraint cannot be met at any threshold.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)

    # Build evaluation grid from actual score values plus fine grid
    candidates = np.unique(np.concatenate([
        np.linspace(0.001, 0.999, 1000),
        np.unique(p),
    ]))

    best_thr  = float("nan")
    best_spec = -1.0

    for t in candidates:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if sens >= min_sensitivity:
            # prefer higher specificity; on ties prefer higher threshold
            if spec > best_spec or (
                math.isclose(spec, best_spec, abs_tol=1e-9) and t > best_thr
            ):
                best_thr, best_spec = float(t), spec

    return best_thr


def find_npv_constrained_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_npv: float,
) -> float:
    """Return the largest threshold that keeps NPV >= min_npv.

    NPV is not monotone in threshold, so all candidates are evaluated.
    Returns NaN when the constraint cannot be met.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)

    candidates = np.unique(np.concatenate([
        np.linspace(0.001, 0.999, 1000),
        np.unique(p),
    ]))

    best_thr  = float("nan")
    best_spec = -1.0

    for t in candidates:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        npv_v = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if npv_v >= min_npv:
            if spec > best_spec or (
                math.isclose(spec, best_spec, abs_tol=1e-9) and t > best_thr
            ):
                best_thr, best_spec = float(t), spec

    return best_thr


def compute_distribution_diagnostics(y_prob: np.ndarray) -> Dict:
    """Probability distribution statistics and degeneracy flags.

    All pct_* values are fractions in [0, 1] (not percentages).
    """
    p = np.asarray(y_prob, dtype=float)
    pv = p[~np.isnan(p)]
    n = len(pv)
    if n == 0:
        return {k: float("nan") for k in [
            "prob_p01", "prob_p05", "prob_p25", "prob_p50",
            "prob_p75", "prob_p95", "prob_p99",
            "pct_below_2", "pct_below_5", "pct_below_8",
            "pct_above_15", "pct_above_30",
            "n_unique_probabilities", "n_prob_exact_0", "n_prob_exact_1",
        ]}
    pctiles = np.percentile(pv, [1, 5, 25, 50, 75, 95, 99])
    return {
        "prob_p01":  float(pctiles[0]), "prob_p05": float(pctiles[1]),
        "prob_p25":  float(pctiles[2]), "prob_p50": float(pctiles[3]),
        "prob_p75":  float(pctiles[4]), "prob_p95": float(pctiles[5]),
        "prob_p99":  float(pctiles[6]),
        "pct_below_2":  float((pv < 0.02).mean()),
        "pct_below_5":  float((pv < 0.05).mean()),
        "pct_below_8":  float((pv < 0.08).mean()),
        "pct_above_15": float((pv > 0.15).mean()),
        "pct_above_30": float((pv > 0.30).mean()),
        "n_unique_probabilities": int(len(np.unique(pv))),
        "n_prob_exact_0":         int((pv == 0.0).sum()),
        "n_prob_exact_1":         int((pv == 1.0).sum()),
    }


def compute_brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier Skill Score = 1 − Brier(model) / Brier(prevalence baseline).

    BSS > 0: model outperforms predicting the prevalence for every patient.
    Delegates to the canonical implementation in stats_compare.
    """
    return float(_bss_fn(y_true, y_prob))


# ── Threshold policy dispatcher ───────────────────────────────────────────────

_NOT_AVAIL: Dict = {
    "selected_threshold": float("nan"),
    "sensitivity": float("nan"), "specificity": float("nan"),
    "PPV": float("nan"), "NPV": float("nan"),
    "TP": float("nan"), "FP": float("nan"),
    "TN": float("nan"), "FN": float("nan"),
    "n_flagged": float("nan"), "flag_rate": float("nan"),
    "event_rate_above": float("nan"), "event_rate_below": float("nan"),
    "status": "not_available",
    "error_message": "",
}


def _apply_threshold_policy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    policy: Dict,
) -> Dict:
    """Resolve and evaluate one threshold policy; returns a metrics dict."""
    policy_name = policy["name"]
    policy_type = policy["type"]

    try:
        if policy_type == "fixed":
            return compute_threshold_metrics(y_true, y_prob, policy["value"])

        if policy_type == "youden":
            thr = find_youden_threshold(y_true, y_prob)
            return compute_threshold_metrics(y_true, y_prob, thr)

        if policy_type == "sensitivity":
            thr = find_sensitivity_constrained_threshold(
                y_true, y_prob, policy["target"]
            )
            if math.isnan(thr):
                m = dict(_NOT_AVAIL)
                m["error_message"] = (
                    f"Cannot achieve sensitivity>={policy['target']:.2f}"
                )
                return m
            return compute_threshold_metrics(y_true, y_prob, thr)

        if policy_type == "npv":
            thr = find_npv_constrained_threshold(
                y_true, y_prob, policy["target"]
            )
            if math.isnan(thr):
                m = dict(_NOT_AVAIL)
                m["error_message"] = (
                    f"Cannot achieve NPV>={policy['target']:.2f}"
                )
                return m
            return compute_threshold_metrics(y_true, y_prob, thr)

        m = dict(_NOT_AVAIL)
        m["status"] = "failed"
        m["error_message"] = f"Unknown policy type: {policy_type!r}"
        return m

    except Exception as exc:
        m = dict(_NOT_AVAIL)
        m["status"] = "failed"
        m["error_message"] = str(exc)
        return m


# Fixed threshold name -> probability value map (for string-based dispatch)
_FIXED_MAP: Dict[str, float] = {
    "fixed_2": 0.02, "fixed_5": 0.05, "fixed_8": 0.08,
    "fixed_10": 0.10, "fixed_15": 0.15,
}


def apply_threshold_policy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    policy: str,
) -> Dict:
    """String-based threshold policy dispatcher (public, for tests and ad-hoc use).

    Supports: "fixed_N" (N in percent), "youden",
    "sensitivity_constrained_NN", "npv_constrained_NN".
    """
    if policy in _FIXED_MAP:
        return _apply_threshold_policy(
            y_true, y_prob, {"name": policy, "type": "fixed", "value": _FIXED_MAP[policy]}
        )
    if policy == "youden":
        return _apply_threshold_policy(y_true, y_prob, {"name": "youden", "type": "youden"})
    if policy.startswith("sensitivity_constrained_"):
        pct_str = policy.replace("sensitivity_constrained_", "")
        try:
            target = float(pct_str) / 100.0
        except ValueError:
            m = dict(_NOT_AVAIL)
            m["status"] = f"unknown:{policy}"
            return m
        return _apply_threshold_policy(
            y_true, y_prob,
            {"name": policy, "type": "sensitivity", "target": target},
        )
    if policy.startswith("npv_constrained_"):
        pct_str = policy.replace("npv_constrained_", "")
        try:
            target = float(pct_str) / 100.0
        except ValueError:
            m = dict(_NOT_AVAIL)
            m["status"] = f"unknown:{policy}"
            return m
        return _apply_threshold_policy(
            y_true, y_prob,
            {"name": policy, "type": "npv", "target": target},
        )
    m = dict(_NOT_AVAIL)
    m["status"] = f"unknown:{policy}"
    return m


# ── Candidate builder (seed-aware) ────────────────────────────────────────────


def _build_local_candidates(seed: int) -> Dict[str, object]:
    """Build fresh candidate estimators seeded from the given value.

    Temporarily sets AppConfig.RANDOM_SEED so that get_model_params picks
    up the correct random_state for this run.
    """
    original = AppConfig.RANDOM_SEED
    AppConfig.RANDOM_SEED = seed
    try:
        candidates: Dict[str, object] = {
            "LogisticRegression": LogisticRegression(
                **get_model_params("LogisticRegression")
            ),
            "RandomForest": RandomForestClassifier(
                **get_model_params("RandomForest")
            ),
        }
        if _HAS_XGB:
            candidates["XGBoost"] = XGBClassifier(**get_model_params("XGBoost"))
        if _HAS_LGBM:
            candidates["LightGBM"] = LGBMClassifier(**get_model_params("LightGBM"))
        if _HAS_CATBOOST:
            candidates["CatBoost"] = CatBoostClassifier(
                **get_model_params("CatBoost")
            )

        stack_base = [
            ("lr", LogisticRegression(**get_model_params("LogisticRegression"))),
            ("rf", RandomForestClassifier(**get_model_params("RandomForest"))),
        ]
        if _HAS_XGB:
            stack_base.append(
                ("xgb", XGBClassifier(**get_model_params("XGBoost")))
            )
        elif _HAS_LGBM:
            stack_base.append(
                ("lgbm", LGBMClassifier(**get_model_params("LightGBM")))
            )

        candidates["StackingEnsemble"] = StackingClassifier(
            estimators=stack_base,
            final_estimator=LogisticRegression(
                C=1.0, max_iter=3000, random_state=seed
            ),
            stack_method="predict_proba",
            passthrough=True,
            cv=AppConfig.CV_SPLITS,
            n_jobs=AppConfig.N_JOBS,
        )
        return candidates
    finally:
        AppConfig.RANDOM_SEED = original


# ── OOF cross-validation with configurable calibration ───────────────────────


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

            # build_preprocessor is called on training fold only
            prep      = build_preprocessor(X_tr)
            est_copy  = clone(estimator)
            fold_pipe = Pipeline([("prep", prep), ("clf", est_copy)])

            if cal_method is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fold_pipe.fit(X_tr, y_tr)
                oof[test_idx] = fold_pipe.predict_proba(X_te)[:, 1]
            else:
                fold_pos    = int(y_tr.sum())
                fold_neg    = int(len(y_tr) - fold_pos)
                eff_cv      = min(cal_cv, fold_pos, fold_neg)
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


# ── Per-combination metric computation ───────────────────────────────────────


def _compute_global_metrics(
    y: np.ndarray,
    p: np.ndarray,
    auprc_baseline: float,
) -> Dict:
    """Discrimination + calibration metrics on valid (non-NaN) OOF array."""
    valid = ~np.isnan(p)
    yv, pv = y[valid], p[valid]

    nan_row = {
        "AUC": float("nan"), "AUPRC": float("nan"),
        "AUPRC_baseline": auprc_baseline,
        "Brier": float("nan"), "Brier_skill_score": float("nan"),
        "calibration_intercept": float("nan"),
        "calibration_slope": float("nan"),
        "CIL": float("nan"), "ICI": float("nan"),
    }
    if len(yv) == 0 or int(yv.sum()) == 0 or int((1 - yv).sum()) == 0:
        return nan_row

    out = {"AUPRC_baseline": auprc_baseline}
    try:
        out["AUC"]   = float(roc_auc_score(yv, pv))
        out["AUPRC"] = float(average_precision_score(yv, pv))
        out["Brier"] = float(brier_score_loss(yv, pv))
        out["Brier_skill_score"] = compute_brier_skill_score(yv, pv)
    except Exception:
        for k in ("AUC", "AUPRC", "Brier", "Brier_skill_score"):
            out.setdefault(k, float("nan"))

    try:
        cal = calibration_intercept_slope(yv, pv)
        out["calibration_intercept"] = cal.get("Calibration intercept", float("nan"))
        out["calibration_slope"]     = cal.get("Calibration slope",     float("nan"))
    except Exception:
        out["calibration_intercept"] = out["calibration_slope"] = float("nan")

    out["CIL"] = float(np.mean(pv) - np.mean(yv))

    try:
        out["ICI"] = float(integrated_calibration_index(yv, pv))
    except Exception:
        out["ICI"] = float("nan")

    return out


# ── Policy list builder (CLI-driven) ─────────────────────────────────────────


def _build_policies(
    fixed_thresholds: List[float],
    sensitivity_targets: List[float],
    npv_targets: List[float],
) -> List[Dict]:
    """Return the ordered list of threshold policy descriptors."""
    policies: List[Dict] = []
    for t in sorted(fixed_thresholds):
        t_pct = int(round(t * 100))
        policies.append({"name": f"fixed_{t_pct}", "type": "fixed", "value": t})
    policies.append({"name": "youden", "type": "youden"})
    for s in sorted(sensitivity_targets):
        s_pct = int(round(s * 100))
        policies.append({
            "name": f"sensitivity_constrained_{s_pct}",
            "type": "sensitivity",
            "target": s,
        })
    for n in sorted(npv_targets):
        n_pct = int(round(n * 100))
        policies.append({
            "name": f"npv_constrained_{n_pct}",
            "type": "npv",
            "target": n,
        })
    return policies


# ── Main ablation loop ────────────────────────────────────────────────────────

_RESULTS_COL_ORDER = [
    "seed", "model_name", "calibrator_name", "calibration_method",
    "calibration_cv",
    "AUC", "AUPRC", "AUPRC_baseline", "Brier", "Brier_skill_score",
    "calibration_intercept", "calibration_slope", "CIL", "ICI",
    "prob_p01", "prob_p05", "prob_p25", "prob_p50",
    "prob_p75", "prob_p95", "prob_p99",
    "pct_below_2", "pct_below_5", "pct_below_8",
    "pct_above_15", "pct_above_30",
    "n_unique_probabilities", "n_prob_exact_0", "n_prob_exact_1",
    "threshold_policy", "selected_threshold",
    "sensitivity", "specificity", "PPV", "NPV",
    "TP", "FP", "TN", "FN",
    "n_flagged", "flag_rate", "event_rate_above", "event_rate_below",
    "status", "error_message",
]


def run_ablation(
    data_path: str,
    n_seeds: int,
    outdir: Path,
    models_filter: Optional[List[str]],
    fixed_thresholds: List[float],
    sensitivity_targets: List[float],
    npv_targets: List[float],
    n_boot: int = 0,
    n_splits: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """One row per seed × model × calibrator × threshold_policy."""
    outdir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading data: {data_path}")
    prepared  = prepare_master_dataset(data_path)
    df        = prepared.data
    feat_cols = prepared.feature_columns

    y_col  = "morte_30d"
    g_col  = "_patient_key"
    y      = df[y_col].values.astype(int)
    groups = df[g_col].values if g_col in df.columns else None

    n_total    = len(y)
    n_events   = int(y.sum())
    auprc_bl   = float(y.mean())

    if verbose:
        print(
            f"  n={n_total}  events={n_events}  "
            f"prevalence={auprc_bl:.1%}  features={len(feat_cols)}"
        )

    policies = _build_policies(fixed_thresholds, sensitivity_targets, npv_targets)
    seeds    = [_BASE_SEED + i for i in range(n_seeds)]

    _have_tqdm = False
    try:
        from tqdm import tqdm as _tqdm  # type: ignore
        _have_tqdm = True
    except ImportError:
        pass

    rows: List[Dict] = []
    total_combos = n_seeds * (len(models_filter) if models_filter else 6) * len(CALIBRATOR_CONFIGS)
    run_i = 0

    for seed in seeds:
        if verbose:
            print(f"\n=== Seed {seed} ===")

        candidates = _build_local_candidates(seed)
        if models_filter:
            candidates = {k: v for k, v in candidates.items() if k in models_filter}

        # Clean features once per seed (column set stable, random_state irrelevant here)
        original_seed         = AppConfig.RANDOM_SEED
        AppConfig.RANDOM_SEED = seed
        try:
            X_clean = clean_features(df[feat_cols])
            non_empty = [c for c in X_clean.columns if not X_clean[c].isna().all()]
            X_clean   = X_clean[non_empty].copy()
        finally:
            AppConfig.RANDOM_SEED = original_seed

        model_items = list(candidates.items())
        if _have_tqdm:
            model_items = _tqdm(model_items, desc=f"seed={seed}", unit="model")

        for model_name, estimator in model_items:
            for cal_cfg in CALIBRATOR_CONFIGS:
                run_i += 1
                cal_name   = cal_cfg["name"]
                cal_method = cal_cfg["method"]
                cal_cv     = cal_cfg["cv"]

                if verbose and not _have_tqdm:
                    print(
                        f"  [{run_i}] {model_name:20s}  {cal_name:14s} ...",
                        end=" ", flush=True,
                    )

                oof, err = _run_oof_cv(
                    X_clean, y, groups,
                    estimator=estimator,
                    cal_method=cal_method,
                    cal_cv=cal_cv,
                    seed=seed,
                    n_splits=n_splits,
                )

                base = {
                    "seed":               seed,
                    "model_name":         model_name,
                    "calibrator_name":    cal_name,
                    "calibration_method": str(cal_method) if cal_method else "none",
                    "calibration_cv":     int(cal_cv) if cal_cv else 0,
                }

                has_valid = (not err) and np.any(~np.isnan(oof))

                if has_valid:
                    valid_mask = ~np.isnan(oof)
                    yv, pv = y[valid_mask], oof[valid_mask]
                    metrics = _compute_global_metrics(yv, pv, auprc_bl)
                    dist    = compute_distribution_diagnostics(pv)
                else:
                    metrics = {
                        "AUC": float("nan"), "AUPRC": float("nan"),
                        "AUPRC_baseline": auprc_bl,
                        "Brier": float("nan"), "Brier_skill_score": float("nan"),
                        "calibration_intercept": float("nan"),
                        "calibration_slope":     float("nan"),
                        "CIL": float("nan"), "ICI": float("nan"),
                    }
                    dist    = compute_distribution_diagnostics(np.array([]))
                    yv = pv = np.array([], dtype=float)

                if verbose and not _have_tqdm:
                    auc_v = metrics.get("AUC", float("nan"))
                    if not math.isnan(auc_v):
                        print(
                            f"AUC={auc_v:.4f}  "
                            f"slope={metrics.get('calibration_slope', float('nan')):.3f}"
                        )
                    else:
                        print(f"FAILED: {err.splitlines()[-1] if err else 'NaN OOF'}")

                for policy in policies:
                    if not has_valid or len(yv) == 0:
                        thr_m = dict(_NOT_AVAIL)
                        thr_m["status"] = "failed"
                        thr_m["error_message"] = err or "Empty OOF"
                    else:
                        thr_m = _apply_threshold_policy(yv, pv, policy)

                    rows.append({
                        **base,
                        "threshold_policy": policy["name"],
                        **metrics,
                        **dist,
                        **thr_m,
                    })

    results = pd.DataFrame(rows)

    # Reorder columns
    present = [c for c in _RESULTS_COL_ORDER if c in results.columns]
    extra   = [c for c in results.columns if c not in _RESULTS_COL_ORDER]
    results = results[present + extra]

    r_csv = outdir / "ablation_calibration_threshold_policy_results.csv"
    results.to_csv(r_csv, index=False)
    if verbose:
        print(f"\nResults   -> {r_csv}  ({len(results)} rows)")

    summary = summarize_results(results)
    s_csv = outdir / "ablation_calibration_threshold_policy_summary.csv"
    summary.to_csv(s_csv, index=False)
    if verbose:
        print(f"Summary   -> {s_csv}")

    recs = build_recommendations(summary)
    rec_csv = outdir / "ablation_calibration_threshold_policy_recommendations.csv"
    recs.to_csv(rec_csv, index=False)
    if verbose:
        print(f"Recs      -> {rec_csv}")

    xlsx = outdir / "ablation_calibration_threshold_policy.xlsx"
    try:
        write_xlsx_report(results, summary, recs, xlsx)
        if verbose:
            print(f"XLSX      -> {xlsx}")
    except Exception as exc:
        warnings.warn(f"XLSX export failed: {exc}")

    if verbose:
        _print_console_summary(summary)

    return results


# ── Summary aggregation ───────────────────────────────────────────────────────

_MEAN_SD_COLS = [
    "AUC", "AUPRC", "Brier", "Brier_skill_score",
    "calibration_intercept", "calibration_slope", "CIL", "ICI",
    "selected_threshold", "sensitivity", "specificity",
    "PPV", "NPV", "flag_rate", "event_rate_above", "event_rate_below",
    "FN", "FP",
]


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across seeds per (model_name, calibrator_name, threshold_policy)."""
    group_cols = ["model_name", "calibrator_name", "threshold_policy"]
    rows: List[Dict] = []

    for keys, grp in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(grp)

        for col in _MEAN_SD_COLS:
            if col not in grp.columns:
                continue
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            row[f"mean_{col}"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"sd_{col}"]   = float(vals.std())  if len(vals) > 1 else float("nan")

        # pass_discrimination_guardrail_rate: AUC >= 0.60
        if "AUC" in grp.columns:
            auc_v = pd.to_numeric(grp["AUC"], errors="coerce").dropna()
            row["pass_discrimination_guardrail_rate"] = (
                float((auc_v >= 0.60).mean()) if len(auc_v) else float("nan")
            )

        # pass_calibration_rate: slope in [0.7, 1.3] AND |CIL| < 0.03
        if "calibration_slope" in grp.columns and "CIL" in grp.columns:
            slopes = pd.to_numeric(grp["calibration_slope"], errors="coerce")
            cils   = pd.to_numeric(grp["CIL"],               errors="coerce")
            valid_mask = slopes.notna() & cils.notna()
            if valid_mask.any():
                row["pass_calibration_rate"] = float(
                    ((slopes[valid_mask] >= 0.7) &
                     (slopes[valid_mask] <= 1.3) &
                     (cils[valid_mask].abs() < 0.03)).mean()
                )
            else:
                row["pass_calibration_rate"] = float("nan")

        # pass_distribution_guardrail_rate: n_unique > 10 AND pct_below_8 > 0
        if "n_unique_probabilities" in grp.columns and "pct_below_8" in grp.columns:
            uniq   = pd.to_numeric(grp["n_unique_probabilities"], errors="coerce")
            pb8    = pd.to_numeric(grp["pct_below_8"],            errors="coerce")
            vmask  = uniq.notna() & pb8.notna()
            if vmask.any():
                row["pass_distribution_guardrail_rate"] = float(
                    ((uniq[vmask] > 10) & (pb8[vmask] > 0.0)).mean()
                )
            else:
                row["pass_distribution_guardrail_rate"] = float("nan")

        # valid_run_rate: fraction where status == "ok"
        if "status" in grp.columns:
            row["valid_run_rate"] = float((grp["status"] == "ok").mean())
        else:
            row["valid_run_rate"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


# ── Recommendations ───────────────────────────────────────────────────────────


def build_recommendations(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Guardrail-based recommendation per (model, calibrator, threshold_policy)."""
    if summary_df.empty:
        return pd.DataFrame(columns=[
            "model_name", "calibrator_name", "threshold_policy",
            "recommended_for_review", "recommendation_type", "reason", "warnings",
        ])

    # Best AUC and AUPRC per model across all calibrators (for relative guardrails)
    model_best_auc   = summary_df.groupby("model_name")["mean_AUC"].max().to_dict()
    model_best_auprc = summary_df.groupby("model_name")["mean_AUPRC"].max().to_dict()

    rows: List[Dict] = []
    for _, r in summary_df.iterrows():
        model  = str(r.get("model_name", ""))
        cal    = str(r.get("calibrator_name", ""))
        policy = str(r.get("threshold_policy", ""))

        auc    = float(r.get("mean_AUC",   float("nan")))
        auprc  = float(r.get("mean_AUPRC", float("nan")))
        slope  = float(r.get("mean_calibration_slope", float("nan")))
        cil    = float(r.get("mean_CIL",   float("nan")))
        sens   = float(r.get("mean_sensitivity", float("nan")))
        vr     = float(r.get("valid_run_rate", float("nan")))

        rejects, warns = [], []

        # Validity
        if math.isnan(vr) or vr < 0.5:
            rows.append({
                "model_name": model, "calibrator_name": cal, "threshold_policy": policy,
                "recommended_for_review": False,
                "recommendation_type": "reject",
                "reason": f"valid_run_rate={vr:.0%} < 50%",
                "warnings": "",
            })
            continue

        # Absolute AUC floor
        if math.isnan(auc) or auc < 0.60:
            rows.append({
                "model_name": model, "calibrator_name": cal, "threshold_policy": policy,
                "recommended_for_review": False,
                "recommendation_type": "reject",
                "reason": f"AUC={auc:.4f} below 0.60 floor",
                "warnings": "",
            })
            continue

        # Relative AUC: not more than 0.005 below best for this model
        best_auc = model_best_auc.get(model, float("nan"))
        if not math.isnan(best_auc) and not math.isnan(auc):
            delta_auc = best_auc - auc
            if delta_auc > 0.005:
                rejects.append(f"AUC degraded {delta_auc:.4f} vs best calibrator for {model}")

        # Relative AUPRC: not more than 0.010 below best
        best_auprc = model_best_auprc.get(model, float("nan"))
        if not math.isnan(best_auprc) and not math.isnan(auprc):
            delta_auprc = best_auprc - auprc
            if delta_auprc > 0.010:
                rejects.append(
                    f"AUPRC degraded {delta_auprc:.4f} vs best calibrator for {model}"
                )

        # Calibration slope
        if not math.isnan(slope):
            if slope < 0.7 or slope > 1.3:
                rejects.append(f"slope={slope:.3f} outside acceptable [0.7, 1.3]")
            elif slope < 0.8 or slope > 1.2:
                warns.append(f"slope={slope:.3f} outside ideal [0.8, 1.2]")

        # CIL
        if not math.isnan(cil):
            if abs(cil) > 0.03:
                rejects.append(f"|CIL|={abs(cil):.4f} > 0.03 (unacceptable bias)")
            elif abs(cil) > 0.02:
                warns.append(f"|CIL|={abs(cil):.4f} > 0.02 ideal")

        # Distribution degeneracy
        pdr = float(r.get("pass_distribution_guardrail_rate", float("nan")))
        if not math.isnan(pdr) and pdr < 0.8:
            rejects.append(
                f"Degenerate distribution in {1 - pdr:.0%} of runs"
            )

        # Policy-specific sensitivity constraints
        if "sensitivity_constrained_90" in policy and not math.isnan(sens):
            if sens < 0.90:
                rejects.append(f"mean_sensitivity={sens:.3f} < 0.90 (constraint not met)")
        if "sensitivity_constrained_95" in policy and not math.isnan(sens):
            if sens < 0.95:
                rejects.append(f"mean_sensitivity={sens:.3f} < 0.95 (constraint not met)")
        if "sensitivity_constrained_85" in policy and not math.isnan(sens):
            if sens < 0.85:
                rejects.append(f"mean_sensitivity={sens:.3f} < 0.85 (constraint not met)")

        # Classify
        if policy == "fixed_8":
            rec_type   = "reference_only"
            reason_str = (
                "fixed_8 is the production reference threshold. "
                "Preserved as reference_only regardless of guardrail outcome."
            )
            if rejects:
                reason_str += " - Issues:" + "; ".join(rejects)
        elif policy == "youden":
            rec_type   = "reference_only"
            reason_str = (
                "Youden threshold is data-driven / exploratory. "
                "Labelled reference_only; not a primary clinical recommendation."
            )
            if rejects:
                reason_str += " - Issues:" + "; ".join(rejects)
        elif rejects:
            rec_type   = "reject"
            reason_str = "; ".join(rejects)
        else:
            if policy == "sensitivity_constrained_90":
                rec_type   = "primary_candidate"
                reason_str = (
                    "Passes all guardrails. "
                    "sensitivity_constrained_90 is the primary target policy."
                )
            else:
                rec_type   = "supplementary_candidate"
                reason_str = "Passes all guardrails."

        rows.append({
            "model_name":           model,
            "calibrator_name":      cal,
            "threshold_policy":     policy,
            "recommended_for_review": rec_type in (
                "primary_candidate", "supplementary_candidate"
            ),
            "recommendation_type":  rec_type,
            "reason":               reason_str,
            "warnings":             "; ".join(warns) if warns else "",
        })

    return pd.DataFrame(rows)


# ── XLSX export ───────────────────────────────────────────────────────────────

_README_DATA = [
    ("Key", "Value"),
    ("Sheet",       "Description"),
    ("README",             "Objective, methodology notes, and sheet index."),
    ("RESULTS_LONG",       "One row per seed × model × calibrator × threshold_policy."),
    ("SUMMARY",            "Statistics aggregated across seeds."),
    ("RECOMMENDATIONS",    "Guardrail-based recommendation per combination."),
    ("MODEL_CALIBRATOR_RANKING", "Models ranked by mean AUC at fixed_8 policy."),
    ("THRESHOLD_POLICY_RANKING", "Threshold policies ranked by average operating metrics."),
    ("SENS90_FOCUS",       "sensitivity_constrained_90 policy rows only."),
    ("SENS95_FOCUS",       "sensitivity_constrained_95 policy rows only."),
    ("FIXED8_REFERENCE",   "fixed_8 policy rows only (production threshold reference)."),
    ("YOUDEN_REFERENCE",   "youden policy rows only (data-driven, exploratory)."),
    ("DISTRIBUTION_DIAGNOSTICS", "Probability distribution statistics per seed × model × calibrator."),
    ("", ""),
    ("Objective",
     "Evaluate model × calibrator × threshold_policy on honest OOF predictions."),
    ("Production model",
     "This ablation does NOT alter the production model, bundle, or MODEL_VERSION."),
    ("OOF evaluation",
     "All metrics are honest OOF - calibrator fitted inside outer folds, no leakage."),
    ("8% reference",
     "The 8% threshold is the production reference. Not replaced automatically."),
    ("sensitivity_constrained_90",
     "Primary candidate policy: largest threshold keeping sensitivity >= 90%."),
    ("sensitivity_constrained_95",
     "Conservative alternative: largest threshold keeping sensitivity >= 95%."),
    ("NPV-constrained",
     "npv_constrained_95/97: largest threshold keeping NPV >= target."),
    ("Youden",
     "Data-driven / exploratory. Labelled reference_only; not a primary recommendation."),
    ("Threshold change rule",
     "Any change to the primary operational threshold requires explicit approval."),
]


def write_xlsx_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    recs: pd.DataFrame,
    path: Path,
) -> None:
    """Write multi-sheet XLSX workbook."""
    readme_df = pd.DataFrame(_README_DATA[1:], columns=_README_DATA[0])

    def _sub(df: pd.DataFrame, policy: str) -> pd.DataFrame:
        if "threshold_policy" not in df.columns:
            return pd.DataFrame()
        return df[df["threshold_policy"] == policy].copy()

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        readme_df.to_excel(w, sheet_name="README", index=False)
        results.to_excel(w, sheet_name="RESULTS_LONG", index=False)
        summary.to_excel(w, sheet_name="SUMMARY", index=False)
        recs.to_excel(w, sheet_name="RECOMMENDATIONS", index=False)

        # MODEL_CALIBRATOR_RANKING (at fixed_8 for fair comparison)
        rank_cols = [
            "model_name", "calibrator_name",
            "n_runs", "mean_AUC", "sd_AUC", "mean_AUPRC", "sd_AUPRC",
            "mean_Brier", "mean_calibration_slope", "mean_CIL", "mean_ICI",
            "pass_calibration_rate", "pass_discrimination_guardrail_rate",
            "pass_distribution_guardrail_rate", "valid_run_rate",
        ]
        f8 = _sub(summary, "fixed_8")
        if not f8.empty:
            present_r = [c for c in rank_cols if c in f8.columns]
            f8[present_r].sort_values("mean_AUC", ascending=False).to_excel(
                w, sheet_name="MODEL_CALIBRATOR_RANKING", index=False
            )

        # THRESHOLD_POLICY_RANKING
        thr_cols = [
            "threshold_policy",
            "mean_AUC", "mean_sensitivity", "mean_specificity",
            "mean_flag_rate", "mean_NPV",
        ]
        if not summary.empty and "threshold_policy" in summary.columns:
            thr_present = [c for c in thr_cols if c in summary.columns]
            thr_rank = (
                summary.groupby("threshold_policy")[
                    [c for c in thr_present if c != "threshold_policy"]
                ].mean()
                .reset_index()
                .sort_values("mean_AUC", ascending=False)
            )
            thr_rank.to_excel(w, sheet_name="THRESHOLD_POLICY_RANKING", index=False)

        # Policy-focused tabs
        for sheet, policy in [
            ("SENS90_FOCUS",    "sensitivity_constrained_90"),
            ("SENS95_FOCUS",    "sensitivity_constrained_95"),
            ("FIXED8_REFERENCE", "fixed_8"),
            ("YOUDEN_REFERENCE", "youden"),
        ]:
            sub = _sub(summary, policy)
            if not sub.empty:
                sub.sort_values("mean_AUC", ascending=False).to_excel(
                    w, sheet_name=sheet, index=False
                )

        # DISTRIBUTION_DIAGNOSTICS (one row per seed × model × calibrator)
        dist_cols = [
            "seed", "model_name", "calibrator_name",
            "prob_p01", "prob_p05", "prob_p25", "prob_p50",
            "prob_p75", "prob_p95", "prob_p99",
            "pct_below_2", "pct_below_5", "pct_below_8",
            "pct_above_15", "pct_above_30",
            "n_unique_probabilities", "n_prob_exact_0", "n_prob_exact_1",
        ]
        if not results.empty and "threshold_policy" in results.columns:
            ref_pol = (
                "fixed_8"
                if "fixed_8" in results["threshold_policy"].values
                else results["threshold_policy"].iloc[0]
            )
            dist_df = results[results["threshold_policy"] == ref_pol][
                [c for c in dist_cols if c in results.columns]
            ]
            dist_df.to_excel(w, sheet_name="DISTRIBUTION_DIAGNOSTICS", index=False)


# ── Console summary ───────────────────────────────────────────────────────────


def _print_console_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    print("\n" + "=" * 90)
    print("SUMMARY - top 10 combinations by mean AUC")
    print("=" * 90)
    cols = [
        "model_name", "calibrator_name", "threshold_policy",
        "mean_AUC", "sd_AUC", "mean_sensitivity", "mean_specificity",
        "mean_calibration_slope", "mean_CIL",
    ]
    avail = [c for c in cols if c in summary.columns]
    top = summary.sort_values("mean_AUC", ascending=False).head(10)[avail]
    print(top.to_string(index=False, float_format="{:.4f}".format))

    if "threshold_policy" in summary.columns:
        s90 = summary[summary["threshold_policy"] == "sensitivity_constrained_90"]
        if not s90.empty:
            print("\nsensitivity_constrained_90:")
            cols90 = [
                "model_name", "calibrator_name",
                "mean_sensitivity", "mean_specificity", "mean_flag_rate",
                "mean_AUC",
            ]
            a90 = [c for c in cols90 if c in s90.columns]
            print(
                s90.sort_values("mean_AUC", ascending=False)[a90]
                .to_string(index=False, float_format="{:.4f}".format)
            )

    print("\nNotes:")
    print("  fixed_8 = production reference only")
    print("  youden  = data-driven / exploratory")
    print("  Any threshold change requires explicit methodological approval.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Calibration × Threshold Policy ablation (OOF, no production changes). "
            "Does NOT modify app, bundle, or MODEL_VERSION."
        )
    )
    p.add_argument(
        "--data",
        default="local_data/Dataset_2025.xlsx",
        help="Path to dataset XLSX/CSV file",
    )
    p.add_argument(
        "--seeds", type=int, default=20,
        help="Number of seeds (default: 20, uses seeds 42..42+N-1)",
    )
    p.add_argument(
        "--outdir",
        default="ablation_outputs/calibration_threshold_policy",
        help="Output directory",
    )
    p.add_argument(
        "--models", default=None,
        help="Comma-separated model names (default: all available)",
    )
    p.add_argument(
        "--thresholds", default="2,5,8,10,15",
        help="Fixed threshold percentages (comma-separated, default: 2,5,8,10,15)",
    )
    p.add_argument(
        "--sensitivity-targets", default="0.85,0.90,0.95",
        help="Sensitivity lower bounds (comma-separated, default: 0.85,0.90,0.95)",
    )
    p.add_argument(
        "--npv-targets", default="0.95,0.97",
        help="NPV lower bounds (comma-separated, default: 0.95,0.97)",
    )
    p.add_argument(
        "--n-boot", type=int, default=0,
        help="Bootstrap resamples for CI (0 = point estimates only, default: 0)",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = p.parse_args()

    if not Path(args.data).exists():
        print(f"Error: dataset not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    fixed_thresholds = [
        float(t.strip()) / 100.0
        for t in args.thresholds.split(",")
        if t.strip()
    ]
    sensitivity_targets = [
        float(s.strip())
        for s in args.sensitivity_targets.split(",")
        if s.strip()
    ]
    npv_targets = [
        float(n.strip())
        for n in args.npv_targets.split(",")
        if n.strip()
    ]
    models_filter = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else None
    )

    verbose = not args.quiet
    if verbose:
        print("=== Calibration x Threshold Policy Ablation ===")
        print(f"  data              : {args.data}")
        print(f"  seeds             : {args.seeds} (base={_BASE_SEED})")
        print(f"  models            : {models_filter or 'all'}")
        print(f"  calibrators       : {[c['name'] for c in CALIBRATOR_CONFIGS]}")
        print(f"  fixed thresholds  : {[f'{t*100:.0f}%' for t in fixed_thresholds]}")
        print(f"  sensitivity targets: {sensitivity_targets}")
        print(f"  NPV targets       : {npv_targets}")
        print(f"  n_boot            : {args.n_boot}")
        print(f"  outdir            : {args.outdir}")

    run_ablation(
        data_path=args.data,
        n_seeds=args.seeds,
        outdir=Path(args.outdir),
        models_filter=models_filter,
        fixed_thresholds=fixed_thresholds,
        sensitivity_targets=sensitivity_targets,
        npv_targets=npv_targets,
        n_boot=args.n_boot,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
