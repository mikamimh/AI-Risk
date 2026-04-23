"""Statistical evaluation and comparison of risk prediction models.

This module provides functions for:
- Computing performance metrics (AUC, calibration, sensitivity/specificity)
- Comparing models statistically (DeLong test, bootstrap CI, NRI, IDI)
- Calibration assessment (Brier score, Hosmer-Lemeshow test)
- Clinical utility analysis (decision curve analysis)
- ROC and calibration curves for visualization

Example:
    >>> from stats_compare import evaluate_scores, bootstrap_auc_diff
    >>> results = evaluate_scores(df, "outcome", ["model1_pred", "model2_pred"])
    >>> delta_auc = bootstrap_auc_diff(y, model1_pred, model2_pred)
"""

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def class_risk(prob: float) -> str:
    if prob < 0.05:
        return "Low"
    if prob <= 0.15:
        return "Intermediate"
    return "High"


def basic_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y)
    p = np.asarray(p)
    return {
        "n": int(len(y)),
        "AUC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
    }


def classification_metrics_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    pred = (p >= threshold).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    sens = float(tp / (tp + fn)) if (tp + fn) else np.nan
    spec = float(tn / (tn + fp)) if (tn + fp) else np.nan
    ppv = float(tp / (tp + fp)) if (tp + fp) else np.nan
    npv = float(tn / (tn + fn)) if (tn + fn) else np.nan

    return {
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def evaluate_scores_with_threshold(
    df: pd.DataFrame,
    y_col: str,
    score_cols: List[str],
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for c in score_cols:
        sub = df[[y_col, c]].dropna()
        if len(sub) < 30 or sub[y_col].nunique() < 2:
            continue
        y = sub[y_col].values
        p = sub[c].values
        m = basic_metrics(y, p)
        cls = classification_metrics_at_threshold(y, p, threshold)
        ici = integrated_calibration_index(y, p)
        rows.append({
            "Score": c,
            "n": int(len(sub)),
            **m,
            "AUPRC_baseline": float(np.mean(y)),
            "ICI": ici,
            **cls,
        })
    cols = [
        "Score", "n", "AUC", "AUPRC", "AUPRC_baseline",
        "Brier", "ICI",
        "Sensitivity", "Specificity", "PPV", "NPV",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values("AUC", ascending=False)


def evaluate_scores(df: pd.DataFrame, y_col: str, score_cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in score_cols:
        sub = df[[y_col, c]].dropna()
        if len(sub) < 30 or sub[y_col].nunique() < 2:
            continue
        y = sub[y_col].values
        m = basic_metrics(y, sub[c].values)
        m["Score"] = c
        m["AUPRC_baseline"] = float(np.mean(y))
        rows.append(m)
    if not rows:
        return pd.DataFrame(columns=["Score", "n", "AUC", "AUPRC", "AUPRC_baseline", "Brier"])
    return pd.DataFrame(rows)[
        ["Score", "n", "AUC", "AUPRC", "AUPRC_baseline", "Brier"]
    ].sort_values("AUC", ascending=False)


def bootstrap_metrics_ci(
    y: np.ndarray,
    p: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)

    # Point estimates from original data (not bootstrap mean)
    point_auc = float(roc_auc_score(y, p))
    point_auprc = float(average_precision_score(y, p))
    point_brier = float(brier_score_loss(y, p))

    aucs = []
    auprcs = []
    briers = []
    n = len(y)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        ps = p[idx]
        if len(np.unique(ys)) < 2:
            continue
        aucs.append(roc_auc_score(ys, ps))
        auprcs.append(average_precision_score(ys, ps))
        briers.append(brier_score_loss(ys, ps))

    if not aucs:
        return {
            "AUC": point_auc,
            "AUC_IC95_inf": np.nan,
            "AUC_IC95_sup": np.nan,
            "AUPRC": point_auprc,
            "AUPRC_IC95_inf": np.nan,
            "AUPRC_IC95_sup": np.nan,
            "Brier": point_brier,
            "Brier_IC95_inf": np.nan,
            "Brier_IC95_sup": np.nan,
        }

    auc_arr = np.array(aucs)
    auprc_arr = np.array(auprcs)
    brier_arr = np.array(briers)

    return {
        "AUC": point_auc,
        "AUC_IC95_inf": float(np.percentile(auc_arr, 2.5)),
        "AUC_IC95_sup": float(np.percentile(auc_arr, 97.5)),
        "AUPRC": point_auprc,
        "AUPRC_IC95_inf": float(np.percentile(auprc_arr, 2.5)),
        "AUPRC_IC95_sup": float(np.percentile(auprc_arr, 97.5)),
        "Brier": point_brier,
        "Brier_IC95_inf": float(np.percentile(brier_arr, 2.5)),
        "Brier_IC95_sup": float(np.percentile(brier_arr, 97.5)),
    }


def evaluate_scores_with_ci(
    df: pd.DataFrame,
    y_col: str,
    score_cols: List[str],
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for c in score_cols:
        sub = df[[y_col, c]].dropna()
        if len(sub) < 30 or sub[y_col].nunique() < 2:
            continue
        y = sub[y_col].values
        m = bootstrap_metrics_ci(y, sub[c].values, n_boot=n_boot, seed=seed)
        m["Score"] = c
        m["n"] = int(len(sub))
        m["AUPRC_baseline"] = float(np.mean(y))
        rows.append(m)

    cols = [
        "Score",
        "n",
        "AUC",
        "AUC_IC95_inf",
        "AUC_IC95_sup",
        "AUPRC",
        "AUPRC_baseline",
        "AUPRC_IC95_inf",
        "AUPRC_IC95_sup",
        "Brier",
        "Brier_IC95_inf",
        "Brier_IC95_sup",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values("AUC", ascending=False)


def bootstrap_auc_diff(
    y: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    deltas = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        d = roc_auc_score(ys, p1[idx]) - roc_auc_score(ys, p2[idx])
        deltas.append(d)

    if not deltas:
        return {"delta_auc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}

    arr = np.array(deltas)
    delta = float(roc_auc_score(y, p1) - roc_auc_score(y, p2))
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    p = float(2 * min((arr <= 0).mean(), (arr >= 0).mean()))
    return {"delta_auc": delta, "ci_low": ci_low, "ci_high": ci_high, "p": p}


def roc_data(y: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fpr, tpr, _ = roc_curve(y, p)
    return fpr, tpr


def calibration_data(y: np.ndarray, p: np.ndarray, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    unique_p = np.unique(np.round(p, 8))
    n_bins = min(bins, max(len(unique_p), 1), len(p))
    if n_bins < 2 or len(np.unique(y)) < 2:
        return np.array([float(np.mean(p))]), np.array([float(np.mean(y))])
    try:
        prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    except Exception:
        prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    return prob_pred, prob_true


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return {"Calibration intercept": np.nan, "Calibration slope": np.nan}
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Setting penalty", category=UserWarning)
            model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000)
            model.fit(logit_p, y)
        return {
            "Calibration intercept": float(model.intercept_[0]),
            "Calibration slope": float(model.coef_[0][0]),
        }
    except Exception:
        return {"Calibration intercept": np.nan, "Calibration slope": np.nan}


def hosmer_lemeshow_test(y: np.ndarray, p: np.ndarray, n_groups: int = 10) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    df = pd.DataFrame({"y": y, "p": p}).sort_values("p").reset_index(drop=True)
    try:
        df["group"] = pd.qcut(df.index + 1, q=min(n_groups, len(df)), labels=False, duplicates="drop")
    except Exception:
        df["group"] = pd.cut(df.index + 1, bins=min(n_groups, len(df)), labels=False, duplicates="drop")
    grouped = df.groupby("group", observed=True)
    if grouped.ngroups < 3:
        return {"HL chi-square": np.nan, "HL dof": np.nan, "HL p-value": np.nan}
    obs = grouped["y"].sum()
    exp = grouped["p"].sum()
    n = grouped.size()
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = (((obs - exp) ** 2) / exp.clip(lower=1e-6) + (((n - obs) - (n - exp)) ** 2) / (n - exp).clip(lower=1e-6)).sum()
    dof = max(int(grouped.ngroups) - 2, 1)
    pval = float(chi2.sf(float(hl), dof))
    return {"HL chi-square": float(hl), "HL dof": dof, "HL p-value": pval}


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1
        ranks[order[i:j]] = mid
        i = j
    return ranks


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


# Minimum class counts required for a stable DeLong covariance estimate.
# np.cov over v01 / v10 needs at least two samples per class; with a single
# positive (or single negative) the covariance collapses to a 0/0 or 1/inf
# shape and raises the three warnings seen on n=24 / events=1 cohorts
# ("Degrees of freedom <= 0", "divide by zero", "invalid value in multiply").
# Two is the hard mathematical floor, not a clinical-sufficiency threshold.
_DELONG_MIN_PER_CLASS = 2
_DELONG_SKIP_REASON_SPARSE = (
    "DeLong not computed: fewer than 2 events or fewer than 2 non-events "
    "in validation cohort."
)
_DELONG_SKIP_REASON_DEGENERATE_VAR = (
    "DeLong not computed: non-positive variance of the AUC difference."
)


def delong_roc_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Dict[str, float]:
    """DeLong test for two correlated ROC AUCs on the same cohort.

    Returns a dict with ``AUC_1``, ``AUC_2``, ``delta_auc``, ``z``, ``p``,
    plus a ``reason`` field (``None`` on success, a short human-readable
    string when the test is skipped for a methodological reason).  Sparse
    validation cohorts — fewer than :data:`_DELONG_MIN_PER_CLASS` positives
    or negatives — are skipped up front to avoid numpy RuntimeWarnings from
    the covariance step; callers receive ``p = NaN`` and the reason string
    and are expected to render an em dash / footnote instead of a p-value.
    """
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    label_1_count = int(y.sum())
    label_0_count = int(y.size - label_1_count)

    # Hard safety rule: the DeLong covariance is only defined when each
    # class contributes at least two observations.  Guard before touching
    # numpy so no RuntimeWarning is emitted for sparse cohorts.
    if label_1_count < _DELONG_MIN_PER_CLASS or label_0_count < _DELONG_MIN_PER_CLASS:
        return {
            "AUC_1": np.nan,
            "AUC_2": np.nan,
            "delta_auc": np.nan,
            "z": np.nan,
            "p": np.nan,
            "reason": _DELONG_SKIP_REASON_SPARSE,
        }

    order = np.argsort(-y)
    preds = np.vstack([p1[order], p2[order]])

    aucs, cov = _fast_delong(preds, label_1_count)
    diff = float(aucs[0] - aucs[1])
    if np.ndim(cov) == 0:
        var = float(cov)
    else:
        var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        return {
            "AUC_1": float(aucs[0]),
            "AUC_2": float(aucs[1]),
            "delta_auc": diff,
            "z": np.nan,
            "p": np.nan,
            "reason": _DELONG_SKIP_REASON_DEGENERATE_VAR,
        }
    se = float(np.sqrt(max(var, 0)))
    z = diff / se
    pval = float(2 * norm.sf(abs(z)))
    return {
        "AUC_1": float(aucs[0]),
        "AUC_2": float(aucs[1]),
        "delta_auc": diff,
        "delta_auc_se": se,
        "z": float(z),
        "p": pval,
        "reason": None,
    }


def net_benefit(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    if threshold <= 0 or threshold >= 1:
        return np.nan
    pred = p >= threshold
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    n = len(y)
    w = threshold / (1 - threshold)
    return float(tp / n - fp / n * w)


def decision_curve(y: np.ndarray, score_map: Dict[str, np.ndarray], thresholds: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    rows = []
    prevalence = float(y.mean())
    for t in thresholds:
        rows.append({"Threshold": float(t), "Strategy": "Treat all", "Net Benefit": prevalence - (1 - prevalence) * (t / (1 - t))})
        rows.append({"Threshold": float(t), "Strategy": "Treat none", "Net Benefit": 0.0})
        for name, preds in score_map.items():
            rows.append({"Threshold": float(t), "Strategy": name, "Net Benefit": net_benefit(y, preds, float(t))})
    return pd.DataFrame(rows)


def categorize_risk(prob: float, cutoffs: Tuple[float, float] = (0.05, 0.15)) -> str:
    low_cut, high_cut = cutoffs
    if prob < low_cut:
        return "low"
    if prob <= high_cut:
        return "intermediate"
    return "high"


def compute_nri(y: np.ndarray, p_old: np.ndarray, p_new: np.ndarray, cutoffs: Tuple[float, float] = (0.05, 0.15)) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p_old = np.asarray(p_old)
    p_new = np.asarray(p_new)
    order = {"low": 0, "intermediate": 1, "high": 2}

    old_num = np.array([order[categorize_risk(float(x), cutoffs)] for x in p_old])
    new_num = np.array([order[categorize_risk(float(x), cutoffs)] for x in p_new])

    event_mask = y == 1
    nonevent_mask = y == 0

    event_up = float(np.mean(new_num[event_mask] > old_num[event_mask])) if event_mask.sum() else np.nan
    event_down = float(np.mean(new_num[event_mask] < old_num[event_mask])) if event_mask.sum() else np.nan
    nonevent_down = float(np.mean(new_num[nonevent_mask] < old_num[nonevent_mask])) if nonevent_mask.sum() else np.nan
    nonevent_up = float(np.mean(new_num[nonevent_mask] > old_num[nonevent_mask])) if nonevent_mask.sum() else np.nan

    nri_events = event_up - event_down
    nri_nonevents = nonevent_down - nonevent_up
    return {
        "NRI events": nri_events,
        "NRI non-events": nri_nonevents,
        "NRI total": nri_events + nri_nonevents,
    }


def compute_idi(y: np.ndarray, p_old: np.ndarray, p_new: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p_old = np.asarray(p_old)
    p_new = np.asarray(p_new)

    event_mask = y == 1
    nonevent_mask = y == 0
    old_disc = float(p_old[event_mask].mean() - p_old[nonevent_mask].mean())
    new_disc = float(p_new[event_mask].mean() - p_new[nonevent_mask].mean())
    return {
        "Old discrimination slope": old_disc,
        "New discrimination slope": new_disc,
        "IDI": new_disc - old_disc,
    }


def compute_nri_with_ci(
    y: np.ndarray,
    p_old: np.ndarray,
    p_new: np.ndarray,
    cutoffs: Tuple[float, float] = (0.05, 0.15),
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """NRI (categorical) with 95% bootstrap CI and two-sided p-value.

    Args:
        y: Binary outcome array.
        p_old: Predicted probabilities from the reference model.
        p_new: Predicted probabilities from the new model.
        cutoffs: Low/high risk thresholds (default 5%/15%).
        n_boot: Bootstrap iterations.
        seed: RNG seed.

    Returns:
        Dict with NRI total/events/non-events plus CI and p-value keys.
    """
    y = np.asarray(y).astype(int)
    p_old = np.asarray(p_old)
    p_new = np.asarray(p_new)
    point = compute_nri(y, p_old, p_new, cutoffs)
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_total, boot_events, boot_nonevents = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        b = compute_nri(y[idx], p_old[idx], p_new[idx], cutoffs)
        boot_total.append(b["NRI total"])
        boot_events.append(b["NRI events"])
        boot_nonevents.append(b["NRI non-events"])
    if not boot_total:
        point.update({
            "NRI_CI_low": np.nan, "NRI_CI_high": np.nan, "NRI_p": np.nan,
            "NRI_events_CI_low": np.nan, "NRI_events_CI_high": np.nan,
            "NRI_nonevents_CI_low": np.nan, "NRI_nonevents_CI_high": np.nan,
        })
        return point
    arr = np.array(boot_total)
    point["NRI_CI_low"] = float(np.percentile(arr, 2.5))
    point["NRI_CI_high"] = float(np.percentile(arr, 97.5))
    point["NRI_p"] = float(min(1.0, 2 * min(
        np.clip((arr <= 0).mean(), 1e-10, 1),
        np.clip((arr >= 0).mean(), 1e-10, 1),
    )))
    point["NRI_events_CI_low"] = float(np.percentile(boot_events, 2.5))
    point["NRI_events_CI_high"] = float(np.percentile(boot_events, 97.5))
    point["NRI_nonevents_CI_low"] = float(np.percentile(boot_nonevents, 2.5))
    point["NRI_nonevents_CI_high"] = float(np.percentile(boot_nonevents, 97.5))
    return point


def compute_idi_with_ci(
    y: np.ndarray,
    p_old: np.ndarray,
    p_new: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """IDI with 95% bootstrap CI and two-sided p-value.

    Args:
        y: Binary outcome array.
        p_old: Predicted probabilities from the reference model.
        p_new: Predicted probabilities from the new model.
        n_boot: Bootstrap iterations.
        seed: RNG seed.

    Returns:
        Dict with IDI plus CI and p-value keys.
    """
    y = np.asarray(y).astype(int)
    p_old = np.asarray(p_old)
    p_new = np.asarray(p_new)
    point = compute_idi(y, p_old, p_new)
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_idi = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        b = compute_idi(y[idx], p_old[idx], p_new[idx])
        boot_idi.append(b["IDI"])
    if not boot_idi:
        point.update({"IDI_CI_low": np.nan, "IDI_CI_high": np.nan, "IDI_p": np.nan})
        return point
    arr = np.array(boot_idi)
    point["IDI_CI_low"] = float(np.percentile(arr, 2.5))
    point["IDI_CI_high"] = float(np.percentile(arr, 97.5))
    point["IDI_p"] = float(min(1.0, 2 * min(
        np.clip((arr <= 0).mean(), 1e-10, 1),
        np.clip((arr >= 0).mean(), 1e-10, 1),
    )))
    return point


def calibration_intercept_slope_with_ci(
    y: np.ndarray,
    p: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Calibration intercept and slope with 95% bootstrap CI.

    Args:
        y: Binary outcome array.
        p: Predicted probabilities.
        n_boot: Bootstrap iterations.
        seed: RNG seed.

    Returns:
        Dict with Calibration intercept/slope + CI keys.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    point = calibration_intercept_slope(y, p)
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_int, boot_slope = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        c = calibration_intercept_slope(y[idx], p[idx])
        if not np.isnan(c["Calibration intercept"]):
            boot_int.append(c["Calibration intercept"])
            boot_slope.append(c["Calibration slope"])
    result = dict(point)
    if boot_int:
        result["Calibration_intercept_CI_low"] = float(np.percentile(boot_int, 2.5))
        result["Calibration_intercept_CI_high"] = float(np.percentile(boot_int, 97.5))
        result["Calibration_slope_CI_low"] = float(np.percentile(boot_slope, 2.5))
        result["Calibration_slope_CI_high"] = float(np.percentile(boot_slope, 97.5))
    else:
        result["Calibration_intercept_CI_low"] = np.nan
        result["Calibration_intercept_CI_high"] = np.nan
        result["Calibration_slope_CI_low"] = np.nan
        result["Calibration_slope_CI_high"] = np.nan
    return result


def calibration_in_the_large(
    y: np.ndarray,
    p: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Mean predicted minus mean observed, with 95% bootstrap CI.

    Positive CIL = model overestimates risk on average.

    Args:
        y: Binary outcome array.
        p: Predicted probabilities.
        n_boot: Bootstrap iterations.
        seed: RNG seed.

    Returns:
        Dict with CIL, CIL_CI_low, CIL_CI_high.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    cil = float(np.mean(p) - np.mean(y))
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(float(np.mean(p[idx]) - np.mean(y[idx])))
    arr = np.array(boots)
    return {
        "CIL": cil,
        "CIL_CI_low": float(np.percentile(arr, 2.5)),
        "CIL_CI_high": float(np.percentile(arr, 97.5)),
    }


def integrated_calibration_index(y: np.ndarray, p: np.ndarray) -> float:
    """Integrated Calibration Index via isotonic regression.

    ICI = mean |smoothed(p) - p|. Lower is better; 0 = perfect calibration.

    Args:
        y: Binary outcome array.
        p: Predicted probabilities.

    Returns:
        ICI scalar (float).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    ir = IsotonicRegression(out_of_bounds="clip")
    p_smooth = ir.fit_transform(p, y)
    return float(np.mean(np.abs(p_smooth - p)))


# ---------------------------------------------------------------------------
# Temporal validation composite functions
# ---------------------------------------------------------------------------

def evaluate_scores_temporal(
    df: pd.DataFrame,
    y_col: str,
    score_cols: List[str],
    threshold: float,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Comprehensive score evaluation for temporal validation.

    Combines bootstrap CI (AUC, AUPRC, Brier), calibration
    (intercept/slope, Hosmer-Lemeshow), and classification metrics
    at a locked threshold into a single table per score.
    """
    rows = []
    for c in score_cols:
        sub = df[[y_col, c]].dropna()
        if len(sub) < 10 or sub[y_col].nunique() < 2:
            continue
        y = sub[y_col].values
        p = sub[c].values

        m = bootstrap_metrics_ci(y, p, n_boot=n_boot, seed=seed)
        cal = calibration_intercept_slope_with_ci(y, p, n_boot=n_boot, seed=seed)
        cil = calibration_in_the_large(y, p, n_boot=n_boot, seed=seed)
        ici = integrated_calibration_index(y, p)
        hl = hosmer_lemeshow_test(y, p)
        cls = classification_metrics_at_threshold(y, p, threshold)

        rows.append({
            "Score": c,
            "n": int(len(sub)),
            "AUC": m["AUC"],
            "AUC_IC95_inf": m["AUC_IC95_inf"],
            "AUC_IC95_sup": m["AUC_IC95_sup"],
            "AUPRC": m["AUPRC"],
            "AUPRC_baseline": float(np.mean(y)),
            "AUPRC_IC95_inf": m["AUPRC_IC95_inf"],
            "AUPRC_IC95_sup": m["AUPRC_IC95_sup"],
            "Brier": m["Brier"],
            "Calibration_Intercept": cal["Calibration intercept"],
            "Calibration_Intercept_CI_low": cal.get("Calibration_intercept_CI_low", np.nan),
            "Calibration_Intercept_CI_high": cal.get("Calibration_intercept_CI_high", np.nan),
            "Calibration_Slope": cal["Calibration slope"],
            "Calibration_Slope_CI_low": cal.get("Calibration_slope_CI_low", np.nan),
            "Calibration_Slope_CI_high": cal.get("Calibration_slope_CI_high", np.nan),
            "CIL": cil["CIL"],
            "CIL_CI_low": cil["CIL_CI_low"],
            "CIL_CI_high": cil["CIL_CI_high"],
            "ICI": ici,
            "HL_p": hl["HL p-value"],
            "Sensitivity": cls["Sensitivity"],
            "Specificity": cls["Specificity"],
            "PPV": cls["PPV"],
            "NPV": cls["NPV"],
        })

    cols = [
        "Score", "n", "AUC", "AUC_IC95_inf", "AUC_IC95_sup",
        "AUPRC", "AUPRC_baseline", "AUPRC_IC95_inf", "AUPRC_IC95_sup", "Brier",
        "Calibration_Intercept", "Calibration_Intercept_CI_low", "Calibration_Intercept_CI_high",
        "Calibration_Slope", "Calibration_Slope_CI_low", "Calibration_Slope_CI_high",
        "CIL", "CIL_CI_low", "CIL_CI_high",
        "ICI", "HL_p",
        "Sensitivity", "Specificity", "PPV", "NPV",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values("AUC", ascending=False)


def pairwise_score_comparison(
    df: pd.DataFrame,
    y_col: str,
    score_pairs: List[Tuple[str, str]],
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Pairwise comparison between scores for temporal validation.

    For each pair (A, B), computes DeLong test, bootstrap ΔAUC with CI,
    NRI, and IDI on the common-available subset.
    """
    rows = []
    for s1, s2 in score_pairs:
        needed = [y_col, s1, s2]
        sub = df[needed].dropna()
        if len(sub) < 10 or sub[y_col].nunique() < 2:
            continue
        y = sub[y_col].values
        p1 = sub[s1].values
        p2 = sub[s2].values

        dl = delong_roc_test(y, p1, p2)
        bs = bootstrap_auc_diff(y, p1, p2, n_boot=n_boot, seed=seed)
        nri = compute_nri_with_ci(y, p2, p1, n_boot=n_boot, seed=seed)
        idi = compute_idi_with_ci(y, p2, p1, n_boot=n_boot, seed=seed)

        # Primary CI: DeLong when available (analytically exact for AUC
        # differences), bootstrap otherwise.  Both are always reported.
        _dl_se = dl.get("delta_auc_se", np.nan)
        if dl.get("reason") is None and np.isfinite(_dl_se) and _dl_se > 0:
            _ci_low = dl["delta_auc"] - 1.96 * _dl_se
            _ci_high = dl["delta_auc"] + 1.96 * _dl_se
            _ci_source = "DeLong"
        else:
            _ci_low = bs["ci_low"]
            _ci_high = bs["ci_high"]
            _ci_source = "Bootstrap"

        rows.append({
            "Comparison": f"{s1} vs {s2}",
            "n": int(len(sub)),
            "Delta_AUC": dl["delta_auc"],
            "Delta_AUC_IC95_inf": _ci_low,
            "Delta_AUC_IC95_sup": _ci_high,
            "Delta_AUC_CI_source": _ci_source,
            "Bootstrap_CI_low": bs["ci_low"],
            "Bootstrap_CI_high": bs["ci_high"],
            "Bootstrap_p": bs["p"],
            "DeLong_p": dl["p"],
            "DeLong_SE": _dl_se,
            # ``DeLong_skip_reason`` is non-null only when the test was not
            # computed for a methodological reason (e.g. <2 events or <2
            # non-events).  Consumers can render this as a footnote; when
            # DeLong_p is a number, this column is empty.
            "DeLong_skip_reason": dl.get("reason"),
            "NRI": nri["NRI total"],
            "NRI_CI_low": nri.get("NRI_CI_low", np.nan),
            "NRI_CI_high": nri.get("NRI_CI_high", np.nan),
            "NRI_p": nri.get("NRI_p", np.nan),
            "IDI": idi["IDI"],
            "IDI_CI_low": idi.get("IDI_CI_low", np.nan),
            "IDI_CI_high": idi.get("IDI_CI_high", np.nan),
            "IDI_p": idi.get("IDI_p", np.nan),
        })

    cols = [
        "Comparison", "n", "Delta_AUC",
        "Delta_AUC_IC95_inf", "Delta_AUC_IC95_sup", "Delta_AUC_CI_source",
        "Bootstrap_CI_low", "Bootstrap_CI_high", "Bootstrap_p",
        "DeLong_p", "DeLong_SE", "DeLong_skip_reason",
        "NRI", "NRI_CI_low", "NRI_CI_high", "NRI_p",
        "IDI", "IDI_CI_low", "IDI_CI_high", "IDI_p",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


def risk_category_table(
    df: pd.DataFrame,
    y_col: str,
    score_cols: List[str],
    cutoffs: Tuple[float, float] = (0.05, 0.15),
) -> pd.DataFrame:
    """Distribution of patients across risk categories for each score.

    Returns DataFrame with columns: Score, Category, n, %, Observed_mortality.
    """
    category_labels = {
        "low": "Low (<5%)",
        "intermediate": "Intermediate (5-15%)",
        "high": "High (>15%)",
    }
    rows = []
    for c in score_cols:
        sub = df[[y_col, c]].dropna()
        if sub.empty:
            continue
        sub = sub.copy()
        sub["_cat"] = sub[c].map(lambda x: categorize_risk(float(x), cutoffs))
        for cat_key in ["low", "intermediate", "high"]:
            mask = sub["_cat"] == cat_key
            n = int(mask.sum())
            pct = n / len(sub) * 100 if len(sub) > 0 else 0
            obs_mort = float(sub.loc[mask, y_col].mean()) if n > 0 else np.nan
            rows.append({
                "Score": c,
                "Category": category_labels[cat_key],
                "n": n,
                "%": round(pct, 1),
                "Observed_mortality": obs_mort,
            })
    if not rows:
        return pd.DataFrame(columns=["Score", "Category", "n", "%", "Observed_mortality"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Youden's J optimal threshold
# ---------------------------------------------------------------------------

def youden_threshold(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Find optimal threshold by Youden's J = Sensitivity + Specificity − 1.

    Returns ``(optimal_threshold, j_score)``.

    Important: computed on the evaluation cohort → treat as exploratory /
    data-driven / optimistic.  Do NOT use as a prospective locked threshold.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    fpr, tpr, thresholds = roc_curve(y, p)
    j = tpr + (1.0 - fpr) - 1.0
    idx = int(np.argmax(j))
    return float(thresholds[idx]), float(j[idx])


# ---------------------------------------------------------------------------
# Multi-threshold classification analysis
# ---------------------------------------------------------------------------

def threshold_analysis_table(
    y: np.ndarray,
    p: np.ndarray,
    thresholds: List[float],
) -> pd.DataFrame:
    """Classification metrics at multiple thresholds.

    Columns: Threshold, Sensitivity, Specificity, PPV, NPV,
    TP, FP, TN, FN, N_Flagged, Positives_per_1000, Flag_Rate_pct.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    n = len(y)
    rows = []
    for t in thresholds:
        m = classification_metrics_at_threshold(y, p, t)
        n_flagged = int((p >= t).sum())
        rows.append({
            "Threshold": float(t),
            "Sensitivity": m["Sensitivity"],
            "Specificity": m["Specificity"],
            "PPV": m["PPV"],
            "NPV": m["NPV"],
            "TP": m["TP"],
            "FP": m["FP"],
            "TN": m["TN"],
            "FN": m["FN"],
            "N_Flagged": n_flagged,
            "Positives_per_1000": round(n_flagged / n * 1000, 1) if n else np.nan,
            "Flag_Rate_pct": round(n_flagged / n * 100, 1) if n else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Detailed calibration bins (for interactive plots)
# ---------------------------------------------------------------------------

def calibration_bins_detail(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Calibration bins with per-bin count and Wilson 95% CI.

    Columns: Bin, N, Mean_Predicted, Obs_Frequency, CI_lower, CI_upper.
    strategy: 'quantile' (equal-size) or 'uniform' (equal-width).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    df_cal = pd.DataFrame({"y": y, "p": p})

    try:
        if strategy == "quantile":
            df_cal["bin"] = pd.qcut(df_cal["p"], q=n_bins, labels=False, duplicates="drop")
        else:
            df_cal["bin"] = pd.cut(df_cal["p"], bins=n_bins, labels=False, duplicates="drop")
    except Exception:
        df_cal["bin"] = pd.cut(df_cal["p"], bins=n_bins, labels=False, duplicates="drop")

    rows = []
    z = 1.96
    for bin_label, grp in df_cal.groupby("bin", observed=True):
        n = len(grp)
        if n == 0:
            continue
        mean_pred = float(grp["p"].mean())
        obs_freq = float(grp["y"].mean())
        # Wilson binomial CI
        center = (obs_freq + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * np.sqrt(obs_freq * (1 - obs_freq) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        ci_low = max(0.0, float(center - half))
        ci_high = min(1.0, float(center + half))
        rows.append({
            "Bin": int(bin_label) + 1,
            "N": n,
            "Mean_Predicted": mean_pred,
            "Obs_Frequency": obs_freq,
            "CI_lower": ci_low,
            "CI_upper": ci_high,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Post-hoc recalibration (exploratory — do not use for primary reporting)
# ---------------------------------------------------------------------------

def recalibrate_intercept_only(y: np.ndarray, p: np.ndarray) -> Dict:
    """Intercept-only (calibration-in-the-large) recalibration.

    Fits: logit(p_new) = a + 1.0 × logit(p_original)  (slope fixed at 1).

    Returns dict: recalibrated_probs, intercept_offset, brier_before/after,
    cal_intercept/slope before/after.

    IMPORTANT: exploratory post-hoc analysis only.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    p_clipped = np.clip(p, 1e-7, 1 - 1e-7)
    logit_p = np.log(p_clipped / (1 - p_clipped))

    def _nll(a: float) -> float:
        lp = a + logit_p
        pn = 1.0 / (1.0 + np.exp(-lp))
        pn = np.clip(pn, 1e-9, 1 - 1e-9)
        return float(-np.sum(y * np.log(pn) + (1 - y) * np.log(1 - pn)))

    res = minimize_scalar(_nll, bounds=(-10, 10), method="bounded")
    offset = float(res.x)
    p_new = np.clip(1.0 / (1.0 + np.exp(-(offset + logit_p))), 0.0, 1.0)

    cal_before = calibration_intercept_slope(y, p)
    cal_after = calibration_intercept_slope(y, p_new)
    return {
        "recalibrated_probs": p_new,
        "method": "intercept_only",
        "intercept_offset": offset,
        "brier_before": float(brier_score_loss(y, p)),
        "brier_after": float(brier_score_loss(y, p_new)),
        "cal_intercept_before": cal_before["Calibration intercept"],
        "cal_intercept_after": cal_after["Calibration intercept"],
        "cal_slope_before": cal_before["Calibration slope"],
        "cal_slope_after": cal_after["Calibration slope"],
    }


def recalibrate_logistic(y: np.ndarray, p: np.ndarray) -> Dict:
    """Intercept + slope logistic recalibration.

    Fits: logit(p_new) = a + b × logit(p_original).

    IMPORTANT: exploratory post-hoc analysis only.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        brier = float(brier_score_loss(y, p))
        return {
            "recalibrated_probs": p.copy(),
            "method": "logistic",
            "intercept": np.nan,
            "slope": np.nan,
            "brier_before": brier,
            "brier_after": np.nan,
            "cal_intercept_before": np.nan,
            "cal_intercept_after": np.nan,
            "cal_slope_before": np.nan,
            "cal_slope_after": np.nan,
        }

    p_clipped = np.clip(p, 1e-7, 1 - 1e-7)
    logit_p = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000)
        lr.fit(logit_p, y)
    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_[0][0])
    p_new = np.clip(lr.predict_proba(logit_p)[:, 1], 0.0, 1.0)
    cal_before = calibration_intercept_slope(y, p)
    cal_after = calibration_intercept_slope(y, p_new)
    return {
        "recalibrated_probs": p_new,
        "method": "logistic",
        "intercept": intercept,
        "slope": slope,
        "brier_before": float(brier_score_loss(y, p)),
        "brier_after": float(brier_score_loss(y, p_new)),
        "cal_intercept_before": cal_before["Calibration intercept"],
        "cal_intercept_after": cal_after["Calibration intercept"],
        "cal_slope_before": cal_before["Calibration slope"],
        "cal_slope_after": cal_after["Calibration slope"],
    }


def recalibrate_isotonic(y: np.ndarray, p: np.ndarray) -> Dict:
    """Isotonic regression recalibration (non-parametric, monotone).

    Most aggressive method. May overfit small datasets.
    IMPORTANT: exploratory post-hoc analysis only.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p, y)
    p_new = np.clip(ir.predict(p), 0.0, 1.0)
    cal_before = calibration_intercept_slope(y, p)
    cal_after = calibration_intercept_slope(y, p_new)
    return {
        "recalibrated_probs": p_new,
        "method": "isotonic",
        "brier_before": float(brier_score_loss(y, p)),
        "brier_after": float(brier_score_loss(y, p_new)),
        "cal_intercept_before": cal_before["Calibration intercept"],
        "cal_intercept_after": cal_after["Calibration intercept"],
        "cal_slope_before": cal_before["Calibration slope"],
        "cal_slope_after": cal_after["Calibration slope"],
    }


def compute_cohort_drift(
    df_train: pd.DataFrame,
    df_temporal: pd.DataFrame,
    y_col: str,
    feature_columns: list,
) -> "Dict[str, Any]":
    """Compare training and temporal cohorts on key distributional properties.

    Args:
        df_train: Training cohort DataFrame.
        df_temporal: Temporal validation cohort DataFrame.
        y_col: Name of the binary outcome column.
        feature_columns: List of feature column names to analyze.

    Returns:
        Dict with prevalence info, missingness_shift (sorted by |delta|),
        and numeric_shift (sorted by |relative shift|).
    """
    result: Dict[str, Any] = {}

    # Prevalence
    prev_train = float(df_train[y_col].mean()) if y_col in df_train.columns else float("nan")
    prev_temp = float(df_temporal[y_col].mean()) if y_col in df_temporal.columns else float("nan")
    result["prevalence_train"] = prev_train
    result["prevalence_temporal"] = prev_temp
    result["prevalence_delta"] = (prev_temp - prev_train) if not (np.isnan(prev_train) or np.isnan(prev_temp)) else float("nan")

    # Missingness shift
    miss_rows = []
    for col in feature_columns:
        miss_train = float(df_train[col].isna().mean()) if col in df_train.columns else float("nan")
        miss_temp = float(df_temporal[col].isna().mean()) if col in df_temporal.columns else float("nan")
        delta = (miss_temp - miss_train) if not (np.isnan(miss_train) or np.isnan(miss_temp)) else float("nan")
        miss_rows.append({
            "variable": col,
            "missing_train": miss_train,
            "missing_temporal": miss_temp,
            "delta": delta,
        })
    result["missingness_shift"] = sorted(
        miss_rows,
        key=lambda x: abs(x.get("delta", 0) or 0),
        reverse=True,
    )

    # Numeric distribution shift (median, IQR)
    num_rows = []
    for col in feature_columns:
        if col in df_train.columns and col in df_temporal.columns:
            s_train = pd.to_numeric(df_train[col], errors="coerce")
            s_temp = pd.to_numeric(df_temporal[col], errors="coerce")
            if s_train.notna().sum() > 10 and s_temp.notna().sum() > 5:
                p25_train = float(s_train.quantile(0.25))
                p75_train = float(s_train.quantile(0.75))
                iqr_train = p75_train - p25_train
                median_train = float(s_train.median())
                median_temp = float(s_temp.median())
                rel_shift = abs(median_temp - median_train) / (iqr_train + 1e-8)
                num_rows.append({
                    "variable": col,
                    "median_train": median_train,
                    "median_temporal": median_temp,
                    "p25_train": p25_train,
                    "p75_train": p75_train,
                    "p25_temporal": float(s_temp.quantile(0.25)),
                    "p75_temporal": float(s_temp.quantile(0.75)),
                    "rel_shift_over_iqr": rel_shift,
                })
    result["numeric_shift"] = sorted(
        num_rows,
        key=lambda x: x.get("rel_shift_over_iqr", 0),
        reverse=True,
    )

    return result
