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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.calibration import calibration_curve
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
        m = basic_metrics(sub[y_col].values, sub[c].values)
        cls = classification_metrics_at_threshold(sub[y_col].values, sub[c].values, threshold)
        rows.append({"Score": c, "n": int(len(sub)), **m, **cls})
    cols = ["Score", "n", "AUC", "AUPRC", "Brier", "Sensitivity", "Specificity", "PPV", "NPV"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values("AUC", ascending=False)


def evaluate_scores(df: pd.DataFrame, y_col: str, score_cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in score_cols:
        sub = df[[y_col, c]].dropna()
        if len(sub) < 30 or sub[y_col].nunique() < 2:
            continue
        m = basic_metrics(sub[y_col].values, sub[c].values)
        m["Score"] = c
        rows.append(m)
    if not rows:
        return pd.DataFrame(columns=["Score", "n", "AUC", "AUPRC", "Brier"])
    return pd.DataFrame(rows)[["Score", "n", "AUC", "AUPRC", "Brier"]].sort_values("AUC", ascending=False)


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
        m = bootstrap_metrics_ci(
            sub[y_col].values,
            sub[c].values,
            n_boot=n_boot,
            seed=seed,
        )
        m["Score"] = c
        m["n"] = int(len(sub))
        rows.append(m)

    cols = [
        "Score",
        "n",
        "AUC",
        "AUC_IC95_inf",
        "AUC_IC95_sup",
        "AUPRC",
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
    delta = float(np.mean(arr))
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


def delong_roc_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    order = np.argsort(-y)
    y_sorted = y[order]
    preds = np.vstack([p1[order], p2[order]])
    label_1_count = int(y_sorted.sum())
    if label_1_count == 0 or label_1_count == len(y_sorted):
        return {"AUC_1": np.nan, "AUC_2": np.nan, "delta_auc": np.nan, "z": np.nan, "p": np.nan}

    aucs, cov = _fast_delong(preds, label_1_count)
    diff = float(aucs[0] - aucs[1])
    if np.ndim(cov) == 0:
        var = float(cov)
    else:
        var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        return {"AUC_1": float(aucs[0]), "AUC_2": float(aucs[1]), "delta_auc": diff, "z": np.nan, "p": np.nan}
    z = diff / np.sqrt(var)
    pval = float(2 * norm.sf(abs(z)))
    return {"AUC_1": float(aucs[0]), "AUC_2": float(aucs[1]), "delta_auc": diff, "z": float(z), "p": pval}


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
        cal = calibration_intercept_slope(y, p)
        hl = hosmer_lemeshow_test(y, p)
        cls = classification_metrics_at_threshold(y, p, threshold)

        rows.append({
            "Score": c,
            "n": int(len(sub)),
            "AUC": m["AUC"],
            "AUC_IC95_inf": m["AUC_IC95_inf"],
            "AUC_IC95_sup": m["AUC_IC95_sup"],
            "AUPRC": m["AUPRC"],
            "AUPRC_IC95_inf": m["AUPRC_IC95_inf"],
            "AUPRC_IC95_sup": m["AUPRC_IC95_sup"],
            "Brier": m["Brier"],
            "Calibration_Intercept": cal["Calibration intercept"],
            "Calibration_Slope": cal["Calibration slope"],
            "HL_p": hl["HL p-value"],
            "Sensitivity": cls["Sensitivity"],
            "Specificity": cls["Specificity"],
            "PPV": cls["PPV"],
            "NPV": cls["NPV"],
        })

    cols = [
        "Score", "n", "AUC", "AUC_IC95_inf", "AUC_IC95_sup",
        "AUPRC", "AUPRC_IC95_inf", "AUPRC_IC95_sup", "Brier",
        "Calibration_Intercept", "Calibration_Slope", "HL_p",
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
        nri = compute_nri(y, p2, p1)
        idi = compute_idi(y, p2, p1)

        rows.append({
            "Comparison": f"{s1} vs {s2}",
            "n": int(len(sub)),
            "Delta_AUC": dl["delta_auc"],
            "Delta_AUC_IC95_inf": bs["ci_low"],
            "Delta_AUC_IC95_sup": bs["ci_high"],
            "Bootstrap_p": bs["p"],
            "DeLong_p": dl["p"],
            "NRI": nri["NRI total"],
            "IDI": idi["IDI"],
        })

    cols = [
        "Comparison", "n", "Delta_AUC", "Delta_AUC_IC95_inf",
        "Delta_AUC_IC95_sup", "Bootstrap_p", "DeLong_p", "NRI", "IDI",
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
