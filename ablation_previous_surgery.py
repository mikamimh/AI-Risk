"""Ablation: Previous surgery — free-text vs binary (Yes/No).

Standalone script — does NOT modify any production file.
Run: python ablation_previous_surgery.py
"""
import sys, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, ".")
from risk_data import prepare_master_dataset, MISSING_TOKENS
from modeling import train_and_select_model
from stats_compare import calibration_intercept_slope, classification_metrics_at_threshold

XLSX = "local_data/Dataset_2025.xlsx"
THRESHOLD_CLINICAL = 0.08
COL = "Previous surgery"


# ── Binary parser ─────────────────────────────────────────────────────────────
# Rules:
#   "No"  → "No"   (explicit absence)
#   blank/NaN/MISSING_TOKEN → "No"  (missing in context of binary presence)
#   anything else → "Yes"  (any mention of a procedure = redo cardiac surgery)
# The grammar details (;, +, years, xN) are metadata on the same fact:
# patient had at least one prior cardiac surgery.

def binarize_previous_surgery(series: pd.Series) -> pd.Series:
    def _map(v):
        if pd.isna(v):
            return "No"
        txt = str(v).strip()
        if not txt or txt.lower() in MISSING_TOKENS:
            return "No"
        if txt.lower() == "no":
            return "No"
        return "Yes"
    return series.map(_map)


# ── Metrics helpers ───────────────────────────────────────────────────────────

def full_metrics(variant, best_model, n_features, y, proba, youden_thresh):
    auc   = roc_auc_score(y, proba)
    auprc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    cal   = calibration_intercept_slope(y, proba)
    m8    = classification_metrics_at_threshold(np.asarray(y), np.asarray(proba), THRESHOLD_CLINICAL)
    my    = classification_metrics_at_threshold(np.asarray(y), np.asarray(proba), youden_thresh)
    return {
        "variant": variant, "best_model": best_model, "n_features": n_features,
        "auc": auc, "auprc": auprc, "brier": brier,
        "cal_intercept": cal["Calibration intercept"],
        "cal_slope":     cal["Calibration slope"],
        "sens_8": m8["Sensitivity"], "spec_8": m8["Specificity"],
        "ppv_8":  m8["PPV"],         "npv_8":  m8["NPV"],
        "youden_thresh": youden_thresh,
        "sens_y": my["Sensitivity"], "spec_y": my["Specificity"],
        "ppv_y":  my["PPV"],         "npv_y":  my["NPV"],
    }


def print_result(r):
    print(f"  Best model : {r['best_model']}  (n_features={r['n_features']})")
    print(f"  AUC        : {r['auc']:.4f}")
    print(f"  AUPRC      : {r['auprc']:.4f}")
    print(f"  Brier      : {r['brier']:.4f}")
    print(f"  Cal interc : {r['cal_intercept']:.4f}")
    print(f"  Cal slope  : {r['cal_slope']:.4f}")
    print(f"  @8%  Sens:{r['sens_8']:.3f} Spec:{r['spec_8']:.3f} PPV:{r['ppv_8']:.3f} NPV:{r['npv_8']:.3f}")
    print(f"  Youden t:{r['youden_thresh']:.3f} Sens:{r['sens_y']:.3f} Spec:{r['spec_y']:.3f} PPV:{r['ppv_y']:.3f} NPV:{r['npv_y']:.3f}")


def print_comparison(results):
    keys   = ["auc", "auprc", "brier", "cal_intercept", "cal_slope"]
    labels = ["AUC", "AUPRC", "Brier", "Cal intercept", "Cal slope"]
    header = f"{'Metric':<22}"
    for r in results:
        header += f"  {r['variant'][:16]:>16}"
    print(header)
    print("-" * (22 + 18 * len(results)))
    for k, lab in zip(keys, labels):
        row = f"{lab:<22}"
        for r in results:
            row += f"  {r[k]:>16.4f}"
        print(row)
    print()
    for lab, k in [("Best model", "best_model"), ("N features", "n_features")]:
        row = f"{lab:<22}"
        for r in results:
            row += f"  {str(r[k]):>16}"
        print(row)
    print()
    print("@ 8% threshold:")
    for lab, k in [("  Sensitivity","sens_8"),("  Specificity","spec_8"),("  PPV","ppv_8"),("  NPV","npv_8")]:
        row = f"{lab:<22}"
        for r in results:
            row += f"  {r[k]:>16.3f}"
        print(row)
    print()
    print("@ Youden threshold:")
    for lab, k in [("  Threshold","youden_thresh"),("  Sensitivity","sens_y"),("  Specificity","spec_y"),("  PPV","ppv_y"),("  NPV","npv_y")]:
        row = f"{lab:<22}"
        for r in results:
            row += f"  {r[k]:>16.3f}"
        print(row)


def get_importance(arts, col):
    try:
        m = arts.model
        m = getattr(m, "_pipeline", m)
        m = getattr(m, "estimator", m)
        m = getattr(m, "calibrated_classifiers_", [m])[0]
        m = getattr(m, "estimator", m)
        if hasattr(m, "named_steps"):
            m = m.named_steps.get("model", m)
        fi = getattr(m, "feature_importances_", None)
        cols = arts.feature_columns
        if fi is not None and len(fi) == len(cols):
            imp = pd.Series(fi, index=cols).sort_values(ascending=False)
            if col in imp.index:
                rank = imp.index.tolist().index(col) + 1
                return imp[col], rank, len(imp)
    except Exception:
        pass
    return None, None, None


def run():
    print("Loading data...")
    prepared = prepare_master_dataset(XLSX)
    df_orig  = prepared.data.copy()
    fc_all   = list(prepared.feature_columns)
    y        = df_orig["morte_30d"].astype(int).values
    print(f"  n={len(df_orig)}, events={y.sum()}, features={len(fc_all)}")

    # ── Diagnostic ────────────────────────────────────────────────────────────
    s = df_orig[COL]
    vc = s.value_counts()
    n_singleton = int((vc == 1).sum())
    n_rare      = int(((vc > 1) & (vc < 5)).sum())
    n_redo      = int((s != "No").sum())
    mort_no     = y[s == "No"].mean()
    mort_redo   = y[s != "No"].mean()
    print(f"\n{COL} diagnostic:")
    print(f"  Unique values    : {s.nunique()}")
    print(f"  Singleton (n=1)  : {n_singleton}")
    print(f"  Rare (n=2-4)     : {n_rare}")
    print(f"  No               : {(s=='No').sum()} ({(s=='No').mean():.1%})")
    print(f"  Redo (any)       : {n_redo} ({n_redo/len(s):.1%})")
    print(f"  Mortality No     : {mort_no:.1%}")
    print(f"  Mortality Redo   : {mort_redo:.1%}  (delta +{mort_redo-mort_no:.1%})")

    # Validate the binary parser
    binary = binarize_previous_surgery(s)
    n_yes = int((binary == "Yes").sum())
    n_no  = int((binary == "No").sum())
    print(f"\n  Binary mapping: No={n_no}, Yes={n_yes} — matches redo count: {n_yes == n_redo}")

    # ── Baseline — free-text as-is ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE — Previous surgery as free-text (36 categories)")
    print("=" * 60)
    arts_base = train_and_select_model(df_orig, fc_all)
    proba_base = arts_base.oof_predictions[arts_base.best_model_name]
    res_base = full_metrics("Baseline", arts_base.best_model_name,
                            len(fc_all), y, proba_base,
                            arts_base.best_youden_threshold)
    print_result(res_base)

    imp_val, imp_rank, imp_total = get_importance(arts_base, COL)
    if imp_val is not None:
        print(f"  '{COL}' importance: {imp_val:.6f}  rank {imp_rank}/{imp_total}")

    # ── Ablation — binarized Yes/No ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION — Previous surgery binarized to Yes/No")
    print("=" * 60)
    df_bin = df_orig.copy()
    df_bin[COL] = binarize_previous_surgery(df_bin[COL])
    print(f"  Values after binarization: {df_bin[COL].value_counts().to_dict()}")

    arts_abl = train_and_select_model(df_bin, fc_all)
    proba_abl = arts_abl.oof_predictions[arts_abl.best_model_name]
    res_abl = full_metrics("Binary Yes/No", arts_abl.best_model_name,
                           len(fc_all), y, proba_abl,
                           arts_abl.best_youden_threshold)
    print_result(res_abl)

    imp_val2, imp_rank2, imp_total2 = get_importance(arts_abl, COL)
    if imp_val2 is not None:
        print(f"  '{COL}' importance: {imp_val2:.6f}  rank {imp_rank2}/{imp_total2}")

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON  (D = Ablation - Baseline)")
    print("=" * 60)
    print_comparison([res_base, res_abl])

    print("\nLeaderboard [Baseline]:")
    lb = arts_base.leaderboard[["Modelo","AUC","AUPRC","Brier"]].copy()
    lb.columns = ["Model","AUC","AUPRC","Brier"]
    print(lb.to_string(index=False))

    print("\nLeaderboard [Binary]:")
    lb2 = arts_abl.leaderboard[["Modelo","AUC","AUPRC","Brier"]].copy()
    lb2.columns = ["Model","AUC","AUPRC","Brier"]
    print(lb2.to_string(index=False))

    return res_base, res_abl


if __name__ == "__main__":
    run()
