"""Ablation: procedure_group vs. no procedure_group.

Standalone script — does NOT modify any production file.
Run once with:  python ablation_procedure_group.py
"""
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# Suppress non-critical calibration / convergence warnings during cross-val
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, ".")
from risk_data import prepare_master_dataset
from modeling import train_and_select_model
from stats_compare import calibration_intercept_slope, classification_metrics_at_threshold

XLSX = "local_data/Dataset_2025.xlsx"
THRESHOLD_CLINICAL = 0.08


# ── helpers ───────────────────────────────────────────────────────────────────

def metrics_at(y, proba, threshold):
    m = classification_metrics_at_threshold(np.asarray(y), np.asarray(proba), threshold)
    return m


def full_metrics(name, y, proba, youden_thresh):
    auc   = roc_auc_score(y, proba)
    auprc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    cal   = calibration_intercept_slope(y, proba)
    m8    = metrics_at(y, proba, THRESHOLD_CLINICAL)
    my    = metrics_at(y, proba, youden_thresh)
    return {
        "variant": name,
        "best_model": None,      # filled after training
        "auc": auc,
        "auprc": auprc,
        "brier": brier,
        "cal_intercept": cal["Calibration intercept"],
        "cal_slope": cal["Calibration slope"],
        # threshold 8%
        "sens_8pct": m8.get("Sensitivity", np.nan),
        "spec_8pct": m8.get("Specificity", np.nan),
        "ppv_8pct":  m8.get("PPV", np.nan),
        "npv_8pct":  m8.get("NPV", np.nan),
        # Youden threshold
        "youden_thresh": youden_thresh,
        "sens_youden": my.get("Sensitivity", np.nan),
        "spec_youden": my.get("Specificity", np.nan),
        "ppv_youden":  my.get("PPV", np.nan),
        "npv_youden":  my.get("NPV", np.nan),
    }


def try_get_importance(artifacts):
    """Best-effort: extract feature importances from final model."""
    try:
        model = artifacts.model
        cols  = artifacts.feature_columns
        # CalibratedClassifierCV wraps estimator
        inner = getattr(model, "_pipeline", model)
        inner = getattr(inner, "estimator", inner)
        inner = getattr(inner, "calibrated_classifiers_", [inner])[0]
        inner = getattr(inner, "estimator", inner)
        # final step of the pipeline
        est   = inner
        if hasattr(est, "named_steps"):
            est = est.named_steps.get("model", est)
        fi    = getattr(est, "feature_importances_", None)
        if fi is None:
            fi = getattr(est, "coef_", None)
            if fi is not None:
                fi = np.abs(fi).ravel()
        if fi is not None and len(fi) == len(cols):
            return pd.Series(fi, index=cols).sort_values(ascending=False)
    except Exception:
        pass
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    print("Loading data…")
    prepared = prepare_master_dataset(XLSX)
    df = prepared.data
    feature_cols_all = prepared.feature_columns
    y = df["morte_30d"].astype(int).values
    n_pos = int(y.sum())
    n_total = len(y)
    print(f"  n={n_total}, events={n_pos}, prevalence={n_pos/n_total:.1%}")
    print(f"  feature_columns (all): {len(feature_cols_all)}")
    pg_in_feats = "procedure_group" in feature_cols_all
    print(f"  procedure_group in feature_columns: {pg_in_feats}")
    print()

    # ── Version A: WITH procedure_group ──────────────────────────────────────
    print("=" * 60)
    print("VERSION A — WITH procedure_group")
    print("=" * 60)
    feat_with = feature_cols_all
    arts_with = train_and_select_model(df, feat_with)
    y_oof_with = arts_with.oof_predictions[arts_with.best_model_name]
    thresh_with = arts_with.best_youden_threshold
    res_with = full_metrics("WITH procedure_group", y, y_oof_with, thresh_with)
    res_with["best_model"] = arts_with.best_model_name
    print(f"  Best model : {arts_with.best_model_name}")
    print(f"  AUC        : {res_with['auc']:.4f}")
    print(f"  AUPRC      : {res_with['auprc']:.4f}")
    print(f"  Brier      : {res_with['brier']:.4f}")
    print(f"  Cal interc : {res_with['cal_intercept']:.4f}")
    print(f"  Cal slope  : {res_with['cal_slope']:.4f}")
    print(f"  @ 8%  Sens : {res_with['sens_8pct']:.3f}  Spec : {res_with['spec_8pct']:.3f}  PPV : {res_with['ppv_8pct']:.3f}  NPV : {res_with['npv_8pct']:.3f}")
    print(f"  Youden t   : {thresh_with:.3f}")
    print(f"  @ Youden Sens : {res_with['sens_youden']:.3f}  Spec : {res_with['spec_youden']:.3f}  PPV : {res_with['ppv_youden']:.3f}  NPV : {res_with['npv_youden']:.3f}")

    imp_with = try_get_importance(arts_with)
    pg_rank_with = None
    if imp_with is not None and "procedure_group" in imp_with.index:
        pg_rank_with = imp_with.index.tolist().index("procedure_group") + 1
        pg_imp_with = imp_with["procedure_group"]
        print(f"  procedure_group importance: {pg_imp_with:.6f}  (rank {pg_rank_with}/{len(imp_with)})")

    # ── Version B: WITHOUT procedure_group ───────────────────────────────────
    print()
    print("=" * 60)
    print("VERSION B — WITHOUT procedure_group")
    print("=" * 60)
    feat_without = [c for c in feature_cols_all if c != "procedure_group"]
    print(f"  Removed 'procedure_group'. Features: {len(feat_without)}")
    arts_without = train_and_select_model(df, feat_without)
    y_oof_without = arts_without.oof_predictions[arts_without.best_model_name]
    thresh_without = arts_without.best_youden_threshold
    res_without = full_metrics("WITHOUT procedure_group", y, y_oof_without, thresh_without)
    res_without["best_model"] = arts_without.best_model_name
    print(f"  Best model : {arts_without.best_model_name}")
    print(f"  AUC        : {res_without['auc']:.4f}")
    print(f"  AUPRC      : {res_without['auprc']:.4f}")
    print(f"  Brier      : {res_without['brier']:.4f}")
    print(f"  Cal interc : {res_without['cal_intercept']:.4f}")
    print(f"  Cal slope  : {res_without['cal_slope']:.4f}")
    print(f"  @ 8%  Sens : {res_without['sens_8pct']:.3f}  Spec : {res_without['spec_8pct']:.3f}  PPV : {res_without['ppv_8pct']:.3f}  NPV : {res_without['npv_8pct']:.3f}")
    print(f"  Youden t   : {thresh_without:.3f}")
    print(f"  @ Youden Sens : {res_without['sens_youden']:.3f}  Spec : {res_without['spec_youden']:.3f}  PPV : {res_without['ppv_youden']:.3f}  NPV : {res_without['npv_youden']:.3f}")

    # ── Side-by-side comparison ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("COMPARISON  (Delta = WITH - WITHOUT)")
    print("=" * 60)
    delta = {k: res_with[k] - res_without[k] for k in ["auc", "auprc", "brier", "cal_intercept", "cal_slope"]}
    fmt = (
        f"{'Metric':<22} {'WITH':>8} {'WITHOUT':>8} {'D':>8}\n"
        f"{'-'*22} {'-'*8} {'-'*8} {'-'*8}\n"
        f"{'AUC':<22} {res_with['auc']:>8.4f} {res_without['auc']:>8.4f} {delta['auc']:>+8.4f}\n"
        f"{'AUPRC':<22} {res_with['auprc']:>8.4f} {res_without['auprc']:>8.4f} {delta['auprc']:>+8.4f}\n"
        f"{'Brier':<22} {res_with['brier']:>8.4f} {res_without['brier']:>8.4f} {delta['brier']:>+8.4f}\n"
        f"{'Cal intercept':<22} {res_with['cal_intercept']:>8.4f} {res_without['cal_intercept']:>8.4f} {delta['cal_intercept']:>+8.4f}\n"
        f"{'Cal slope':<22} {res_with['cal_slope']:>8.4f} {res_without['cal_slope']:>8.4f} {delta['cal_slope']:>+8.4f}\n"
        f"\n"
        f"{'Best model':<22} {res_with['best_model']:>8} {res_without['best_model']:>8}\n"
        f"\n"
        f"@ 8% threshold:\n"
        f"{'  Sensitivity':<22} {res_with['sens_8pct']:>8.3f} {res_without['sens_8pct']:>8.3f} {res_with['sens_8pct']-res_without['sens_8pct']:>+8.3f}\n"
        f"{'  Specificity':<22} {res_with['spec_8pct']:>8.3f} {res_without['spec_8pct']:>8.3f} {res_with['spec_8pct']-res_without['spec_8pct']:>+8.3f}\n"
        f"{'  PPV':<22} {res_with['ppv_8pct']:>8.3f} {res_without['ppv_8pct']:>8.3f} {res_with['ppv_8pct']-res_without['ppv_8pct']:>+8.3f}\n"
        f"{'  NPV':<22} {res_with['npv_8pct']:>8.3f} {res_without['npv_8pct']:>8.3f} {res_with['npv_8pct']-res_without['npv_8pct']:>+8.3f}\n"
        f"\n"
        f"@ Youden threshold:\n"
        f"{'  Threshold':<22} {res_with['youden_thresh']:>8.3f} {res_without['youden_thresh']:>8.3f}\n"
        f"{'  Sensitivity':<22} {res_with['sens_youden']:>8.3f} {res_without['sens_youden']:>8.3f} {res_with['sens_youden']-res_without['sens_youden']:>+8.3f}\n"
        f"{'  Specificity':<22} {res_with['spec_youden']:>8.3f} {res_without['spec_youden']:>8.3f} {res_with['spec_youden']-res_without['spec_youden']:>+8.3f}\n"
        f"{'  PPV':<22} {res_with['ppv_youden']:>8.3f} {res_without['ppv_youden']:>8.3f} {res_with['ppv_youden']-res_without['ppv_youden']:>+8.3f}\n"
        f"{'  NPV':<22} {res_with['npv_youden']:>8.3f} {res_without['npv_youden']:>8.3f} {res_with['npv_youden']-res_without['npv_youden']:>+8.3f}\n"
    )
    print(fmt)

    # ── Leaderboard both versions ─────────────────────────────────────────────
    print("Leaderboard — WITH procedure_group:")
    lb_with = arts_with.leaderboard[["Modelo", "AUC", "AUPRC", "Brier"]].copy()
    lb_with.columns = ["Model", "AUC", "AUPRC", "Brier"]
    print(lb_with.to_string(index=False))
    print()
    print("Leaderboard — WITHOUT procedure_group:")
    lb_without = arts_without.leaderboard[["Modelo", "AUC", "AUPRC", "Brier"]].copy()
    lb_without.columns = ["Model", "AUC", "AUPRC", "Brier"]
    print(lb_without.to_string(index=False))

    return res_with, res_without, arts_with, arts_without


if __name__ == "__main__":
    run()
