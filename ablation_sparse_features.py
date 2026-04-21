"""Ablation: high-sparsity echo features + KDIGO redundancy.

Standalone script — does NOT modify any production file.
Run: python ablation_sparse_features.py
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

# ── Candidate columns for Ablation 1: high-sparsity valve/echo measures ───────
# Matched by substring to avoid encoding issues with ² and †
ABLATION_1_PATTERNS = [
    "Vena contracta",       # 94.1% missing — AR proximal zone measure
    "MVA",                  # 93.2% missing — mitral valve area (MS severity)
    "Mitral Mean gradient", # 93.2% missing — MS gradient (MS severity)
    "Mitral Stenosis",      # 92.3% missing — MS ordinal (same pathology as MVA)
    "PHT Aortic",           # 91.2% missing — AR pressure half-time
    "Vena contracta (mm)",  # 89.2% missing — MR proximal zone measure
]

# Ablation 2: KDIGO only — redundancy argument (Spearman rho=-0.953 with CrClearance)
ABLATION_2_PATTERNS = ["KDIGO"]


def match_cols(feature_columns, patterns):
    matched = []
    for pat in patterns:
        for c in feature_columns:
            if pat.lower() in c.lower() and c not in matched:
                matched.append(c)
                break
    return matched


def full_metrics(variant, best_model, y, proba, youden_thresh):
    auc   = roc_auc_score(y, proba)
    auprc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    cal   = calibration_intercept_slope(y, proba)
    m8    = classification_metrics_at_threshold(np.asarray(y), np.asarray(proba), THRESHOLD_CLINICAL)
    my    = classification_metrics_at_threshold(np.asarray(y), np.asarray(proba), youden_thresh)
    return {
        "variant": variant,
        "best_model": best_model,
        "n_features": None,
        "auc": auc,
        "auprc": auprc,
        "brier": brier,
        "cal_intercept": cal["Calibration intercept"],
        "cal_slope":     cal["Calibration slope"],
        "sens_8": m8["Sensitivity"], "spec_8": m8["Specificity"],
        "ppv_8":  m8["PPV"],         "npv_8":  m8["NPV"],
        "youden_thresh": youden_thresh,
        "sens_y": my["Sensitivity"], "spec_y": my["Specificity"],
        "ppv_y":  my["PPV"],         "npv_y":  my["NPV"],
    }


def run_variant(label, df, feature_cols, y):
    arts = train_and_select_model(df, feature_cols)
    proba = arts.oof_predictions[arts.best_model_name]
    res = full_metrics(label, arts.best_model_name, y, proba, arts.best_youden_threshold)
    res["n_features"] = len(feature_cols)
    return res, arts


def print_result(r):
    print(f"  Best model : {r['best_model']}")
    print(f"  N features : {r['n_features']}")
    print(f"  AUC        : {r['auc']:.4f}")
    print(f"  AUPRC      : {r['auprc']:.4f}")
    print(f"  Brier      : {r['brier']:.4f}")
    print(f"  Cal interc : {r['cal_intercept']:.4f}")
    print(f"  Cal slope  : {r['cal_slope']:.4f}")
    print(f"  @8% Sens:{r['sens_8']:.3f}  Spec:{r['spec_8']:.3f}  PPV:{r['ppv_8']:.3f}  NPV:{r['npv_8']:.3f}")
    print(f"  Youden t:{r['youden_thresh']:.3f}  Sens:{r['sens_y']:.3f}  Spec:{r['spec_y']:.3f}  PPV:{r['ppv_y']:.3f}  NPV:{r['npv_y']:.3f}")


def print_comparison(results):
    keys = ["auc", "auprc", "brier", "cal_intercept", "cal_slope"]
    labels = ["AUC", "AUPRC", "Brier", "Cal intercept", "Cal slope"]
    base = results[0]
    header = f"{'Metric':<22}"
    for r in results:
        header += f"  {r['variant'][:14]:>14}"
    print(header)
    print("-" * (22 + 16 * len(results)))
    for k, lab in zip(keys, labels):
        row = f"{lab:<22}"
        for r in results:
            row += f"  {r[k]:>14.4f}"
        print(row)
    print()
    row = f"{'Best model':<22}"
    for r in results:
        row += f"  {r['best_model']:>14}"
    print(row)
    row = f"{'N features':<22}"
    for r in results:
        row += f"  {str(r['n_features']):>14}"
    print(row)
    print()
    print("@ 8% threshold:")
    for metric, k in [("  Sensitivity","sens_8"),("  Specificity","spec_8"),("  PPV","ppv_8"),("  NPV","npv_8")]:
        row = f"{metric:<22}"
        for r in results:
            row += f"  {r[k]:>14.3f}"
        print(row)
    print()
    print("@ Youden threshold:")
    for metric, k in [("  Threshold","youden_thresh"),("  Sensitivity","sens_y"),("  Specificity","spec_y"),("  PPV","ppv_y"),("  NPV","npv_y")]:
        row = f"{metric:<22}"
        for r in results:
            v = r[k]
            row += f"  {v:>14.3f}"
        print(row)


def run():
    print("Loading data...")
    prepared = prepare_master_dataset(XLSX)
    df = prepared.data
    fc_all = prepared.feature_columns
    y = df["morte_30d"].astype(int).values
    print(f"  n={len(df)}, events={y.sum()}, features={len(fc_all)}")

    # Resolve actual column names for ablation blocks
    a1_cols = match_cols(fc_all, ABLATION_1_PATTERNS)
    # Vena contracta (mm) matched via "Vena contracta" already — need to be careful
    # Re-resolve to get unique, correct matches
    a1_cols_resolved = []
    for pat in ABLATION_1_PATTERNS:
        hits = [c for c in fc_all if pat.lower() in c.lower() and c not in a1_cols_resolved]
        if hits:
            a1_cols_resolved.append(hits[0])

    a2_cols = [c for c in fc_all if "kdigo" in c.lower()]

    print(f"\nAblation 1 targets ({len(a1_cols_resolved)} cols):")
    for c in a1_cols_resolved:
        s = df[c]
        is_miss = s.isna() | s.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)
        safe = c.encode("ascii","replace").decode()
        print(f"  {safe:<42} {is_miss.mean():.1%} missing")

    print(f"\nAblation 2 targets ({len(a2_cols)} cols):")
    for c in a2_cols:
        s = df[c]
        is_miss = s.isna() | s.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)
        safe = c.encode("ascii","replace").decode()
        print(f"  {safe:<42} {is_miss.mean():.1%} missing  (Spearman rho=-0.953 with CrClearance)")

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE — all 61 features")
    print("=" * 60)
    res_base, arts_base = run_variant("Baseline", df, fc_all, y)
    print_result(res_base)

    # ── Ablation 1: remove 6 high-sparsity echo features ─────────────────────
    print("\n" + "=" * 60)
    print(f"ABLATION 1 — remove {len(a1_cols_resolved)} sparse echo features")
    print("=" * 60)
    fc_a1 = [c for c in fc_all if c not in a1_cols_resolved]
    res_a1, arts_a1 = run_variant("Ablation1", df, fc_a1, y)
    print_result(res_a1)

    # ── Ablation 2: remove KDIGO only ────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"ABLATION 2 — remove KDIGO only")
    print("=" * 60)
    fc_a2 = [c for c in fc_all if c not in a2_cols]
    res_a2, arts_a2 = run_variant("Ablation2", df, fc_a2, y)
    print_result(res_a2)

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print_comparison([res_base, res_a1, res_a2])

    # ── Leaderboards ─────────────────────────────────────────────────────────
    for label, arts in [("Baseline", arts_base), ("Ablation1", arts_a1), ("Ablation2", arts_a2)]:
        print(f"\nLeaderboard [{label}]:")
        lb = arts.leaderboard[["Modelo","AUC","AUPRC","Brier"]].copy()
        lb.columns = ["Model","AUC","AUPRC","Brier"]
        print(lb.to_string(index=False))

    # ── Feature importance of ablated cols in baseline ────────────────────────
    print("\n=== Baseline RF importance of ablation candidates ===")
    try:
        model_obj = arts_base.model
        inner = getattr(model_obj, "_pipeline", model_obj)
        inner = getattr(inner, "estimator", inner)
        inner = getattr(inner, "calibrated_classifiers_", [inner])[0]
        inner = getattr(inner, "estimator", inner)
        est = inner
        if hasattr(est, "named_steps"):
            est = est.named_steps.get("model", est)
        fi = getattr(est, "feature_importances_", None)
        base_cols = arts_base.feature_columns
        if fi is not None and len(fi) == len(base_cols):
            imp = pd.Series(fi, index=base_cols).sort_values(ascending=False)
            all_a_cols = a1_cols_resolved + a2_cols
            for c in all_a_cols:
                if c in imp.index:
                    rank = imp.index.tolist().index(c) + 1
                    safe = c.encode("ascii","replace").decode()
                    print(f"  {safe:<42} imp={imp[c]:.6f}  rank={rank}/{len(imp)}")
        else:
            print("  (importance not available for this model)")
    except Exception as e:
        print(f"  (error: {e})")

    return res_base, res_a1, res_a2


if __name__ == "__main__":
    run()
