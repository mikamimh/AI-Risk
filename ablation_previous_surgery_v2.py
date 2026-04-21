"""Ablation v2: Previous surgery — baseline com semantica None corrigida.

Standalone script — NAO modifica nenhum arquivo de producao.
Executar: python ablation_previous_surgery_v2.py

Diferenca em relacao a v1 (ablation_previous_surgery.py):
  - Baseline agora usa pipeline com BLANK_MEANS_NONE_COLUMNS e
    _normalize_previous_surgery_column ativo nos paths flat/master.
    "No" e blank -> "None"; textos de redo permanecem inalterados.
  - Binarizacao usa "None"/"Yes" (vocabulary canonico) para consistencia
    com normalize_previous_surgery_value na inferencia individual.
  - Mostra comparativo com numeros da ablacao anterior (v1).

Resultados de referencia (ablacao v1, baseline antigo):
  AUC  baseline=0.7510  binario=0.7540  D=+0.0030
  AUPRC baseline=0.3376 binario=0.3399  D=+0.0023
  Brier baseline=0.1150 binario=0.1148  D=-0.0002
  Cal intercept baseline=0.0640 binario=0.0952 (pior)
  Cal slope     baseline=1.0281 binario=1.0468 (pior)
  Conclusao v1: INCONCLUSIVO (diferenca menor que ruido de CV)
"""
import sys, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, ".")
from risk_data import prepare_master_dataset, MISSING_TOKENS, normalize_previous_surgery_value
from modeling import train_and_select_model
from stats_compare import calibration_intercept_slope, classification_metrics_at_threshold

XLSX = "local_data/Dataset_2025.xlsx"
THRESHOLD_CLINICAL = 0.08
COL = "Previous surgery"

# Numeros da ablacao v1 (baseline antigo, pre-correcao semantica)
V1_REF = {
    "Baseline": {"auc": 0.7510, "auprc": 0.3376, "brier": 0.1150,
                 "cal_intercept": 0.0640, "cal_slope": 1.0281},
    "Binary Yes/No": {"auc": 0.7540, "auprc": 0.3399, "brier": 0.1148,
                      "cal_intercept": 0.0952, "cal_slope": 1.0468},
}


# ── Binarizacao com semantica canonicada ────────────────────────────────────
# Saida: "None" (ausencia) ou "Yes" (presenca de cirugia previa)
# "None" e "Yes" sao os tokens canonicos do pipeline atual, alinhados com
# normalize_previous_surgery_value — garante consistencia treino/inferencia.
#
# Mapeamento:
#   NaN           -> "None"  (dado ausente = ausencia implicita)
#   MISSING_TOKEN -> "None"  (inclui "none", "-", etc.)
#   "no" / "No"   -> "None"  (ausencia explicita, belt-and-suspenders)
#   "None"        -> "None"  (ja canonico)
#   "Yes"         -> "Yes"   (presenca explicita — dado via form de inferencia)
#   qualquer texto de redo (AVR, CABG...) -> "Yes"

def binarize_previous_surgery_canonical(series: pd.Series) -> pd.Series:
    def _map(v):
        if pd.isna(v):
            return "None"
        txt = str(v).strip()
        lower = txt.lower()
        if not lower:
            return "None"
        if lower in MISSING_TOKENS:     # inclui "none"
            return "None"
        if lower == "no":               # belt-and-suspenders (apos normalize ja sera "None")
            return "None"
        return "Yes"                    # qualquer texto de redo ou "Yes" explicito
    return series.map(_map)


# ── Metricas ────────────────────────────────────────────────────────────────

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


def print_v1_comparison(results):
    """Compara metricas desta rodada com os numeros da ablacao v1."""
    keys   = ["auc", "auprc", "brier", "cal_intercept", "cal_slope"]
    labels = ["AUC", "AUPRC", "Brier", "Cal intercept", "Cal slope"]
    mapping = {"Baseline (None)": "Baseline", "Binary None/Yes": "Binary Yes/No"}
    print(f"\n{'Metric':<22}  {'Baseline v1':>12}  {'Baseline v2':>12}  {'D(v2-v1)':>10}  {'Binary v1':>12}  {'Binary v2':>12}  {'D(v2-v1)':>10}")
    print("-" * 100)
    v1_b  = V1_REF["Baseline"]
    v1_bi = V1_REF["Binary Yes/No"]
    r_b   = next((r for r in results if "baseline" in r["variant"].lower()), None)
    r_bi  = next((r for r in results if "binary" in r["variant"].lower()), None)
    if not r_b or not r_bi:
        print("  (nao foi possivel comparar)")
        return
    for k, lab in zip(keys, labels):
        v2b  = r_b[k];  v2bi = r_bi[k]
        db   = v2b  - v1_b[k]
        dbi  = v2bi - v1_bi[k]
        print(f"{lab:<22}  {v1_b[k]:>12.4f}  {v2b:>12.4f}  {db:>+10.4f}  {v1_bi[k]:>12.4f}  {v2bi:>12.4f}  {dbi:>+10.4f}")


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
    print("=" * 70)
    print("REABLACAO v2 — Previous surgery (baseline com semantica None corrigida)")
    print("=" * 70)

    print("\nLoading data...")
    prepared = prepare_master_dataset(XLSX)
    df_orig  = prepared.data.copy()
    fc_all   = list(prepared.feature_columns)
    y        = df_orig["morte_30d"].astype(int).values
    print(f"  n={len(df_orig)}, events={y.sum()}, features={len(fc_all)}")

    # ── Parte A: Diagnostico da coluna atual ─────────────────────────────────
    print("\n" + "=" * 70)
    print("PARTE A — Diagnostico de 'Previous surgery' no baseline v2")
    print("=" * 70)

    s = df_orig[COL]
    vc = s.value_counts(dropna=False)
    print(f"\nDistribuicao atual (pos-normalizacao, n={len(s)}):")
    for val, cnt in vc.items():
        label = repr(val) if pd.isna(val) else f'"{val}"'
        mort  = y[s == val].mean() if not pd.isna(val) and (s == val).sum() > 0 else float("nan")
        mort_str = f"  mort={mort:.1%}" if not np.isnan(mort) else ""
        print(f"  {label:<30} n={cnt:>4}{mort_str}")

    n_absence = int((s == "None").sum())
    n_redo    = int(s.notna().sum()) - n_absence - int((s == "No").sum()) - int((s == "Yes").sum())
    n_yes_explicit = int((s == "Yes").sum())
    print(f"\n  Ausencia (\"None\"): {n_absence}")
    print(f"  Texto redo (outros): {n_redo}")
    print(f"  \"Yes\" explicito: {n_yes_explicit}")
    print(f"  NaN (missing): {int(s.isna().sum())}")

    # Binarizacao diagnostica
    binary = binarize_previous_surgery_canonical(s)
    n_none_bin = int((binary == "None").sum())
    n_yes_bin  = int((binary == "Yes").sum())
    print(f"\n  Binarizacao (None/Yes): None={n_none_bin} ({n_none_bin/len(s):.1%}), Yes={n_yes_bin} ({n_yes_bin/len(s):.1%})")

    mort_absence = y[binary == "None"].mean()
    mort_redo    = y[binary == "Yes"].mean()
    print(f"  Mortalidade None (ausencia): {mort_absence:.1%}")
    print(f"  Mortalidade Yes (redo):      {mort_redo:.1%}  (delta +{mort_redo - mort_absence:.1%})")

    print("\n  Inferencia individual (app.py -> normalize_previous_surgery_value):")
    print("    Sem cirugia previa: app envia 'No' -> normalize -> 'None' -> categoria VISTA no treino (v2) [OK]")
    print("    Com cirugia previa: app envia 'Yes' -> normalize -> 'Yes'")
    print("    No baseline v2: 'Yes' NAO e categoria de treino (treino tem textos redo) -> TargetEncoder usa media global [MISMATCH]")
    print("    Na ablacao binaria: 'Yes' SERIA categoria de treino -> consistencia total [FIX]")

    # ── Baseline v2 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE v2 — Previous surgery canonico (None / textos redo)")
    print("=" * 70)
    arts_base = train_and_select_model(df_orig, fc_all)
    proba_base = arts_base.oof_predictions[arts_base.best_model_name]
    res_base = full_metrics("Baseline (None)", arts_base.best_model_name,
                            len(fc_all), y, proba_base,
                            arts_base.best_youden_threshold)
    print_result(res_base)

    imp_val, imp_rank, imp_total = get_importance(arts_base, COL)
    if imp_val is not None:
        print(f"  '{COL}' importancia: {imp_val:.6f}  rank {imp_rank}/{imp_total}")

    # ── Ablacao binaria None/Yes ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ABLACAO — Previous surgery binarizado (None / Yes)")
    print("=" * 70)
    df_bin = df_orig.copy()
    df_bin[COL] = binarize_previous_surgery_canonical(df_bin[COL])
    vc_bin = df_bin[COL].value_counts()
    print(f"  Valores pos-binarizacao: {vc_bin.to_dict()}")

    arts_abl = train_and_select_model(df_bin, fc_all)
    proba_abl = arts_abl.oof_predictions[arts_abl.best_model_name]
    res_abl = full_metrics("Binary None/Yes", arts_abl.best_model_name,
                           len(fc_all), y, proba_abl,
                           arts_abl.best_youden_threshold)
    print_result(res_abl)

    imp_val2, imp_rank2, imp_total2 = get_importance(arts_abl, COL)
    if imp_val2 is not None:
        print(f"  '{COL}' importancia: {imp_val2:.6f}  rank {imp_rank2}/{imp_total2}")

    # ── Comparativo v2 ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARATIVO v2  (D = Ablacao - Baseline)")
    print("=" * 70)
    print_comparison([res_base, res_abl])

    # ── Comparativo com v1 ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARATIVO v2 vs v1  (mudanca do baseline)")
    print("=" * 70)
    print_v1_comparison([res_base, res_abl])

    # ── Leaderboards ─────────────────────────────────────────────────────────
    print("\nLeaderboard [Baseline v2]:")
    lb = arts_base.leaderboard[["Modelo","AUC","AUPRC","Brier"]].copy()
    lb.columns = ["Model","AUC","AUPRC","Brier"]
    print(lb.to_string(index=False))

    print("\nLeaderboard [Binary v2]:")
    lb2 = arts_abl.leaderboard[["Modelo","AUC","AUPRC","Brier"]].copy()
    lb2.columns = ["Model","AUC","AUPRC","Brier"]
    print(lb2.to_string(index=False))

    return res_base, res_abl


if __name__ == "__main__":
    run()
