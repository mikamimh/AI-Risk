"""Regenera o bundle oficial do AI Risk com o pipeline atual corrigido.

Standalone script — modifica ia_risk_bundle.joblib (acao intencional).
Executar: python regenerate_bundle.py

O que este script faz:
  1. Carrega e prepara os dados com o pipeline atual
  2. Treina e seleciona o melhor modelo (mesma logica do app.py)
  3. Calcula metricas completas (OOF)
  4. Salva o novo bundle em ia_risk_bundle.joblib
  5. Compara com o bundle anterior (se existir)

Mudancas metodologicas ja incorporadas neste baseline:
  - NEVER_FEATURE_COLUMNS (leakage temporal + pós-op + scores comparadores + metadados)
  - procedure_group removido de treino/inferencia (ablação confirmou: atrapalha)
  - BLANK_MEANS_NONE_COLUMNS (Previous surgery, HF, Arrhythmia Remote, Aortic Stenosis)
  - normalize_previous_surgery_value ativo no path flat e master
  - Filtro de 95% missingness (validado empiricamente: PHT Aortic rank 4/61)
  - 61 features ativas

Baseline de referencia anterior (bundle v12, CSV flat path):
  RandomForest: AUC=0.7454, AUPRC=0.3400, Brier=0.1149  Youden=0.0936
"""
import sys, warnings, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, ".")
from risk_data import prepare_master_dataset
from modeling import train_and_select_model
from stats_compare import calibration_intercept_slope, classification_metrics_at_threshold
from bundle_io import bundle_signature, serialize_bundle, BUNDLE_SCHEMA_VERSION
from config import AppConfig

XLSX      = "local_data/Dataset_2025.xlsx"
BUNDLE    = AppConfig.MODEL_CACHE_FILE        # ia_risk_bundle.joblib
THRESHOLD = 0.08

# Numeros de referencia (bundle v12, treinado do CSV flat, sem garantia de semantica)
V12_REF = {
    "auc":           0.7454,
    "auprc":         0.3400,
    "brier":         0.1149,
    "youden_thresh": 0.0936,
    "n_features":    61,
    "model_version": "2026-03-29-v12-calibrated-oof",
    "source_type":   "flat (CSV)",
}


def compute_full_metrics(y, proba, youden_thresh):
    auc   = roc_auc_score(y, proba)
    auprc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    cal   = calibration_intercept_slope(y, proba)
    m8    = classification_metrics_at_threshold(np.asarray(y), np.asarray(proba), THRESHOLD)
    my    = classification_metrics_at_threshold(np.asarray(y), np.asarray(proba), youden_thresh)
    return {
        "auc": auc, "auprc": auprc, "brier": brier,
        "cal_intercept": cal["Calibration intercept"],
        "cal_slope":     cal["Calibration slope"],
        "sens_8": m8["Sensitivity"], "spec_8": m8["Specificity"],
        "ppv_8":  m8["PPV"],         "npv_8":  m8["NPV"],
        "youden_thresh": youden_thresh,
        "sens_y": my["Sensitivity"], "spec_y": my["Specificity"],
        "ppv_y":  my["PPV"],         "npv_y":  my["NPV"],
    }


def run():
    print("=" * 70)
    print("REGENERACAO DO BUNDLE AI RISK — baseline v13")
    print(f"MODEL_VERSION: {AppConfig.MODEL_VERSION}")
    print("=" * 70)

    # ── Inspecao do bundle anterior ───────────────────────────────────────────
    print("\n[0] Bundle anterior:")
    if BUNDLE.exists():
        try:
            old_payload = joblib.load(BUNDLE)
            old_sig = old_payload.get("signature", {})
            old_arts = old_payload.get("bundle", {}).get("artifacts", {})
            old_lb = old_arts.get("leaderboard")
            old_version = old_sig.get("model_version", "?")
            old_saved  = old_payload.get("saved_at", "?")
            old_source = old_payload.get("training_source", "?")
            print(f"  Versao:  {old_version}")
            print(f"  Salvo:   {old_saved}")
            print(f"  Fonte:   {old_source}")
            if old_lb is not None:
                lb_df = pd.DataFrame(old_lb) if isinstance(old_lb, list) else old_lb
                cols = [c for c in ["Modelo", "AUC", "AUPRC", "Brier"] if c in lb_df.columns]
                if cols:
                    print("  Leaderboard anterior:")
                    for _, row in lb_df[cols].iterrows():
                        modelo = str(row.get("Modelo", "?"))
                        print(f"    {modelo:<22} AUC={row.get('AUC',0):.4f} AUPRC={row.get('AUPRC',0):.4f} Brier={row.get('Brier',0):.4f}")
        except Exception as e:
            print(f"  (nao foi possivel ler: {e})")
    else:
        print("  (bundle nao encontrado)")

    # ── Preparacao dos dados ──────────────────────────────────────────────────
    print("\n[1] Preparando dados com pipeline atual...")
    prepared = prepare_master_dataset(XLSX)
    df       = prepared.data.copy()
    fc       = list(prepared.feature_columns)
    y        = df["morte_30d"].astype(int).values
    info     = prepared.info

    print(f"  n={len(df)}, events={y.sum()} ({y.mean():.1%}), features={len(fc)}")
    print(f"  source_type: {info.get('source_type', '?')}")
    nf = info.get("never_feature_intercepted", [])
    nf_in_data = info.get("never_feature_columns_in_source", [])
    print(f"  never_feature_columns_in_source: {len(nf_in_data)} colunas")
    print(f"  never_feature_intercepted: {len(nf)} (deve ser 0)")
    if nf:
        print(f"  ALERTA: {nf}")

    # Validacoes pre-treino
    assert "procedure_group" not in fc, "procedure_group NAO deve estar em features!"
    assert "morte_30d" not in fc, "morte_30d NAO deve estar em features!"
    assert "sts_score" not in fc, "sts_score NAO deve estar em features!"
    assert "EuroSCORE II" not in fc, "EuroSCORE II NAO deve estar em features!"
    assert nf == [], f"Never-feature interceptado: {nf}"
    print("  Pre-validacoes: OK")

    # Auditoria semantica de Previous surgery
    if "Previous surgery" in df.columns:
        s = df["Previous surgery"]
        n_none = int((s == "None").sum())
        n_redo = int(s.notna().sum()) - n_none - int((s == "No").sum()) - int((s == "Yes").sum())
        print(f"  Previous surgery: None={n_none} ({n_none/len(s):.1%}), redo={n_redo} ({n_redo/len(s):.1%})")
        # Confirmar que "No" nao existe (foi canonicalizado para "None")
        n_raw_no = int((s == "No").sum())
        if n_raw_no > 0:
            print(f"  AVISO: {n_raw_no} linhas com valor 'No' (esperado: canonicalizado para 'None')")
        else:
            print("  Semantica None: OK (nenhum 'No' residual)")

    # ── Treino ────────────────────────────────────────────────────────────────
    print("\n[2] Treinando modelo (pipeline atual)...")
    arts = train_and_select_model(df, fc)
    best = arts.best_model_name
    proba = arts.oof_predictions[best]
    youden = arts.best_youden_threshold

    print(f"  Melhor modelo: {best}")
    print("\n  Leaderboard completo:")
    lb = arts.leaderboard[["Modelo", "AUC", "AUPRC", "Brier"]].copy()
    for _, row in lb.iterrows():
        print(f"    {str(row['Modelo']):<22} AUC={row['AUC']:.4f} AUPRC={row['AUPRC']:.4f} Brier={row['Brier']:.4f}")

    # ── Metricas completas ────────────────────────────────────────────────────
    print("\n[3] Metricas completas do novo baseline:")
    m = compute_full_metrics(y, proba, youden)
    print(f"  AUC          : {m['auc']:.4f}")
    print(f"  AUPRC        : {m['auprc']:.4f}")
    print(f"  Brier        : {m['brier']:.4f}")
    print(f"  Cal intercept: {m['cal_intercept']:.4f}")
    print(f"  Cal slope    : {m['cal_slope']:.4f}")
    print(f"  @8%  Sens:{m['sens_8']:.3f} Spec:{m['spec_8']:.3f} PPV:{m['ppv_8']:.3f} NPV:{m['npv_8']:.3f}")
    print(f"  Youden t:{m['youden_thresh']:.3f} Sens:{m['sens_y']:.3f} Spec:{m['spec_y']:.3f} PPV:{m['ppv_y']:.3f} NPV:{m['npv_y']:.3f}")

    # ── Comparativo v12 vs v13 ────────────────────────────────────────────────
    print("\n[4] Comparativo baseline v12 vs v13:")
    keys = ["auc", "auprc", "brier"]
    labels = ["AUC", "AUPRC", "Brier"]
    ref = V12_REF
    print(f"  {'Metric':<16} {'v12 (CSV)':>12} {'v13 (XLSX)':>12} {'D':>8}")
    print("  " + "-" * 50)
    for k, lab in zip(keys, labels):
        d = m[k] - ref[k]
        print(f"  {lab:<16} {ref[k]:>12.4f} {m[k]:>12.4f} {d:>+8.4f}")
    print(f"  {'Cal intercept':<16} {'---':>12} {m['cal_intercept']:>12.4f} {'---':>8}")
    print(f"  {'Cal slope':<16} {'---':>12} {m['cal_slope']:>12.4f} {'---':>8}")
    print(f"  {'Youden thresh':<16} {ref['youden_thresh']:>12.4f} {m['youden_thresh']:>12.4f} {m['youden_thresh']-ref['youden_thresh']:>+8.4f}")
    print(f"  {'N features':<16} {ref['n_features']:>12} {len(fc):>12} {len(fc)-ref['n_features']:>+8}")
    print(f"  {'Source':<16} {ref['source_type']:>12} {'master(XLSX)':>12}")

    # ── Construcao e salvamento do bundle ─────────────────────────────────────
    print("\n[5] Construindo e salvando bundle...")

    sig = bundle_signature(XLSX)
    print(f"  Assinatura XLSX: mtime_ns={sig['xlsx_mtime_ns']}, size={sig['xlsx_size']}")
    print(f"  MODEL_VERSION: {sig['model_version']}")

    bundle_dict = {
        "prepared": prepared,
        "artifacts": arts,
        "data": df,
    }
    serialized = serialize_bundle(bundle_dict)

    saved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "signature": sig,
        "bundle": serialized,
        "saved_at": saved_at,
        "training_source": Path(XLSX).name,
    }
    joblib.dump(payload, BUNDLE)
    bundle_size_mb = BUNDLE.stat().st_size / 1024 / 1024
    print(f"  Salvo: {BUNDLE}  ({bundle_size_mb:.1f} MB)")
    print(f"  saved_at: {saved_at}")

    # ── Validacao pos-salvamento ──────────────────────────────────────────────
    print("\n[6] Validacao pos-salvamento...")
    reloaded = joblib.load(BUNDLE)
    r_sig = reloaded.get("signature", {})
    r_arts = reloaded.get("bundle", {}).get("artifacts", {})
    r_fc   = r_arts.get("feature_columns", [])
    assert r_sig.get("model_version") == AppConfig.MODEL_VERSION, "MODEL_VERSION mismatch!"
    assert r_sig.get("xlsx_path") == str(Path(XLSX).resolve()), "path mismatch!"
    assert len(r_fc) == len(fc), f"feature_columns count mismatch: {len(r_fc)} vs {len(fc)}"
    assert "procedure_group" not in r_fc, "procedure_group vazou para features!"
    assert "morte_30d" not in r_fc, "morte_30d vazou para features!"
    assert r_arts.get("best_model_name") == best, "best_model mismatch!"
    print("  Assinatura: OK")
    print("  Features count: OK")
    print("  Leakage check: OK")
    print("  Best model: OK")
    print("\nBundle v13 salvo e validado com sucesso.")

    return m, fc


if __name__ == "__main__":
    run()
