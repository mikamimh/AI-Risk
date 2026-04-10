"""Model metadata, audit trail, and reporting for AI Risk.

Provides:
- Structured model version metadata (bundle info)
- Analysis audit trail logging
- Individual patient report generation
- Input completeness assessment
- Statistical summary export
- Validation readiness helpers
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import AppConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_QUARTER_TO_MONTH = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
_QUARTER_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def _extract_year_quarter_range(data: pd.DataFrame) -> tuple:
    """Return (start, end) strings from surgery_year / surgery_quarter columns.

    Format: ``"2024-Q1"`` or ``"Unknown"`` when columns are absent.
    """
    if "surgery_year" not in data.columns:
        return ("Unknown", "Unknown")

    years = pd.to_numeric(data["surgery_year"], errors="coerce").dropna()
    if years.empty:
        return ("Unknown", "Unknown")

    if "surgery_quarter" in data.columns:
        temp = data[["surgery_year", "surgery_quarter"]].dropna()
        temp = temp.copy()
        temp["_y"] = pd.to_numeric(temp["surgery_year"], errors="coerce")
        temp["_q"] = temp["surgery_quarter"].map(_QUARTER_ORDER)
        temp = temp.dropna(subset=["_y", "_q"])
        if not temp.empty:
            temp["_sort"] = temp["_y"] * 10 + temp["_q"]
            earliest = temp.loc[temp["_sort"].idxmin()]
            latest = temp.loc[temp["_sort"].idxmax()]
            start = f"{int(earliest['_y'])}-{earliest['surgery_quarter']}"
            end = f"{int(latest['_y'])}-{latest['surgery_quarter']}"
            return (start, end)

    # Fallback: year only
    return (str(int(years.min())), str(int(years.max())))


def _yq_to_timestamp(yq: str) -> Optional[pd.Timestamp]:
    """Convert ``'2024-Q2'`` or ``'2024'`` to a pd.Timestamp (start of period)."""
    if not yq or yq == "Unknown":
        return None
    try:
        if "-Q" in yq:
            parts = yq.split("-Q")
            year = int(parts[0])
            month = _QUARTER_TO_MONTH.get(f"Q{parts[1]}", 1)
            return pd.Timestamp(year=year, month=month, day=1)
        return pd.Timestamp(year=int(yq), month=1, day=1)
    except Exception:
        return None


def _yq_to_end_timestamp(yq: str) -> Optional[pd.Timestamp]:
    """Convert ``'2024-Q2'`` to end-of-quarter Timestamp."""
    if not yq or yq == "Unknown":
        return None
    try:
        if "-Q" in yq:
            parts = yq.split("-Q")
            year = int(parts[0])
            q = int(parts[1])
            # End of quarter: Q1→Mar31, Q2→Jun30, Q3→Sep30, Q4→Dec31
            end_months = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
            m, d = end_months.get(q, (12, 31))
            return pd.Timestamp(year=year, month=m, day=d)
        return pd.Timestamp(year=int(yq), month=12, day=31)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model version metadata
# ---------------------------------------------------------------------------

def build_model_metadata(
    prepared_info: dict,
    artifacts_leaderboard: pd.DataFrame,
    best_model_name: str,
    feature_columns: list,
    xlsx_path: str,
    sts_available: bool = False,
    bundle_saved_at: str = None,
    training_source_file: str = None,
    calibration_method: str = "sigmoid",
    training_data: pd.DataFrame = None,
) -> dict:
    """Build structured metadata about the current model bundle.

    Args:
        bundle_saved_at: ISO timestamp of when the bundle was actually saved to disk.
        training_source_file: filename used when the bundle was trained.
        training_data: prepared DataFrame — used to extract temporal range
            from ``surgery_year`` and ``surgery_quarter`` columns.
    """
    current_analysis_file = Path(xlsx_path).name

    # Extract training temporal range from surgery_year / surgery_quarter
    training_start_date = "Unknown"
    training_end_date = "Unknown"
    if training_data is not None:
        _range = _extract_year_quarter_range(training_data)
        training_start_date = _range[0]
        training_end_date = _range[1]

    n_patients = int(prepared_info.get("n_rows", 0))
    event_rate = float(prepared_info.get("positive_rate", 0))

    return {
        "model_version": AppConfig.MODEL_VERSION,
        "bundle_saved_at": bundle_saved_at or "Unknown",
        "training_source_file": training_source_file or current_analysis_file,
        "current_analysis_file": current_analysis_file,
        "metadata_generated_at": datetime.now(timezone.utc).isoformat(),
        "n_patients": n_patients,
        "n_events": int(round(n_patients * event_rate)),
        "event_rate": event_rate,
        "n_features": int(prepared_info.get("n_features", 0)),
        "feature_columns": list(feature_columns),
        "best_model": best_model_name,
        "candidate_models": artifacts_leaderboard["Modelo"].tolist() if not artifacts_leaderboard.empty else [],
        "cv_strategy": AppConfig.CV_STRATEGY,
        "cv_splits": AppConfig.CV_SPLITS,
        "random_seed": AppConfig.RANDOM_SEED,
        "preprocessing": {
            "numeric": "median imputation + StandardScaler",
            "valve_severity": "OrdinalEncoder (None<Trivial<Mild<Moderate<Severe) + median imputation + StandardScaler",
            "categorical": "mode imputation + TargetEncoder (smooth=auto) + median post-imputation",
            "probability_clipping": "numerical-stability epsilon only (1e-6)",
        },
        "calibration": {
            "method": f"{calibration_method} (Platt scaling)" if calibration_method == "sigmoid" else calibration_method,
            "applied_to": "tree-based models (RandomForest, XGBoost, LightGBM, CatBoost)",
            "oof_evaluation": "calibrated inside each CV fold (inner cv≤3, StratifiedKFold)",
            "grouping_note": "inner calibration CV does not enforce patient grouping (sklearn limitation)",
        },
        "thresholds": {
            "leaderboard": "Youden's J (optimal per model, on calibrated OOF)",
            "clinical_default": "8%",
        },
        "oof_used_in_leaderboard": "calibrated (Platt-scaled for tree models, raw for others)",
        "source_type": Path(xlsx_path).suffix.lstrip("."),
        "sts_method": "Automated query to the STS web calculator (acsdriskcalc.research.sts.org)" if sts_available else "Not available",
        "euroscore_method": "Published logistic equation (Nashef et al., 2012)",
        # Temporal validation fields
        "training_start_date": training_start_date,
        "training_end_date": training_end_date,
        "locked_for_temporal_validation": True,
        "locked_threshold": 0.08,
    }


def format_metadata_for_display(metadata: dict, language: str = "English") -> pd.DataFrame:
    """Convert metadata dict to a display-friendly DataFrame."""
    def _tr(en, pt):
        return en if language == "English" else pt

    def _fmt_ts(ts):
        if not ts or ts == "Unknown":
            return "Unknown"
        return ts[:19].replace("T", " ")

    # Training date range
    t_start = metadata.get("training_start_date", "Unknown")
    t_end = metadata.get("training_end_date", "Unknown")
    if t_start != "Unknown" and t_end != "Unknown":
        date_range = f"{t_start} — {t_end}"
    else:
        date_range = "Unknown"

    calib = metadata.get("calibration", {})
    calib_method = calib.get("method", "N/A") if isinstance(calib, dict) else str(calib)
    locked_thr = metadata.get("locked_threshold")
    locked_str = f"{locked_thr:.0%}" if locked_thr is not None else "8%"
    locked_status = _tr("Yes", "Sim") if metadata.get("locked_for_temporal_validation") else _tr("No", "Não")

    rows = [
        (_tr("Model version", "Versão do modelo"), metadata.get("model_version", "N/A")),
        (_tr("Bundle saved at", "Bundle salvo em"), _fmt_ts(metadata.get("bundle_saved_at"))),
        (_tr("Training data source", "Fonte de dados do treino"), metadata.get("training_source_file", "N/A")),
        (_tr("Current analysis file", "Arquivo de análise atual"), metadata.get("current_analysis_file", "N/A")),
        (_tr("Patients in training", "Pacientes no treinamento"), str(metadata.get("n_patients", "N/A"))),
        (_tr("Deaths in training (primary outcome)", "Óbitos no treinamento (desfecho primário)"), str(metadata.get("n_events", "N/A"))),
        (_tr("Mortality rate", "Taxa de mortalidade"), f"{metadata.get('event_rate', 0):.1%}"),
        (_tr("Number of features", "Número de variáveis"), str(metadata.get("n_features", "N/A"))),
        (_tr("Best model", "Melhor modelo"), metadata.get("best_model", "N/A")),
        (_tr("CV strategy", "Estratégia de CV"), metadata.get("cv_strategy", "N/A")),
        (_tr("CV splits", "Folds de CV"), str(metadata.get("cv_splits", "N/A"))),
        (_tr("Calibration method", "Método de calibração"), calib_method),
        (_tr("Training date range", "Período do treinamento"), date_range),
        (_tr("Locked clinical threshold", "Limiar clínico bloqueado"), locked_str),
        (_tr("Locked for temporal validation", "Bloqueado para validação temporal"), locked_status),
        (_tr("STS method", "Método STS"), metadata.get("sts_method", "N/A")),
        (_tr("EuroSCORE II method", "Método EuroSCORE II"), metadata.get("euroscore_method", "N/A")),
    ]
    return pd.DataFrame(rows, columns=[_tr("Property", "Propriedade"), _tr("Value", "Valor")])


# ---------------------------------------------------------------------------
# Input completeness assessment
# ---------------------------------------------------------------------------

# Variables with highest clinical relevance for risk prediction
HIGH_RELEVANCE_VARIABLES = {
    "Age (years)", "Sex", "LVEF, %", "Creatinine (mg/dL)", "Cr clearance, ml/min *",
    "Surgery", "Surgical Priority", "Diabetes", "Dialysis", "Previous surgery",
    "Preoperative NYHA", "Hematocrit (%)",
}

MODERATE_RELEVANCE_VARIABLES = {
    "Height (cm)", "Weight (kg)", "PSAP", "TAPSE",
    "IE", "PVD", "CVA", "Chronic Lung Disease", "HF",
    "No. of Diseased Vessels", "Coronary Symptom",
    "Aortic Stenosis", "Mitral Regurgitation", "Tricuspid Regurgitation",
    "WBC Count (10³/μL)", "Platelet Count (cells/μL)",
}

# Detailed echo measurements — optional in the individual form.
# Their absence should not penalize the completeness indicator.
OPTIONAL_DETAILED_VARIABLES = {
    "AVA (cm²)", "MVA (cm²)",
    "Aortic Mean gradient (mmHg)", "Mitral Mean gradient (mmHg)",
    "PHT Aortic", "PHT Mitral",
    "Vena contracta", "Vena contracta (mm)",
}


def assess_input_completeness(
    feature_columns: list,
    input_row: pd.DataFrame,
    language: str = "English",
) -> dict:
    """Assess the completeness and reliability of a patient's input data.

    Returns:
        dict with keys: level, label, color, n_total, n_informed, n_imputed,
                        n_high_missing, n_moderate_missing, missing_high, missing_moderate, missing_other
    """
    def _tr(en, pt):
        return en if language == "English" else pt

    n_total = len(feature_columns)
    informed_mask = input_row[feature_columns].notna().iloc[0]
    n_informed = int(informed_mask.sum())
    n_imputed = n_total - n_informed

    missing_cols = [c for c in feature_columns if not informed_mask[c]]
    missing_high = [c for c in missing_cols if c in HIGH_RELEVANCE_VARIABLES]
    missing_moderate = [c for c in missing_cols if c in MODERATE_RELEVANCE_VARIABLES]
    missing_optional = [c for c in missing_cols if c in OPTIONAL_DETAILED_VARIABLES]
    missing_other = [c for c in missing_cols if c not in HIGH_RELEVANCE_VARIABLES and c not in MODERATE_RELEVANCE_VARIABLES]

    n_high_missing = len(missing_high)
    n_moderate_missing = len(missing_moderate)

    # For classification, exclude optional detailed measurements from the imputation count.
    # These are echo measurements (AVA, MVA, PHT, gradients, Vena contracta) that are rarely
    # available in a basic clinical form — their absence should not penalize the indicator.
    n_imputed_for_classification = n_imputed - len(missing_optional)

    # Classification logic — conservative: moderate-relevance variables also count
    if n_high_missing == 0 and n_moderate_missing == 0 and n_imputed_for_classification <= 3:
        level = "complete"
        label = _tr("Complete data", "Dados completos")
        color = "green"
    elif n_high_missing == 0 and n_moderate_missing <= 2 and n_imputed_for_classification <= 10:
        level = "adequate"
        label = _tr("Adequate — minor imputation", "Adequada — imputação menor")
        color = "yellow"
    elif n_high_missing <= 1 and n_imputed_for_classification <= 20:
        level = "partial"
        label = _tr("Partially imputed — interpret with caution", "Parcialmente imputado — interpretar com cautela")
        color = "orange"
    else:
        level = "low"
        label = _tr("Heavily imputed — low reliability", "Muito imputado — baixa confiabilidade")
        color = "red"

    return {
        "level": level,
        "label": label,
        "color": color,
        "n_total": n_total,
        "n_informed": n_informed,
        "n_imputed": n_imputed,
        "n_high_missing": n_high_missing,
        "n_moderate_missing": n_moderate_missing,
        "missing_high": missing_high,
        "missing_moderate": missing_moderate,
        "missing_other": missing_other,
    }


def format_imputation_detail(
    feature_columns: list,
    input_row: pd.DataFrame,
    language: str = "English",
) -> pd.DataFrame:
    """Return a DataFrame listing each variable and whether it was informed or imputed."""
    def _tr(en, pt):
        return en if language == "English" else pt

    rows = []
    for col in feature_columns:
        val = input_row[col].iloc[0] if col in input_row.columns else None
        is_missing = pd.isna(val)
        if col in HIGH_RELEVANCE_VARIABLES:
            relevance = "High"
        elif col in MODERATE_RELEVANCE_VARIABLES:
            relevance = "Moderate"
        elif col in OPTIONAL_DETAILED_VARIABLES:
            relevance = "Optional"
        else:
            relevance = "Standard"
        rows.append({
            _tr("Variable", "Variável"): col,
            _tr("Status", "Status"): _tr("Imputed (median/mode)", "Imputado (mediana/moda)") if is_missing else _tr("Informed", "Informado"),
            _tr("Relevance", "Relevância"): _tr(relevance, {"High": "Alta", "Moderate": "Moderada", "Optional": "Opcional", "Standard": "Padrão"}.get(relevance, relevance)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis audit trail
# ---------------------------------------------------------------------------

AUDIT_LOG_FILE = AppConfig.APP_CACHE_DIR / "audit_log.jsonl"


def log_analysis(
    analysis_type: str,
    source_file: str,
    model_version: str,
    n_patients: int = 0,
    n_imputed: int = 0,
    completeness_level: str = "",
    sts_method: str = "websocket",
    extra: Optional[dict] = None,
) -> dict:
    """Log an analysis event to the audit trail.

    Returns the logged entry.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "analysis_type": analysis_type,
        "source_file": str(source_file),
        "model_version": model_version,
        "n_patients": n_patients,
        "n_imputed": n_imputed,
        "completeness_level": completeness_level,
        "sts_method": sts_method,
    }
    if extra:
        entry["extra"] = extra

    try:
        AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # Audit logging should never break the app

    return entry


def read_audit_log(max_entries: int = 100) -> List[dict]:
    """Read the most recent audit log entries."""
    if not AUDIT_LOG_FILE.exists():
        return []
    entries = []
    try:
        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception:
        return []
    return entries[-max_entries:]


# ---------------------------------------------------------------------------
# Individual patient report
# ---------------------------------------------------------------------------

def generate_individual_report(
    patient_id: str,
    form_map: dict,
    ia_prob: float,
    euro_prob: float,
    sts_prob: float,
    risk_class: str,
    model_version: str,
    model_name: str,
    completeness: dict,
    pos_factors: pd.DataFrame,
    neg_factors: pd.DataFrame,
    sts_result: dict,
    language: str = "English",
    bundle_saved_at: str = None,
    training_source_file: str = None,
    current_analysis_file: str = None,
) -> str:
    """Generate an individual patient report as Markdown text."""
    def _tr(en, pt):
        return en if language == "English" else pt

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    _saved = (bundle_saved_at or "Unknown")[:19].replace("T", " ") if bundle_saved_at else "Unknown"

    lines = [
        f"# {_tr('Individual Risk Report', 'Relatório Individual de Risco')}",
        "",
        f"**{_tr('Patient/Code', 'Paciente/Código')}:** {patient_id}",
        f"**{_tr('Analysis date', 'Data da análise')}:** {now}",
        f"**{_tr('Model version', 'Versão do modelo')}:** {model_version}",
        f"**{_tr('Selected model', 'Modelo selecionado')}:** {model_name}",
        f"**{_tr('Bundle saved at', 'Bundle salvo em')}:** {_saved}",
        f"**{_tr('Training data source', 'Fonte de dados do treino')}:** {training_source_file or 'Unknown'}",
        f"**{_tr('Current analysis file', 'Arquivo de análise atual')}:** {current_analysis_file or 'Unknown'}",
        "",
        f"## {_tr('Risk Scores', 'Escores de Risco')}",
        "",
        f"| {_tr('Score', 'Escore')} | {_tr('Value', 'Valor')} | {_tr('Risk class', 'Classe de risco')} |",
        "|:--|:--|:--|",
        f"| AI Risk ({model_name}) | {ia_prob*100:.2f}% | {risk_class} |",
        f"| EuroSCORE II | {euro_prob*100:.2f}% | {_classify_risk(euro_prob)} |",
        f"| STS PROM | {'-' if np.isnan(sts_prob) else f'{sts_prob*100:.2f}%'} | {'-' if np.isnan(sts_prob) else _classify_risk(sts_prob)} |",
        "",
    ]

    # STS sub-scores
    if sts_result:
        from sts_calculator import STS_LABELS
        lines.append(f"### {_tr('STS Sub-scores', 'Sub-escores STS')}")
        lines.append("")
        lines.append(f"| {_tr('Endpoint', 'Desfecho')} | {_tr('Value', 'Valor')} |")
        lines.append("|:--|:--|")
        for key, label in STS_LABELS.items():
            val = sts_result.get(key, np.nan)
            lines.append(f"| {label} | {'-' if (isinstance(val, float) and np.isnan(val)) else f'{val*100:.2f}%'} |")
        lines.append("")

    # Input completeness
    lines.append(f"## {_tr('Input Completeness', 'Completude da Entrada')}")
    lines.append("")
    lines.append(f"- **{_tr('Status', 'Status')}:** {completeness['label']}")
    lines.append(f"- **{_tr('Informed variables', 'Variáveis informadas')}:** {completeness['n_informed']}/{completeness['n_total']}")
    lines.append(f"- **{_tr('Imputed variables', 'Variáveis imputadas')}:** {completeness['n_imputed']}")
    if completeness["missing_high"]:
        lines.append(f"- **{_tr('Missing high-relevance', 'Alta relevância ausentes')}:** {', '.join(completeness['missing_high'])}")
    if completeness["missing_moderate"]:
        lines.append(f"- **{_tr('Missing moderate-relevance', 'Moderada relevância ausentes')}:** {', '.join(completeness['missing_moderate'])}")
    lines.append("")

    # Key variables used
    lines.append(f"## {_tr('Key Variables', 'Variáveis-Chave')}")
    lines.append("")
    key_vars = [
        ("Age (years)", _tr("Age", "Idade")),
        ("Sex", _tr("Sex", "Sexo")),
        ("Surgery", _tr("Surgery", "Cirurgia")),
        ("Surgical Priority", _tr("Priority", "Prioridade")),
        ("LVEF, %", "LVEF"),
        ("Creatinine (mg/dL)", _tr("Creatinine", "Creatinina")),
        ("Cr clearance, ml/min *", _tr("Cr clearance", "Clearance de Cr")),
        ("Diabetes", "Diabetes"),
        ("Preoperative NYHA", "NYHA"),
        ("PSAP", "PSAP"),
    ]
    lines.append(f"| {_tr('Variable', 'Variável')} | {_tr('Value', 'Valor')} |")
    lines.append("|:--|:--|")
    for var_key, var_label in key_vars:
        val = form_map.get(var_key, "-")
        if isinstance(val, float) and np.isnan(val):
            val = "-"
        lines.append(f"| {var_label} | {val} |")
    lines.append("")

    # Clinical explanation summary
    _clinical = generate_clinical_explanation(pos_factors, neg_factors, ia_prob, language)
    lines.append(f"## {_tr('Clinical Interpretation', 'Interpretação Clínica')}")
    lines.append("")
    lines.append(_clinical)
    lines.append("")

    # Detailed risk factors
    lines.append(f"## {_tr('Risk Factors (Interpretable Layer)', 'Fatores de Risco (Camada Interpretável)')}")
    lines.append("")
    if not pos_factors.empty:
        lines.append(f"### {_tr('Factors increasing risk', 'Fatores que aumentam o risco')}")
        lines.append("")
        for _, row in pos_factors.iterrows():
            factor_col = [c for c in pos_factors.columns if "Factor" in c or "Fator" in c]
            if factor_col:
                lines.append(f"- {row[factor_col[0]]}")
        lines.append("")
    if not neg_factors.empty:
        lines.append(f"### {_tr('Factors decreasing risk', 'Fatores que reduzem o risco')}")
        lines.append("")
        for _, row in neg_factors.iterrows():
            factor_col = [c for c in neg_factors.columns if "Factor" in c or "Fator" in c]
            if factor_col:
                lines.append(f"- {row[factor_col[0]]}")
        lines.append("")

    # Methodological notes and disclaimer
    lines.append("---")
    lines.append("")
    lines.append(f"## {_tr('Methodological Notes', 'Notas Metodológicas')}")
    lines.append("")
    lines.append(f"- {_tr('This report was generated by AI Risk for research purposes only. It should not be used as the sole basis for clinical decisions.', 'Este relatório foi gerado pelo AI Risk apenas para fins de pesquisa. Não deve ser utilizado como base única para decisões clínicas.')}")
    lines.append(f"- **{_tr('Imputation', 'Imputação')}:** {_tr('Variables not informed by the user were replaced by the training dataset median (numeric) or mode (categorical). This is a standard approach in predictive modeling, but predictions with many imputed variables should be interpreted with greater caution.', 'Variáveis não informadas pelo usuário foram substituídas pela mediana (numéricas) ou moda (categóricas) do dataset de treinamento. Esta é uma abordagem padrão em modelagem preditiva, mas predições com muitas variáveis imputadas devem ser interpretadas com maior cautela.')}")
    lines.append(f"- **{_tr('Input completeness', 'Completude da entrada')}:** {completeness['label']} ({completeness['n_informed']}/{completeness['n_total']} {_tr('informed', 'informadas')}, {completeness['n_imputed']} {_tr('imputed', 'imputadas')})")
    lines.append(f"- **{_tr('EuroSCORE II', 'EuroSCORE II')}:** {_tr('Calculated by the app from the published logistic equation (Nashef et al., 2012). Not read from the input file.', 'Calculado pelo app pela equação logística publicada (Nashef et al., 2012). Não lido do arquivo de entrada.')}")
    lines.append(f"- **{_tr('STS Score', 'Score STS')}:** {_tr('Obtained via automated query to the official STS web calculator. The STS does not publish a documented public API; this value reflects the same calculation available to clinicians through the web interface. Not read from the input file.', 'Obtido via consulta automatizada à calculadora web oficial do STS. O STS não disponibiliza uma API pública documentada; este valor reflete o mesmo cálculo disponível aos clínicos pela interface web. Não lido do arquivo de entrada.')}")
    lines.append(f"- **{_tr('Risk factors', 'Fatores de risco')}:** {_tr('Based on the logistic regression interpretable layer. These reflect estimated statistical associations, not causal relationships.', 'Baseados na camada interpretável de regressão logística. Refletem associações estatísticas estimadas, não relações causais.')}")

    return "\n".join(lines)


def _classify_risk(prob: float) -> str:
    if np.isnan(prob):
        return "-"
    if prob < 0.05:
        return "Low"
    if prob <= 0.15:
        return "Intermediate"
    return "High"


# ---------------------------------------------------------------------------
# Statistical analysis summary export
# ---------------------------------------------------------------------------

def build_statistical_summary(
    triple_ci: pd.DataFrame,
    calib_df: pd.DataFrame,
    formal_df: pd.DataFrame,
    delong_df: pd.DataFrame,
    reclass_df: pd.DataFrame,
    threshold: float,
    threshold_metrics: pd.DataFrame,
    n_triple: int,
    model_version: str,
    language: str = "English",
) -> str:
    """Build an exportable statistical summary as Markdown."""
    def _tr(en, pt):
        return en if language == "English" else pt

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# {_tr('Statistical Analysis Summary', 'Resumo da Análise Estatística')}",
        "",
        f"**{_tr('Generated', 'Gerado em')}:** {now}",
        f"**{_tr('Model version', 'Versão do modelo')}:** {model_version}",
        f"**{_tr('Triple comparison sample', 'Amostra da comparação tripla')}:** n = {n_triple}",
        f"**{_tr('Decision threshold', 'Limiar de decisão')}:** {threshold:.0%}",
        "",
    ]

    # Discrimination with CI
    if not triple_ci.empty:
        lines.append(f"## {_tr('Discrimination (95% CI)', 'Discriminação (IC 95%)')}")
        lines.append("")
        lines.append(f"| {_tr('Score', 'Escore')} | n | AUC (95% CI) | AUPRC (95% CI) | Brier (95% CI) |")
        lines.append("|:--|:--|:--|:--|:--|")
        for _, row in triple_ci.iterrows():
            auc_ci = f"{row['AUC']:.3f} ({row.get('AUC_IC95_inf', np.nan):.3f}-{row.get('AUC_IC95_sup', np.nan):.3f})"
            auprc_ci = f"{row['AUPRC']:.3f} ({row.get('AUPRC_IC95_inf', np.nan):.3f}-{row.get('AUPRC_IC95_sup', np.nan):.3f})"
            brier_ci = f"{row['Brier']:.4f} ({row.get('Brier_IC95_inf', np.nan):.4f}-{row.get('Brier_IC95_sup', np.nan):.4f})"
            lines.append(f"| {row['Score']} | {row.get('n', '')} | {auc_ci} | {auprc_ci} | {brier_ci} |")
        lines.append("")

    # Threshold metrics
    if not threshold_metrics.empty:
        lines.append(f"## {_tr('Classification at threshold', 'Classificação no limiar')} {threshold:.0%}")
        lines.append("")
        lines.append(f"| {_tr('Score', 'Escore')} | {_tr('Sensitivity', 'Sensibilidade')} | {_tr('Specificity', 'Especificidade')} | PPV | NPV |")
        lines.append("|:--|:--|:--|:--|:--|")
        for _, row in threshold_metrics.iterrows():
            ppv = f"{row['PPV']:.3f}" if pd.notna(row.get('PPV')) else "-"
            npv = f"{row['NPV']:.3f}" if pd.notna(row.get('NPV')) else "-"
            lines.append(f"| {row['Score']} | {row.get('Sensitivity', np.nan):.3f} | {row.get('Specificity', np.nan):.3f} | {ppv} | {npv} |")
        lines.append("")

    # Calibration
    if not calib_df.empty:
        lines.append(f"## {_tr('Calibration', 'Calibração')}")
        lines.append("")
        lines.append(f"| {_tr('Score', 'Escore')} | {_tr('Intercept', 'Intercepto')} | Slope | HL chi² | HL p |")
        lines.append("|:--|:--|:--|:--|:--|")
        for _, row in calib_df.iterrows():
            lines.append(f"| {row['Score']} | {row.get('Calibration intercept', np.nan):.4f} | {row.get('Calibration slope', np.nan):.4f} | {row.get('HL chi-square', np.nan):.2f} | {row.get('HL p-value', np.nan):.4f} |")
        lines.append("")

    # DeLong
    if not delong_df.empty:
        lines.append(f"## {_tr('DeLong Test', 'Teste de DeLong')}")
        lines.append("")
        comp_col = [c for c in delong_df.columns if "Comparison" in c or "Comparação" in c]
        comp_key = comp_col[0] if comp_col else "Comparison"
        lines.append(f"| {_tr('Comparison', 'Comparação')} | ΔAUC | z | p |")
        lines.append("|:--|:--|:--|:--|")
        for _, row in delong_df.iterrows():
            lines.append(f"| {row.get(comp_key, '')} | {row.get('Delta AUC', np.nan):.3f} | {row.get('z', np.nan):.2f} | {row.get('p (DeLong)', np.nan):.4f} |")
        lines.append("")

    # Bootstrap comparison
    if not formal_df.empty:
        lines.append(f"## {_tr('Bootstrap AUC Comparison', 'Comparação de AUC por Bootstrap')}")
        lines.append("")
        comp_col = [c for c in formal_df.columns if "Comparison" in c or "Comparação" in c]
        comp_key = comp_col[0] if comp_col else "Comparison"
        ci_lo_col = [c for c in formal_df.columns if "CI low" in c or "IC95% inf" in c]
        ci_hi_col = [c for c in formal_df.columns if "CI high" in c or "IC95% sup" in c]
        lo_key = ci_lo_col[0] if ci_lo_col else "95% CI low"
        hi_key = ci_hi_col[0] if ci_hi_col else "95% CI high"
        lines.append(f"| {_tr('Comparison', 'Comparação')} | ΔAUC | 95% CI | p |")
        lines.append("|:--|:--|:--|:--|")
        for _, row in formal_df.iterrows():
            lines.append(f"| {row.get(comp_key, '')} | {row.get('Delta AUC (A-B)', np.nan):.3f} | {row.get(lo_key, np.nan):.3f}-{row.get(hi_key, np.nan):.3f} | {row.get('p (bootstrap)', np.nan):.4f} |")
        lines.append("")

    # NRI/IDI
    if not reclass_df.empty:
        lines.append(f"## {_tr('Reclassification (NRI/IDI)', 'Reclassificação (NRI/IDI)')}")
        lines.append("")
        comp_col = [c for c in reclass_df.columns if "Comparison" in c or "Comparação" in c]
        comp_key = comp_col[0] if comp_col else "Comparison"
        lines.append(f"| {_tr('Comparison', 'Comparação')} | NRI events | NRI non-events | NRI total | IDI |")
        lines.append("|:--|:--|:--|:--|:--|")
        for _, row in reclass_df.iterrows():
            lines.append(f"| {row.get(comp_key, '')} | {row.get('NRI events', np.nan):.3f} | {row.get('NRI non-events', np.nan):.3f} | {row.get('NRI total', np.nan):.3f} | {row.get('IDI', np.nan):.4f} |")
        lines.append("")

    lines.append("---")
    lines.append(f"*{_tr('Generated by AI Risk', 'Gerado pelo AI Risk')}*")

    return "\n".join(lines)


def _parse_md_tables(md_text: str) -> List[dict]:
    """Extract Markdown tables from summary text as list of {title, headers, rows}."""
    import re
    tables = []
    lines = md_text.split("\n")
    i = 0
    current_title = ""
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("## "):
            current_title = line.lstrip("# ").strip()
        elif line.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            headers = [c.strip() for c in line.strip("|").split("|")]
            i += 1  # skip separator
            rows = []
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                rows.append(cells)
                i += 1
            tables.append({"title": current_title, "headers": headers, "rows": rows})
            continue
        i += 1
    return tables


def statistical_summary_to_dataframes(md_text: str) -> Dict[str, pd.DataFrame]:
    """Convert the Markdown statistical summary into a dict of DataFrames (one per table)."""
    tables = _parse_md_tables(md_text)
    result = {}
    for t in tables:
        key = t["title"] or f"Table_{len(result) + 1}"
        df = pd.DataFrame(t["rows"], columns=t["headers"])
        result[key] = df
    return result


def statistical_summary_to_xlsx(md_text: str) -> bytes:
    """Convert statistical summary to XLSX with one sheet per table."""
    import io
    dfs = statistical_summary_to_dataframes(md_text)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in dfs.items():
            import re
            sheet_name = re.sub(r'[\\/*?\[\]:/]', '_', name)[:31]  # sanitize for Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buf.getvalue()


def statistical_summary_to_csv(md_text: str) -> str:
    """Convert statistical summary to CSV (all tables concatenated with section headers)."""
    dfs = statistical_summary_to_dataframes(md_text)
    parts = []
    for name, df in dfs.items():
        parts.append(f"# {name}")
        parts.append(df.to_csv(index=False))
    return "\n".join(parts)


def statistical_summary_to_pdf(md_text: str) -> bytes:
    """Convert statistical summary to PDF."""
    try:
        from fpdf import FPDF
    except ImportError:
        return b""

    def _latin_safe(text: str) -> str:
        """Replace Unicode characters unsupported by Helvetica with ASCII equivalents."""
        replacements = {
            "\u0394": "Delta ",  # Δ
            "\u2264": "<=",      # ≤
            "\u2265": ">=",      # ≥
            "\u00b2": "2",       # ²
            "\u2013": "-",       # –
            "\u2014": "--",      # —
            "\u00b3": "3",       # ³
            "\u03c7": "chi",     # χ
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    tables = _parse_md_tables(md_text)

    # Extract header metadata from markdown
    title = "Statistical Summary"
    header_lines = []
    for line in md_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("**") and ":**" in stripped:
            header_lines.append(stripped.replace("**", ""))
        elif stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped.lstrip("# ").strip()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _latin_safe(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Header metadata
    pdf.set_font("Helvetica", "", 9)
    for hl in header_lines:
        pdf.cell(0, 5, _latin_safe(hl), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    for t in tables:
        # Section title
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, _latin_safe(t["title"]), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        if not t["headers"]:
            continue

        n_cols = len(t["headers"])
        available_width = pdf.w - pdf.l_margin - pdf.r_margin
        col_w = available_width / n_cols

        # Header row
        pdf.set_font("Helvetica", "B", 8)
        for h in t["headers"]:
            pdf.cell(col_w, 6, _latin_safe(h[:20]), border=1, align="C")
        pdf.ln()

        # Data rows
        pdf.set_font("Helvetica", "", 8)
        for row in t["rows"]:
            for j, cell in enumerate(row):
                txt = _latin_safe(cell[:22] if j > 0 else cell[:25])
                pdf.cell(col_w, 5, txt, border=1, align="C" if j > 0 else "L")
            pdf.ln()

        pdf.ln(4)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Data quality panel helpers
# ---------------------------------------------------------------------------

def compute_data_quality_summary(
    df: pd.DataFrame,
    feature_columns: list,
    language: str = "English",
) -> dict:
    """Compute data quality metrics for the dataset panel."""
    def _tr(en, pt):
        return en if language == "English" else pt

    n_total = len(df)
    n_events = int(df["morte_30d"].sum()) if "morte_30d" in df.columns else 0
    event_rate = n_events / n_total if n_total > 0 else 0

    # Missing rate per variable
    missing_rates = {}
    for col in feature_columns:
        if col in df.columns:
            missing_rates[col] = float(df[col].isna().mean())

    # Patients with EuroSCORE available
    n_euro_sheet = int(df["euroscore_sheet"].notna().sum()) if "euroscore_sheet" in df.columns else 0
    n_euro_auto = int(df["euroscore_auto_sheet_clean"].notna().sum()) if "euroscore_auto_sheet_clean" in df.columns else 0
    n_euro_calc = int(df["euroscore_calc"].notna().sum()) if "euroscore_calc" in df.columns else 0

    # Patients with STS available
    n_sts = int(df["sts_score"].notna().sum()) if "sts_score" in df.columns else 0
    n_sts_sheet = int(df["sts_score_sheet"].notna().sum()) if "sts_score_sheet" in df.columns else 0

    # Triple cohort
    triple_cols = ["ia_risk_oof", "euroscore_calc", "sts_score"]
    triple_cols_available = [c for c in triple_cols if c in df.columns]
    n_triple = int(df[triple_cols_available].notna().all(axis=1).sum()) if triple_cols_available else 0

    # Surgery type distribution
    surgery_dist = {}
    if "Surgery" in df.columns:
        from risk_data import split_surgery_procedures
        procedure_counts = {}
        for s in df["Surgery"].dropna():
            parts = split_surgery_procedures(s)
            for p in parts:
                procedure_counts[p] = procedure_counts.get(p, 0) + 1
        surgery_dist = dict(sorted(procedure_counts.items(), key=lambda x: x[1], reverse=True)[:15])

    return {
        "n_total": n_total,
        "n_events": n_events,
        "event_rate": event_rate,
        "missing_rates": missing_rates,
        "n_euro_sheet": n_euro_sheet,
        "n_euro_auto": n_euro_auto,
        "n_euro_calc": n_euro_calc,
        "n_sts": n_sts,
        "n_sts_sheet": n_sts_sheet,
        "n_triple": n_triple,
        "surgery_dist": surgery_dist,
    }


# ---------------------------------------------------------------------------
# Clinical explainability text generation
# ---------------------------------------------------------------------------

def generate_clinical_explanation(
    pos_factors: pd.DataFrame,
    neg_factors: pd.DataFrame,
    ia_prob: float,
    language: str = "English",
) -> str:
    """Generate a clinical text explanation based on the top risk factors."""
    def _tr(en, pt):
        return en if language == "English" else pt

    lines = []

    risk_level = _tr("low", "baixo") if ia_prob < 0.05 else (_tr("intermediate", "intermediário") if ia_prob <= 0.15 else _tr("high", "alto"))
    lines.append(_tr(
        f"The estimated 30-day mortality risk is {ia_prob*100:.1f}%, classified as {risk_level} risk.",
        f"O risco estimado de mortalidade em 30 dias é de {ia_prob*100:.1f}%, classificado como risco {risk_level}.",
    ))
    lines.append("")

    if not pos_factors.empty:
        factor_col = [c for c in pos_factors.columns if "Factor" in c or "Fator" in c]
        if factor_col:
            factors = pos_factors[factor_col[0]].tolist()
            if len(factors) == 1:
                lines.append(_tr(
                    f"The main factor contributing to increased risk was: {factors[0].lower()}.",
                    f"O principal fator que contribuiu para o aumento do risco foi: {factors[0].lower()}.",
                ))
            else:
                factor_list = ", ".join(f.lower() for f in factors[:-1]) + _tr(f" and {factors[-1].lower()}", f" e {factors[-1].lower()}")
                lines.append(_tr(
                    f"The main factors contributing to increased risk were: {factor_list}.",
                    f"Os principais fatores que contribuíram para o aumento do risco foram: {factor_list}.",
                ))

    if not neg_factors.empty:
        factor_col = [c for c in neg_factors.columns if "Factor" in c or "Fator" in c]
        if factor_col:
            factors = neg_factors[factor_col[0]].tolist()
            if len(factors) == 1:
                lines.append(_tr(
                    f"The main protective factor was: {factors[0].lower()}.",
                    f"O principal fator protetor foi: {factors[0].lower()}.",
                ))
            else:
                factor_list = ", ".join(f.lower() for f in factors[:-1]) + _tr(f" and {factors[-1].lower()}", f" e {factors[-1].lower()}")
                lines.append(_tr(
                    f"Protective factors included: {factor_list}.",
                    f"Fatores protetores incluíram: {factor_list}.",
                ))

    lines.append("")
    lines.append(_tr(
        "Note: This explanation is based on the logistic regression interpretable layer and reflects estimated statistical associations, not causal relationships. It should be used as a complement to clinical judgment.",
        "Nota: Esta explicação é baseada na camada interpretável de regressão logística e reflete associações estatísticas estimadas, não relações causais. Deve ser usada como complemento ao julgamento clínico.",
    ))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation readiness helpers
# ---------------------------------------------------------------------------

def export_model_bundle_metadata(metadata: dict, output_path: str) -> None:
    """Export model metadata to a JSON file for version tracking and future validation."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass


def check_validation_readiness(metadata: dict, language: str = "English") -> List[dict]:
    """Check how ready the current model is for different validation scenarios."""
    def _tr(en, pt):
        return en if language == "English" else pt

    checks = []
    n = metadata.get("n_patients", 0)
    n_events = metadata.get("n_events", 0)

    checks.append({
        "check": _tr("Temporal validation", "Validação temporal"),
        "status": _tr("Ready", "Pronto") if n >= 100 else _tr("Needs more data", "Precisa de mais dados"),
        "note": _tr(
            "Requires a held-out temporal subset. Current bundle can serve as the development cohort.",
            "Requer subconjunto temporal separado. O bundle atual pode servir como coorte de desenvolvimento.",
        ),
    })
    checks.append({
        "check": _tr("External validation", "Validação externa"),
        "status": _tr("Infrastructure ready", "Infraestrutura pronta"),
        "note": _tr(
            "The model can accept external datasets in CSV/Parquet/Excel format with the same variable schema.",
            "O modelo aceita datasets externos em CSV/Parquet/Excel com o mesmo esquema de variáveis.",
        ),
    })
    checks.append({
        "check": _tr("Bundle comparison", "Comparação de bundles"),
        "status": _tr("Supported", "Suportado"),
        "note": _tr(
            f"Current bundle: version {metadata.get('model_version', 'N/A')}, {n} patients, {n_events} events. Export metadata JSON for version tracking.",
            f"Bundle atual: versão {metadata.get('model_version', 'N/A')}, {n} pacientes, {n_events} eventos. Exporte o JSON de metadados para rastreamento de versões.",
        ),
    })
    checks.append({
        "check": _tr("Feature compatibility", "Compatibilidade de variáveis"),
        "status": _tr("Documented", "Documentado"),
        "note": _tr(
            f"Model uses {metadata.get('n_features', 0)} features. Variable dictionary available for schema matching.",
            f"O modelo usa {metadata.get('n_features', 0)} variáveis. Dicionário de variáveis disponível para matching de esquema.",
        ),
    })

    return checks


# ---------------------------------------------------------------------------
# Temporal validation helpers
# ---------------------------------------------------------------------------

def check_temporal_overlap(
    training_start: str,
    training_end: str,
    validation_start: str,
    validation_end: str,
) -> dict:
    """Compare training and validation temporal ranges for overlap.

    Accepts year-quarter strings (``"2024-Q1"``) or plain years (``"2024"``).
    Returns dict with keys: overlap, status, severity, message_en, message_pt.
    """
    result = {
        "training_range": (training_start, training_end),
        "validation_range": (validation_start, validation_end),
        "overlap": False,
        "status": "unknown",
        "severity": "info",
        "message_en": "",
        "message_pt": "",
    }
    t_start = _yq_to_timestamp(training_start)
    t_end = _yq_to_end_timestamp(training_end)
    v_start = _yq_to_timestamp(validation_start)
    v_end = _yq_to_end_timestamp(validation_end)

    if t_start is None or t_end is None or v_start is None or v_end is None:
        result["status"] = "unknown"
        result["severity"] = "warning"
        result["message_en"] = "Could not parse temporal ranges — overlap check skipped."
        result["message_pt"] = "Não foi possível interpretar os períodos — verificação de sobreposição ignorada."
        return result

    if v_end < t_start:
        result["status"] = "validation_before_training"
        result["severity"] = "error"
        result["message_en"] = (
            f"The validation cohort ({validation_start} — {validation_end}) is entirely "
            f"BEFORE the training cohort ({training_start} — {training_end}). "
            "This is NOT temporal validation — it is retrograde validation and "
            "severely compromises methodological validity."
        )
        result["message_pt"] = (
            f"A coorte de validação ({validation_start} — {validation_end}) é inteiramente "
            f"ANTERIOR à coorte de treinamento ({training_start} — {training_end}). "
            "Isso NÃO é validação temporal — é validação retrógrada e "
            "compromete gravemente a validade metodológica."
        )
    elif v_start > t_end:
        result["status"] = "no_overlap"
        result["severity"] = "success"
        result["message_en"] = (
            f"No temporal overlap detected. The validation cohort ({validation_start} — {validation_end}) "
            f"is strictly after the training cohort ({training_start} — {training_end}) "
            "— ideal for temporal validation."
        )
        result["message_pt"] = (
            f"Sem sobreposição temporal detectada. A coorte de validação ({validation_start} — {validation_end}) "
            f"é estritamente posterior à coorte de treinamento ({training_start} — {training_end}) "
            "— ideal para validação temporal."
        )
    else:
        result["overlap"] = True
        result["status"] = "overlap"
        result["severity"] = "warning"
        result["message_en"] = (
            f"Temporal overlap detected between training ({training_start} — {training_end}) "
            f"and validation ({validation_start} — {validation_end}). "
            "Patients in the overlapping period may have been used for training, "
            "which weakens the validity of temporal validation."
        )
        result["message_pt"] = (
            f"Sobreposição temporal detectada entre treinamento ({training_start} — {training_end}) "
            f"e validação ({validation_start} — {validation_end}). "
            "Pacientes no período sobreposto podem ter sido usados no treinamento, "
            "o que enfraquece a validade da validação temporal."
        )

    return result


def format_locked_model_for_display(
    metadata: dict,
    language: str = "English",
) -> pd.DataFrame:
    """Format locked model metadata for the Temporal Validation tab."""
    def _tr(en, pt):
        return en if language == "English" else pt

    def _fmt_ts(ts):
        if not ts or ts == "Unknown":
            return "Unknown"
        return ts[:19].replace("T", " ")

    t_start = metadata.get("training_start_date", "Unknown")
    t_end = metadata.get("training_end_date", "Unknown")
    date_range = f"{t_start} — {t_end}" if t_start != "Unknown" and t_end != "Unknown" else "Unknown"

    calib = metadata.get("calibration", {})
    calib_method = calib.get("method", "N/A") if isinstance(calib, dict) else str(calib)
    locked_thr = metadata.get("locked_threshold", 0.08)

    rows = [
        (_tr("Model version", "Versão do modelo"), metadata.get("model_version", "N/A")),
        (_tr("Bundle saved at", "Bundle salvo em"), _fmt_ts(metadata.get("bundle_saved_at"))),
        (_tr("Training data source", "Fonte de dados do treino"), metadata.get("training_source_file", "N/A")),
        (_tr("Best model", "Melhor modelo"), metadata.get("best_model", "N/A")),
        (_tr("Patients in training", "Pacientes no treinamento"), str(metadata.get("n_patients", "N/A"))),
        (_tr("Events in training", "Eventos no treinamento"), str(metadata.get("n_events", "N/A"))),
        (_tr("Event rate in training", "Taxa de eventos no treinamento"), f"{metadata.get('event_rate', 0):.1%}"),
        (_tr("Number of features", "Número de variáveis"), str(metadata.get("n_features", "N/A"))),
        (_tr("CV strategy", "Estratégia de CV"), metadata.get("cv_strategy", "N/A")),
        (_tr("CV splits", "Folds de CV"), str(metadata.get("cv_splits", "N/A"))),
        (_tr("Calibration method", "Método de calibração"), calib_method),
        (_tr("Training date range", "Período do treinamento"), date_range),
        (_tr("Locked clinical threshold", "Limiar clínico bloqueado"), f"{locked_thr:.0%}"),
        (_tr("Lock status", "Status de bloqueio"), _tr("Locked — no retraining allowed", "Bloqueado — sem retreinamento")),
    ]
    return pd.DataFrame(rows, columns=[_tr("Property", "Propriedade"), _tr("Value", "Valor")])


def build_temporal_validation_summary(
    cohort_summary: dict,
    performance_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    risk_category_df: pd.DataFrame,
    metadata: dict,
    threshold: float,
    language: str = "English",
) -> str:
    """Build Markdown summary for temporal validation results."""
    def _tr(en, pt):
        return en if language == "English" else pt

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cs = cohort_summary

    lines = [
        f"# {_tr('Temporal Validation Report', 'Relatório de Validação Temporal')}",
        "",
        f"**{_tr('Generated', 'Gerado em')}:** {now}",
        f"**{_tr('Model version', 'Versão do modelo')}:** {metadata.get('model_version', 'N/A')}",
        f"**{_tr('Locked threshold', 'Limiar bloqueado')}:** {threshold:.0%}",
        f"**{_tr('Training cohort', 'Coorte de treinamento')}:** n = {metadata.get('n_patients', 'N/A')}, "
        f"{_tr('events', 'eventos')} = {metadata.get('n_events', 'N/A')} "
        f"({metadata.get('event_rate', 0):.1%})",
        f"**{_tr('Validation cohort', 'Coorte de validação')}:** n = {cs.get('n_total', 0)}, "
        f"{_tr('events', 'eventos')} = {cs.get('n_events', 0)} "
        f"({cs.get('event_rate', 0):.1%})",
        "",
        f"**{_tr('Methodological note', 'Nota metodológica')}:** "
        + _tr(
            "This report applies a previously locked model to an independent temporal cohort. "
            "No retraining, recalibration, or threshold adjustment was performed.",
            "Este relatório aplica um modelo previamente congelado a uma coorte temporal independente. "
            "Não houve retreinamento, recalibração ou ajuste de limiar.",
        ),
        "",
    ]

    # Cohort summary table
    lines.append(f"## {_tr('Cohort Summary', 'Resumo da Coorte')}")
    lines.append("")
    lines.append(f"| {_tr('Property', 'Propriedade')} | {_tr('Value', 'Valor')} |")
    lines.append("|:--|:--|")
    lines.append(f"| {_tr('Total patients', 'Total de pacientes')} | {cs.get('n_total', 0)} |")
    lines.append(f"| {_tr('Events (30-day mortality)', 'Eventos (mortalidade 30 dias)')} | {cs.get('n_events', 0)} |")
    lines.append(f"| {_tr('Event rate', 'Taxa de eventos')} | {cs.get('event_rate', 0):.1%} |")
    lines.append(f"| {_tr('Date range', 'Período')} | {cs.get('date_range', 'Unknown')} |")
    for level_key, level_label_en, level_label_pt in [
        ("n_complete", "Complete data", "Dados completos"),
        ("n_adequate", "Adequate", "Adequados"),
        ("n_partial", "Partially imputed", "Parcialmente imputados"),
        ("n_low", "Heavily imputed", "Muito imputados"),
    ]:
        n = cs.get(level_key, 0)
        pct = n / cs["n_total"] * 100 if cs.get("n_total", 0) > 0 else 0
        lines.append(f"| {_tr(level_label_en, level_label_pt)} | {n} ({pct:.1f}%) |")
    lines.append("")

    # Performance table
    if not performance_df.empty:
        lines.append(f"## {_tr('Discrimination and Calibration', 'Discriminação e Calibração')}")
        lines.append("")
        hdr = " | ".join(performance_df.columns)
        lines.append(f"| {hdr} |")
        lines.append("|" + "|".join(":--" for _ in performance_df.columns) + "|")
        for _, row in performance_df.iterrows():
            cells = " | ".join(str(v) for v in row.values)
            lines.append(f"| {cells} |")
        lines.append("")

    # Pairwise comparison
    if not pairwise_df.empty:
        lines.append(f"## {_tr('Pairwise Comparison', 'Comparação Pareada')}")
        lines.append("")
        hdr = " | ".join(pairwise_df.columns)
        lines.append(f"| {hdr} |")
        lines.append("|" + "|".join(":--" for _ in pairwise_df.columns) + "|")
        for _, row in pairwise_df.iterrows():
            cells = " | ".join(str(v) for v in row.values)
            lines.append(f"| {cells} |")
        lines.append("")

    # Risk categories
    if not risk_category_df.empty:
        lines.append(f"## {_tr('Risk Category Distribution', 'Distribuição por Classe de Risco')}")
        lines.append("")
        hdr = " | ".join(risk_category_df.columns)
        lines.append(f"| {hdr} |")
        lines.append("|" + "|".join(":--" for _ in risk_category_df.columns) + "|")
        for _, row in risk_category_df.iterrows():
            cells = " | ".join(str(v) for v in row.values)
            lines.append(f"| {cells} |")
        lines.append("")

    lines.append("---")
    lines.append(f"*{_tr('Generated by AI Risk — Temporal Validation Module', 'Gerado pelo AI Risk — Módulo de Validação Temporal')}*")

    return "\n".join(lines)
