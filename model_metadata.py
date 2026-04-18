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
# Re-exports from extracted modules
# ---------------------------------------------------------------------------
# Statistical summary and export functions live in export_helpers.py.
# Temporal helpers and validation functions live in temporal_validation.py.
# All symbols are re-exported here so existing importers (e.g. app.py) require
# no changes.

from export_helpers import (
    build_statistical_summary,
    statistical_summary_to_xlsx,
    statistical_summary_to_csv,
    statistical_summary_to_pdf,
)

from temporal_validation import (
    extract_year_quarter_range,
    check_temporal_overlap,
    format_locked_model_for_display,
    build_temporal_validation_summary,
    build_exploratory_recalibration_summary,
    build_exploratory_threshold_summary,
    build_exploratory_temporal_validation_section,
    build_sts_accounting_table,
    is_surrogate_timeline,
    build_surrogate_timeline_note,
    chronological_state_label,
    CHRONO_STATE_NO_OVERLAP,
    CHRONO_STATE_OVERLAP,
    CHRONO_STATE_RETROGRADE,
    CHRONO_STATE_UNKNOWN,
)
# Backward-compat alias: build_model_metadata (below) and any caller that uses
# the private-style name both continue to work without modification.
_extract_year_quarter_range = extract_year_quarter_range


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
            "oof_evaluation": "calibrated inside each CV fold (RandomForest: sigmoid inner cv≤5; LightGBM/CatBoost: isotonic inner cv≤5; XGBoost: isotonic inner cv≤3)",
            "grouping_note": "inner calibration CV does not enforce patient grouping (sklearn limitation)",
        },
        "thresholds": {
            "leaderboard": "Youden's J (optimal per model, on calibrated OOF)",
            "clinical_default": "8%",
        },
        "oof_used_in_leaderboard": "calibrated per-model inside each CV fold (see calibration.oof_evaluation for per-model method); LogisticRegression and StackingEnsemble are used uncalibrated",
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
    value_col = _tr("Value", "Valor")
    out = pd.DataFrame(rows, columns=[_tr("Property", "Propriedade"), value_col])
    # Force the Value column to string dtype.  Rows mix Python types
    # (ints, floats, strings such as "14.98%", "RandomForest", "N/A"),
    # and pyarrow's object-column type inference inside Streamlit picks
    # int64 from early numeric rows and then raises ArrowInvalid on the
    # later string rows ("Could not convert '14.98%' ... to int64").
    # The column is rendered as plain text anyway — casting to str keeps
    # the display identical while making Arrow serialization deterministic.
    out[value_col] = out[value_col].astype(str)
    return out


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

    def _localize_risk(label: str) -> str:
        """Translate a risk-class English label to the report language."""
        if language == "English":
            return label
        return _RISK_CLASS_PT.get(label, label)

    def _display_val(val) -> str:
        """Return a display-ready string for a form-map value.

        Translates common English categorical values (Yes/No, sex, priority)
        to Portuguese when the report language is Portuguese.  Numeric and
        unknown values are returned as-is.
        """
        if isinstance(val, float) and np.isnan(val):
            return "-"
        s = str(val)
        if language == "English":
            return s
        return _FORM_VALUE_PT.get(s, s)

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
        f"| AI Risk ({model_name}) | {ia_prob*100:.2f}% | {_localize_risk(risk_class)} |",
        f"| EuroSCORE II | {euro_prob*100:.2f}% | {_classify_risk(euro_prob, language)} |",
        f"| STS Score PROM | {'-' if np.isnan(sts_prob) else f'{sts_prob*100:.2f}%'} | {'-' if np.isnan(sts_prob) else _classify_risk(sts_prob, language)} |",
        "",
    ]

    # STS sub-scores
    if sts_result:
        from sts_calculator import STS_LABELS
        lines.append(f"### {_tr('STS Score Sub-scores', 'Sub-escores STS Score')}")
        lines.append("")
        lines.append(f"| {_tr('Endpoint', 'Desfecho')} | {_tr('Value', 'Valor')} |")
        lines.append("|:--|:--|")
        # Use Portuguese endpoint labels when the report language is Portuguese.
        _sts_label_map = STS_LABELS if language == "English" else _STS_LABELS_PT
        for key, label in _sts_label_map.items():
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
        val = _display_val(form_map.get(var_key, "-"))
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
    lines.append(f"- **{_tr('STS Score', 'STS Score')}:** {_tr('Obtained via automated query to the official STS Score web calculator. The STS does not publish a documented public API; this value reflects the same calculation available to clinicians through the web interface. Not read from the input file.', 'Obtido via consulta automatizada à calculadora web oficial do STS Score. O STS não disponibiliza uma API pública documentada; este valor reflete o mesmo cálculo disponível aos clínicos pela interface web. Não lido do arquivo de entrada.')}")
    lines.append(f"- **{_tr('Risk factors', 'Fatores de risco')}:** {_tr('Based on the logistic regression interpretable layer. These reflect estimated statistical associations, not causal relationships.', 'Baseados na camada interpretável de regressão logística. Refletem associações estatísticas estimadas, não relações causais.')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Display-layer translation tables (used only in generate_individual_report)
# ---------------------------------------------------------------------------

# Risk class labels — keyed by English canonical value.
_RISK_CLASS_PT = {"Low": "Baixo", "Intermediate": "Intermediário", "High": "Alto"}

# Form-map display values that may appear in Portuguese reports.
_FORM_VALUE_PT = {
    # Sex
    "Male": "Masculino",
    "Female": "Feminino",
    # Surgical priority
    "Elective": "Eletiva",
    "Urgent": "Urgente",
    "Emergency": "Emergência",
    "Emergent Salvage": "Emergente/Salvamento",
    # Boolean
    "Yes": "Sim",
    "No": "Não",
    # Completeness status (generated by assess_input_completeness)
    "Complete": "Completa",
    "Adequate": "Adequada",
    "Partial": "Parcial",
    "Low": "Baixa",
}

# STS Score endpoint labels in Portuguese.
_STS_LABELS_PT = {
    "predmort": "Mortalidade Operatória",
    "predmm": "Morbidade e Mortalidade",
    "predstro": "AVC",
    "predrenf": "Insuficiência Renal",
    "predreop": "Reoperação",
    "predvent": "Ventilação Prolongada",
    "preddeep": "Infecção Esternal Profunda",
    "pred14d": "Internação Prolongada (>14 dias)",
    "pred6d": "Internação Curta (<6 dias)",
}


def _classify_risk(prob: float, language: str = "English") -> str:
    if np.isnan(prob):
        return "-"
    if prob < 0.05:
        label = "Low"
    elif prob <= 0.15:
        label = "Intermediate"
    else:
        label = "High"
    if language != "English":
        return _RISK_CLASS_PT.get(label, label)
    return label


# ---------------------------------------------------------------------------
# Statistical analysis summary export — moved to export_helpers.py
# (re-exported above for backward compatibility)
# ---------------------------------------------------------------------------


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
# Temporal validation helpers — moved to temporal_validation.py
# (re-exported at the top of this file for backward compatibility)
# ---------------------------------------------------------------------------
# Functions removed: check_temporal_overlap, format_locked_model_for_display,
# build_temporal_validation_summary.  See temporal_validation.py.
