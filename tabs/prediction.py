"""Individual Prediction tab — extracted from app.py (tab index 1).

Pure extraction: all logic, text, i18n, and UI elements are identical to the
original inline code.  The only structural change is that shared state is
accessed through ``ctx`` (:class:`tabs.TabContext`) instead of bare local
variables in ``app.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict

import numpy as np
import pandas as pd
import streamlit as st

from ai_risk_inference import (
    _get_numeric_columns_from_pipeline,
    _patient_identifier_from_row,
    _run_ai_risk_inference_row,
    _safe_select_features,
)
from model_metadata import (
    assess_input_completeness,
    format_imputation_detail,
    generate_clinical_explanation,
    generate_individual_report,
    log_analysis,
    statistical_summary_to_csv,
    statistical_summary_to_pdf,
    statistical_summary_to_xlsx,
)
from modeling import clean_features
from explainability import ModelExplainer
from risk_data import (
    MISSINGNESS_INDICATOR_COLUMNS,
    normalize_arrhythmia_recent_value,
    normalize_arrhythmia_remote_value,
    normalize_cva_value,
    normalize_hf_value,
    normalize_pneumonia_value,
    parse_number,
)
from stats_compare import class_risk
from euroscore import euroscore_from_inputs
from sts_calculator import (
    STS_LABELS,
    calculate_sts,
)

if TYPE_CHECKING:
    from tabs import TabContext


# ---------------------------------------------------------------------------
# Feature display-name helper (mirrors app.py)
# ---------------------------------------------------------------------------

_FEAT_PREFIXES = (
    "cat__onehot__",
    "cat__target_enc__",
    "cat__ordinal__",
    "cat__",
    "num__",
    "valve__",
    "ord__",
)


def _feat_display_name(name: str) -> str:
    s = str(name)
    for prefix in _FEAT_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def _resolve_base_feature(encoded_feature: str, feature_columns: list) -> str:
    if encoded_feature in feature_columns:
        return encoded_feature
    for feat in sorted(feature_columns, key=len, reverse=True):
        if encoded_feature.startswith(feat + "_"):
            return feat
    return encoded_feature


def _feature_group(base_feature: str, tr) -> str:
    clinical = {
        "Age (years)", "Sex", "Preoperative NYHA", "CCS4", "Diabetes", "PVD", "Previous surgery",
        "Dialysis", "IE", "HF", "Hypertension", "Dyslipidemia", "CVA", "Cancer ≤ 5 yrs",
        "Arrhythmia Remote", "Arrhythmia Recent", "Family Hx of CAD", "Smoking (Pack-year)",
        "Alcohol", "Pneumonia", "Chronic Lung Disease", "Poor mobility",
        "Critical preoperative state", "Coronary Symptom", "Left Main Stenosis ≥ 50%",
        "Proximal LAD Stenosis ≥ 70%", "No. of Diseased Vessels",
    }
    lab = {
        "Weight (kg)", "Height (cm)", "Cr clearance, ml/min *", "Creatinine (mg/dL)", "Hematocrit (%)",
        "WBC Count (10³/μL)", "Platelet Count (cells/μL)", "INR", "PTT", "KDIGO †",
        *MISSINGNESS_INDICATOR_COLUMNS,
    }
    echo = {
        "Pré-LVEF, %", "PSAP", "TAPSE", "Aortic Stenosis", "Aortic Regurgitation",
        "Mitral Stenosis", "Mitral Regurgitation", "Tricuspid Regurgitation", "Aortic Root Abscess",
        "AVA (cm²)", "MVA (cm²)", "Aortic Mean gradient (mmHg)", "Mitral Mean gradient (mmHg)",
        "PHT Aortic", "PHT Mitral", "Vena contracta", "Vena contracta (mm)",
    }
    procedure = {
        "Surgery", "Surgical Priority", "cirurgia_combinada", "peso_procedimento", "thoracic_aorta_flag",
        "Anticoagulation/ Antiaggregation", "Suspension of Anticoagulation (day)", "Preoperative Medications",
    }
    if base_feature in clinical:
        return tr("Clinical", "Clínico")
    if base_feature in lab:
        return tr("Laboratory", "Laboratorial")
    if base_feature in echo:
        return tr("Echocardiographic", "Ecocardiográfico")
    if base_feature in procedure:
        return tr("Procedure", "Procedimento")
    return tr("Other", "Outro")


def _patient_factor_label(base_feature: str, form_map: Dict, tr) -> str:
    val = form_map.get(base_feature)
    if base_feature == "Age (years)":
        age = parse_number(val)
        if pd.notna(age):
            if float(age) >= 75:
                return tr("Very advanced age", "Idade muito avançada")
            if float(age) >= 65:
                return tr("Advanced age", "Idade avançada")
            return tr("Age below 65 years", "Idade abaixo de 65 anos")
    if base_feature == "Pré-LVEF, %":
        lvef = parse_number(val)
        if pd.notna(lvef):
            if float(lvef) <= 40:
                return tr("Reduced LVEF", "FEVE reduzida")
            if float(lvef) <= 49:
                return tr("Mildly reduced LVEF", "FEVE levemente reduzida")
            return tr("Preserved LVEF", "FEVE preservada")
    if base_feature == "Cr clearance, ml/min *":
        cc = parse_number(val)
        if pd.notna(cc):
            if float(cc) <= 50:
                return tr("Severely impaired renal function", "Função renal gravemente comprometida")
            if float(cc) <= 85:
                return tr("Moderately impaired renal function", "Função renal moderadamente comprometida")
            return tr("Preserved renal function", "Função renal preservada")
    if base_feature == "Creatinine (mg/dL)":
        c = parse_number(val)
        if pd.notna(c):
            if float(c) >= 2.0:
                return tr("Elevated creatinine", "Creatinina elevada")
            return tr("Lower creatinine level", "Creatinina mais baixa")
    if base_feature == "PSAP":
        ps = parse_number(val)
        if pd.notna(ps):
            if float(ps) >= 55:
                return tr("Pulmonary hypertension", "Hipertensão pulmonar")
            if float(ps) >= 31:
                return tr("Mild-to-moderate pulmonary pressure elevation", "Elevação leve a moderada da pressão pulmonar")
            return tr("Lower pulmonary pressure", "Pressão pulmonar mais baixa")
    if base_feature == "TAPSE":
        tp = parse_number(val)
        if pd.notna(tp):
            if float(tp) < 17:
                return tr("Reduced right ventricular systolic function", "Função sistólica do ventrículo direito reduzida")
            return tr("Preserved right ventricular systolic function", "Função sistólica do ventrículo direito preservada")
    if base_feature == "cirurgia_combinada":
        return tr("Combined surgery", "Cirurgia combinada")
    if base_feature == "thoracic_aorta_flag":
        return tr("Thoracic aorta surgery", "Cirurgia da aorta torácica")
    mapping = {
        "Critical preoperative state": tr("Critical preoperative state", "Estado crítico pré-operatório"),
        "Dialysis": tr("Dialysis", "Diálise"),
        "IE": tr("Active/probable endocarditis", "Endocardite ativa/provável"),
        "PVD": tr("Peripheral vascular disease", "Doença vascular periférica"),
        "Diabetes": tr("Diabetes treatment burden", "Carga clínica do diabetes"),
        "Surgical Priority": tr("Surgical priority", "Prioridade cirúrgica"),
        "No. of Diseased Vessels": tr("Extent of coronary disease", "Extensão da doença coronariana"),
        "Left Main Stenosis ≥ 50%": tr("Left main stenosis ≥ 50%", "Estenose de tronco ≥ 50%"),
        "Proximal LAD Stenosis ≥ 70%": tr("Proximal LAD stenosis ≥ 70%", "Estenose proximal de DA ≥ 70%"),
        "Preoperative NYHA": tr("NYHA functional class", "Classe funcional NYHA"),
        "CCS4": tr("CCS class 4", "CCS classe 4"),
        "Poor mobility": tr("Poor mobility", "Mobilidade reduzida"),
        "Chronic Lung Disease": tr("Chronic lung disease", "Doença pulmonar crônica"),
        "Aortic Root Abscess": tr("Aortic root abscess", "Abscesso de raiz aórtica"),
        "Aortic Stenosis": tr("Aortic stenosis", "Estenose aórtica"),
        "Aortic Regurgitation": tr("Aortic regurgitation", "Insuficiência aórtica"),
        "Mitral Stenosis": tr("Mitral stenosis", "Estenose mitral"),
        "Mitral Regurgitation": tr("Mitral regurgitation", "Insuficiência mitral"),
        "Tricuspid Regurgitation": tr("Tricuspid regurgitation", "Insuficiência tricúspide"),
        "HF": tr("Heart failure", "Insuficiência cardíaca"),
        "Hypertension": tr("Hypertension", "Hipertensão"),
        "Dyslipidemia": tr("Dyslipidemia", "Dislipidemia"),
        "CVA": tr("Cerebrovascular disease", "Doença cerebrovascular"),
        "Cancer ≤ 5 yrs": tr("Cancer within 5 years", "Câncer nos últimos 5 anos"),
        "Arrhythmia Remote": tr("Remote arrhythmia history", "História de arritmia remota"),
        "Arrhythmia Recent": tr("Recent arrhythmia", "Arritmia recente"),
        "Family Hx of CAD": tr("Family history of CAD", "História familiar de DAC"),
        "Alcohol": tr("Relevant alcohol use", "Uso relevante de álcool"),
        "Pneumonia": tr("Recent pneumonia", "Pneumonia recente"),
        "Anticoagulation/ Antiaggregation": tr("Anticoagulation or antiplatelet therapy", "Anticoagulação ou antiagregação"),
    }
    return mapping.get(base_feature, base_feature)


def _explain_patient_risk(artifacts, input_features: pd.DataFrame, form_map: Dict, tr, top_n: int = 5):
    if "LogisticRegression" not in artifacts.fitted_models:
        return pd.DataFrame(), pd.DataFrame()
    pipe = artifacts.fitted_models["LogisticRegression"]
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    transformed = prep.transform(input_features)
    arr = np.asarray(transformed)
    if arr.ndim == 2:
        arr = arr[0]
    feature_names = [_feat_display_name(n) for n in prep.get_feature_names_out()]
    coef = np.asarray(model.coef_).ravel()
    rows = []
    for feat, val, c in zip(feature_names, arr, coef):
        contrib = float(val * c)
        if abs(contrib) < 1e-9:
            continue
        base = _resolve_base_feature(feat, artifacts.feature_columns)
        rows.append({
            "base": base,
            tr("Contribution", "Contribuição"): contrib,
        })
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    expl = pd.DataFrame(rows).groupby("base", as_index=False)[tr("Contribution", "Contribuição")].sum()
    expl[tr("Factor", "Fator")] = expl["base"].map(lambda x: _patient_factor_label(x, form_map, tr))
    expl[tr("Group", "Grupo")] = expl["base"].map(lambda x: _feature_group(x, tr))
    pos = expl[expl[tr("Contribution", "Contribuição")] > 0].sort_values(tr("Contribution", "Contribuição"), ascending=False)
    neg = expl[expl[tr("Contribution", "Contribuição")] < 0].sort_values(tr("Contribution", "Contribuição"), ascending=True)
    pos = pos[[tr("Factor", "Fator"), tr("Group", "Grupo"), tr("Contribution", "Contribuição")]].head(top_n)
    neg = neg[[tr("Factor", "Fator"), tr("Group", "Grupo"), tr("Contribution", "Contribuição")]].head(top_n)
    return pos, neg


def _explain_table_column_config(tr) -> dict:
    return {
        tr("Factor", "Fator"): st.column_config.TextColumn(
            tr("Factor", "Fator"),
            help=tr(
                "Patient-specific factor summarized in clinically interpretable language.",
                "Fator específico do paciente resumido em linguagem clinicamente interpretável.",
            ),
        ),
        tr("Group", "Grupo"): st.column_config.TextColumn(
            tr("Group", "Grupo"),
            help=tr(
                "Clinical domain of the factor.",
                "Domínio clínico do fator.",
            ),
        ),
        tr("Contribution", "Contribuição"): st.column_config.NumberColumn(
            tr("Contribution", "Contribuição"),
            help=tr(
                "Approximate contribution from the interpretable logistic layer. Higher positive values suggest stronger association with increased risk; more negative values suggest stronger association with reduced risk.",
                "Contribuição aproximada da camada logística interpretável. Valores positivos maiores sugerem associação mais forte com aumento do risco; valores mais negativos sugerem associação mais forte com redução do risco.",
            ),
            format="%.4f",
        ),
    }


def _data_quality_alerts(form_map: Dict, prepared, tr) -> list:
    alerts: list = []
    df_ref = prepared.data
    age = parse_number(form_map.get("Age (years)"))
    weight = parse_number(form_map.get("Weight (kg)"))
    height = parse_number(form_map.get("Height (cm)"))
    creatinine = parse_number(form_map.get("Creatinine (mg/dL)"))
    lvef = parse_number(form_map.get("Pré-LVEF, %"))
    psap = parse_number(form_map.get("PSAP"))
    hct = parse_number(form_map.get("Hematocrit (%)"))

    if pd.isna(hct):
        alerts.append((tr("Warning", "Atenção"), tr("Hematocrit is missing. This may reduce confidence in the prediction.", "Hematócrito ausente. Isso pode reduzir a confiança na predição.")))
    if pd.isna(lvef):
        alerts.append((tr("Warning", "Atenção"), tr("LVEF is missing. This is an important cardiac risk predictor.", "FEVE ausente. Este é um preditor cardíaco importante de risco.")))
    if pd.notna(lvef) and not (5 <= float(lvef) <= 90):
        alerts.append((tr("Critical", "Crítico"), tr("LVEF is outside the plausible clinical range. Please verify the value.", "A FEVE está fora da faixa clínica plausível. Verifique o valor informado.")))
    if pd.notna(creatinine) and creatinine > 20:
        alerts.append((tr("Critical", "Crítico"), tr("Creatinine appears incompatible with the expected mg/dL unit. Please verify the unit.", "A creatinina parece incompatível com a unidade esperada em mg/dL. Verifique a unidade.")))
    if pd.notna(creatinine) and creatinine < 0.2:
        alerts.append((tr("Warning", "Atenção"), tr("Creatinine is unusually low. Please verify the value and unit.", "A creatinina está muito baixa. Verifique o valor e a unidade.")))
    if pd.notna(psap) and not (10 <= float(psap) <= 150):
        alerts.append((tr("Warning", "Atenção"), tr("PSAP is outside the plausible clinical range. Please verify the value.", "A PSAP está fora da faixa clínica plausível. Verifique o valor informado.")))
    if pd.notna(weight) and not (20 <= float(weight) <= 250):
        alerts.append((tr("Warning", "Atenção"), tr("Weight is outside the expected input range.", "O peso está fora da faixa de entrada esperada.")))
    if pd.notna(height) and not (120 <= float(height) <= 230):
        alerts.append((tr("Warning", "Atenção"), tr("Height is outside the expected input range.", "A altura está fora da faixa de entrada esperada.")))
    if pd.notna(age):
        ref_age = pd.to_numeric(df_ref["Age (years)"], errors="coerce")
        if age < ref_age.min() or age > ref_age.max():
            alerts.append((tr("Informative", "Informativo"), tr(f"Age is outside the range observed in the training dataset ({ref_age.min():.0f}-{ref_age.max():.0f} years).", f"A idade está fora da faixa observada na base de treinamento ({ref_age.min():.0f}-{ref_age.max():.0f} anos).")))
    return alerts


def _prediction_uncertainty(patient_pred_df: pd.DataFrame, prob_col: str, imputed_features: int, tr) -> tuple:
    vals = patient_pred_df[prob_col].str.replace("%", "", regex=False).astype(float) / 100.0
    low = float(vals.quantile(0.25))
    high = float(vals.quantile(0.75))
    spread = high - low
    if spread < 0.04:
        confidence = tr("High model agreement", "Alta concordância entre modelos")
    elif spread < 0.10:
        confidence = tr("Moderate model agreement", "Concordância moderada entre modelos")
    else:
        confidence = tr("Low model agreement", "Baixa concordância entre modelos")
    range_text = tr(
        f"Interquartile risk range: {low*100:.1f}% to {high*100:.1f}%.",
        f"Faixa interquartil de risco: {low*100:.1f}% a {high*100:.1f}%.",
    )
    return range_text, confidence


_MODEL_DISAGREEMENT_RANGE_THRESHOLD = 0.10


def _candidate_model_disagreement_summary(model_probs: Dict) -> dict:
    clean = {
        str(name): float(prob)
        for name, prob in model_probs.items()
        if prob is not None and np.isfinite(prob)
    }
    if len(clean) < 2:
        return {
            "n": len(clean),
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "median": np.nan,
            "iqr": np.nan,
            "high": False,
            "low_end_boosting": [],
        }

    vals = pd.Series(clean, dtype=float)
    low_end_boosting = [
        name
        for name in ("LightGBM", "XGBoost")
        if name in clean and clean[name] <= 1e-4
    ]
    return {
        "n": int(len(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "range": float(vals.max() - vals.min()),
        "median": float(vals.median()),
        "iqr": float(vals.quantile(0.75) - vals.quantile(0.25)),
        "high": bool((vals.max() - vals.min()) > _MODEL_DISAGREEMENT_RANGE_THRESHOLD),
        "low_end_boosting": low_end_boosting,
    }


def _risk_badge(p: float, tr) -> str:
    if np.isnan(p):
        return "Not available"
    label = class_risk(float(p))
    label_map = {"Low": tr("Low", "Baixo"), "Intermediate": tr("Intermediate", "Intermediário"), "High": tr("High", "Alto")}
    return f"{label_map.get(label, label)} ({100*p:.1f}%)"


def _sts_score_patient_id(row: dict):
    if not isinstance(row, dict):
        return None
    for k in ("_patient_key", "patient_id", "Name", "Nome"):
        v = row.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in ("nan", "none", "-"):
            return s
    return None


def render(ctx: "TabContext") -> None:
    tr = ctx.tr
    hp = ctx.hp
    artifacts = ctx.artifacts
    prepared = ctx.prepared
    df = ctx.df
    forced_model = ctx.forced_model
    best_model_name = ctx.best_model_name
    bundle_info = ctx.bundle_info
    xlsx_path = ctx.xlsx_path
    _default_threshold = ctx.default_threshold
    MODEL_VERSION = ctx.model_version
    language = ctx.language
    _bytes_download_btn = ctx.bytes_download_btn
    _txt_download_btn = ctx.txt_download_btn
    _safe_prob = ctx.safe_prob
    HAS_STS = ctx.has_sts
    general_table_column_config = ctx.general_table_column_config
    stats_table_column_config = ctx.stats_table_column_config

    # Rebuild model_options from artifacts (same logic as app.py)
    model_options = artifacts.leaderboard["Modelo"].tolist()

    # Rebuild surgery_component_options (same logic as app.py)
    from risk_data import split_surgery_procedures
    _procedure_item_map: dict = {}
    for _row in prepared.data["Surgery"].dropna().unique():
        for _part in split_surgery_procedures(str(_row)):
            _part = _part.strip()
            if _part:
                _procedure_item_map[_part.lower()] = _part
    surgery_component_options = sorted(_procedure_item_map.values(), key=str.lower)

    st.subheader(tr("Prediction", "Predição"))
    st.caption(tr(
        "Individual patient prediction with AI Risk, EuroSCORE II, STS Score, case completeness, and local interpretation.",
        "Predição individual com AI Risk, EuroSCORE II, STS Score, completude do caso e interpretação local.",
    ))
    _pred_top1, _pred_top2, _pred_top3, _pred_top4 = st.columns(4)
    _pred_top1.metric(tr("Primary AI model", "Modelo IA principal"), forced_model, border=True)
    _pred_top2.metric(tr("Comparators", "Comparadores"), "EuroSCORE II / STS", border=True)
    _pred_top3.metric(tr("Operational threshold", "Limiar operacional"), f"{_default_threshold:.0%}", border=True)
    _pred_top4.metric(tr("Model predictors", "Preditores do modelo"), f"{len(artifacts.feature_columns)}", border=True)

    def yn_pt_to_en(v: str) -> str:
        return "Yes" if str(v).strip().lower() in {"sim", "yes"} else "No"

    yn_options = [tr("No", "Não"), tr("Yes", "Sim")]

    def cockcroft_gault(age_years: float, weight_kg: float, scr_mg_dl: float, sex_code: str) -> float:
        if scr_mg_dl <= 0:
            return np.nan
        factor = 0.85 if sex_code == "F" else 1.0
        return ((140.0 - age_years) * weight_kg * factor) / (72.0 * scr_mg_dl)

    def kdigo_from_clearance(crcl: float) -> str:
        if pd.isna(crcl):
            return "Unknown"
        if crcl >= 90:
            return "G1"
        if crcl >= 60:
            return "G2"
        if crcl >= 45:
            return "G3a"
        if crcl >= 30:
            return "G3b"
        if crcl >= 15:
            return "G4"
        return "G5"

    diabetes_map = {
        "Não": "No",
        "No": "No",
        "Oral": "Oral",
        "Insulina": "Insulin",
        "Insulin": "Insulin",
        "Dieta": "Diet Only",
        "Diet": "Diet Only",
        "Sem método de controle": "No Control Method",
        "No Control Method": "No Control Method",
    }
    urgency_map = {
        "Eletiva": "Elective",
        "Elective": "Elective",
        "Urgente": "Urgent",
        "Urgent": "Urgent",
        "Emergência": "Emergent",
        "Emergency": "Emergent",
        "Salvamento": "Emergent Salvage",
        "Salvage": "Emergent Salvage",
    }
    coronary_map = {
        "Sem sintomas coronarianos": "No coronary symptoms",
        "No coronary symptoms": "No coronary symptoms",
        "Angina estável": "Stable Angina",
        "Stable angina": "Stable Angina",
        "Angina instável": "Unstable Angina",
        "Unstable angina": "Unstable Angina",
        "IAM sem supra de ST": "Non-STEMI",
        "NSTEMI": "Non-STEMI",
        "IAM com supra de ST": "STEMI",
        "STEMI": "STEMI",
        "Equivalente anginoso": "Angina Equivalent",
        "Angina Equivalent": "Angina Equivalent",
        "Outro": "Other",
        "Other": "Other",
    }

    st.divider()
    st.markdown(tr("### Inputs", "### Entradas"))
    st.caption(tr(
        "Fill in the clinical profile below. Detailed echo and lab measurements can remain blank when unavailable; missing model inputs are handled by the trained pipeline and summarized after prediction.",
        "Preencha o perfil clínico abaixo. Medidas detalhadas de eco e laboratório podem ficar em branco quando indisponíveis; entradas ausentes são tratadas pelo pipeline treinado e resumidas após a predição.",
    ))

    with st.container():
        st.markdown(tr("**General data**", "**Dados gerais**"))
        gx1, gx2, gx3 = st.columns(3)

        with gx1:
            st.markdown(tr("**Demographics and anthropometrics**", "**Demografia e antropometria**"))
            age = st.number_input(tr("Age (years)", "Idade (anos)"), 18, 100, 65, help=hp("Chronological age at surgery. Older age usually increases baseline operative risk.", "Idade cronológica no momento da cirurgia. Idade mais avançada geralmente aumenta o risco operatório basal."))
            sex = st.selectbox(tr("Sex", "Sexo"), ["M", "F"], help=hp("Biological sex used in clinical risk equations such as EuroSCORE II and Cockcroft-Gault.", "Sexo biológico usado em equações clínicas como EuroSCORE II e Cockcroft-Gault."))
            race = st.selectbox(
                tr("Race", "Raça"),
                ["White", "Black", "Asian", "Mixed", "Other"],
                index=0,
                help=hp("Self-reported race/ethnicity. Used in the STS calculator and as a predictor in the ML model.", "Raça/etnia autorreferida. Usada na calculadora STS e como preditor no modelo de ML."),
            )
            weight_kg = st.number_input(tr("Weight (kg)", "Peso (kg)"), 20.0, 250.0, 75.0, help=hp("Body weight in kilograms. Used in renal function estimation and may affect procedural risk indirectly.", "Peso corporal em quilogramas. Usado na estimativa da função renal e pode influenciar o risco de forma indireta."))
            height_cm = st.number_input(tr("Height (cm)", "Altura (cm)"), 120.0, 230.0, 168.0, help=hp("Height in centimeters. Helps characterize anthropometry and supports risk stratification.", "Altura em centímetros. Ajuda a caracterizar a antropometria e pode apoiar a estratificação de risco."))
            bsa = 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)
            st.metric(tr("BSA (Du Bois)", "ASC (Du Bois)"), f"{bsa:.2f} m²")

        with gx2:
            st.markdown(tr("**Preoperative clinical status**", "**Condição clínica pré-operatória**"))
            nyha = st.selectbox("NYHA", ["I", "II", "III", "IV"], index=0, help=hp("Functional class of heart failure symptoms. Higher class usually indicates worse clinical status.", "Classe funcional dos sintomas de insuficiência cardíaca. Classes mais altas geralmente indicam pior condição clínica."))
            ccs4 = st.selectbox(tr("CCS class 4", "CCS classe 4"), yn_options, index=0, help=hp("Canadian Cardiovascular Society class 4 angina: symptoms at rest or with minimal activity.", "Classe 4 da Canadian Cardiovascular Society: angina em repouso ou com atividade mínima."))
            prior = st.selectbox(tr("Previous cardiac surgery", "Cirurgia cardíaca prévia"), yn_options, index=0, help=hp("Indicates previous major cardiac surgery. Reoperation usually increases surgical complexity and risk.", "Indica cirurgia cardíaca prévia de grande porte. Reoperação geralmente aumenta a complexidade e o risco cirúrgico."))
            urgency_pt = st.selectbox(tr("Surgical priority", "Prioridade cirúrgica"), [tr("Elective", "Eletiva"), tr("Urgent", "Urgente"), tr("Emergency", "Emergência"), tr("Salvage", "Salvamento")], index=0, help=hp("Urgency of the operation. More urgent procedures are usually associated with higher perioperative risk.", "Urgência do procedimento. Cirurgias mais urgentes costumam estar associadas a maior risco perioperatório."))

        with gx3:
            st.markdown(tr("**Procedure and external scores**", "**Procedimento e escores externos**"))
            surgery_selected = st.multiselect(
                tr("Planned surgery", "Cirurgia planejada"),
                surgery_component_options,
                default=["CABG"] if "CABG" in surgery_component_options else [],
                help=hp("Select one or more planned major procedures. The combination is used for procedural weighting in EuroSCORE II and in the local model.", "Selecione um ou mais procedimentos maiores planejados. A combinação é usada para o peso do procedimento no EuroSCORE II e no modelo local."),
            )
            use_custom_surgery = st.checkbox(
                tr("Add custom procedure text", "Adicionar procedimento personalizado"),
                value=False,
            )
            custom_surgery = ""
            if use_custom_surgery:
                custom_surgery = st.text_input(
                    tr("Custom procedure", "Procedimento personalizado"),
                    value="",
                ).strip()

            surgery_parts = list(surgery_selected)
            if custom_surgery:
                surgery_parts.append(custom_surgery)
            surgery = ", ".join([p for p in surgery_parts if str(p).strip()])
            st.caption(tr("STS Score will be calculated automatically via the STS Score web calculator.", "O STS Score será calculado automaticamente via a calculadora web do STS Score."))

        st.markdown(tr("**Coronary artery disease**", "**Doença coronariana**"))
        cor1, cor2 = st.columns(2)
        with cor1:
            coronary_pt = st.selectbox(
                tr("Coronary symptom", "Sintoma coronariano"),
                [
                    tr("No coronary symptoms", "Sem sintomas coronarianos"),
                    tr("Stable angina", "Angina estável"),
                    tr("Unstable angina", "Angina instável"),
                    tr("NSTEMI", "IAM sem supra de ST"),
                    tr("STEMI", "IAM com supra de ST"),
                    tr("Angina Equivalent", "Equivalente anginoso"),
                    tr("Other", "Outro"),
                ],
                index=0,
                help=hp("Current coronary presentation. It may proxy recent ischemic instability in the risk models.", "Apresentação coronariana atual. Pode funcionar como marcador indireto de instabilidade isquêmica recente nos modelos."),
            )
            diseased_vessels = st.number_input(tr("Number of diseased vessels", "Nº de vasos doentes"), 0, 3, 0, help=hp("Number of major coronary vessels with significant disease. Higher burden suggests more extensive coronary disease.", "Número de vasos coronarianos principais com doença significativa. Valores maiores sugerem doença coronariana mais extensa."))
        with cor2:
            left_main_50 = st.selectbox(tr("Left main stenosis ≥ 50%", "Estenose de tronco da coronária esquerda ≥ 50%"), yn_options, index=0, help=hp("Relevant stenosis in the left main coronary artery. Usually indicates higher anatomical risk.", "Estenose relevante no tronco da coronária esquerda. Geralmente indica maior gravidade anatômica."))
            prox_lad_70 = st.selectbox(tr("Proximal LAD stenosis ≥ 70%", "Estenose proximal de DA ≥ 70%"), yn_options, index=0, help=hp("Relevant proximal LAD lesion. Often associated with higher ischemic burden.", "Lesão relevante na DA proximal. Frequentemente associada a maior carga isquêmica."))

        st.markdown(tr("**Comorbidities**", "**Comorbidades**"))
        cx1, cx2, cx3, cx4 = st.columns(4)
        with cx1:
            hf = st.selectbox(tr("Heart failure (HF)", "Insuficiência cardíaca (HF)"), [tr("None", "Nenhuma"), tr("Acute", "Aguda"), tr("Chronic", "Crônica"), tr("Both", "Ambas")], index=0, help=hp("Clinical diagnosis of heart failure before surgery. Specify timing: Acute, Chronic, or Both.", "Diagnóstico clínico de insuficiência cardíaca antes da cirurgia. Especifique: Aguda, Crônica ou Ambas."))
            htn = st.selectbox(tr("Hypertension", "Hipertensão"), yn_options, index=0, help=hp("History of systemic arterial hypertension.", "História de hipertensão arterial sistêmica."))
            dlp = st.selectbox(tr("Dyslipidemia", "Dislipidemia"), yn_options, index=0, help=hp("Presence of dyslipidemia or lipid-lowering treatment.", "Presença de dislipidemia ou uso de tratamento redutor de lipídios."))
            diabetes_pt = st.selectbox(tr("Diabetes", "Diabetes"), [tr("No", "Não"), "Oral", tr("Insulin", "Insulina"), tr("Diet", "Dieta"), tr("No Control Method", "Sem método de controle")], index=0, help=hp("Diabetes treatment category. Insulin usually indicates more severe metabolic disease in risk models.", "Categoria de tratamento do diabetes. Uso de insulina costuma indicar doença metabólica mais grave nos modelos de risco."))
        with cx2:
            cva = st.selectbox(tr("Cerebrovascular disease (CVA)", "Doença cerebrovascular (CVA)"), [tr("No", "Não"), "<= 30 days", ">= 30 days", tr("Timing unk", "Timing desconhecido"), "TIA", tr("Other CVD", "Outra DCV")], index=0, help=hp("History of stroke or cerebrovascular disease. Specify timing when known.", "História de AVC ou doença cerebrovascular. Especifique o timing quando conhecido."))
            pvd2 = st.selectbox(tr("Peripheral vascular disease (PVD)", "Doença vascular periférica (PVD)"), yn_options, index=0, key="pvd_comorb", help=hp("Peripheral arterial disease. In EuroSCORE II it is used as an approximation of extracardiac arteriopathy.", "Doença arterial periférica. No EuroSCORE II é usada como aproximação de extracardiac arteriopathy."))
            cancer5 = st.selectbox(tr("Cancer <= 5 years", "Câncer <= 5 anos"), yn_options, index=0, help=hp("History of cancer diagnosed or treated within the last 5 years.", "História de câncer diagnosticado ou tratado nos últimos 5 anos."))
            dialysis = st.selectbox(tr("Dialysis", "Diálise"), yn_options, index=0, help=hp("Indicates established dialysis therapy. Strong marker of severe renal dysfunction.", "Indica terapia dialítica estabelecida. Marcador forte de disfunção renal grave."))
        with cx3:
            _arrhythmia_options = ["None", tr("Atrial Fibrillation", "Fibrilação Atrial"), tr("Atrial Flutter", "Flutter Atrial"), "V. Tach / V. Fib", tr("3rd Degree Block", "Bloqueio 3º Grau")]
            arr_rem = st.selectbox(tr("Remote arrhythmia", "Arritmia remota"), _arrhythmia_options, index=0, help=hp("Past arrhythmia type. 'None' means no history of arrhythmia.", "Tipo de arritmia pregressa. 'None' indica ausência de histórico de arritmia."))
            arr_rec = st.selectbox(tr("Recent arrhythmia", "Arritmia recente"), _arrhythmia_options, index=0, help=hp("Recent arrhythmia before surgery. 'None' means no recent arrhythmia.", "Arritmia recente antes da cirurgia. 'None' indica ausência de arritmia recente."))
            fam_cad = st.selectbox(tr("Family history of CAD", "História familiar de DAC"), yn_options, index=0, help=hp("Family history of coronary artery disease.", "História familiar de doença arterial coronariana."))
            ie_pt = st.selectbox(tr("Active/probable endocarditis", "Endocardite ativa/provável"), [tr("No", "Não"), tr("Yes", "Sim"), tr("Possible", "Possível")], index=0, help=hp("Active or probable infective endocarditis at the time of surgery. In this app, 'Possible' is treated as positive for EuroSCORE II operationalization.", "Endocardite infecciosa ativa ou provável no momento da cirurgia. Neste app, 'Possível' é tratada como positiva na operacionalização do EuroSCORE II."))
        with cx4:
            smoker = st.selectbox(tr("Smoking", "Tabagismo"), [tr("Never", "Nunca"), tr("Current", "Atual"), tr("Former", "Ex-tabagista")], index=0, help=hp("Smoking status. Active or former smoking may reflect pulmonary and vascular risk burden.", "Situação tabágica. Tabagismo atual ou prévio pode refletir maior carga de risco pulmonar e vascular."))
            _smoker_en = str(smoker).strip().lower()
            _smoking_status = "Current" if _smoker_en in {"atual", "current"} else "Former" if _smoker_en in {"ex-tabagista", "former"} else "Never"
            alcohol = st.selectbox(tr("Alcohol", "Álcool"), yn_options, index=0, help=hp("History of relevant alcohol use.", "História de uso relevante de álcool."))
            recent_pneum = st.selectbox(tr("Recent pneumonia", "Pneumonia recente"), [tr("No", "Não"), tr("Under treatment", "Em tratamento"), tr("Treated", "Tratada")], index=0, help=hp("Recent pneumonia before surgery. Specify if still under treatment or already treated.", "Pneumonia recente antes da cirurgia. Especifique se ainda em tratamento ou já tratada."))
            cpd = st.selectbox(tr("Chronic lung disease", "Doença pulmonar crônica"), yn_options, index=0, help=hp("Chronic pulmonary disease. This usually increases operative risk and is used in EuroSCORE II operationalization.", "Doença pulmonar crônica. Geralmente aumenta o risco operatório e é usada na operacionalização do EuroSCORE II."))

        ox1, ox2 = st.columns(2)
        with ox1:
            mobility = st.selectbox(tr("Severely reduced mobility", "Mobilidade reduzida grave"), yn_options, index=0, help=hp("Marked limitation due to musculoskeletal or neurologic disease. Relevant in EuroSCORE II.", "Limitação importante por doença musculoesquelética ou neurológica. Relevante no EuroSCORE II."))
        with ox2:
            critical = st.selectbox(tr("Critical preoperative state", "Estado crítico pré-operatório"), yn_options, index=0, help=hp("Severe preoperative instability such as shock, inotropes, ventilation, resuscitation, or other critical support.", "Instabilidade pré-operatória grave, como choque, uso de inotrópicos, ventilação, reanimação ou outro suporte crítico."))

        st.markdown(tr("**Laboratory data**", "**Dados laboratoriais**"))
        lx1, lx2, lx3 = st.columns(3)
        with lx1:
            creatinine = st.number_input(tr("Creatinine (mg/dL)", "Creatinina (mg/dL)"), 0.1, 20.0, 1.0, help=hp("Serum creatinine used for renal function estimation. Higher values usually indicate worse kidney function.", "Creatinina sérica usada na estimativa da função renal. Valores maiores geralmente indicam pior função renal."))
            creat_clear = cockcroft_gault(float(age), float(weight_kg), float(creatinine), str(sex))
            kdigo = kdigo_from_clearance(creat_clear)
            st.metric(tr("Creatinine clearance (Cockcroft-Gault)", "Clearance (Cockcroft-Gault)"), f"{creat_clear:.1f} ml/min")
            st.metric(tr("Estimated KDIGO", "KDIGO estimado"), kdigo)
        with lx2:
            hematocrit = st.number_input(tr("Hematocrit (%)", "Hematócrito (%)"), 10.0, 65.0, 40.0, help=hp("Percentage of blood volume occupied by red cells. Low values may indicate anemia and reduced physiological reserve.", "Percentual do volume sanguíneo ocupado por hemácias. Valores baixos podem indicar anemia e menor reserva fisiológica."))
            wbc = st.number_input(tr("WBC count (10^3/μL)", "Leucócitos (10^3/μL)"), 0.5, 40.0, 7.0, help=hp("White blood cell count. Marked elevations may reflect inflammation or infection.", "Contagem de leucócitos. Elevações importantes podem refletir inflamação ou infecção."))
        with lx3:
            platelets = st.number_input(tr("Platelets (cells/μL)", "Plaquetas (cells/μL)"), 10000.0, 1000000.0, 250000.0, step=1000.0, help=hp("Platelet count. Low values may suggest bleeding risk or systemic illness.", "Contagem de plaquetas. Valores baixos podem sugerir risco de sangramento ou doença sistêmica."))
            inr = st.number_input("INR", 0.5, 10.0, 1.0, help=hp("International normalized ratio. Reflects coagulation status and anticoagulation effect.", "Razão normalizada internacional. Reflete o estado de coagulação e o efeito de anticoagulação."))
            ptt = st.number_input("PTT", 10.0, 180.0, 30.0, help=hp("Partial thromboplastin time. Useful for coagulation assessment.", "Tempo de tromboplastina parcial. Útil para avaliação da coagulação."))
            anticoag_pt = st.selectbox(tr("Anticoagulation / antiplatelet therapy", "Anticoagulação / antiagregação"), yn_options, index=0, help=hp("Indicates current anticoagulant or antiplatelet treatment before surgery.", "Indica uso atual de anticoagulante ou antiagregante antes da cirurgia."))
            suspension_days = 0
            if yn_pt_to_en(anticoag_pt) == "Yes":
                _susp_yn = st.selectbox(
                    tr("Suspended before surgery?", "Suspensa antes da cirurgia?"),
                    yn_options, index=0, key="susp_anticoag_yn",
                    help=hp("Whether anticoagulation was suspended before the procedure.", "Se a anticoagulação foi suspensa antes do procedimento."),
                )
                if yn_pt_to_en(_susp_yn) == "Yes":
                    suspension_days = st.number_input(
                        tr("Days since suspension", "Dias desde a suspensão"),
                        min_value=1, max_value=90, value=5, step=1,
                        help=hp("Number of days since anticoagulation was suspended.", "Número de dias desde a suspensão da anticoagulação."),
                    )

        st.markdown(tr("**Echocardiographic data**", "**Dados ecocardiográficos**"))
        st.caption(tr("Valve disease grading: None, Trivial, Mild, Moderate, Severe, Unknown", "Classificação de valvopatias: None, Trivial, Mild, Moderate, Severe, Unknown"))
        st.markdown(tr("**Ventricular function and hemodynamics**", "**Função ventricular e hemodinâmica**"))
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            lvef = st.number_input(tr("LVEF (%)", "FEVE (%)"), 5.0, 90.0, 50.0, help=hp("Left ventricular ejection fraction. Lower values suggest poorer systolic function and higher operative risk.", "Fração de ejeção do ventrículo esquerdo. Valores mais baixos sugerem pior função sistólica e maior risco operatório."))
        with ex2:
            psap = st.number_input("PSAP (mmHg)", 10.0, 150.0, 35.0, help=hp("Pulmonary artery systolic pressure. Higher values suggest pulmonary hypertension.", "Pressão sistólica da artéria pulmonar. Valores altos sugerem hipertensão pulmonar."))
        with ex3:
            tapse = st.number_input("TAPSE (mm)", 5.0, 40.0, 20.0, help=hp("Tricuspid annular plane systolic excursion. Lower values suggest worse right ventricular systolic function.", "Excursão sistólica do plano do anel tricúspide. Valores baixos sugerem pior função sistólica do ventrículo direito."))

        st.markdown(tr("**Valve disease**", "**Valvopatias**"))
        vx1, vx2, vx3 = st.columns(3)
        sev_choices = ["None", "Trivial", "Mild", "Moderate", "Severe", "Unknown"]
        with vx1:
            st.markdown(tr("**Aortic valve**", "**Valva Aórtica**"))
            a_stenosis = st.selectbox(tr("Aortic stenosis", "Estenose aórtica"), sev_choices, index=0, help=hp("Severity of aortic stenosis. Higher grades indicate more severe valve obstruction.", "Gravidade da estenose aórtica. Graus mais altos indicam obstrução valvar mais importante."))
            a_reg = st.selectbox(tr("Aortic regurgitation", "Insuficiência aórtica"), sev_choices, index=0, help=hp("Severity of aortic regurgitation. Higher grades indicate larger regurgitant burden.", "Gravidade da insuficiência aórtica. Graus mais altos indicam maior carga regurgitante."))
        with vx2:
            st.markdown(tr("**Mitral valve**", "**Valva Mitral**"))
            m_stenosis = st.selectbox(tr("Mitral stenosis", "Estenose mitral"), sev_choices, index=0, help=hp("Severity of mitral stenosis.", "Gravidade da estenose mitral."))
            m_reg = st.selectbox(tr("Mitral regurgitation", "Insuficiência mitral"), sev_choices, index=0, help=hp("Severity of mitral regurgitation.", "Gravidade da insuficiência mitral."))
        with vx3:
            st.markdown(tr("**Tricuspid valve**", "**Valva Tricúspide**"))
            t_reg = st.selectbox(tr("Tricuspid regurgitation", "Insuficiência tricúspide"), sev_choices, index=0, help=hp("Severity of tricuspid regurgitation.", "Gravidade da insuficiência tricúspide."))

        with st.expander(tr("Detailed valve measurements (optional)", "Medidas valvares detalhadas (opcional)"), expanded=False):
            st.caption(tr(
                "Fill in only if echocardiographic measurements are available. Leave empty to use training median (imputation).",
                "Preencha apenas se as medidas ecocardiográficas estiverem disponíveis. Deixe vazio para usar a mediana do treinamento (imputação).",
            ))
            vm1, vm2 = st.columns(2)
            with vm1:
                st.markdown(tr("**Aortic valve**", "**Valva Aórtica**"))
                ava = st.number_input(tr("AVA (cm²)", "AVA (cm²)"), min_value=0.0, max_value=6.0, value=None, step=0.1, format="%.2f", help=hp("Aortic valve area by planimetry or continuity equation.", "Área valvar aórtica por planimetria ou equação de continuidade."))
                ao_grad = st.number_input(tr("Aortic mean gradient (mmHg)", "Gradiente médio aórtico (mmHg)"), min_value=0.0, max_value=100.0, value=None, step=1.0, format="%.1f", help=hp("Mean transaortic pressure gradient.", "Gradiente de pressão transaórtico médio."))
                pht_ao = st.number_input(tr("PHT Aortic (ms)", "PHT Aórtico (ms)"), min_value=0.0, max_value=1000.0, value=None, step=10.0, format="%.0f", help=hp("Pressure half-time of aortic regurgitation.", "Tempo de meia-pressão da insuficiência aórtica."))
                vc_ao = st.number_input(tr("Vena contracta aortic (mm)", "Vena contracta aórtica (mm)"), min_value=0.0, max_value=15.0, value=None, step=0.5, format="%.1f", help=hp("Vena contracta width of aortic regurgitation jet.", "Largura da vena contracta do jato de insuficiência aórtica."))
            with vm2:
                st.markdown(tr("**Mitral valve**", "**Valva Mitral**"))
                mva = st.number_input(tr("MVA (cm²)", "AVM (cm²)"), min_value=0.0, max_value=6.0, value=None, step=0.1, format="%.2f", help=hp("Mitral valve area by planimetry or PHT.", "Área valvar mitral por planimetria ou PHT."))
                mi_grad = st.number_input(tr("Mitral mean gradient (mmHg)", "Gradiente médio mitral (mmHg)"), min_value=0.0, max_value=30.0, value=None, step=1.0, format="%.1f", help=hp("Mean transmitral pressure gradient.", "Gradiente de pressão transmitral médio."))
                pht_mi = st.number_input(tr("PHT Mitral (ms)", "PHT Mitral (ms)"), min_value=0.0, max_value=1000.0, value=None, step=10.0, format="%.0f", help=hp("Pressure half-time of mitral stenosis.", "Tempo de meia-pressão da estenose mitral."))
                vc_mi = st.number_input(tr("Vena contracta mitral (mm)", "Vena contracta mitral (mm)"), min_value=0.0, max_value=15.0, value=None, step=0.5, format="%.1f", help=hp("Vena contracta width of mitral regurgitation jet.", "Largura da vena contracta do jato de insuficiência mitral."))

        st.markdown(tr("**Other echocardiographic findings**", "**Outros achados ecocardiográficos**"))
        a_root_abs = st.selectbox(tr("Aortic root abscess", "Abscesso de raiz aórtica"), yn_options, index=0, help=hp("Echocardiographic evidence of aortic root abscess, usually associated with active infective disease and higher risk.", "Evidência ecocardiográfica de abscesso de raiz aórtica, geralmente associada a doença infecciosa ativa e maior risco."))

        submitted = True

    if submitted:
        form_map = {
            "Age (years)": age,
            "Sex": sex,
            "Race": race,
            "Weight (kg)": weight_kg,
            "Height (cm)": height_cm,
            "BSA, m2": bsa,
            "Preoperative NYHA": nyha,
            "CCS4": yn_pt_to_en(ccs4),
            "Diabetes": diabetes_map[diabetes_pt],
            "Previous surgery": "Yes" if yn_pt_to_en(prior) == "Yes" else "No",
            "Dialysis": yn_pt_to_en(dialysis),
            "IE": "Possible" if str(ie_pt).strip().lower() in {"possível", "possible"} else yn_pt_to_en(ie_pt),
            "Cr clearance, ml/min *": creat_clear,
            "KDIGO †": kdigo,
            "Creatinine (mg/dL)": creatinine,
            "Hematocrit (%)": hematocrit,
            "WBC Count (10³/μL)": wbc,
            "Platelet Count (cells/μL)": platelets,
            "INR": inr,
            "PTT": ptt,
            "Pré-LVEF, %": lvef,
            "PSAP": psap,
            "TAPSE": tapse,
            "Surgical Priority": urgency_map[urgency_pt],
            "Surgery": surgery,
            "Coronary Symptom": coronary_map[coronary_pt],
            "No. of Diseased Vessels": float(diseased_vessels),
            "Left Main Stenosis ≥ 50%": yn_pt_to_en(left_main_50),
            "Proximal LAD Stenosis ≥ 70%": yn_pt_to_en(prox_lad_70),
            "Chronic Lung Disease": yn_pt_to_en(cpd),
            "Poor mobility": yn_pt_to_en(mobility),
            "Critical preoperative state": yn_pt_to_en(critical),
            "HF": normalize_hf_value(hf),
            "Hypertension": yn_pt_to_en(htn),
            "Dyslipidemia": yn_pt_to_en(dlp),
            "CVA": normalize_cva_value(cva),
            "PVD": yn_pt_to_en(pvd2),
            "Cancer ≤ 5 yrs": yn_pt_to_en(cancer5),
            "Arrhythmia Remote": normalize_arrhythmia_remote_value(arr_rem),
            "Arrhythmia Recent": normalize_arrhythmia_recent_value(arr_rec),
            "Family Hx of CAD": yn_pt_to_en(fam_cad),
            "Smoking (Pack-year)": _smoking_status,
            "Alcohol": yn_pt_to_en(alcohol),
            "Pneumonia": normalize_pneumonia_value(recent_pneum),
            "Anticoagulation/ Antiaggregation": yn_pt_to_en(anticoag_pt),
            "Suspension of Anticoagulation (day)": float(suspension_days),
            "Aortic Stenosis": a_stenosis,
            "Aortic Regurgitation": a_reg,
            "Mitral Stenosis": m_stenosis,
            "Mitral Regurgitation": m_reg,
            "Tricuspid Regurgitation": t_reg,
            "Aortic Root Abscess": yn_pt_to_en(a_root_abs),
            "AVA (cm²)": ava if ava is not None else np.nan,
            "Aortic Mean gradient (mmHg)": ao_grad if ao_grad is not None else np.nan,
            "PHT Aortic": pht_ao if pht_ao is not None else np.nan,
            "Vena contracta": vc_ao if vc_ao is not None else np.nan,
            "MVA (cm²)": mva if mva is not None else np.nan,
            "Mitral Mean gradient (mmHg)": mi_grad if mi_grad is not None else np.nan,
            "PHT Mitral": pht_mi if pht_mi is not None else np.nan,
            "Vena contracta (mm)": vc_mi if vc_mi is not None else np.nan,
        }

        _num_cols = _get_numeric_columns_from_pipeline(artifacts.fitted_models[forced_model])
        _patient_id_sp = _patient_identifier_from_row(form_map, 0)
        _infer_sp = _run_ai_risk_inference_row(
            model_pipeline=artifacts.fitted_models[forced_model],
            feature_columns=artifacts.feature_columns,
            reference_df=prepared.data,
            row_dict=form_map,
            patient_id=_patient_id_sp,
            numeric_cols=_num_cols,
            language=language,
        )
        if _infer_sp["incident"] is not None:
            st.error(tr(
                f"AI Risk inference failed: {_infer_sp['incident']['reason']}",
                f"Falha na inferência AI Risk: {_infer_sp['incident']['reason']}",
            ))
            st.stop()
        model_input = _infer_sp["model_input"]
        ia_prob = _infer_sp["probability"]
        informed_features = int(model_input.notna().sum(axis=1).iloc[0])
        imputed_features = int(model_input.shape[1] - informed_features)
        euro_prob = float(euroscore_from_inputs(form_map))
        # Calculate STS Score via the web calculator, routed through the
        # Phase 2 persistent STS Score cache (14-day TTL, revalidation).
        sts_result = {}
        sts_exec_record = None
        if HAS_STS:
            _sts_pid = _sts_score_patient_id(form_map) or form_map.get("Name") or None
            with st.spinner(tr("Querying STS Score web calculator...", "Consultando calculadora web do STS Score...")):
                sts_result = calculate_sts(form_map, patient_id=_sts_pid)
            # Grab the most recent execution record (appended by calculate_sts)
            try:
                sts_exec_record = calculate_sts.last_execution_log[-1]  # type: ignore[attr-defined]
            except Exception:
                sts_exec_record = None
        sts_prob = sts_result.get("predmort", np.nan)

        patient_pred = []
        _patient_model_probs: Dict[str, float] = {}
        for model_name in model_options:
            p = float(artifacts.fitted_models[model_name].predict_proba(model_input)[:, 1][0])
            _patient_model_probs[model_name] = p
            patient_pred.append({tr("Model", "Modelo"): model_name, tr("Probability", "Probabilidade"): p})
        _model_disagreement = _candidate_model_disagreement_summary(_patient_model_probs)
        patient_pred_df = pd.DataFrame(patient_pred).sort_values(tr("Probability", "Probabilidade"), ascending=False)
        patient_pred_df[tr("Probability", "Probabilidade")] = patient_pred_df[tr("Probability", "Probabilidade")].map(lambda x: f"{x*100:.2f}%")
        quality_alerts = _data_quality_alerts(form_map, prepared, tr)
        likely_range_text, confidence_text = _prediction_uncertainty(patient_pred_df, tr("Probability", "Probabilidade"), imputed_features, tr)

        # --- Input completeness indicator ---
        # Use model_input (post clean_features) but restore values that
        # clean_features wrongly converted to NaN (e.g. categorical "No"→NaN).
        # A feature is "informed" if the user provided a value in form_map.
        _derived = {
            "cirurgia_combinada",
            "peso_procedimento",
            "thoracic_aorta_flag",
            *MISSINGNESS_INDICATOR_COLUMNS,
        }
        _completeness_row = model_input.copy()
        for _fc in artifacts.feature_columns:
            if _fc in _completeness_row.columns and pd.isna(_completeness_row.at[_completeness_row.index[0], _fc]):
                # Check if user actually provided a value
                if _fc in _derived:
                    _completeness_row.at[_completeness_row.index[0], _fc] = 1  # always informed
                elif _fc in form_map:
                    _v = form_map[_fc]
                    if _v is not None and str(_v).strip() not in ("", "nan"):
                        _completeness_row.at[_completeness_row.index[0], _fc] = 1  # mark as informed
        _completeness = assess_input_completeness(artifacts.feature_columns, _completeness_row, language)
        _comp_icon = {"green": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴"}.get(_completeness["color"], "⚪")
        _comp_border = {"green": "#28a745", "yellow": "#ffc107", "orange": "#fd7e14", "red": "#dc3545"}.get(_completeness["color"], "#6c757d")
        _pct_informed = round(100 * _completeness["n_informed"] / max(_completeness["n_total"], 1))

        best_model_prob = float(artifacts.fitted_models[best_model_name].predict_proba(model_input)[:, 1][0])

        # ── PREDICTED RISK — dominant primary block ───────────────────────────
        st.divider()
        st.markdown(tr("### Predicted Risk", "### Risco Predito"))
        st.caption(tr(
            f"Primary prediction generated by: **{forced_model}**",
            f"Predição principal gerada por: **{forced_model}**",
        ))
        r1, r2, r3 = st.columns(3)
        r1.metric("\U0001f916 AI Risk", f"{ia_prob*100:.2f}%", _risk_badge(ia_prob, tr), delta_color="off")
        r2.metric("\U0001f4ca EuroSCORE II", f"{euro_prob*100:.2f}%", _risk_badge(euro_prob, tr), delta_color="off")
        r3.metric("\U0001f310 STS Score", "-" if np.isnan(sts_prob) else f"{sts_prob*100:.2f}%", _risk_badge(sts_prob, tr), delta_color="off")

        # ── COMPLETENESS / RELIABILITY — anchored immediately below the result ─
        st.markdown(
            f"""
            <div style="border-left: 4px solid {_comp_border}; padding: 12px 16px; border-radius: 6px;
                        background: rgba(255,255,255,0.03); margin: 8px 0 12px 0;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 1.4rem;">{_comp_icon}</span>
                    <span style="font-size: 1.05rem; font-weight: 600;">{_completeness['label']}</span>
                </div>
                <div style="background: #23272b; border-radius: 4px; height: 8px; width: 100%; margin-bottom: 8px;">
                    <div style="background: {_comp_border}; border-radius: 4px; height: 8px; width: {_pct_informed}%;"></div>
                </div>
                <div style="display: flex; gap: 24px; font-size: 0.9rem; color: #adb5bd;">
                    <span>{tr("Informed", "Informadas")}: <b style="color:#e9ecef;">{_completeness['n_informed']}</b> / {_completeness['n_total']}</span>
                    <span>{tr("Imputed", "Imputadas")}: <b style="color:#e9ecef;">{_completeness['n_imputed']}</b></span>
                    <span style="margin-left: auto; font-size: 0.85rem;">{_pct_informed}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if _completeness["missing_high"]:
            st.error(tr(
                f"⚠ High-relevance variables missing: **{', '.join(_completeness['missing_high'])}** — predictions may be unreliable",
                f"⚠ Variáveis de alta relevância ausentes: **{', '.join(_completeness['missing_high'])}** — predições podem ser pouco confiáveis",
            ))
        if _completeness["missing_moderate"]:
            st.warning(tr(
                f"Moderate-relevance variables missing: {', '.join(_completeness['missing_moderate'])}",
                f"Variáveis de relevância moderada ausentes: {', '.join(_completeness['missing_moderate'])}",
            ))

        if _model_disagreement.get("high"):
            _boost_note_en = ""
            _boost_note_pt = ""
            _low_boost = _model_disagreement.get("low_end_boosting") or []
            if _low_boost:
                _boost_names = ", ".join(_low_boost)
                _boost_note_en = (
                    f" Very low calibrated estimates from {_boost_names} are present; "
                    "this can occur with isotonic-calibrated boosting models at the low end of the risk distribution."
                )
                _boost_note_pt = (
                    f" Estimativas calibradas muito baixas de {_boost_names} estão presentes; "
                    "isso pode ocorrer em modelos boosting com calibração isotônica no extremo inferior da distribuição de risco."
                )
            st.warning(tr(
                "High disagreement between candidate AI models. "
                f"The primary prediction remains the selected model (**{forced_model}**), but candidate estimates range from "
                f"{_model_disagreement['min']*100:.1f}% to {_model_disagreement['max']*100:.1f}% "
                f"(range {_model_disagreement['range']*100:.1f} percentage points; median {_model_disagreement['median']*100:.1f}%). "
                "Interpret this estimate with caution and review input completeness and key risk factors."
                + _boost_note_en,
                "Alta discordância entre os modelos candidatos de IA. "
                f"A predição principal continua sendo o modelo selecionado (**{forced_model}**), mas as estimativas variam de "
                f"{_model_disagreement['min']*100:.1f}% a {_model_disagreement['max']*100:.1f}% "
                f"(amplitude {_model_disagreement['range']*100:.1f} pontos percentuais; mediana {_model_disagreement['median']*100:.1f}%). "
                "Interprete esta estimativa com cautela e revise a completude dos dados e os principais fatores de risco."
                + _boost_note_pt,
            ))

        st.caption(tr(
            "Individual predictions reflect the model's estimate for this specific combination of variables. Population-level performance (AUC, calibration) does not guarantee accuracy for any single patient. This tool is for research purposes and does not replace clinical judgment.",
            "Predições individuais refletem a estimativa do modelo para esta combinação específica de variáveis. O desempenho populacional (AUC, calibração) não garante acurácia para um paciente isolado. Esta ferramenta é para fins de pesquisa e não substitui o julgamento clínico.",
        ))
        if np.isnan(sts_prob):
            st.caption(tr(
                "STS Score unavailable — the web calculator could not be reached or did not return a result for this patient. No dataset fallback is used.",
                "STS Score indisponível — a calculadora web não pôde ser acessada ou não retornou resultado para este paciente. Nenhum fallback do dataset é utilizado.",
            ))

        # STS Score cache transparency: extract all record fields once so they
        # are accessible both for inline failure warnings and inside expanders.
        _st_status = "unknown"
        _st_age = None
        _st_stage = ""
        _st_reason = ""
        _st_retry = False
        _st_used_prev = False
        _age_txt = ""
        if sts_exec_record is not None:
            _st_status = getattr(sts_exec_record, "status", "unknown")
            _st_age = getattr(sts_exec_record, "cache_age_days", None)
            _st_stage = getattr(sts_exec_record, "stage", "")
            _st_reason = getattr(sts_exec_record, "reason", "")
            _st_retry = getattr(sts_exec_record, "retry_attempted", False)
            _st_used_prev = getattr(sts_exec_record, "used_previous_cache", False)
            _age_txt = f" (age {_st_age:.1f} d)" if isinstance(_st_age, (int, float)) else ""
            # Failure states are shown inline so the user sees them immediately.
            if _st_status == "stale_fallback":
                st.warning(tr(
                    f"STS Score: web calculator failed; using a previous cached result{_age_txt}. "
                    f"Stage: {_st_stage}. Reason: {_st_reason}. "
                    f"Retry attempted: {_st_retry}. Previous cache used: {_st_used_prev}.",
                    f"STS Score: a calculadora web falhou; usando um resultado anterior em cache{_age_txt}. "
                    f"Etapa: {_st_stage}. Motivo: {_st_reason}. "
                    f"Tentativa de repetição: {_st_retry}. Cache anterior utilizado: {_st_used_prev}.",
                ))
            elif _st_status == "failed":
                st.warning(tr(
                    f"STS Score: calculation failed and no previous cache is available. "
                    f"Stage: {_st_stage}. Reason: {_st_reason}. Retry attempted: {_st_retry}.",
                    f"STS Score: cálculo falhou e não há cache anterior disponível. "
                    f"Etapa: {_st_stage}. Motivo: {_st_reason}. Tentativa de repetição: {_st_retry}.",
                ))

        out = pd.DataFrame(
            {
                tr("Score", "Escore"): ["\U0001f916 AI Risk", "\U0001f4ca EuroSCORE II", "\U0001f310 STS Score"],
                tr("Probability", "Probabilidade"): [ia_prob, euro_prob, sts_prob],
                tr("Class", "Classe"): [_risk_badge(ia_prob, tr), _risk_badge(euro_prob, tr), _risk_badge(sts_prob, tr)],
            }
        )
        out[tr("Probability", "Probabilidade")] = out[tr("Probability", "Probabilidade")].map(lambda x: "-" if np.isnan(x) else f"{x*100:.2f}%")
        with st.expander(tr("Detailed score table", "Tabela detalhada dos escores"), expanded=False):
            st.dataframe(out, width="stretch", column_config=general_table_column_config("patient_scores"))
            if sts_result:
                st.markdown(tr("**STS Score Sub-scores (web calculator)**", "**Sub-escores STS Score (calculadora web)**"))
                sts_rows = []
                for key, label in STS_LABELS.items():
                    val = sts_result.get(key, np.nan)
                    sts_rows.append({
                        tr("STS Score Endpoint", "Desfecho STS Score"): label,
                        tr("Value", "Valor"): f"{val*100:.2f}%" if not (isinstance(val, float) and np.isnan(val)) else "-",
                    })
                st.dataframe(pd.DataFrame(sts_rows), width="stretch", column_config=general_table_column_config("sts_subscores"))
            # OK-state STS cache status shown here to avoid cluttering the main result area.
            if _st_status == "cached":
                st.caption(tr(
                    f"STS Score: cached result returned{_age_txt}. No network call.",
                    f"STS Score: resultado em cache{_age_txt}. Sem chamada de rede.",
                ))
            elif _st_status == "fresh":
                st.caption(tr(
                    "STS Score: fresh calculation from the web calculator (cached for 14 days).",
                    "STS Score: cálculo novo via calculadora web (armazenado em cache por 14 dias).",
                ))
            elif _st_status == "refreshed":
                st.caption(tr(
                    "STS Score: cached entry expired and was refreshed from the web calculator.",
                    "STS Score: entrada em cache expirada e atualizada via calculadora web.",
                ))

        with st.expander(tr("Prediction by each AI model", "Predição por cada modelo de IA"), expanded=False):
            st.info(tr(
                f"The risk summary uses **{forced_model}**, selected as the best-performing model in internal cross-validation (highest AUC). "
                "The table below shows predictions from all candidate models for comparison. Differences are expected — each algorithm learns patterns differently.",
                f"O resumo de risco usa **{forced_model}**, selecionado como o modelo de melhor desempenho na validação cruzada interna (maior AUC). "
                "A tabela abaixo mostra as predições de todos os modelos candidatos para comparação. Diferenças são esperadas — cada algoritmo aprende padrões de forma diferente.",
            ))
            st.caption(tr(
                f"Candidate-model spread: min {_model_disagreement['min']*100:.1f}%, "
                f"median {_model_disagreement['median']*100:.1f}%, "
                f"max {_model_disagreement['max']*100:.1f}%, "
                f"range {_model_disagreement['range']*100:.1f} percentage points.",
                f"Dispersão entre modelos candidatos: mínimo {_model_disagreement['min']*100:.1f}%, "
                f"mediana {_model_disagreement['median']*100:.1f}%, "
                f"máximo {_model_disagreement['max']*100:.1f}%, "
                f"amplitude {_model_disagreement['range']*100:.1f} pontos percentuais.",
            ))
            _model_col = tr("Model", "Modelo")
            _styled_pred = patient_pred_df.copy()
            st.dataframe(
                _styled_pred.style.apply(
                    lambda row: [
                        "background-color: rgba(40, 167, 69, 0.15); font-weight: bold" if str(row[_model_col]) == forced_model else ""
                        for _ in row
                    ],
                    axis=1,
                ),
                width="stretch",
                column_config=general_table_column_config("patient_scores"),
            )

        # ── INPUT QUALITY — consolidated into one expander ────────────────────
        with st.expander(tr("Input quality & imputation detail", "Qualidade do input e detalhe de imputação"), expanded=False):
            q1, q2 = st.columns(2)
            with q1:
                st.info(tr(
                    f"Model input: {informed_features} features informed, {imputed_features} imputed (detailed echo/lab values not available in the form use training median).",
                    f"Entrada do modelo: {informed_features} variáveis informadas, {imputed_features} imputadas (valores detalhados de eco/lab não disponíveis no formulário usam a mediana do treino).",
                ))
            with q2:
                st.info(tr(f"{likely_range_text} {confidence_text}.", f"{likely_range_text} {confidence_text}."))
            _imp_detail = format_imputation_detail(artifacts.feature_columns, model_input, language)
            st.dataframe(_imp_detail, width="stretch", hide_index=True)
            st.markdown(tr(
                """
**Informed vs. imputed variables:** The model uses 61 predictor variables. In the individual form, some variables (detailed echocardiographic measurements, specific lab values) are not available as input fields — for these, the model uses the training dataset median. More imputed variables means less personalized prediction, but the core clinical variables (age, surgery type, renal function, LVEF) are always informed.

**Interquartile risk range (IQR):** Shows the range between the 25th and 75th percentile of predictions across the available AI Risk candidate models. A narrow IQR means models agree; a wide IQR means disagreement.

| Agreement | IQR spread | Interpretation |
|:--|:--|:--|
| **High** | < 4 percentage points | Models converge — prediction is robust |
| **Moderate** | 4–10 percentage points | Some disagreement — consider the range, not just the point estimate |
| **Low** | > 10 percentage points | Large disagreement — prediction is uncertain, clinical judgment should prevail |

**Why do models disagree?** Tree-based models (RandomForest, CatBoost, XGBoost, LightGBM) capture non-linear interactions and may predict differently from linear models (Logistic Regression) or from the StackingEnsemble meta-learner. For atypical patients, disagreement is more likely.
""",
                """
**Variáveis informadas vs. imputadas:** O modelo usa 61 variáveis preditoras. No formulário individual, algumas variáveis (medidas ecocardiográficas detalhadas, exames laboratoriais específicos) não estão disponíveis como campos — para estas, o modelo usa a mediana do dataset de treinamento. Mais variáveis imputadas significa predição menos personalizada, mas as variáveis clínicas centrais (idade, tipo de cirurgia, função renal, FEVE) são sempre informadas.

**Faixa interquartil de risco (IQR):** Mostra o intervalo entre o percentil 25 e o percentil 75 das predições dos modelos candidatos do AI Risk disponíveis. IQR estreito significa que os modelos concordam; IQR largo significa discordância.

| Concordância | Amplitude do IQR | Interpretação |
|:--|:--|:--|
| **Alta** | < 4 pontos percentuais | Modelos convergem — predição robusta |
| **Moderada** | 4–10 pontos percentuais | Alguma discordância — considere a faixa, não apenas o valor pontual |
| **Baixa** | > 10 pontos percentuais | Grande discordância — predição incerta, o julgamento clínico deve prevalecer |

**Por que os modelos discordam?** Modelos baseados em árvore (RandomForest, CatBoost, XGBoost, LightGBM) capturam interações não-lineares e podem predizer diferente de modelos lineares (Regressão Logística) ou do meta-aprendiz StackingEnsemble. Para pacientes atípicos, a discordância é mais provável.
""",
            ))

        st.caption(tr(
            "Note: EuroSCORE II and STS Score are calculated automatically. For EuroSCORE II, variables not available in the spreadsheet (e.g., poor mobility, critical preoperative state) are entered manually in this form. STS Score is obtained via automated interaction with the STS web calculator.",
            "Observação: EuroSCORE II e STS Score são calculados automaticamente. No EuroSCORE II, variáveis não presentes na planilha (ex.: mobilidade, estado crítico) são informadas manualmente neste formulário. O STS Score é obtido via interação automatizada com a calculadora web do STS.",
        ))

        if quality_alerts:
            with st.expander(tr("Data quality alerts", "Alertas de qualidade do dado"), expanded=False):
                for level, message in quality_alerts:
                    if level == tr("Critical", "Crítico"):
                        st.error(f"{level}: {message}")
                    elif level == tr("Warning", "Atenção"):
                        st.warning(f"{level}: {message}")
                    else:
                        st.info(f"{level}: {message}")

        # ── CLINICAL INTERPRETATION ───────────────────────────────────────────
        st.divider()
        st.markdown(tr("### Clinical Interpretation", "### Interpretação Clínica"))
        pos_factors, neg_factors = _explain_patient_risk(artifacts, model_input, form_map, tr, top_n=5)
        st.caption(tr(
            "This explanation uses the logistic regression reference model as an interpretable layer. It reflects estimated associations with risk, not causal relationships.",
            "Esta explicação usa o modelo de regressão logística como camada interpretável de referência. Ela reflete associações estimadas com o risco, e não relações causais.",
        ))
        c_pos, c_neg = st.columns(2)
        with c_pos:
            st.markdown(tr("**Factors associated with higher risk**", "**Fatores associados ao aumento do risco**"))
            if pos_factors.empty:
                st.info(tr("No strong increasing-risk factors were identified by the interpretable layer.", "Nenhum fator forte de aumento de risco foi identificado pela camada interpretável."))
            else:
                st.dataframe(pos_factors, width="stretch", column_config=_explain_table_column_config(tr))
        with c_neg:
            st.markdown(tr("**Factors associated with lower risk**", "**Fatores associados à redução do risco**"))
            if neg_factors.empty:
                st.info(tr("No strong lower-risk factors were identified by the interpretable layer.", "Nenhum fator forte de redução de risco foi identificado pela camada interpretável."))
            else:
                st.dataframe(neg_factors, width="stretch", column_config=_explain_table_column_config(tr))

        _clinical_text = generate_clinical_explanation(pos_factors, neg_factors, ia_prob, language)
        st.info(_clinical_text)

        with st.expander(tr("Why this prediction? (SHAP — actual model)", "Por que essa predição? (SHAP — modelo real)"), expanded=False):
            st.caption(tr(
                "Unlike the interpretable layer above (logistic regression proxy), this explanation uses SHAP values computed directly from the selected model. Each row shows how that variable pushed the predicted probability up or down for this specific patient.",
                "Ao contrário da camada interpretável acima (proxy de regressão logística), esta explicação usa valores SHAP calculados diretamente do modelo selecionado. Cada linha mostra quanto aquela variável empurrou a probabilidade prevista para cima ou para baixo neste paciente específico.",
            ))
            shap_mode = st.radio(
                tr("Computation mode", "Modo de cálculo"),
                [tr("Automatic", "Automático"), tr("Manual", "Manual")],
                horizontal=True,
                help=tr(
                    "Automatic: runs immediately when the section opens. Manual: lets you adjust parameters before running.",
                    "Automático: calcula ao abrir a seção. Manual: permite ajustar os parâmetros antes de calcular.",
                ),
            )
            shap_n_bg = 50
            shap_top_n = 10
            if shap_mode == tr("Manual", "Manual"):
                sc1, sc2 = st.columns(2)
                shap_n_bg = sc1.slider(
                    tr("Background samples", "Amostras de background"),
                    min_value=10, max_value=min(200, len(prepared.data)),
                    value=50, step=10,
                    help=tr(
                        "More samples = more accurate but slower (~1-2s per 10 samples).",
                        "Mais amostras = mais preciso, porém mais lento (~1-2s por 10 amostras).",
                    ),
                )
                shap_top_n = sc2.slider(
                    tr("Top features to show", "Top variáveis a mostrar"),
                    min_value=3, max_value=20, value=10,
                )
                run_shap = st.button(tr("▶ Calculate SHAP", "▶ Calcular SHAP"), type="primary")
            else:
                run_shap = True

            if run_shap:
                with st.spinner(tr("Computing SHAP explanation…", "Calculando explicação SHAP…")):
                    try:
                        X_bg = clean_features(_safe_select_features(prepared.data, artifacts.feature_columns)).head(shap_n_bg)
                        _shap_explainer = ModelExplainer(
                            artifacts.fitted_models[forced_model], X_bg, max_samples=shap_n_bg
                        )
                        shap_top = _shap_explainer.top_features_for_sample(model_input, idx=0, top_n=shap_top_n)
                        shap_top.columns = [
                            tr("Feature", "Variável"),
                            tr("Value", "Valor"),
                            tr("SHAP", "SHAP"),
                            tr("Impact", "Impacto"),
                        ]
                        shap_top[tr("Value", "Valor")] = shap_top[tr("Value", "Valor")].astype(str)
                        shap_top[tr("Impact", "Impacto")] = shap_top[tr("Impact", "Impacto")].map(
                            lambda v: tr("↑ increases risk", "↑ aumenta risco")
                            if v == "increases"
                            else tr("↓ decreases risk", "↓ reduz risco")
                        )
                        st.dataframe(shap_top, width="stretch")
                    except Exception as _shap_err:
                        st.warning(tr(f"SHAP explanation unavailable: {_shap_err}", f"Explicação SHAP indisponível: {_shap_err}"))

        # ── EXPORT ────────────────────────────────────────────────────────────
        st.divider()
        st.markdown(tr("### Export individual report", "### Exportar relatório individual"))
        st.caption(tr(
            "Generate and download the complete individual report: MD (full text), PDF (formatted), XLSX (structured tables), CSV (flat data).",
            "Gere e baixe o relatório individual completo: MD (texto completo), PDF (formatado), XLSX (tabelas estruturadas), CSV (dados planos).",
        ))
        _patient_id = st.text_input(
            tr("Patient identifier (for report)", "Identificador do paciente (para relatório)"),
            value=tr("Patient_001", "Paciente_001"),
            key="report_patient_id",
        )
        _report_text = generate_individual_report(
            patient_id=_patient_id,
            form_map=form_map,
            ia_prob=ia_prob,
            euro_prob=euro_prob,
            sts_prob=sts_prob,
            risk_class=class_risk(ia_prob),
            model_version=bundle_info.get("model_version") or MODEL_VERSION,
            model_name=forced_model,
            completeness=_completeness,
            pos_factors=pos_factors,
            neg_factors=neg_factors,
            sts_result=sts_result,
            language=language,
            bundle_saved_at=bundle_info.get("saved_at"),
            training_source_file=bundle_info.get("training_source"),
            current_analysis_file=Path(xlsx_path).name,
        )
        _rpt_c1, _rpt_c2, _rpt_c3, _rpt_c4 = st.columns(4)
        with _rpt_c1:
            _txt_download_btn(_report_text, f"report_{_patient_id}.md", tr("Download MD", "Baixar MD"))
        with _rpt_c2:
            _rpt_pdf = statistical_summary_to_pdf(_report_text)
            if _rpt_pdf:
                _bytes_download_btn(
                    _rpt_pdf,
                    f"report_{_patient_id}.pdf",
                    tr("Download PDF", "Baixar PDF"),
                    "application/pdf",
                    key="dl_report_pdf",
                )
        with _rpt_c3:
            _rpt_xlsx = statistical_summary_to_xlsx(_report_text)
            if _rpt_xlsx:
                _bytes_download_btn(
                    _rpt_xlsx,
                    f"report_{_patient_id}.xlsx",
                    tr("Download XLSX", "Baixar XLSX"),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_report_xlsx",
                )
        with _rpt_c4:
            _rpt_csv = statistical_summary_to_csv(_report_text)
            if _rpt_csv:
                _bytes_download_btn(
                    _rpt_csv,
                    f"report_{_patient_id}.csv",
                    tr("Download CSV", "Baixar CSV"),
                    "text/csv",
                    key="dl_report_csv",
                )

        # ── AUDIT TRAIL ───────────────────────────────────────────────────────
        log_analysis(
            analysis_type="individual_prediction",
            source_file="manual_form",
            model_version=bundle_info.get("model_version") or MODEL_VERSION,
            n_patients=1,
            n_imputed=_completeness["n_imputed"],
            completeness_level=_completeness["level"],
            sts_method="websocket" if sts_result else "unavailable",
            extra={"patient_id": _patient_id, "ia_risk": round(ia_prob, 4)},
        )
