"""Structured variable dictionary for AI Risk.

Provides clinical definitions, origins, units, and model usage metadata
for all variables used in the cardiac surgery risk prediction system.
"""

from typing import List, Dict, Sequence
import pandas as pd


# Each entry: (variable_name, clinical_definition, origin, unit, transformation, in_model)
VARIABLE_DICTIONARY: List[Dict[str, str]] = [
    # --- Demographics & Anthropometrics ---
    {"variable": "Age (years)", "definition": "Chronological age at surgery", "origin": "Preoperative", "unit": "years", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Demographics"},
    {"variable": "Sex", "definition": "Biological sex", "origin": "Preoperative", "unit": "M/F", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Demographics"},
    {"variable": "Height (cm)", "definition": "Body height", "origin": "Preoperative", "unit": "cm", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Demographics"},
    {"variable": "Weight (kg)", "definition": "Body weight", "origin": "Preoperative", "unit": "kg", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Demographics"},
    {"variable": "BSA, m2", "definition": "Body surface area", "origin": "Preoperative", "unit": "m²", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Demographics"},
    {"variable": "Race", "definition": "Self-reported race/ethnicity", "origin": "Preoperative", "unit": "categorical", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Demographics"},

    # --- Surgical ---
    {"variable": "Surgery", "definition": "Planned surgical procedure(s), comma-separated", "origin": "Preoperative", "unit": "text", "transformation": "Used for procedure weight, thoracic aorta flag, combined surgery flag; procedure_group derived for DQ audit only", "in_model": "Derived features only", "domain": "Procedure"},
    {"variable": "Surgical Priority", "definition": "Urgency classification: Elective, Urgent, Emergency, Salvage", "origin": "Preoperative", "unit": "categorical", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Procedure"},
    {"variable": "cirurgia_combinada", "definition": "Whether surgery involves multiple procedures", "origin": "Derived from Surgery", "unit": "0/1", "transformation": "Binary", "in_model": "Yes", "domain": "Procedure"},
    {"variable": "peso_procedimento", "definition": "EuroSCORE II procedure weight category", "origin": "Derived from Surgery", "unit": "categorical", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Procedure"},
    {"variable": "thoracic_aorta_flag", "definition": "Whether surgery involves thoracic aorta", "origin": "Derived from Surgery", "unit": "0/1", "transformation": "Binary", "in_model": "Yes", "domain": "Procedure"},
    {"variable": "procedure_group", "definition": "Intermediate procedure group (CABG_OPCAB, AORTIC_VALVE, MITRAL_TRICUSPID, AORTA_ROOT, AORTA_ANEURYSM, HF_TRANSPLANT, CONGENITAL_STRUCTURAL, CARDIAC_MASS_THROMBUS, OTHER_CARDIAC, OTHER, UNKNOWN)", "origin": "Derived from Surgery", "unit": "categorical", "transformation": "Computed for Data Quality audit and surgery coverage reporting only — not used in training or inference", "in_model": "No (excluded by controlled ablation: consistent degradation at n=454 — AUC -0.017, AUPRC -0.020; TargetEncoder unstable at this cohort size)", "domain": "Procedure"},

    # --- Clinical Status ---
    {"variable": "Preoperative NYHA", "definition": "New York Heart Association functional class (I-IV)", "origin": "Preoperative", "unit": "I/II/III/IV", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Clinical"},
    {"variable": "CCS4", "definition": "Canadian Cardiovascular Society class 4 angina", "origin": "Preoperative", "unit": "Yes/No", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Clinical"},
    {"variable": "Coronary Symptom", "definition": "Current coronary presentation (no symptoms, angina presentation, or ACS/MI label such as Non-STEMI/STEMI)", "origin": "Preoperative", "unit": "categorical", "transformation": "Categorical (TargetEncoder); literal 'None' is canonicalized to 'No coronary symptoms'", "in_model": "Yes", "domain": "Clinical"},
    {"variable": "HF", "definition": "Clinical diagnosis of heart failure", "origin": "Preoperative", "unit": "None/Acute/Chronic/Both", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Clinical"},
    {"variable": "Critical preoperative state", "definition": "Severe preoperative instability (shock, inotropes, ventilation, resuscitation)", "origin": "Preoperative", "unit": "Yes/No", "transformation": "Categorical (TargetEncoder)", "in_model": "No (not retained in current model)", "domain": "Clinical"},
    {"variable": "Poor mobility", "definition": "Severely reduced mobility (musculoskeletal/neurologic)", "origin": "Preoperative", "unit": "Yes/No", "transformation": "Categorical (TargetEncoder); set to False for all in EuroSCORE II (not reliably collected)", "in_model": "No (not retained in current model)", "domain": "Clinical"},

    # --- Comorbidities ---
    {"variable": "Diabetes", "definition": "Diabetes mellitus and treatment type", "origin": "Preoperative", "unit": "No/Oral/Insulin/Diet", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Comorbidities"},
    {"variable": "Hypertension", "definition": "History of systemic arterial hypertension", "origin": "Preoperative", "unit": "Yes/No", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Comorbidities"},
    {"variable": "Dyslipidemia", "definition": "Presence of dyslipidemia", "origin": "Preoperative", "unit": "Yes/No", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Comorbidities"},
    {"variable": "CVA", "definition": "History of stroke or cerebrovascular disease", "origin": "Preoperative", "unit": "No/≤30d/≥30d/Timing unk/TIA/Other CVD", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Comorbidities"},
    {"variable": "PVD", "definition": "Peripheral vascular disease (also proxy for ECA in EuroSCORE II)", "origin": "Preoperative", "unit": "Yes/No", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Comorbidities"},
    {"variable": "Dialysis", "definition": "Established dialysis therapy", "origin": "Preoperative", "unit": "Yes/No", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Comorbidities"},
    {"variable": "Chronic Lung Disease", "definition": "COPD or other chronic pulmonary disease", "origin": "Preoperative", "unit": "Yes/No (or severity)", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Comorbidities"},
    {"variable": "IE", "definition": "Infective endocarditis (active/probable/possible)", "origin": "Preoperative", "unit": "Yes/No/Possible", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Comorbidities"},
    {"variable": "Previous surgery", "definition": "History of prior cardiac surgery. Training data: free-text (36 categories); individual inference form: binary Yes/No. TargetEncoder(smooth='auto') shrinks sparse categories toward global mean, so both representations converge to the same effective signal. Five audit-only columns are derived via parse_previous_surgery() — not model features.", "origin": "Preoperative", "unit": "Free-text (training) / Yes-No (inference form)", "transformation": "Categorical (TargetEncoder); audit columns: previous_surgery_any, previous_surgery_count_est, previous_surgery_has_combined, previous_surgery_has_repeat_marker, previous_surgery_has_year_marker", "in_model": "Yes", "domain": "Comorbidities"},
    {"variable": "Cancer ≤ 5 yrs", "definition": "Cancer diagnosed or treated within 5 years", "origin": "Preoperative", "unit": "Yes/No", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Comorbidities"},
    {"variable": "Pneumonia", "definition": "Recent pneumonia before surgery", "origin": "Preoperative", "unit": "No/Under treatment/Treated", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Comorbidities"},
    {"variable": "Arrhythmia Remote", "definition": "Past history of arrhythmia type", "origin": "Preoperative", "unit": "None/AF/Flutter/VT-VF/3rd Block", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Comorbidities"},
    {"variable": "Arrhythmia Recent", "definition": "Recent arrhythmia before surgery", "origin": "Preoperative", "unit": "None/AF/Flutter/VT-VF/3rd Block", "transformation": "Categorical (TargetEncoder)", "in_model": "No (not retained in current model)", "domain": "Comorbidities"},
    {"variable": "Family Hx of CAD", "definition": "Family history of coronary artery disease", "origin": "Preoperative", "unit": "Yes/No", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Comorbidities"},
    {"variable": "Alcohol", "definition": "Relevant alcohol use history", "origin": "Preoperative", "unit": "Yes/No", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Comorbidities"},
    {"variable": "Smoking (Pack-year)", "definition": "Smoking status", "origin": "Preoperative", "unit": "Never/Former/Current", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Comorbidities"},

    # --- Coronary Anatomy ---
    {"variable": "No. of Diseased Vessels", "definition": "Number of major coronary vessels with significant disease (0-3)", "origin": "Preoperative", "unit": "0/1/2/3", "transformation": "Numeric or categorical", "in_model": "Yes", "domain": "Coronary"},
    {"variable": "Left Main Stenosis ≥ 50%", "definition": "Significant left main coronary stenosis", "origin": "Preoperative", "unit": "Yes/No", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Coronary"},
    {"variable": "Proximal LAD Stenosis ≥ 70%", "definition": "Significant proximal LAD lesion", "origin": "Preoperative", "unit": "Yes/No", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Coronary"},

    # --- Laboratory ---
    {"variable": "Creatinine (mg/dL)", "definition": "Serum creatinine", "origin": "Preoperative", "unit": "mg/dL", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Laboratory"},
    {"variable": "Cr clearance, ml/min *", "definition": "Creatinine clearance (Cockcroft-Gault)", "origin": "Preoperative / Calculated", "unit": "ml/min", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Laboratory"},
    {"variable": "KDIGO †", "definition": "Estimated KDIGO stage from clearance", "origin": "Derived", "unit": "G1-G5", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Laboratory"},
    {"variable": "Hematocrit (%)", "definition": "Hematocrit percentage", "origin": "Preoperative", "unit": "%", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Laboratory"},
    {"variable": "WBC Count (10³/μL)", "definition": "White blood cell count", "origin": "Preoperative", "unit": "10³/μL", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Laboratory"},
    {"variable": "Platelet Count (cells/μL)", "definition": "Platelet count", "origin": "Preoperative", "unit": "cells/μL", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Laboratory"},
    {"variable": "INR", "definition": "International normalized ratio", "origin": "Preoperative", "unit": "ratio", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Laboratory"},
    {"variable": "PTT", "definition": "Partial thromboplastin time", "origin": "Preoperative", "unit": "seconds", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Laboratory"},

    # --- Echocardiographic ---
    {"variable": "Pré-LVEF, %", "definition": "Left ventricular ejection fraction (preoperative)", "origin": "Pre-Echocardiogram", "unit": "%", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Echocardiographic"},
    {"variable": "PSAP", "definition": "Pulmonary artery systolic pressure", "origin": "Pre-Echocardiogram", "unit": "mmHg", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Echocardiographic"},
    {"variable": "TAPSE", "definition": "Tricuspid annular plane systolic excursion", "origin": "Pre-Echocardiogram", "unit": "mm", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Echocardiographic"},
    {"variable": "Aortic Stenosis", "definition": "Severity of aortic stenosis", "origin": "Pre-Echocardiogram", "unit": "None/Trivial/Mild/Moderate/Severe", "transformation": "OrdinalEncoder (None<Trivial<Mild<Moderate<Severe)", "in_model": "Yes", "domain": "Echocardiographic"},
    {"variable": "Aortic Regurgitation", "definition": "Severity of aortic regurgitation", "origin": "Pre-Echocardiogram", "unit": "None/Trivial/Mild/Moderate/Severe", "transformation": "OrdinalEncoder (None<Trivial<Mild<Moderate<Severe)", "in_model": "Yes", "domain": "Echocardiographic"},
    {"variable": "Mitral Stenosis", "definition": "Severity of mitral stenosis", "origin": "Pre-Echocardiogram", "unit": "None/Trivial/Mild/Moderate/Severe", "transformation": "OrdinalEncoder (None<Trivial<Mild<Moderate<Severe)", "in_model": "Yes", "domain": "Echocardiographic"},
    {"variable": "Mitral Regurgitation", "definition": "Severity of mitral regurgitation", "origin": "Pre-Echocardiogram", "unit": "None/Trivial/Mild/Moderate/Severe", "transformation": "OrdinalEncoder (None<Trivial<Mild<Moderate<Severe)", "in_model": "Yes", "domain": "Echocardiographic"},
    {"variable": "Tricuspid Regurgitation", "definition": "Severity of tricuspid regurgitation", "origin": "Pre-Echocardiogram", "unit": "None/Trivial/Mild/Moderate/Severe", "transformation": "OrdinalEncoder (None<Trivial<Mild<Moderate<Severe)", "in_model": "Yes", "domain": "Echocardiographic"},
    {"variable": "Aortic Root Abscess", "definition": "Echocardiographic evidence of aortic root abscess", "origin": "Pre-Echocardiogram", "unit": "Yes/No", "transformation": "Categorical (TargetEncoder)", "in_model": "No (not retained in current model)", "domain": "Echocardiographic"},
    {"variable": "AVA (cm²)", "definition": "Aortic valve area", "origin": "Pre-Echocardiogram", "unit": "cm²", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Echocardiographic"},
    {"variable": "MVA (cm²)", "definition": "Mitral valve area", "origin": "Pre-Echocardiogram", "unit": "cm²", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Echocardiographic"},
    {"variable": "Aortic Mean gradient (mmHg)", "definition": "Mean transvalvular aortic gradient", "origin": "Pre-Echocardiogram", "unit": "mmHg", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Echocardiographic"},
    {"variable": "Mitral Mean gradient (mmHg)", "definition": "Mean transmitral gradient", "origin": "Pre-Echocardiogram", "unit": "mmHg", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Echocardiographic"},
    {"variable": "PHT Aortic", "definition": "Aortic pressure half-time", "origin": "Pre-Echocardiogram", "unit": "ms", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Echocardiographic"},
    {"variable": "PHT Mitral", "definition": "Mitral pressure half-time", "origin": "Pre-Echocardiogram", "unit": "ms", "transformation": "Numeric, continuous", "in_model": "No (not retained in current model)", "domain": "Echocardiographic"},
    {"variable": "Vena contracta", "definition": "Vena contracta (aortic)", "origin": "Pre-Echocardiogram", "unit": "mm", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Echocardiographic"},
    {"variable": "Vena contracta (mm)", "definition": "Vena contracta (mitral)", "origin": "Pre-Echocardiogram", "unit": "mm", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Echocardiographic"},

    # --- Other ---
    {"variable": "Anticoagulation/ Antiaggregation", "definition": "Current anticoagulant or antiplatelet therapy", "origin": "Preoperative", "unit": "Yes/No", "transformation": "Categorical (TargetEncoder)", "in_model": "Yes", "domain": "Medication"},
    {"variable": "Suspension of Anticoagulation (day)", "definition": "Days since anticoagulation was suspended", "origin": "Preoperative", "unit": "days", "transformation": "Numeric, continuous", "in_model": "Yes", "domain": "Medication"},
    {"variable": "Preoperative Medications", "definition": "List of preoperative medications", "origin": "Preoperative", "unit": "text", "transformation": "Used for STS medication mapping only", "in_model": "No (STS input only)", "domain": "Medication"},

    # --- Outcome ---
    {"variable": "morte_30d", "definition": "30-day or in-hospital mortality (binary outcome)", "origin": "Derived from Postoperative Death column", "unit": "0/1", "transformation": "Binary (Operative or ≤30 days = 1; >30 or missing = 0)", "in_model": "Outcome (not predictor)", "domain": "Outcome"},
]


def _align_model_usage(df: pd.DataFrame, model_feature_columns: Sequence[str] | None) -> pd.DataFrame:
    """Align the In model flag with the currently loaded bundle, when available."""
    if model_feature_columns is None:
        return df

    feature_set = set(model_feature_columns)
    df = df.copy()
    for idx, row in df.iterrows():
        variable = row["variable"]
        current_value = str(row["in_model"])
        if variable == "morte_30d":
            df.at[idx, "in_model"] = "Outcome (not predictor)"
        elif variable in feature_set:
            df.at[idx, "in_model"] = "Yes"
        elif current_value == "Yes":
            df.at[idx, "in_model"] = "No (not retained in current model)"
    return df


def get_dictionary_dataframe(
    language: str = "English",
    model_feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the variable dictionary as a DataFrame."""
    df = pd.DataFrame(VARIABLE_DICTIONARY)
    df = _align_model_usage(df, model_feature_columns)
    if language != "English":
        df.columns = ["Variável", "Definição clínica", "Origem", "Unidade", "Transformação", "No modelo", "Domínio"]
    else:
        df.columns = ["Variable", "Clinical definition", "Origin", "Unit", "Transformation", "In model", "Domain"]
    return df


def get_dictionary_by_domain(
    language: str = "English",
    model_feature_columns: Sequence[str] | None = None,
) -> dict:
    """Return dictionary grouped by domain."""
    df = get_dictionary_dataframe(language, model_feature_columns=model_feature_columns)
    domain_col = "Domínio" if language != "English" else "Domain"
    return {name: group for name, group in df.groupby(domain_col)}
