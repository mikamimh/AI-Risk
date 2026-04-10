"""Data loading, validation, and preparation for AI Risk.

This module handles:
- Reading Excel, CSV, SQLite, and Parquet data sources
- Column mapping and standardization
- Data validation and eligibility criteria
- Patient matching across multiple data tables
- Feature engineering and preprocessing

Example:
    >>> from risk_data import prepare_master_dataset
    >>> prepared = prepare_master_dataset("patient_data.xlsx")
    >>> df = prepared.data
    >>> features = prepared.feature_columns
"""

import re
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


MISSING_TOKENS = {
    "",
    "-",
    "nan",
    "none",
    "not applicable",
    "unknown",
    "not informed",
}

REQUIRED_SOURCE_TABLES = [
    "Preoperative",
    "Pre-Echocardiogram",
    "Postoperative",
]

OPTIONAL_SOURCE_TABLES = [
    "EuroSCORE II",
    "EuroSCORE II Automático",
    "STS Score",
]

FLAT_ALIAS_TO_APP_COLUMNS = {
    "patient_id": "Name",
    "patient_name": "Name",
    "sex": "Sex",
    "age_years": "Age (years)",
    "height_cm": "Height (cm)",
    "weight_kg": "Weight (kg)",
    "bsa_m2": "BSA, m2",
    "race": "Race",
    "surgical_priority": "Surgical Priority",
    "surgery_pre": "Surgery",
    "surgery": "Surgery",
    "procedure_date": "Procedure Date",
    "infective_endocarditis_pre": "IE",
    "infective_endocarditis": "IE",
    "diseased_vessels_count": "No. of Diseased Vessels",
    "num_diseased_vessels": "No. of Diseased Vessels",
    "left_main_stenosis_ge_50_pct": "Left Main Stenosis ≥ 50%",
    "left_main_stenosis_ge_50": "Left Main Stenosis ≥ 50%",
    "proximal_lad_stenosis_ge_70_pct": "Proximal LAD Stenosis ≥ 70%",
    "proximal_lad_stenosis_ge_70": "Proximal LAD Stenosis ≥ 70%",
    "coronary_symptom": "Coronary Symptom",
    "nyha_pre": "Preoperative NYHA",
    "preoperative_nyha": "Preoperative NYHA",
    "ccs4": "CCS4",
    "pre_lvef_pct": "LVEF, %",
    "lvef_pre_pct": "LVEF, %",
    "hf_class_by_ef": "Classification of Heart Failure According to Ejection Fraction",
    "previous_surgery": "Previous surgery",
    "preoperative_medications": "Preoperative Medications",
    "heart_failure": "HF",
    "hf": "HF",
    "arrhythmia_remote": "Arrhythmia Remote",
    "arrhythmia_recent": "Arrhythmia Recent",
    "hypertension": "Hypertension",
    "diabetes": "Diabetes",
    "dyslipidemia": "Dyslipidemia",
    "stroke_history": "CVA",
    "cva": "CVA",
    "peripheral_vascular_disease": "PVD",
    "pvd": "PVD",
    "alcohol_use": "Alcohol",
    "alcohol": "Alcohol",
    "smoking_pack_years": "Smoking (Pack-year)",
    "ex_smoker_pack_years": "Ex-Smoker (Pack-year)",
    "smoking": "_smoking_status_csv",
    "smoking_status": "_smoking_status_csv",
    "cancer_le_5_years": "Cancer ≤ 5 yrs",
    "family_history_cad": "Family Hx of CAD",
    "anticoagulation_antiplatelet": "Anticoagulation/ Antiaggregation",
    "days_anticoagulation_suspension": "Suspension of Anticoagulation (day)",
    "pneumonia_pre": "Pneumonia",
    "other_information": "Others informations",
    "dialysis_pre": "Dialysis",
    "kdigo_stage": "KDIGO †",
    "copd": "Chronic Lung Disease",
    "dpoc": "Chronic Lung Disease",
    "chronic_lung_disease_pre": "Chronic Lung Disease",
    "dpoc_pre": "Chronic Lung Disease",
    "critical_preop_state": "Critical preoperative state",
    "critical_state_pre": "Critical preoperative state",
    "critical_condition": "Critical preoperative state",
    "critical_condition_pre": "Critical preoperative state",
    "estado_critico_pre": "Critical preoperative state",
    "poor_mobility": "Poor mobility",
    "reduced_mobility": "Poor mobility",
    "mobilidade_reduzida": "Poor mobility",
    "creatinine_clearance_ml_min": "Cr clearance, ml/min *",
    "creatinine_mg_dl": "Creatinine (mg/dL)",
    "creatinine_pre_mg_dl": "Creatinine (mg/dL)",
    "hematocrit_pct": "Hematocrit (%)",
    "hematocrit_pre_pct": "Hematocrit (%)",
    "wbc_count_10e3_ul": "WBC Count (10³/μL)",
    "wbc_count_pre_10e3_ul": "WBC Count (10³/μL)",
    "platelet_count_cells_ul": "Platelet Count (cells/μL)",
    "platelet_count_pre_cells_ul": "Platelet Count (cells/μL)",
    "inr": "INR",
    "ptt": "PTT",
    "exam_date": "Exam date",
    "lvef_pct": "LVEF, %",
    "aortic_stenosis": "Aortic Stenosis",
    "aortic_stenosis_pre": "Aortic Stenosis",
    "aortic_mean_gradient_mmhg": "Aortic Mean gradient (mmHg)",
    "aortic_mean_gradient_pre_mmhg": "Aortic Mean gradient (mmHg)",
    "ava_cm2": "AVA (cm²)",
    "aortic_valve_area_pre_cm2": "AVA (cm²)",
    "aortic_regurgitation": "Aortic Regurgitation",
    "aortic_regurgitation_pre": "Aortic Regurgitation",
    "vena_contracta": "Vena contracta",
    "vena_contracta_pre": "Vena contracta",
    "pht_aortic": "PHT Aortic",
    "pht_aortic_pre": "PHT Aortic",
    "mitral_stenosis": "Mitral Stenosis",
    "mitral_stenosis_pre": "Mitral Stenosis",
    "mitral_mean_gradient_mmhg": "Mitral Mean gradient (mmHg)",
    "mitral_mean_gradient_pre_mmhg": "Mitral Mean gradient (mmHg)",
    "mva_cm2": "MVA (cm²)",
    "mitral_valve_area_pre_cm2": "MVA (cm²)",
    "mitral_regurgitation": "Mitral Regurgitation",
    "mitral_regurgitation_pre": "Mitral Regurgitation",
    "vena_contracta_mm": "Vena contracta (mm)",
    "vena_contracta_mm_pre": "Vena contracta (mm)",
    "pht_mitral": "PHT Mitral",
    "pht_mitral_pre": "PHT Mitral",
    "tricuspid_regurgitation": "Tricuspid Regurgitation",
    "tricuspid_regurgitation_pre": "Tricuspid Regurgitation",
    "psap": "PSAP",
    "psap_pre": "PSAP",
    "tapse": "TAPSE",
    "tapse_pre": "TAPSE",
    "aortic_root_abscess": "Aortic Root Abscess",
    "aortic_root_abscess_pre": "Aortic Root Abscess",
    "death": "Death",
    "euroscore_ii": "EuroSCORE II",
    "euroscore_ii_automatic": "EuroSCORE II Automático",
    "operative_mortality_sts": "Operative Mortality",
    "morbidity_mortality_sts": "Morbidity & Mortality",
    "euroscore": "EuroSCORE II",
    "operative": "Operative Mortality",
    "morbidity": "Morbidity & Mortality",
    "stroke_sts_short": "Stroke",
    "renal_fail": "Renal Failure",
    "reoperati": "Reoperation",
    "prolonged": "Prolonged Ventilation",
    "deep_ster": "Deep Sternal Wound Infection",
    "long_hosp": "Long Hospital Stay (>14 days)",
    "short_hos": "Short Hospital Stay (<6 days)",
    "stroke_sts": "Stroke",
    "renal_failure_sts": "Renal Failure",
    "reoperation_sts": "Reoperation",
    "prolonged_ventilation_sts": "Prolonged Ventilation",
    "deep_sternal_wound_infection_sts": "Deep Sternal Wound Infection",
    "long_hospital_stay_gt_14_days_sts": "Long Hospital Stay (>14 days)",
    "short_hospital_stay_lt_6_days_sts": "Short Hospital Stay (<6 days)",
}

FLAT_PREOP_ALLOWED_COLUMNS = {
    "Sex",
    "Age (years)",
    "Height (cm)",
    "Weight (kg)",
    "BSA, m2",
    "Race",
    "Surgical Priority",
    "Surgery",
    "IE",
    "No. of Diseased Vessels",
    "Left Main Stenosis ≥ 50%",
    "Proximal LAD Stenosis ≥ 70%",
    "Coronary Symptom",
    "Preoperative NYHA",
    "CCS4",
    "LVEF, %",
    "Pré-LVEF, %",
    "Classification of Heart Failure According to Ejection Fraction",
    "Previous surgery",
    "HF",
    "Arrhythmia Remote",
    "Arrhythmia Recent",
    "Hypertension",
    "Diabetes",
    "Dyslipidemia",
    "CVA",
    "PVD",
    "Alcohol",
    "Smoking (Pack-year)",
    "Ex-Smoker (Pack-year)",
    "Cancer ≤ 5 yrs",
    "Family Hx of CAD",
    "Anticoagulation/ Antiaggregation",
    "Suspension of Anticoagulation (day)",
    "Pneumonia",
    "Dialysis",
    "KDIGO †",
    "Chronic Lung Disease",
    "Critical preoperative state",
    "Poor mobility",
    "Cr clearance, ml/min *",
    "Creatinine (mg/dL)",
    "Hematocrit (%)",
    "WBC Count (10³/μL)",
    "Platelet Count (cells/μL)",
    "INR",
    "PTT",
    "Aortic Stenosis",
    "Aortic Mean gradient (mmHg)",
    "AVA (cm²)",
    "Aortic Regurgitation",
    "Vena contracta",
    "PHT Aortic",
    "Mitral Stenosis",
    "Mitral Mean gradient (mmHg)",
    "MVA (cm²)",
    "Mitral Regurgitation",
    "Vena contracta (mm)",
    "PHT Mitral",
    "Tricuspid Regurgitation",
    "PSAP",
    "TAPSE",
    "Aortic Root Abscess",
}


def _norm_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_patient(value: object) -> str:
    txt = _norm_text(value).upper()
    txt = re.sub(r"\s+", " ", txt)
    return txt


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def parse_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    txt = str(value).strip()
    if txt.lower() in MISSING_TOKENS:
        return np.nan
    txt = txt.replace(",", ".")
    found = re.findall(r"[-+]?\d*\.?\d+", txt)
    if not found:
        return np.nan
    try:
        return float(found[0])
    except ValueError:
        return np.nan


def map_death_30d(value: object) -> int:
    txt = _norm_text(value)
    txt_l = txt.lower()
    if txt_l in {"", "-", "nan"}:
        return 0
    if txt == "Operative":
        return 1
    if txt == "> 30":
        return 0
    num = parse_number(txt)
    if np.isnan(num):
        warnings.warn(f"map_death_30d: unrecognised value '{txt}' — treated as 0 (survivor). Please review this record.", stacklevel=2)
        return 0
    return int(num <= 30)


def is_combined_surgery(text: object) -> int:
    s = _norm_text(text)
    if not s:
        return 0
    return int("," in s or ";" in s or "+" in s)


MAJOR_PROCEDURES = {
    "cabg",
    "opcab",
    "avr",
    "av repair",
    "mv repair",
    "mvr",
    "tv repair",
    "tvr",
    "asd closure",
    "vsd correction",
    "aortic aneurism repair",
    "aortic dissection repair",
    "bentall-de bono procedure",
    "valve sparing aortic root replacement (david procedure)",
    "ross",
    "pericardiectomy",
    "heart transplant",
    "intracardiac tumor resection",
    "resection of intracardiac and/or pulmonary artery thrombus",
    "pulmonary homograft implantation",
    "left ventricular aneurysmectomy",
    "myectomy",
    "surgical treatment of anomalous aortic origin of coronary",
}

MINOR_PROCEDURES = {
    "pacemaker implantation",
    "pfo closure",
    "laao",
    "thrombus removal",
    "pacemaker electrode extraction",
    "tevar",
    "removal of panus in aortic valve",
}

THORACIC_AORTA_PROCEDURES = {
    "aortic aneurism repair",
    "aortic dissection repair",
    "bentall-de bono procedure",
    "valve sparing aortic root replacement (david procedure)",
}

NON_AORTA_EXCLUSIONS = {
    "surgical treatment of anomalous aortic origin of coronary",
    "removal of panus in aortic valve",
}


def split_surgery_procedures(text: object) -> List[str]:
    s = _norm_text(text)
    if not s:
        return []
    normalized = s.replace(";", ",").replace("+", ",")
    parts = [p.strip().lower() for p in normalized.split(",") if p.strip()]
    return [p for p in parts if p not in MINOR_PROCEDURES]


def procedure_weight(text: object) -> str:
    parts = split_surgery_procedures(text)
    major = [p for p in parts if p in MAJOR_PROCEDURES]
    if not major:
        return "1_non_cabg"
    if len(major) == 1 and major[0] in {"cabg", "opcab"}:
        return "isolated_cabg"
    if len(major) == 1:
        return "1_non_cabg"
    if len(major) == 2:
        return "2_procedures"
    return "3plus_procedures"


def thoracic_aorta_surgery(text: object) -> int:
    parts = split_surgery_procedures(text)
    for p in parts:
        if p in NON_AORTA_EXCLUSIONS:
            continue
        if p in THORACIC_AORTA_PROCEDURES:
            return 1
    return 0


def _choose_echo_for_patient(patient_surgeries: pd.DataFrame, patient_echo: pd.DataFrame) -> pd.DataFrame:
    if patient_echo.empty:
        out = pd.DataFrame(index=patient_surgeries.index)
        return out

    eco = patient_echo.copy().sort_values("_echo_date")
    eco_valid = eco[eco["_echo_date"].notna()]
    rows = []
    for idx, surg_row in patient_surgeries.iterrows():
        d = surg_row["_proc_date"]
        if pd.isna(d) or eco_valid.empty:
            chosen = eco.iloc[-1]
        else:
            eligible = eco_valid[eco_valid["_echo_date"] <= d]
            if not eligible.empty:
                chosen = eligible.iloc[-1]
            else:
                dist = (pd.to_datetime(eco_valid["_echo_date"]) - pd.to_datetime(d)).abs()
                if len(dist) == 0:
                    chosen = eco.iloc[-1]
                else:
                    chosen = eco_valid.loc[dist.idxmin()]
        chosen = chosen.copy()
        chosen.name = idx
        rows.append(chosen)

    return pd.DataFrame(rows)


def _aggregate_score_by_patient(df: pd.DataFrame, patient_col: str, score_col: str) -> pd.Series:
    temp = df.copy()
    temp["_patient_key"] = temp[patient_col].map(normalize_patient)
    temp["_score"] = pd.to_numeric(temp[score_col], errors="coerce")
    return temp.groupby("_patient_key")["_score"].median()


@dataclass
class PreparedData:
    data: pd.DataFrame
    feature_columns: List[str]
    info: Dict[str, object]


def _load_source_tables(source_path: str) -> Dict[str, pd.DataFrame]:
    _COL_FIXES = {
        "Surgical Priorit": "Surgical Priority",
    }

    def _strip_col_whitespace(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
        df.columns = [_COL_FIXES.get(c, c) for c in df.columns]
        return df

    ext = Path(source_path).suffix.lower()
    if ext in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(source_path)
        return {name: _strip_col_whitespace(pd.read_excel(source_path, sheet_name=name)) for name in xls.sheet_names}
    if ext in {".db", ".sqlite", ".sqlite3"}:
        conn = sqlite3.connect(source_path)
        try:
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
            return {name: _strip_col_whitespace(pd.read_sql_query(f'SELECT * FROM "{name}"', conn)) for name in tables}
        finally:
            conn.close()
    raise ValueError(f"Unsupported multi-table source format: {ext}")


def _normalize_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    rename_map = {col: FLAT_ALIAS_TO_APP_COLUMNS[col] for col in renamed.columns if col in FLAT_ALIAS_TO_APP_COLUMNS}
    renamed = renamed.rename(columns=rename_map)
    return renamed


def _read_csv_auto(path: str, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", nrows=nrows)


def prepare_flat_dataset(source_path: str) -> PreparedData:
    ext = Path(source_path).suffix.lower()
    if ext == ".csv":
        df = _read_csv_auto(source_path)
    elif ext == ".parquet":
        df = pd.read_parquet(source_path)
    else:
        raise ValueError(f"Unsupported flat source format: {ext}")

    data = _normalize_flat_columns(df)

    # Expand simplified smoking status into the two pack-year columns
    if "_smoking_status_csv" in data.columns:
        _status = data["_smoking_status_csv"].astype(str).str.strip().str.lower()
        if "Smoking (Pack-year)" not in data.columns:
            data["Smoking (Pack-year)"] = _status.map(
                lambda v: "30" if v == "current" else "Never"
            )
        if "Ex-Smoker (Pack-year)" not in data.columns:
            data["Ex-Smoker (Pack-year)"] = _status.map(
                lambda v: "20" if v in {"former", "ex-smoker"} else "Never"
            )
        data.drop(columns=["_smoking_status_csv"], inplace=True)

    if "morte_30d" not in data.columns:
        if "Death" in data.columns:
            data["morte_30d"] = data["Death"].map(map_death_30d)
        else:
            raise ValueError("Flat dataset must include a 'morte_30d' column or a 'Death' column that can be converted into the outcome.")

    def _parse_pct_score(series: pd.Series) -> pd.Series:
        """Parse score values like '0,91%', '1.23%', '2,06' → float (as %)."""
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.rstrip("%")
            .str.strip()
            .str.replace(",", ".", regex=False)
        )
        result = pd.to_numeric(cleaned, errors="coerce")
        return result

    if "euroscore_sheet" not in data.columns:
        if "EuroSCORE II" in data.columns:
            data["euroscore_sheet"] = _parse_pct_score(data["EuroSCORE II"])
        else:
            data["euroscore_sheet"] = np.nan
    if "euroscore_auto_sheet" not in data.columns:
        if "EuroSCORE II Automático" in data.columns:
            data["euroscore_auto_sheet"] = _parse_pct_score(data["EuroSCORE II Automático"])
        elif "EuroSCORE II" in data.columns:
            data["euroscore_auto_sheet"] = _parse_pct_score(data["EuroSCORE II"])
        else:
            data["euroscore_auto_sheet"] = np.nan
    if "sts_score_sheet" not in data.columns:
        sts_candidates = [c for c in data.columns if "Operative Mortality" in str(c)]
        if not sts_candidates:
            sts_candidates = [c for c in data.columns if c == "STS Score"]
        if sts_candidates:
            data["sts_score_sheet"] = _parse_pct_score(data[sts_candidates[0]])
        else:
            data["sts_score_sheet"] = np.nan

    if "Surgery" in data.columns:
        data["cirurgia_combinada"] = data["Surgery"].map(is_combined_surgery)
        data["peso_procedimento"] = data["Surgery"].map(procedure_weight)
        data["thoracic_aorta_flag"] = data["Surgery"].map(thoracic_aorta_surgery)

    if "_patient_key" not in data.columns:
        if "Name" in data.columns:
            data["_patient_key"] = data["Name"].map(normalize_patient)
        else:
            data["_patient_key"] = pd.Series(range(len(data))).astype(str)

    exclude_cols = {
        "morte_30d",
        "Death",
        "euroscore_sheet",
        "euroscore_auto_sheet",
        "euroscore_calc",
        "sts_score",
        "sts_score_sheet",
        "euroscore_sheet_clean",
        "euroscore_auto_sheet_clean",
        "ia_risk_oof",
        "ia_risk_fullfit",
        "classe_ia",
        "classe_euro",
        "classe_sts",
    }
    allowed_cols = set(FLAT_PREOP_ALLOWED_COLUMNS) | {"cirurgia_combinada", "peso_procedimento", "thoracic_aorta_flag", "_patient_key"}
    _noise_cols_flat = {
        "Others informations", "Others",
        "Classification of Heart Failure According to Ejection Fraction",
        "Preoperative Medications",
    }
    _engineered = {"cirurgia_combinada", "peso_procedimento", "thoracic_aorta_flag"}

    def _should_exclude_flat(col_name: str) -> bool:
        if col_name in _engineered:
            return False
        s = data[col_name]
        # Count true missing: NaN + MISSING_TOKENS (e.g. "-", "nan", "unknown")
        is_missing = s.isna() | s.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)
        if is_missing.sum() / len(s) > 0.95:
            return True
        # Zero variance (only 1 unique non-missing value)
        real_values = s[~is_missing]
        if real_values.nunique() <= 1:
            return True
        return False

    feature_columns = [
        c for c in data.columns
        if c in allowed_cols and c not in exclude_cols and c != "_patient_key"
        and c not in _noise_cols_flat and not _should_exclude_flat(c)
    ]
    info = {
        "n_rows": int(len(data)),
        "n_features": int(len(feature_columns)),
        "positive_rate": float(pd.to_numeric(data["morte_30d"], errors="coerce").mean()),
        "source_type": "flat",
    }
    return PreparedData(data=data, feature_columns=feature_columns, info=info)


def prepare_master_dataset(xlsx_path: str, require_surgery_and_date: bool = True) -> PreparedData:
    ext = Path(xlsx_path).suffix.lower()
    if ext in {".csv", ".parquet"}:
        return prepare_flat_dataset(xlsx_path)

    tables = _load_source_tables(xlsx_path)
    missing_tables = [t for t in REQUIRED_SOURCE_TABLES if t not in tables]
    if missing_tables:
        raise ValueError(f"Missing required tables/sheets: {', '.join(missing_tables)}")

    pre = tables["Preoperative"]
    eco = tables["Pre-Echocardiogram"]
    post = tables["Postoperative"]
    eu = tables.get("EuroSCORE II")
    eu_auto = tables.get("EuroSCORE II Automático")
    sts = tables.get("STS Score")

    pre = pre.copy()
    post = post.copy()
    eco = eco.copy()

    pre["_patient_key"] = pre["Name"].map(normalize_patient)
    post["_patient_key"] = post["Patient"].map(normalize_patient)
    eco["_patient_key"] = eco["Patient"].map(normalize_patient)

    pre["_proc_date"] = _to_date(pre["Procedure Date"])
    post["_proc_date"] = _to_date(post["Procedure Date"])
    eco["_echo_date"] = _to_date(eco["Exam date"])

    pre_rows_before = len(pre)
    if require_surgery_and_date:
        pre["_surgery_txt"] = pre["Surgery"].astype(str).str.strip()
        pre = pre[(pre["_proc_date"].notna()) & (pre["_surgery_txt"].str.lower().ne("nan")) & (pre["_surgery_txt"] != "")]
        pre = pre.drop(columns=["_surgery_txt"])
    pre_rows_after = len(pre)
    pre_unique_after = int(pre[["_patient_key", "_proc_date"]].drop_duplicates().shape[0])
    post_unique = int(post[["_patient_key", "_proc_date"]].drop_duplicates().shape[0])

    pre_post = pre.merge(
        post[["_patient_key", "_proc_date", "Death"]],
        on=["_patient_key", "_proc_date"],
        how="inner",
    )
    pre_post_rows = len(pre_post)
    pre_post["morte_30d"] = pre_post["Death"].map(map_death_30d)

    eco_cols = [c for c in eco.columns if c not in {"Patient", "Exam date", "_patient_key", "_echo_date"}]
    echo_joined_rows = []
    for patient_key, grp in pre_post.groupby("_patient_key", sort=False):
        egrp = eco[eco["_patient_key"] == patient_key][["_echo_date"] + eco_cols]
        joined = _choose_echo_for_patient(grp[["_proc_date"]], egrp)
        echo_joined_rows.append(joined)

    if echo_joined_rows:
        echo_aligned = pd.concat(echo_joined_rows).sort_index()
    else:
        echo_aligned = pd.DataFrame(index=pre_post.index)

    for c in eco_cols:
        if c in echo_aligned.columns:
            pre_post[c] = echo_aligned[c]
        else:
            pre_post[c] = np.nan

    pre_post["cirurgia_combinada"] = pre_post["Surgery"].map(is_combined_surgery)
    pre_post["peso_procedimento"] = pre_post["Surgery"].map(procedure_weight)
    pre_post["thoracic_aorta_flag"] = pre_post["Surgery"].map(thoracic_aorta_surgery)

    eu_series = pd.Series(dtype=float)
    if eu is not None and "Patient" in eu.columns and "EuroSCORE II" in eu.columns:
        eu_series = _aggregate_score_by_patient(eu, "Patient", "EuroSCORE II")

    eu_auto_series = pd.Series(dtype=float)
    if eu_auto is not None and "Patient" in eu_auto.columns:
        eu_auto_cols = [c for c in eu_auto.columns if c != "Patient"]
        if eu_auto_cols:
            eu_auto_series = _aggregate_score_by_patient(eu_auto, "Patient", eu_auto_cols[-1])

    sts_series = pd.Series(dtype=float)
    if sts is not None and "Patient" in sts.columns:
        sts_candidates = [c for c in sts.columns if "Operative Mortality" in c]
        if sts_candidates:
            sts_series = _aggregate_score_by_patient(sts, "Patient", sts_candidates[0])

    pre_post["euroscore_sheet"] = pre_post["_patient_key"].map(eu_series)
    pre_post["euroscore_auto_sheet"] = pre_post["_patient_key"].map(eu_auto_series)
    pre_post["sts_score_sheet"] = pre_post["_patient_key"].map(sts_series)

    pre_cols_exclude = {
        "Name",
        "Procedure Date",
        "Death",
        "_patient_key",
        "_proc_date",
    }
    echo_cols_exclude = {"Patient", "Exam date", "_patient_key", "_echo_date"}

    pre_features = [c for c in pre.columns if c not in pre_cols_exclude]
    echo_features = [c for c in eco.columns if c not in echo_cols_exclude and c not in pre_features]
    engineered = ["cirurgia_combinada", "peso_procedimento", "thoracic_aorta_flag"]
    feature_columns = pre_features + echo_features + engineered
    feature_columns = [c for c in feature_columns if c in pre_post.columns]

    # Exclude features that add noise: >95% missing or known non-informative columns
    _noise_cols = {
        "Others informations", "Others",
        "Classification of Heart Failure According to Ejection Fraction",
        "Preoperative Medications",
    }
    def _too_sparse_or_constant(col_name: str) -> bool:
        if col_name in engineered:
            return False
        s = pre_post[col_name]
        is_missing = s.isna() | s.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)
        if is_missing.sum() / len(s) > 0.95:
            return True
        real_values = s[~is_missing]
        if real_values.nunique() <= 1:
            return True
        return False

    feature_columns = [
        c for c in feature_columns
        if c not in _noise_cols and not _too_sparse_or_constant(c)
    ]

    info = {
        "n_rows": len(pre_post),
        "n_features": len(feature_columns),
        "positive_rate": float(pre_post["morte_30d"].mean()),
        "pre_rows_before_criteria": int(pre_rows_before),
        "pre_rows_after_criteria": int(pre_rows_after),
        "excluded_missing_surgery_or_date": int(pre_rows_before - pre_rows_after),
        "require_surgery_and_date": bool(require_surgery_and_date),
        "pre_unique_patient_date_after_criteria": pre_unique_after,
        "post_unique_patient_date": post_unique,
        "matched_pre_post_rows": int(pre_post_rows),
        "excluded_no_pre_post_match": int(pre_unique_after - pre_post_rows),
        "echo_rows": int(len(eco)),
        "available_optional_tables": [t for t in OPTIONAL_SOURCE_TABLES if t in tables],
    }

    return PreparedData(data=pre_post, feature_columns=feature_columns, info=info)
