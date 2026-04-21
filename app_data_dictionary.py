"""Live data dictionary generated from the app ingestion code."""

from io import BytesIO
from typing import Dict, List, Mapping, Sequence

import pandas as pd

from risk_data import (
    BLANK_MEANS_NO_COLUMNS,
    BLANK_MEANS_NONE_COLUMNS,
    FLAT_ALIAS_TO_APP_COLUMNS,
    FLAT_PREOP_ALLOWED_COLUMNS,
    LITERAL_NONE_IS_VALID_COLUMNS,
    MISSINGNESS_INDICATOR_SPECS,
    MISSING_TOKENS,
    OPTIONAL_SOURCE_TABLES,
    REQUIRED_SOURCE_TABLES,
    _CLINICAL_PLAUSIBILITY_RANGES,
)
from variable_dictionary import VARIABLE_DICTIONARY


ENGINEERED_COLUMNS = {
    "cirurgia_combinada": {
        "domain": "Procedure",
        "definition": "Derived flag for procedures with more than one major surgical component.",
        "unit": "0/1",
        "transformation": "Derived from Surgery by separator/procedure parsing.",
    },
    "peso_procedimento": {
        "domain": "Procedure",
        "definition": "Procedure complexity bucket derived from the normalized Surgery text.",
        "unit": "category",
        "transformation": "Derived from Surgery using risk_data.procedure_weight.",
    },
    "thoracic_aorta_flag": {
        "domain": "Procedure",
        "definition": "Derived flag for thoracic-aorta procedures.",
        "unit": "0/1",
        "transformation": "Derived from Surgery using risk_data.thoracic_aorta_surgery.",
    },
    "missing_renal_labs": {
        "domain": "Laboratory",
        "definition": "Panel-level indicator that at least one renal laboratory input was missing.",
        "unit": "0/1",
        "transformation": "Derived before imputation from Creatinine (mg/dL) and Cr clearance, ml/min *.",
    },
    "missing_cbc_labs": {
        "domain": "Laboratory",
        "definition": "Panel-level indicator that at least one complete blood count input was missing.",
        "unit": "0/1",
        "transformation": "Derived before imputation from Hematocrit, WBC Count, and Platelet Count.",
    },
    "missing_coagulation_labs": {
        "domain": "Laboratory",
        "definition": "Panel-level indicator that at least one coagulation laboratory input was missing.",
        "unit": "0/1",
        "transformation": "Derived before imputation from INR and PTT.",
    },
}

SCORE_COLUMNS = {
    "euroscore_calc": {
        "domain": "Scores",
        "definition": "EuroSCORE II calculated internally from the application inputs.",
        "unit": "0-1 probability",
        "transformation": "Reference score; not used as an AI-model predictor.",
    },
    "euroscore_sheet": {
        "domain": "Scores",
        "definition": "EuroSCORE II read from the optional EuroSCORE II sheet or flat score column.",
        "unit": "0-1 probability",
        "transformation": "Reference score parsed from the source file.",
    },
    "euroscore_auto_sheet": {
        "domain": "Scores",
        "definition": "Automatic EuroSCORE II read from the optional external sheet/column.",
        "unit": "0-1 probability",
        "transformation": "Reference score parsed from the source file.",
    },
    "sts_score_sheet": {
        "domain": "Scores",
        "definition": "STS Operative Mortality read from the optional STS Score sheet/column.",
        "unit": "0-1 probability",
        "transformation": "Reference score parsed from the source file.",
    },
}

OUTCOME_COLUMNS = {
    "morte_30d": {
        "domain": "Outcome",
        "definition": "30-day or in-hospital mortality outcome derived from Death.",
        "unit": "0/1",
        "transformation": "Death values are mapped by risk_data.map_death_30d; operative/day 0/day 1-30/death => 1, survivor/>30/missing => 0.",
    },
    "Death": {
        "domain": "Outcome",
        "definition": "Raw postoperative death/timing field used to derive morte_30d.",
        "unit": "text",
        "transformation": "Read from Postoperative in multi-sheet sources or from flat data when morte_30d is absent.",
    },
}

ECHO_SOURCE_COLUMNS = {
    "PrÃ©-LVEF, %",
    "PSAP",
    "TAPSE",
    "Aortic Stenosis",
    "Aortic Regurgitation",
    "Mitral Stenosis",
    "Mitral Regurgitation",
    "Tricuspid Regurgitation",
    "Aortic Root Abscess",
    "AVA (cmÂ²)",
    "AVA (cm²)",
    "MVA (cmÂ²)",
    "MVA (cm²)",
    "Aortic Mean gradient (mmHg)",
    "Mitral Mean gradient (mmHg)",
    "PHT Aortic",
    "PHT Mitral",
    "Vena contracta",
    "Vena contracta (mm)",
}


def _is_pt(language: str) -> bool:
    return language != "English"


def _base_by_variable() -> Dict[str, Dict[str, str]]:
    return {str(row["variable"]): dict(row) for row in VARIABLE_DICTIONARY}


def _aliases_by_target() -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {}
    for alias, target in FLAT_ALIAS_TO_APP_COLUMNS.items():
        final_target = "Smoking (Pack-year)" if target == "_smoking_status_csv" else target
        aliases.setdefault(final_target, []).append(alias)
    return {target: sorted(set(items)) for target, items in aliases.items()}


def get_reading_aliases_dataframe(language: str = "English") -> pd.DataFrame:
    """Return the flat-file alias map currently used by ingestion."""
    rows = [
        {
            "Input column": alias,
            "Canonical app variable": (
                "Smoking (Pack-year)"
                if target == "_smoking_status_csv"
                else target
            ),
            "Applied during": "Flat CSV/Parquet column canonicalization",
        }
        for alias, target in sorted(FLAT_ALIAS_TO_APP_COLUMNS.items())
    ]
    df = pd.DataFrame(rows)
    if _is_pt(language):
        df.columns = ["Coluna de entrada", "Variavel canonica no app", "Aplicado durante"]
    return df


def _reading_source(variable: str, origin: str) -> str:
    if variable in ENGINEERED_COLUMNS:
        return "Derived from Surgery"
    if variable in OUTCOME_COLUMNS:
        return "Postoperative / flat outcome"
    if variable in SCORE_COLUMNS:
        return "Optional score sheet/flat score column"
    if variable in ECHO_SOURCE_COLUMNS:
        return "Pre-Echocardiogram"
    if variable in FLAT_PREOP_ALLOWED_COLUMNS or origin:
        return origin or "Preoperative / flat canonical column"
    return "Flat canonical column or model artifact"


def _characteristic(variable: str, unit: str, transformation: str, pt: bool) -> str:
    text = f"{unit} {transformation}".lower()
    if variable in OUTCOME_COLUMNS:
        return "Binaria - desfecho" if pt else "Binary outcome"
    if variable in ENGINEERED_COLUMNS:
        if unit == "0/1":
            return "Binaria derivada" if pt else "Derived binary"
        return "Derivada categorica" if pt else "Derived categorical"
    numeric_tokens = ("mg/dl", "cm", "kg", "mmhg", "%", "years", "ml/min")
    if "numeric" in text or "continuous" in text or any(tok in text for tok in numeric_tokens):
        return "Numerica continua" if pt else "Continuous numeric"
    if "yes/no" in text or "0/1" in text or "bin" in text:
        return "Binaria" if pt else "Binary"
    ordinal_tokens = ("mild", "moderate", "severe", "elective", "urgent", "emergency")
    if "ordinal" in text or any(tok in text for tok in ordinal_tokens):
        return "Categorica ordinal" if pt else "Ordinal categorical"
    if "text" in text:
        return "Texto" if pt else "Text"
    return "Categorica nominal" if pt else "Nominal categorical"


def _missing_rule(variable: str, pt: bool) -> str:
    generic = (
        "Tokens vazios/desconhecidos viram NaN"
        if pt
        else "Blank/unknown tokens become NaN"
    )
    if variable in BLANK_MEANS_NO_COLUMNS:
        return (
            "Celula em branco vira 'No' apos a normalizacao; tokens desconhecidos continuam NaN."
            if pt
            else "Blank cells become 'No' after normalization; unknown tokens remain NaN."
        )
    if variable in BLANK_MEANS_NONE_COLUMNS:
        return (
            "Celula em branco vira 'None' antes da normalizacao; 'None' e categoria clinica valida."
            if pt
            else "Blank cells become 'None' before normalization; 'None' is a valid clinical category."
        )
    if variable in LITERAL_NONE_IS_VALID_COLUMNS:
        return (
            "Literal 'None' e categoria clinica valida; demais tokens ausentes viram NaN."
            if pt
            else "Literal 'None' is a valid clinical category; other missing tokens become NaN."
        )
    return generic


def _parameterization(variable: str, unit: str, transformation: str, pt: bool) -> str:
    parts = []
    if unit:
        parts.append(("Unidade: " if pt else "Unit: ") + str(unit))
    if transformation:
        parts.append(("Leitura: " if pt else "Read as: ") + str(transformation))
    if variable in _CLINICAL_PLAUSIBILITY_RANGES:
        lo, hi = _CLINICAL_PLAUSIBILITY_RANGES[variable]
        label = "Faixa de plausibilidade" if pt else "Plausibility range"
        parts.append(f"{label}: {lo:g}-{hi:g}")
    if variable == "Surgery":
        parts.append(
            "Separadores ',', ';' e '+' indicam componentes cirurgicos."
            if pt
            else "Separators ',', ';' and '+' indicate surgical components."
        )
    if variable == "morte_30d":
        parts.append("0/1")
    return " | ".join(parts)


def _model_usage(variable: str, static_usage: str, feature_set: set[str] | None, pt: bool) -> str:
    if variable in OUTCOME_COLUMNS:
        return "Desfecho; nao preditor" if pt else "Outcome; not a predictor"
    if variable in SCORE_COLUMNS:
        return "Escore comparativo; nao preditor" if pt else "Comparator score; not a predictor"
    if feature_set is not None:
        if variable in feature_set:
            return "Preditor no modelo ativo" if pt else "Predictor in active model"
        if static_usage == "Yes":
            return "Nao retido no modelo ativo" if pt else "Not retained in active model"
    if static_usage:
        return static_usage
    return "Entrada potencial" if pt else "Potential input"


def get_app_reading_dictionary_dataframe(
    language: str = "English",
    model_feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a dictionary rebuilt from the current ingestion constants."""
    pt = _is_pt(language)
    base = _base_by_variable()
    aliases = _aliases_by_target()
    feature_set = set(model_feature_columns) if model_feature_columns is not None else None

    ordered_variables: List[str] = []
    for row in VARIABLE_DICTIONARY:
        ordered_variables.append(str(row["variable"]))
    for source in (
        sorted(FLAT_PREOP_ALLOWED_COLUMNS),
        list(ENGINEERED_COLUMNS),
        list(SCORE_COLUMNS),
        list(OUTCOME_COLUMNS),
        sorted(set(FLAT_ALIAS_TO_APP_COLUMNS.values())),
        sorted(_CLINICAL_PLAUSIBILITY_RANGES),
    ):
        for variable in source:
            if variable == "_smoking_status_csv":
                variable = "Smoking (Pack-year)"
            if variable not in ordered_variables:
                ordered_variables.append(variable)

    rows = []
    for variable in ordered_variables:
        meta: Mapping[str, str] = (
            ENGINEERED_COLUMNS.get(variable)
            or SCORE_COLUMNS.get(variable)
            or OUTCOME_COLUMNS.get(variable)
            or base.get(variable)
            or {}
        )
        definition = str(meta.get("definition", ""))
        origin = str(meta.get("origin", ""))
        unit = str(meta.get("unit", ""))
        transformation = str(meta.get("transformation", ""))
        static_usage = str(meta.get("in_model", ""))
        domain = str(meta.get("domain", "Other"))
        rows.append({
            "Variable": variable,
            "Domain": domain,
            "Characteristic": _characteristic(variable, unit, transformation, pt),
            "Description": definition,
            "Parameterization": _parameterization(variable, unit, transformation, pt),
            "Source / app input": _reading_source(variable, origin),
            "Accepted flat aliases": ", ".join(aliases.get(variable, [])),
            "Missing / blank handling": _missing_rule(variable, pt),
            "Current app/model usage": _model_usage(variable, static_usage, feature_set, pt),
        })

    df = pd.DataFrame(rows)
    if pt:
        df.columns = [
            "Variavel",
            "Dominio",
            "Caracteristica",
            "Descricao",
            "Parametrizacao",
            "Origem / entrada no app",
            "Aliases aceitos em CSV/Parquet",
            "Tratamento de ausentes/brancos",
            "Uso atual no app/modelo",
        ]
    return df


def get_reading_rules_dataframe(language: str = "English") -> pd.DataFrame:
    """Return high-level rules that explain how the app reads source data."""
    rows = [
        {
            "Rule": "Accepted sources",
            "Current behavior": ".xlsx/.xls/.db/.sqlite multi-table sources; .csv/.parquet and single-sheet .xlsx/.xls flat sources.",
            "Code source": "risk_data.prepare_master_dataset",
        },
        {
            "Rule": "Required multi-sheet tables",
            "Current behavior": ", ".join(REQUIRED_SOURCE_TABLES),
            "Code source": "risk_data.REQUIRED_SOURCE_TABLES",
        },
        {
            "Rule": "Optional score tables",
            "Current behavior": ", ".join(OPTIONAL_SOURCE_TABLES),
            "Code source": "risk_data.OPTIONAL_SOURCE_TABLES",
        },
        {
            "Rule": "Flat column aliases",
            "Current behavior": f"{len(FLAT_ALIAS_TO_APP_COLUMNS)} aliases mapped before normalization.",
            "Code source": "risk_data.FLAT_ALIAS_TO_APP_COLUMNS",
        },
        {
            "Rule": "Missing tokens",
            "Current behavior": ", ".join(sorted(repr(t) for t in MISSING_TOKENS)),
            "Code source": "risk_data.MISSING_TOKENS",
        },
        {
            "Rule": "Blank means No",
            "Current behavior": ", ".join(sorted(BLANK_MEANS_NO_COLUMNS)),
            "Code source": "risk_data.BLANK_MEANS_NO_COLUMNS",
        },
        {
            "Rule": "Blank means None",
            "Current behavior": ", ".join(sorted(BLANK_MEANS_NONE_COLUMNS)),
            "Code source": "risk_data.BLANK_MEANS_NONE_COLUMNS",
        },
        {
            "Rule": "Literal None is valid",
            "Current behavior": ", ".join(sorted(LITERAL_NONE_IS_VALID_COLUMNS)),
            "Code source": "risk_data.LITERAL_NONE_IS_VALID_COLUMNS",
        },
        {
            "Rule": "Clinical plausibility checks",
            "Current behavior": f"{len(_CLINICAL_PLAUSIBILITY_RANGES)} numeric columns have range rescue/clearing after parse_number.",
            "Code source": "risk_data._CLINICAL_PLAUSIBILITY_RANGES",
        },
        {
            "Rule": "Missingness indicators",
            "Current behavior": (
                f"{len(MISSINGNESS_INDICATOR_SPECS)} conservative panel-level "
                "laboratory missingness indicators are derived before imputation."
            ),
            "Code source": "risk_data.MISSINGNESS_INDICATOR_SPECS",
        },
        {
            "Rule": "Multi-sheet patient matching",
            "Current behavior": "Preoperative and Postoperative are inner-joined by normalized patient key plus procedure date.",
            "Code source": "risk_data.prepare_master_dataset",
        },
        {
            "Rule": "Echocardiogram alignment",
            "Current behavior": "For each surgery, the app chooses the latest echo on/before surgery; otherwise the nearest available echo.",
            "Code source": "risk_data._choose_echo_for_patient",
        },
    ]
    df = pd.DataFrame(rows)
    if _is_pt(language):
        df.columns = ["Regra", "Comportamento atual", "Fonte no codigo"]
    return df


def build_dictionary_xlsx_bytes(
    dictionary_df: pd.DataFrame,
    aliases_df: pd.DataFrame,
    rules_df: pd.DataFrame,
) -> bytes:
    """Build the downloadable dictionary workbook."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        dictionary_df.to_excel(writer, sheet_name="Dicionario", index=False)
        aliases_df.to_excel(writer, sheet_name="Aliases_CSV", index=False)
        rules_df.to_excel(writer, sheet_name="Regras_de_Leitura", index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for column_cells in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in column_cells)
                width = min(max(max_len + 2, 12), 60)
                ws.column_dimensions[column_cells[0].column_letter].width = width

    return buf.getvalue()
