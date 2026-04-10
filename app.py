# force-reload-v4: ensure modeling.py and sts_calculator.py changes are picked up
import importlib as _il
import modeling as _modeling_mod
_il.reload(_modeling_mod)
import sts_calculator as _sts_mod
_il.reload(_sts_mod)

import json
from io import BytesIO
from pathlib import Path
from typing import Dict
import re
import shutil
import sqlite3
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance

from config import AppConfig
from euroscore import COEF as EURO_COEF
from euroscore import EURO_CONST, euroscore_from_inputs, euroscore_from_row
from explainability import ModelExplainer, show_explainability_ui
from modeling import clean_features, train_and_select_model
from risk_data import (
    FLAT_ALIAS_TO_APP_COLUMNS,
    PreparedData,
    REQUIRED_SOURCE_TABLES,
    is_combined_surgery,
    parse_number,
    prepare_master_dataset,
    procedure_weight,
    split_surgery_procedures,
    thoracic_aorta_surgery,
)
from sts_calculator import (
    HAS_WEBSOCKETS as HAS_STS,
    STS_LABELS,
    calculate_sts,
    calculate_sts_batch,
)
from stats_compare import (
    bootstrap_auc_diff,
    bootstrap_metrics_ci,
    calibration_data,
    calibration_intercept_slope,
    class_risk,
    compute_idi,
    compute_nri,
    decision_curve,
    delong_roc_test,
    evaluate_scores,
    evaluate_scores_with_threshold,
    evaluate_scores_with_ci,
    evaluate_scores_temporal,
    hosmer_lemeshow_test,
    pairwise_score_comparison,
    risk_category_table,
    roc_data,
)
from model_metadata import (
    build_model_metadata,
    format_metadata_for_display,
    format_locked_model_for_display,
    assess_input_completeness,
    format_imputation_detail,
    log_analysis,
    read_audit_log,
    generate_individual_report,
    generate_clinical_explanation,
    build_statistical_summary,
    build_temporal_validation_summary,
    statistical_summary_to_xlsx,
    statistical_summary_to_csv,
    statistical_summary_to_pdf,
    compute_data_quality_summary,
    check_temporal_overlap,
    check_validation_readiness,
    export_model_bundle_metadata,
)
from variable_dictionary import get_dictionary_dataframe, get_dictionary_by_domain


st.set_page_config(page_title=AppConfig.PAGE_TITLE, layout=AppConfig.LAYOUT)

language = st.sidebar.selectbox(
    "Language",
    AppConfig.LANGUAGES,
    index=0,
    label_visibility="collapsed",
)
st.sidebar.divider()


def tr(en: str, pt: str) -> str:
    return en if language == "English" else pt


def hp(en: str, pt: str) -> str:
    return en if language == "English" else pt

# Use centralized configuration from config module
MODEL_CACHE_FILE = AppConfig.MODEL_CACHE_FILE
MODEL_VERSION = AppConfig.MODEL_VERSION
APP_CACHE_DIR = AppConfig.APP_CACHE_DIR
TEMP_DATA_DIR = AppConfig.TEMP_DATA_DIR
LOCAL_DATA_DIR = AppConfig.LOCAL_DATA_DIR
UPLOAD_CACHE_FILE = AppConfig.UPLOAD_CACHE_FILE
GSHEETS_CACHE_FILE = AppConfig.GSHEETS_CACHE_FILE
REQUIRED_SHEETS = {
    *REQUIRED_SOURCE_TABLES,
}

APP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data(show_spinner=False)
def _cached_eligibility_info(xlsx_path: str) -> dict:
    """Cache only the numeric data; language-dependent formatting happens outside the cache."""
    prepared = prepare_master_dataset(xlsx_path)
    return {k: v for k, v in prepared.info.items() if isinstance(v, (int, float, bool, str))}


def _eligibility_summary(xlsx_path: str) -> pd.DataFrame:
    """Build the eligibility summary table. tr() is called here so language changes take effect immediately."""
    info = _cached_eligibility_info(xlsx_path)
    rows = [
        {
            tr("Step", "Etapa"): tr("Preoperative rows read", "Linhas lidas em Preoperative"),
            tr("Count", "Quantidade"): int(info.get("pre_rows_before_criteria", 0)),
        },
        {
            tr("Step", "Etapa"): tr("Excluded: missing Surgery or Procedure Date", "Excluídos: Surgery ou Procedure Date ausentes"),
            tr("Count", "Quantidade"): int(info.get("excluded_missing_surgery_or_date", 0)),
        },
        {
            tr("Step", "Etapa"): tr("Eligible after inclusion criteria", "Elegíveis após critérios de inclusão"),
            tr("Count", "Quantidade"): int(info.get("pre_rows_after_criteria", 0)),
        },
        {
            tr("Step", "Etapa"): tr("Unique patient-procedure keys in Postoperative", "Chaves paciente-procedimento únicas em Postoperative"),
            tr("Count", "Quantidade"): int(info.get("post_unique_patient_date", 0)),
        },
        {
            tr("Step", "Etapa"): tr("Matched Preoperative-Postoperative rows", "Linhas pareadas Preoperative-Postoperative"),
            tr("Count", "Quantidade"): int(info.get("matched_pre_post_rows", 0)),
        },
        {
            tr("Step", "Etapa"): tr("Excluded: no Preoperative-Postoperative match", "Excluídos: sem pareamento Preoperative-Postoperative"),
            tr("Count", "Quantidade"): int(info.get("excluded_no_pre_post_match", 0)),
        },
    ]
    return pd.DataFrame(rows)


def _to_csv_bytes(df: pd.DataFrame, **kwargs) -> bytes:
    """Export DataFrame to CSV bytes using ';' separator for Portuguese locale compatibility."""
    sep = ";" if language != "English" else ","
    return df.to_csv(index=False, sep=sep, **kwargs).encode("utf-8")


@st.fragment
def _csv_download_btn(df: pd.DataFrame, filename: str, label: str) -> None:
    """Download button isolated in a fragment — clicking it won't rerun the whole page."""
    st.download_button(label, data=df.pipe(_to_csv_bytes), file_name=filename, mime="text/csv")


@st.fragment
def _txt_download_btn(text: str, filename: str, label: str) -> None:
    """Text download button isolated in a fragment."""
    st.download_button(label, data=text.encode("utf-8"), file_name=filename, mime="text/plain")


def _format_ppv_npv(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN in PPV/NPV with '—' for display (Streamlit shows NaN as 'None')."""
    df = df.copy()
    for col in ("PPV", "NPV"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    return df


def _safe_prob(x: object) -> float:
    try:
        raw = str(x).strip().replace("%", "").replace(",", ".")
        v = float(raw)
    except Exception:
        return np.nan
    if v > 1:
        v = v / 100.0
    if v < 0 or v > 1:
        return np.nan
    return float(min(v, 1.0))


def _compute_bundle(xlsx_path: str, progress_callback=None) -> Dict[str, object]:
    # Force-reload modeling module to pick up any code changes without server restart
    import importlib
    import config.model_config as _cfg_mod
    importlib.reload(_cfg_mod)
    import modeling as _mod
    importlib.reload(_mod)
    _fresh_train = _mod.train_and_select_model

    prepared = prepare_master_dataset(xlsx_path)
    artifacts = _fresh_train(prepared.data, prepared.feature_columns, progress_callback=progress_callback)

    df = prepared.data.copy()
    df["euroscore_calc"] = df.apply(euroscore_from_row, axis=1)
    df["euroscore_sheet_clean"] = pd.to_numeric(df["euroscore_sheet"], errors="coerce")
    df["euroscore_auto_sheet_clean"] = pd.to_numeric(df["euroscore_auto_sheet"], errors="coerce")

    # STS: query web calculator for all patients (preferred over CSV values)
    sts_ws_results = []
    if HAS_STS:
        rows_as_dicts = df.to_dict(orient="records")
        sts_ws_results = calculate_sts_batch(rows_as_dicts)

    if sts_ws_results:
        df["sts_score"] = [r.get("predmort", np.nan) for r in sts_ws_results]
        # Store all STS sub-scores for later display
        for key in ["predmort", "predmm", "predstro", "predrenf", "predreop",
                     "predvent", "preddeep", "pred14d", "pred6d"]:
            df[f"sts_{key}"] = [r.get(key, np.nan) for r in sts_ws_results]
    else:
        # No fallback — STS is only available via the web calculator
        df["sts_score"] = np.nan

    return {
        "prepared": prepared,
        "artifacts": artifacts,
        "data": df,
    }


def _bundle_signature(xlsx_path: str) -> Dict[str, object]:
    p = Path(xlsx_path)
    stt = p.stat()
    return {
        "xlsx_path": str(p.resolve()),
        "xlsx_mtime_ns": int(stt.st_mtime_ns),
        "xlsx_size": int(stt.st_size),
        "model_version": MODEL_VERSION,
    }


def _google_sheet_export_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        raise ValueError("empty_url")

    parsed = urlparse(raw)
    if "docs.google.com" not in parsed.netloc or "/spreadsheets/" not in parsed.path:
        raise ValueError("invalid_google_sheets_url")

    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", parsed.path)
    if not m:
        raise ValueError("sheet_id_not_found")
    sheet_id = m.group(1)

    # For this project we need the full workbook (all tabs), so we deliberately
    # ignore any gid from the shared URL.
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"


def _download_to_file(url: str, out_path: Path) -> tuple[bool, str]:
    try:
        with urlopen(url, timeout=40) as r:
            data = r.read()
        if len(data) < 200:
            return False, "downloaded_file_too_small"
        if not data.startswith(b"PK\x03\x04"):
            return False, "downloaded_content_is_not_valid_xlsx"
        out_path.write_bytes(data)
        return True, "ok"
    except Exception as e:
        return False, str(e)


def _validate_source(path: str) -> tuple[bool, str]:
    ext = Path(path).suffix.lower()
    if ext in {".csv", ".parquet"}:
        try:
            if ext == ".csv":
                df = pd.read_csv(path, sep=None, engine="python", nrows=5)
            else:
                df = pd.read_parquet(path)
        except Exception as e:
            return False, f"invalid_dataset: {e}"
        if "morte_30d" not in df.columns and "Death" not in df.columns and "death" not in df.columns:
            return False, "flat_dataset_missing_column: morte_30d_or_Death_or_death"
        return True, "ok"
    if ext in {".db", ".sqlite", ".sqlite3"}:
        try:
            conn = sqlite3.connect(path)
            try:
                names = set(pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist())
            finally:
                conn.close()
        except Exception as e:
            return False, f"invalid_db: {e}"
        missing = sorted(REQUIRED_SHEETS - names)
        if missing:
            found = ", ".join(sorted(names))
            return False, f"missing_tables: {', '.join(missing)} | found_tables: {found}"
        return True, "ok"
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        return False, f"invalid_xlsx: {e}"
    names = set(xls.sheet_names)
    missing = sorted(REQUIRED_SHEETS - names)
    if missing:
        found = ", ".join(sorted(xls.sheet_names))
        return False, f"missing_sheets: {', '.join(missing)} | found_sheets: {found}"
    return True, "ok"


def _clear_temp_data_dir() -> tuple[bool, str]:
    try:
        if TEMP_DATA_DIR.exists():
            shutil.rmtree(TEMP_DATA_DIR)
        TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        if MODEL_CACHE_FILE.exists():
            MODEL_CACHE_FILE.unlink()
        return True, "ok"
    except Exception as e:
        return False, str(e)


def _local_source_candidates() -> list[Path]:
    patterns = ["*.xlsx", "*.xls", "*.csv", "*.parquet", "*.db", "*.sqlite", "*.sqlite3"]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(LOCAL_DATA_DIR.glob(pattern)))
    unique: dict[str, Path] = {}
    for p in candidates:
        unique[str(p.resolve())] = p
    return sorted(unique.values(), key=lambda p: str(p).lower())


def render_analysis_guide(prepared, artifacts, triple_n: int | None = None):
    st.subheader(tr("Analysis Guide", "Guia da Análise"))
    st.caption(
        tr(
            "Complete reference on how the data were selected, how each score was built, and how to interpret every metric in this app.",
            "Referência completa sobre como os dados foram selecionados, como cada escore foi construído e como interpretar cada métrica deste app.",
        )
    )

    # ── 1. Overview ──
    with st.expander(tr("What does this app do?", "O que este app faz?"), expanded=True):
        st.markdown(
            tr(
                """
This app compares **three risk scores** for predicting in-hospital or 30-day mortality after cardiac surgery:

| Score | How it's calculated | Source |
|:--|:--|:--|
| **AI Risk** | Machine learning model trained on your local data | This app |
| **EuroSCORE II** | Published logistic equation (Nashef et al., 2012) | Calculated locally |
| **STS** | STS Risk Calculator (automated web query) | Obtained via automated queries to the official web calculator |

The app evaluates discrimination, calibration, clinical utility, and reclassification — all from preoperative data only.
""",
                """
Este app compara **três escores de risco** para predição de mortalidade hospitalar ou em 30 dias após cirurgia cardíaca:

| Escore | Como é calculado | Fonte |
|:--|:--|:--|
| **AI Risk** | Modelo de machine learning treinado nos seus dados locais | Este app |
| **EuroSCORE II** | Equação logística publicada (Nashef et al., 2012) | Calculado localmente |
| **STS** | STS Risk Calculator (consulta web automatizada) | Obtido via consultas automatizadas à calculadora web oficial |

O app avalia discriminação, calibração, utilidade clínica e reclassificação — tudo a partir de dados exclusivamente pré-operatórios.
""",
            )
        )

    # ── 2. Data ──
    with st.expander(tr("Data and eligibility", "Dados e elegibilidade")):
        st.markdown(
            tr(
                """
**Data sources:** The app accepts Excel files (.xlsx with Preoperative, Pre-Echocardiogram, and Postoperative sheets) or flat CSV/Parquet files. CSV files with either `,` or `;` as field separator and comma or dot as decimal separator are handled automatically.

**Inclusion criteria (Excel):** Only records with 'Surgery' and 'Procedure Date' fields are included. Records are matched across the three sheets using patient identity and procedure date. Name and date are used only for internal linkage and **never as predictors**.

**Inclusion criteria (CSV):** All rows with a valid 'Surgery' field are included. Column names are mapped from snake_case (e.g. `lvef_pre_pct`) to the internal format automatically.

**Primary outcome:** 30-day or in-hospital mortality, extracted from the `Death` column. Values "Operative" or numeric ≤30 (days to death, including 0 = immediate postoperative death) are coded as events; ">30" or "-" are coded as survivors.
""",
                """
**Fontes de dados:** O app aceita arquivos Excel (.xlsx com abas Preoperative, Pre-Echocardiogram e Postoperative) ou arquivos CSV/Parquet. Arquivos CSV com separador `,` ou `;` e decimal com vírgula ou ponto são tratados automaticamente.

**Critérios de inclusão (Excel):** Apenas registros com os campos 'Surgery' e 'Procedure Date' são incluídos. Os registros são pareados entre as três abas usando identidade do paciente e data do procedimento. Nome e data são usados apenas para vinculação interna e **nunca como preditores**.

**Critérios de inclusão (CSV):** Todas as linhas com campo 'Surgery' válido são incluídas. Os nomes das colunas são mapeados automaticamente de snake_case (ex: `lvef_pre_pct`) para o formato interno.

**Desfecho primário:** Mortalidade hospitalar ou em 30 dias, extraída da coluna `Death`. Valores "Operative" ou numéricos ≤30 (dias até o óbito, incluindo 0 = óbito pós-operatório imediato) são codificados como eventos; ">30" ou "-" são codificados como sobreviventes.
""",
            )
        )
        _elig_info_guide = _cached_eligibility_info(xlsx_path)
        if _elig_info_guide.get("source_type") != "flat" and _elig_info_guide.get("pre_rows_before_criteria", 0) > 0:
            st.dataframe(_eligibility_summary(xlsx_path), width="stretch", column_config=general_table_column_config("eligibility"))
        else:
            st.caption(tr(
                f"Flat data source ({Path(xlsx_path).name}): eligibility flow not available.",
                f"Fonte de dados plana ({Path(xlsx_path).name}): fluxo de elegibilidade não disponível.",
            ))

    # ── 3. AI Risk ──
    with st.expander(tr("How AI Risk was built", "Como o AI Risk foi construído")):
        st.markdown(
            tr(
                f"""
**Predictor variables:** Only preoperative data (clinical, laboratory, echocardiographic). Postoperative complications are **never** used as predictors, preventing temporal leakage. Total: {prepared.info['n_features']} variables.

**Preprocessing:**
- Numeric variables: comma-decimal values (e.g. "64,7") are converted automatically; clinically impossible zeros (e.g. BSA=0.00 when height/weight are missing) are treated as missing. Median imputation + StandardScaler normalization
- Valve severity variables: OrdinalEncoder with clinically ordered categories (None < Trivial < Mild < Moderate < Severe) + median imputation + StandardScaler
- Other categorical variables: most-frequent imputation + TargetEncoder (smooth="auto") + median post-imputation for encoded values

**Calibration:** Tree-based models are calibrated via Platt scaling (sigmoid method). The calibrated probability is used directly as the clinical output, with only a minimal numerical-stability bound (1e-6).

**Candidate models:** {', '.join(artifacts.leaderboard['Modelo'].tolist())}

**Validation:** StratifiedGroupKFold with {AppConfig.CV_SPLITS} folds, grouped by patient key (`_patient_key`). This ensures:
- The same patient **never** appears in both training and test folds
- Class balance (mortality rate) is preserved across folds
- Out-of-fold (OOF) predictions — including calibration applied inside each fold — are used for all performance metrics

**Model selection:** The model with the highest AUC on calibrated OOF predictions is selected as the best model. Current best: **{artifacts.best_model_name}**.

**Leaderboard threshold:** Sensitivity and specificity in the leaderboard are computed at the optimal Youden's J threshold for each model. The clinical decision threshold (default 8%) used in the triple comparison tab is independent.
""",
                f"""
**Variáveis preditoras:** Apenas dados pré-operatórios (clínicos, laboratoriais, ecocardiográficos). Complicações pós-operatórias **nunca** são usadas como preditores, evitando vazamento temporal. Total: {prepared.info['n_features']} variáveis.

**Pré-processamento:**
- Variáveis numéricas: valores com vírgula decimal (ex: "64,7") são convertidos automaticamente; zeros clinicamente impossíveis (ex: BSA=0,00 quando altura/peso ausentes) são tratados como ausentes. Imputação pela mediana + normalização StandardScaler
- Variáveis de gravidade valvar: OrdinalEncoder com categorias clinicamente ordenadas (None < Trivial < Mild < Moderate < Severe) + imputação pela mediana + StandardScaler
- Demais variáveis categóricas: imputação pela moda + TargetEncoder (smooth="auto") + imputação pela mediana pós-codificação

**Calibração:** Modelos baseados em árvore são calibrados por Platt scaling (método sigmoid). A probabilidade calibrada é utilizada diretamente como saída clínica, com apenas um limite mínimo de estabilidade numérica (1e-6).

**Modelos candidatos:** {', '.join(artifacts.leaderboard['Modelo'].tolist())}

**Validação:** StratifiedGroupKFold com {AppConfig.CV_SPLITS} folds, agrupado por chave do paciente (`_patient_key`). Isso garante:
- O mesmo paciente **nunca** aparece em treino e teste simultaneamente
- O balanceamento de classes (taxa de mortalidade) é preservado entre os folds
- Predições out-of-fold (OOF) — incluindo calibração aplicada dentro de cada fold — são usadas para todas as métricas de desempenho

**Seleção do modelo:** O modelo com maior AUC nas predições OOF calibradas é selecionado como melhor. Atual: **{artifacts.best_model_name}**.

**Limiar do leaderboard:** Sensibilidade e especificidade no leaderboard usam o limiar ótimo de Youden (J) de cada modelo. O limiar clínico de decisão (padrão 8%) usado na aba de comparação tripla é independente.
""",
            )
        )

    # ── 3b. Imputation and input completeness ──
    with st.expander(tr("What is imputation and input completeness?", "O que é imputação e completude da entrada?")):
        st.markdown(
            tr(
                """
**What is imputation?**

In clinical datasets, some variables may not be available for every patient — for example, a lab test not performed or a field not filled in the form. When a predictive model requires those variables, the missing values must be replaced by reasonable estimates. This process is called **imputation**.

In AI Risk, missing numeric variables are replaced by the **median** of the training dataset, and missing categorical variables by the **mode** (most frequent value). This is a standard and conservative approach, widely used in clinical prediction models (TRIPOD statement, Collins et al., 2015).

**What is input completeness?**

The completeness indicator shows how much of the patient data was actually informed versus imputed. The classification considers both the number and the clinical importance of missing variables:

| Level | Meaning |
|:--|:--|
| **Complete** | All clinically important variables were informed; very few imputations |
| **Adequate** | No critical variables missing; minor imputations in secondary variables |
| **Partially imputed** | Some important variables were imputed — interpret the prediction with caution |
| **Heavily imputed** | Many variables or critical variables were imputed — low reliability of the estimate |

**Why does this matter?**

A prediction based on mostly imputed data is statistically valid but clinically less informative. The completeness indicator helps researchers and clinicians understand how much trust to place in any individual prediction. For a dissertation, this transparency is methodologically important.
""",
                """
**O que é imputação?**

Em bases de dados clínicos, algumas variáveis podem não estar disponíveis para todos os pacientes — por exemplo, um exame laboratorial não realizado ou um campo não preenchido no formulário. Quando um modelo preditivo necessita dessas variáveis, os valores ausentes precisam ser substituídos por estimativas razoáveis. Esse processo se chama **imputação**.

No AI Risk, variáveis numéricas ausentes são substituídas pela **mediana** do dataset de treinamento, e variáveis categóricas pela **moda** (valor mais frequente). Essa é uma abordagem padrão e conservadora, amplamente utilizada em modelos de predição clínica (declaração TRIPOD, Collins et al., 2015).

**O que é completude da entrada?**

O indicador de completude mostra quanto dos dados do paciente foi efetivamente informado versus imputado. A classificação considera tanto o número quanto a importância clínica das variáveis ausentes:

| Nível | Significado |
|:--|:--|
| **Completo** | Todas as variáveis clinicamente importantes foram informadas; pouquíssimas imputações |
| **Adequado** | Sem variáveis críticas ausentes; imputações menores em variáveis secundárias |
| **Parcialmente imputado** | Algumas variáveis importantes foram imputadas — interpretar a predição com cautela |
| **Muito imputado** | Muitas variáveis ou variáveis críticas foram imputadas — baixa confiabilidade da estimativa |

**Por que isso importa?**

Uma predição baseada em dados predominantemente imputados é estatisticamente válida, mas clinicamente menos informativa. O indicador de completude ajuda pesquisadores e clínicos a entenderem quanta confiança depositar em cada predição individual. Para uma dissertação, essa transparência é metodologicamente importante.
""",
            )
        )

    # ── 4. EuroSCORE II ──
    with st.expander(tr("How EuroSCORE II was operationalized", "Como o EuroSCORE II foi operacionalizado")):
        st.markdown(
            tr(
                """
EuroSCORE II is calculated using the **published logistic equation** with 18 risk factors and 27 coefficients (Nashef et al., *Eur J Cardiothorac Surg*, 2012).

**Variable mapping from the dataset:**

| EuroSCORE II variable | Source in dataset | Notes |
|:--|:--|:--|
| Age | `Age (years)` | Continuous, with non-linear transform |
| Sex | `Sex` | Female = 1 |
| Renal function | `Cr clearance, ml/min *`, `Dialysis` | Categories: normal, moderate, severe, dialysis |
| Extracardiac arteriopathy | `PVD` | PVD used as proxy |
| Poor mobility | Not available | Set to False for all patients (limitation) |
| Previous cardiac surgery | `Previous cardiac surgery` | Binary |
| Active endocarditis | `Active endocarditis` | "Possible" is treated as positive |
| Critical preoperative state | `Critical preoperative state` | Binary |
| Diabetes on insulin | `Diabetes`, `Insulin` | Both must be positive |
| NYHA class | `NYHA` | Classes I–IV |
| CCS class 4 angina | `CCS 4` | Binary |
| LVEF | `LVEF, %` | Categories: good (>50%), moderate (31–50%), poor (21–30%), very poor (≤20%) |
| Pulmonary hypertension | `PSAP` | Categories: no (≤30), moderate (31–55), severe (>55 mmHg) |
| Urgency | `Urgency` | Elective, urgent, emergency, salvage |
| Procedure weight | `Surgery` | Derived from procedure list: isolated CABG = 0, 1 major procedure = 1, 2+ = 2, 3+ = 3 |
| Thoracic aorta surgery | `Surgery` | Only for explicit aneurysm/dissection/root procedures |

**Formula:** logit(mortality) = β₀ + Σ(βᵢ × xᵢ), where β₀ = −5.324537 (constant).
""",
                """
O EuroSCORE II é calculado usando a **equação logística publicada** com 18 fatores de risco e 27 coeficientes (Nashef et al., *Eur J Cardiothorac Surg*, 2012).

**Mapeamento das variáveis a partir do dataset:**

| Variável EuroSCORE II | Fonte no dataset | Observações |
|:--|:--|:--|
| Idade | `Age (years)` | Contínua, com transformação não-linear |
| Sexo | `Sex` | Feminino = 1 |
| Função renal | `Cr clearance, ml/min *`, `Dialysis` | Categorias: normal, moderada, grave, diálise |
| Arteriopatia extracardíaca | `PVD` | PVD usado como proxy |
| Mobilidade reduzida | Não disponível | Definido como False para todos (limitação) |
| Cirurgia cardíaca prévia | `Previous cardiac surgery` | Binário |
| Endocardite ativa | `Active endocarditis` | "Possible" é tratado como positivo |
| Estado crítico pré-operatório | `Critical preoperative state` | Binário |
| Diabetes com insulina | `Diabetes`, `Insulin` | Ambos devem ser positivos |
| Classe NYHA | `NYHA` | Classes I–IV |
| Angina CCS classe 4 | `CCS 4` | Binário |
| FEVE | `LVEF, %` | Categorias: boa (>50%), moderada (31–50%), ruim (21–30%), muito ruim (≤20%) |
| Hipertensão pulmonar | `PSAP` | Categorias: não (≤30), moderada (31–55), grave (>55 mmHg) |
| Urgência | `Urgency` | Eletiva, urgente, emergência, salvamento |
| Peso do procedimento | `Surgery` | Derivado da lista: CRM isolada = 0, 1 procedimento maior = 1, 2+ = 2, 3+ = 3 |
| Cirurgia de aorta torácica | `Surgery` | Apenas para aneurisma/dissecção/raiz aórtica explícitos |

**Fórmula:** logit(mortalidade) = β₀ + Σ(βᵢ × xᵢ), onde β₀ = −5,324537 (constante).
""",
            )
        )

    # ── 5. STS Score ──
    with st.expander(tr("How STS Score was operationalized", "Como o STS Score foi operacionalizado")):
        st.markdown(
            tr(
                """
The STS Predicted Risk of Mortality is obtained via **automated interaction with the official STS Risk Calculator web application** (Society of Thoracic Surgeons) hosted at `acsdriskcalc.research.sts.org`. The STS does not publish a documented public API; this implementation automates the same web calculator that clinicians use manually, via its WebSocket interface.

**How it works:**
1. For each patient, the app maps preoperative variables to the STS input format
2. A WebSocket connection is established with the STS Risk Calculator's Shiny server interface
3. The patient data is sent and the server returns the predicted risk
4. For batch analysis, multiple patients are processed concurrently for speed

**Key variable mappings:**

| STS field | Source in dataset | Mapping |
|:--|:--|:--|
| Age | `Age (years)` | Direct |
| Gender | `Sex` | Male/Female |
| Height/Weight | `Height (cm)`, `Weight (kg)` | Direct (converted to cm/kg) |
| Diabetes | `Diabetes` | None / managed by diet / oral / insulin |
| Dialysis | `Dialysis` | Yes/No |
| Creatinine | `Creatinine, mg/dL` | Direct (mg/dL) |
| LVEF | `LVEF, %` | Numeric percentage |
| Procedure | `Surgery` | Mapped to STS procedure categories |
| Previous surgery | `Previous cardiac surgery` | Yes/No |
| Urgency | `Urgency` | Elective / Urgent / Emergency / Salvage |

**STS endpoints returned:** Predicted mortality, morbidity or mortality, stroke, renal failure, reoperation, prolonged ventilation, deep sternal infection, >14-day stay, >6-day stay.

**Limitations:**
- Requires internet connection (web calculator query)
- Some patients (~1–3%) may fail if procedure mapping is ambiguous
- The STS algorithm is proprietary — the coefficients are not publicly available
- Results may differ slightly from the web calculator due to field mapping approximations
""",
                """
O STS Predicted Risk of Mortality é obtido via **interação automatizada com a calculadora web oficial do STS Risk Calculator** (Society of Thoracic Surgeons), hospedada em `acsdriskcalc.research.sts.org`. O STS não publica uma API pública documentada; esta implementação automatiza a mesma calculadora web que os clínicos usam manualmente, via sua interface WebSocket.

**Como funciona:**
1. Para cada paciente, o app mapeia as variáveis pré-operatórias para o formato de entrada do STS
2. Uma conexão WebSocket é estabelecida com a interface Shiny do servidor do STS Risk Calculator
3. Os dados do paciente são enviados e o servidor retorna o risco predito
4. Para análise em lote, múltiplos pacientes são processados concorrentemente para maior velocidade

**Principais mapeamentos de variáveis:**

| Campo STS | Fonte no dataset | Mapeamento |
|:--|:--|:--|
| Age | `Age (years)` | Direto |
| Gender | `Sex` | Male/Female |
| Height/Weight | `Height (cm)`, `Weight (kg)` | Direto (convertido para cm/kg) |
| Diabetes | `Diabetes` | Nenhum / dieta / oral / insulina |
| Dialysis | `Dialysis` | Sim/Não |
| Creatinine | `Creatinine, mg/dL` | Direto (mg/dL) |
| LVEF | `LVEF, %` | Percentual numérico |
| Procedure | `Surgery` | Mapeado para categorias de procedimento STS |
| Previous surgery | `Previous cardiac surgery` | Sim/Não |
| Urgency | `Urgency` | Eletiva / Urgente / Emergência / Salvamento |

**Desfechos retornados pelo STS:** Mortalidade predita, morbimortalidade, AVC, insuficiência renal, reoperação, ventilação prolongada, infecção esternal profunda, internação >14 dias, internação >6 dias.

**Limitações:**
- Requer conexão com a internet (consulta à calculadora web)
- Alguns pacientes (~1–3%) podem falhar se o mapeamento do procedimento for ambíguo
- O algoritmo do STS é proprietário — os coeficientes não são publicamente disponíveis
- Os resultados podem diferir levemente da calculadora web devido a aproximações no mapeamento de campos
""",
            )
        )

    # ── 6. Performance metrics glossary ──
    with st.expander(tr("Performance metrics glossary", "Glossário de métricas de desempenho")):
        st.markdown(
            tr(
                """
**Discrimination** — Does the model rank patients correctly?

| Metric | What it measures | Good value |
|:--|:--|:--|
| **AUC-ROC** | Overall ability to distinguish events from non-events | > 0.70 acceptable, > 0.80 good |
| **AUPRC** | Discrimination focusing on correct identification of events (important when events are rare) | Higher is better; baseline = prevalence |
| **Sensitivity** | Proportion of actual events correctly identified as positive | Depends on threshold |
| **Specificity** | Proportion of actual non-events correctly identified as negative | Depends on threshold |
| **PPV** | Among those classified as positive, how many truly had the event | Depends on prevalence + threshold |
| **NPV** | Among those classified as negative, how many truly did not have the event | Depends on prevalence + threshold |

**Calibration** — Are the predicted probabilities accurate?

| Metric | What it measures | Good value |
|:--|:--|:--|
| **Brier score** | Average squared difference between prediction and outcome | Lower is better; 0 = perfect |
| **Calibration-in-the-large** | Average bias (over- or underprediction) | Close to 0 |
| **Calibration slope** | Whether predictions are too extreme or too compressed | Close to 1 |
| **Hosmer-Lemeshow** | Goodness of fit across risk groups (complementary, not definitive) | p > 0.05 suggests acceptable fit |

**Comparison** — Are differences between scores statistically significant?

| Metric | What it measures |
|:--|:--|
| **Delta AUC (bootstrap)** | Difference in AUC with 95% CI and p-value from 2,000 bootstrap resamples |
| **DeLong test** | Formal test for correlated ROC curves on the same patients |
| **NRI** | Net reclassification improvement across risk categories (low <5%, intermediate 5–15%, high >15%) |
| **IDI** | Integrated discrimination improvement (average separation between events and non-events) |

**Clinical utility** — Is the model useful for clinical decisions?

| Metric | What it measures |
|:--|:--|
| **DCA (net benefit)** | Net clinical benefit compared to treating all or treating none, across thresholds 5–20% |
""",
                """
**Discriminação** — O modelo ordena os pacientes corretamente?

| Métrica | O que mede | Valor bom |
|:--|:--|:--|
| **AUC-ROC** | Capacidade geral de distinguir eventos de não-eventos | > 0,70 aceitável, > 0,80 bom |
| **AUPRC** | Discriminação focada na identificação correta de eventos (importante quando eventos são raros) | Quanto maior, melhor; baseline = prevalência |
| **Sensibilidade** | Proporção de eventos reais corretamente identificados como positivos | Depende do limiar |
| **Especificidade** | Proporção de não-eventos reais corretamente identificados como negativos | Depende do limiar |
| **PPV** | Entre os classificados como positivos, quantos realmente tiveram o evento | Depende da prevalência + limiar |
| **NPV** | Entre os classificados como negativos, quantos realmente não tiveram o evento | Depende da prevalência + limiar |

**Calibração** — As probabilidades preditas são precisas?

| Métrica | O que mede | Valor bom |
|:--|:--|:--|
| **Brier score** | Diferença quadrática média entre predição e desfecho | Quanto menor, melhor; 0 = perfeito |
| **Calibration-in-the-large** | Viés médio (super- ou subestimação) | Próximo de 0 |
| **Slope de calibração** | Se as predições são muito extremas ou muito comprimidas | Próximo de 1 |
| **Hosmer-Lemeshow** | Qualidade do ajuste por grupos de risco (complementar, não definitivo) | p > 0,05 sugere ajuste aceitável |

**Comparação** — As diferenças entre escores são estatisticamente significativas?

| Métrica | O que mede |
|:--|:--|
| **Delta AUC (bootstrap)** | Diferença de AUC com IC95% e valor-p de 2.000 reamostras bootstrap |
| **Teste de DeLong** | Teste formal para curvas ROC correlacionadas nos mesmos pacientes |
| **NRI** | Melhora líquida de reclassificação entre categorias de risco (baixo <5%, intermediário 5–15%, alto >15%) |
| **IDI** | Melhora integrada de discriminação (separação média entre eventos e não-eventos) |

**Utilidade clínica** — O modelo é útil para decisões clínicas?

| Métrica | O que mede |
|:--|:--|
| **DCA (benefício líquido)** | Benefício clínico líquido comparado a tratar todos ou tratar ninguém, nos limiares de 5–20% |
""",
            )
        )
        if triple_n is not None:
            st.info(tr(f"Current triple-comparison sample: n={triple_n}", f"Amostra atual da comparação tripla: n={triple_n}"))

    # ── 7. Interpretability ──
    with st.expander(tr("Interpretability and explainability", "Interpretabilidade e explicabilidade")):
        st.markdown(
            tr(
                """
The app provides multiple layers of interpretability:

| Method | Scope | Tab | What it shows |
|:--|:--|:--|:--|
| **Permutation importance** | Global | Models | How much model performance drops when each variable is randomly shuffled |
| **SHAP beeswarm** | Global | Models | Direction and magnitude of each variable's impact across all patients |
| **SHAP dependence** | Global | Models | How a specific variable's value relates to its SHAP impact |
| **SHAP local** | Individual | Prediction | Which variables pushed this specific patient's risk up or down |
| **Logistic regression coefficients** | Global | Models | Clinically interpretable weights (positive = higher risk, negative = lower risk) |
| **EuroSCORE II coefficients** | Global | Models | Official published coefficients for reference |

**Important:** Interpretability tools show **associations**, not causal relationships. A variable with high importance may be a proxy for an underlying risk factor.
""",
                """
O app oferece múltiplas camadas de interpretabilidade:

| Método | Escopo | Aba | O que mostra |
|:--|:--|:--|:--|
| **Importância por permutação** | Global | Modelos | Quanto o desempenho do modelo cai quando cada variável é embaralhada |
| **SHAP beeswarm** | Global | Modelos | Direção e magnitude do impacto de cada variável em todos os pacientes |
| **SHAP dependência** | Global | Modelos | Como o valor de uma variável específica se relaciona com seu impacto SHAP |
| **SHAP local** | Individual | Predição | Quais variáveis empurraram o risco deste paciente específico para cima ou para baixo |
| **Coeficientes da regressão logística** | Global | Modelos | Pesos clinicamente interpretáveis (positivo = maior risco, negativo = menor risco) |
| **Coeficientes do EuroSCORE II** | Global | Modelos | Coeficientes oficiais publicados para referência |

**Importante:** As ferramentas de interpretabilidade mostram **associações**, e não relações causais. Uma variável com alta importância pode ser um proxy de um fator de risco subjacente.
""",
            )
        )

    # ── 8. Limitations ──
    with st.expander(tr("Limitations and methodological notes", "Limitações e notas metodológicas")):
        st.markdown(
            tr(
                """
- **Single-center data:** AI Risk is trained on local data and may not generalize to other populations without external validation.
- **Internal validation only:** OOF cross-validation reduces overfitting but does not replace validation on an independent cohort.
- **EuroSCORE II approximations:** Some variables (poor mobility, critical preoperative state) may be approximated from available fields rather than captured exactly as in the original form.
- **STS web calculator dependency:** STS calculation requires internet access and depends on the availability of the STS web calculator at acsdriskcalc.research.sts.org. The interface may change without notice. ~1–3% of patients may fail due to procedure mapping ambiguity.
- **Small subgroups:** Results in subgroups with <50 patients or <10 events should be interpreted with caution — confidence intervals may be wide.
- **Calibration slope <1 for OOF predictions:** Can occur with cross-validated predictions, especially in small samples. Tree-based models are calibrated via Platt scaling inside each CV fold, so the OOF calibration metrics reflect the same calibration strategy used in the final model. The final model (used for individual predictions) is refitted on all data and may show slightly different calibration.
- **Missing data and imputation:** Missing variables are replaced by the training dataset median (numeric) or mode (categorical). The input completeness indicator classifies each prediction as complete, adequate, partially imputed, or heavily imputed — considering both the number and clinical relevance of missing variables. Predictions with heavily imputed data should be interpreted with greater caution.
- **TRIPOD/PROBAST:** Methodological transparency follows TRIPOD/TRIPOD-AI principles. Risk of bias should be assessed across PROBAST domains (participants, predictors, outcome, analysis).
""",
                """
- **Dados de centro único:** O AI Risk é treinado em dados locais e pode não generalizar para outras populações sem validação externa.
- **Validação interna apenas:** A validação cruzada OOF reduz o sobreajuste, mas não substitui a validação em uma coorte independente.
- **Aproximações do EuroSCORE II:** Algumas variáveis (mobilidade reduzida, estado crítico pré-operatório) podem ser aproximadas a partir dos campos disponíveis, e não capturadas exatamente como no formulário original.
- **Dependência da calculadora web do STS:** O cálculo do STS requer acesso à internet e depende da disponibilidade da calculadora web do STS em acsdriskcalc.research.sts.org. A interface pode mudar sem aviso. ~1–3% dos pacientes podem falhar por ambiguidade no mapeamento de procedimentos.
- **Subgrupos pequenos:** Resultados em subgrupos com <50 pacientes ou <10 eventos devem ser interpretados com cautela — os intervalos de confiança podem ser amplos.
- **Slope de calibração <1 nas predições OOF:** Pode ocorrer em predições de validação cruzada, especialmente em amostras pequenas. Modelos baseados em árvore são calibrados via Platt scaling dentro de cada fold do CV, portanto as métricas de calibração OOF refletem a mesma estratégia de calibração do modelo final. O modelo final (usado para predições individuais) é reajustado em todos os dados e pode ter calibração ligeiramente diferente.
- **Dados faltantes e imputação:** Variáveis ausentes são substituídas pela mediana (numéricas) ou moda (categóricas) do dataset de treinamento. O indicador de completude classifica cada predição como completa, adequada, parcialmente imputada ou muito imputada — considerando tanto o número quanto a relevância clínica das variáveis ausentes. Predições com dados muito imputados devem ser interpretadas com maior cautela.
- **TRIPOD/PROBAST:** A transparência metodológica segue princípios do TRIPOD/TRIPOD-AI. O risco de viés deve ser avaliado nos domínios do PROBAST (participantes, preditores, desfecho, análise).
""",
            )
        )

    # ── 9. Methods text ──
    st.divider()
    st.markdown(tr("**Statistical Methods for Manuscript**", "**Métodos Estatísticos para Manuscrito**"))
    methods_mode = st.radio(
        tr("Methods text format", "Formato do texto de métodos"),
        [tr("Short", "Curto"), tr("Detailed", "Detalhado")],
        horizontal=True,
        key="methods_mode_analysis_guide",
    )
    st.text_area(
        tr("Methods for manuscript", "Texto para artigo - Métodos"),
        value=build_methods_text(methods_mode),
        height=240,
    )
    _txt_download_btn(build_methods_text(methods_mode), "methods_for_manuscript.txt", tr("Download Methods text (.txt)", "Baixar texto de Métodos (.txt)"))


def _serialize_bundle(bundle: Dict[str, object]) -> Dict[str, object]:
    """Convert dataclasses to plain dicts for pickle compatibility.

    Streamlit may reload modules between runs, creating new class objects.
    Pickle requires the exact same class reference, so we store plain dicts.
    """
    out = dict(bundle)
    prepared = out["prepared"]
    out["prepared"] = {
        "data": prepared.data,
        "feature_columns": prepared.feature_columns,
        "info": prepared.info,
    }
    artifacts = out["artifacts"]
    out["artifacts"] = {
        "model": artifacts.model,
        "leaderboard": artifacts.leaderboard,
        "oof_predictions": artifacts.oof_predictions,       # calibrated OOF (primary)
        "oof_raw": getattr(artifacts, "oof_raw", None),     # uncalibrated, audit only
        "feature_columns": artifacts.feature_columns,
        "fitted_models": artifacts.fitted_models,
        "best_model_name": artifacts.best_model_name,
        "calibration_method": getattr(artifacts, "calibration_method", "sigmoid"),
    }
    return out


def _deserialize_bundle(bundle: Dict[str, object]) -> Dict[str, object]:
    """Reconstruct dataclasses from plain dicts."""
    import importlib
    import modeling as _mod
    importlib.reload(_mod)
    TrainedArtifacts = _mod.TrainedArtifacts
    out = dict(bundle)
    p = out["prepared"]
    if isinstance(p, dict):
        out["prepared"] = PreparedData(
            data=p["data"],
            feature_columns=p["feature_columns"],
            info=p["info"],
        )
    a = out["artifacts"]
    if isinstance(a, dict):
        out["artifacts"] = TrainedArtifacts(
            model=a["model"],
            leaderboard=a["leaderboard"],
            oof_predictions=a["oof_predictions"],           # calibrated OOF
            feature_columns=a["feature_columns"],
            fitted_models=a["fitted_models"],
            best_model_name=a["best_model_name"],
            calibration_method=a.get("calibration_method", "sigmoid"),
            oof_raw=a.get("oof_raw"),                       # may be None in old caches
        )
    return out


def load_train_bundle(xlsx_path: str, force_retrain: bool = False, progress_callback=None) -> tuple[Dict[str, object], str, dict]:
    sig = _bundle_signature(xlsx_path)

    if not force_retrain and MODEL_CACHE_FILE.exists():
        try:
            payload = joblib.load(MODEL_CACHE_FILE)
            if payload.get("signature") == sig:
                bundle_info = {
                    "saved_at": payload.get("saved_at", "Unknown"),
                    "training_source": payload.get("training_source", Path(xlsx_path).name),
                }
                return _deserialize_bundle(payload["bundle"]), "Cache local", bundle_info
        except Exception:
            pass

    from datetime import datetime, timezone
    bundle = _compute_bundle(xlsx_path, progress_callback=progress_callback)
    saved_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "signature": sig,
        "bundle": _serialize_bundle(bundle),
        "saved_at": saved_at,
        "training_source": Path(xlsx_path).name,
    }
    joblib.dump(payload, MODEL_CACHE_FILE)
    return bundle, "Recalculado", {"saved_at": saved_at, "training_source": Path(xlsx_path).name}


@st.cache_resource(show_spinner=False)
def load_cached_bundle_only(xlsx_path: str, _model_version: str = MODEL_VERSION) -> tuple[Dict[str, object] | None, str, dict]:
    sig = _bundle_signature(xlsx_path)
    empty_info = {"saved_at": "Unknown", "training_source": "Unknown"}
    if not MODEL_CACHE_FILE.exists():
        return None, "Sem treino salvo", empty_info

    try:
        payload = joblib.load(MODEL_CACHE_FILE)
    except Exception:
        return None, "Cache inválido", empty_info

    cache_sig = payload.get("signature", {})
    if cache_sig != sig:
        return None, "Treino desatualizado para o arquivo atual", empty_info

    raw = payload.get("bundle")
    if raw is None:
        return None, "Cache corrompido", empty_info
    bundle_info = {
        "saved_at": payload.get("saved_at", "Unknown"),
        "training_source": payload.get("training_source", Path(xlsx_path).name),
    }
    return _deserialize_bundle(raw), "Cache local", bundle_info


def model_weight_table(artifacts, prepared, model_name: str, top_n: int = 20) -> tuple[pd.DataFrame, str]:
    pipe = artifacts.fitted_models[model_name]
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    if model.__class__.__name__ == "StackingClassifier" and hasattr(model, "final_estimator_"):
        final = model.final_estimator_
        if hasattr(final, "coef_") and hasattr(model, "estimators"):
            names = [n for n, _ in model.estimators]
            vals = final.coef_.ravel()
            w = pd.DataFrame({"Model": names[: len(vals)], "Weight": vals[: len(names)]})
            w["Absolute impact"] = w["Weight"].abs()
            return w.sort_values("Absolute impact", ascending=False).head(top_n), "stacking"
        return pd.DataFrame(), "opaque"

    feature_names = prep.get_feature_names_out()
    feature_names = [
        str(n).replace("num__", "").replace("cat__onehot__", "").replace("cat__target_enc__", "") for n in feature_names
    ]

    if hasattr(model, "coef_"):
        vals = np.asarray(model.coef_).ravel()
        n = min(len(vals), len(feature_names))
        w = pd.DataFrame({"Variable": feature_names[:n], "Coefficient": vals[:n]})
        w["Absolute impact"] = w["Coefficient"].abs()
        return w.sort_values("Absolute impact", ascending=False).head(top_n), "coefficient"

    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_).ravel()
        n = min(len(vals), len(feature_names))
        w = pd.DataFrame({"Variable": feature_names[:n], "Importance": vals[:n]})
        w["Absolute impact"] = w["Importance"].abs()
        return w.sort_values("Absolute impact", ascending=False).head(top_n), "importance"

    return pd.DataFrame(), "opaque"


def _feature_group(base_feature: str) -> str:
    clinical = {
        "Age (years)", "Sex", "Preoperative NYHA", "CCS4", "Diabetes", "PVD", "Previous surgery",
        "Dialysis", "IE", "HF", "Hypertension", "Dyslipidemia", "CVA", "Cancer ≤ 5 yrs",
        "Arrhythmia Remote", "Arrhythmia Recent", "Family Hx of CAD", "Smoking (Pack-year)",
        "Ex-Smoker (Pack-year)", "Alcohol", "Pneumonia", "Chronic Lung Disease", "Poor mobility",
        "Critical preoperative state", "Coronary Symptom", "Left Main Stenosis ≥ 50%",
        "Proximal LAD Stenosis ≥ 70%", "No. of Diseased Vessels",
    }
    lab = {
        "Weight (kg)", "Height (cm)", "Cr clearance, ml/min *", "Creatinine (mg/dL)", "Hematocrit (%)",
        "WBC Count (10³/μL)", "Platelet Count (cells/μL)", "INR", "PTT", "KDIGO †",
    }
    echo = {
        "Pré-LVEF, %", "LVEF, %", "PSAP", "TAPSE", "Aortic Stenosis", "Aortic Regurgitation",
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


def _resolve_base_feature(encoded_feature: str, feature_columns: list[str]) -> str:
    if encoded_feature in feature_columns:
        return encoded_feature
    for feat in sorted(feature_columns, key=len, reverse=True):
        if encoded_feature.startswith(feat + "_"):
            return feat
    return encoded_feature


@st.cache_data(show_spinner=False)
def cached_permutation_importance_table(
    xlsx_path: str,
    model_name: str,
    top_n: int,
    show_all: bool,
    _model,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_columns: list[str],
) -> pd.DataFrame:
    result = permutation_importance(_model, X, y, n_repeats=10, random_state=42, scoring="roc_auc")
    imp = pd.DataFrame(
        {
            tr("Variable", "Variável"): feature_columns,
            tr("Importance", "Importância"): result.importances_mean,
        }
    )
    imp[tr("Group", "Grupo")] = imp[tr("Variable", "Variável")].map(_feature_group)
    imp[tr("Ranking", "Ranking")] = imp[tr("Importance", "Importância")].rank(ascending=False, method="dense").astype(int)
    imp = imp.sort_values(tr("Importance", "Importância"), ascending=False)
    return imp if show_all else imp.head(top_n)


@st.cache_data(show_spinner=False)
def _cached_shap_global(
    xlsx_path: str,
    model_name: str,
    top_n: int,
    _pipe,
    _X: pd.DataFrame,
) -> pd.DataFrame:
    """Compute global SHAP importance using TreeExplainer on preprocessed features.

    Uses the pipeline's preprocessing step to transform X, then applies
    TreeExplainer on the fitted estimator directly. Only supported for
    tree-based models (RandomForest, XGBoost, LightGBM, CatBoost).
    Returns an empty DataFrame for unsupported model types.
    """
    try:
        import shap as _shap
    except ImportError:
        return pd.DataFrame()

    estimator = _pipe.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame()

    prep = _pipe.named_steps["prep"]
    X_proc = prep.transform(_X)
    feat_names = [
        str(n).replace("num__", "").replace("cat__onehot__", "").replace("cat__target_enc__", "").replace("cat__target_enc__", "")
        for n in prep.get_feature_names_out()
    ]

    explainer = _shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_proc)
    # Handle multi-class output: list of arrays or 3D array (samples x features x classes)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_dir = shap_values.mean(axis=0)
    df_imp = pd.DataFrame({
        "Feature": feat_names,
        "Mean |SHAP|": mean_abs,
        "Mean SHAP": mean_dir,
    }).sort_values("Mean |SHAP|", ascending=False).head(top_n).reset_index(drop=True)
    return df_imp


@st.cache_data(show_spinner=False)
def _cached_shap_beeswarm(
    xlsx_path: str,
    model_name: str,
    top_n: int,
    _pipe,
    _X: pd.DataFrame,
):
    """Generate SHAP beeswarm plot (summary_plot) for tree-based models."""
    try:
        import shap as _shap
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    estimator = _pipe.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return None

    prep = _pipe.named_steps["prep"]
    X_proc = prep.transform(_X)
    feat_names = [
        str(n).replace("num__", "").replace("cat__onehot__", "").replace("cat__target_enc__", "")
        for n in prep.get_feature_names_out()
    ]

    explainer = _shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_proc)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    fig = plt.figure(figsize=(10, 8))
    _shap.summary_plot(
        shap_values,
        X_proc,
        feature_names=feat_names,
        plot_type="dot",
        show=False,
        max_display=top_n,
    )
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def _cached_shap_dependence(
    xlsx_path: str,
    model_name: str,
    feature_name: str,
    _pipe,
    _X: pd.DataFrame,
):
    """Generate SHAP dependence plot for a specific feature."""
    try:
        import shap as _shap
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    estimator = _pipe.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return None

    prep = _pipe.named_steps["prep"]
    X_proc = prep.transform(_X)
    feat_names = [
        str(n).replace("num__", "").replace("cat__onehot__", "").replace("cat__target_enc__", "")
        for n in prep.get_feature_names_out()
    ]

    explainer = _shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_proc)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    if feature_name not in feat_names:
        return None
    feature_idx = feat_names.index(feature_name)

    fig = plt.figure(figsize=(10, 6))
    _shap.dependence_plot(
        feature_idx,
        shap_values,
        X_proc,
        feature_names=feat_names,
        show=False,
    )
    plt.tight_layout()
    return fig


def logistic_clinical_coefficients_table(artifacts, prepared, top_n: int = 20, show_all: bool = False) -> pd.DataFrame:
    if "LogisticRegression" not in artifacts.fitted_models:
        return pd.DataFrame()
    pipe = artifacts.fitted_models["LogisticRegression"]
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame()

    raw_features = artifacts.feature_columns
    feature_names = [str(n).replace("num__", "").replace("cat__onehot__", "").replace("cat__target_enc__", "") for n in prep.get_feature_names_out()]
    vals = np.asarray(model.coef_).ravel()
    rows = []
    for feat, coef in zip(feature_names, vals):
        base = _resolve_base_feature(feat, raw_features)
        rows.append(
            {
                "encoded_feature": feat,
                tr("Variable", "Variável"): base,
                tr("Coefficient", "Coeficiente"): float(coef),
                tr("Absolute impact", "Impacto absoluto"): abs(float(coef)),
            }
        )
    df_coef = pd.DataFrame(rows)
    if df_coef.empty:
        return df_coef
    idx = df_coef.groupby(tr("Variable", "Variável"))[tr("Absolute impact", "Impacto absoluto")].idxmax()
    out = df_coef.loc[idx].copy()
    out[tr("Direction", "Direção")] = np.where(
        out[tr("Coefficient", "Coeficiente")] >= 0,
        tr("Higher risk tendency", "Tendência a maior risco"),
        tr("Lower risk tendency", "Tendência a menor risco"),
    )
    out[tr("Group", "Grupo")] = out[tr("Variable", "Variável")].map(_feature_group)
    out[tr("Ranking", "Ranking")] = out[tr("Absolute impact", "Impacto absoluto")].rank(ascending=False, method="dense").astype(int)
    out = out.sort_values(tr("Absolute impact", "Impacto absoluto"), ascending=False)
    cols = [
        tr("Variable", "Variável"),
        tr("Group", "Grupo"),
        tr("Coefficient", "Coeficiente"),
        tr("Direction", "Direção"),
        tr("Absolute impact", "Impacto absoluto"),
        tr("Ranking", "Ranking"),
    ]
    out = out[cols]
    return out if show_all else out.head(top_n)


def _patient_factor_label(base_feature: str, form_map: Dict[str, object]) -> str:
    val = form_map.get(base_feature)
    if base_feature == "Age (years)":
        age = parse_number(val)
        if pd.notna(age):
            if float(age) >= 75:
                return tr("Very advanced age", "Idade muito avançada")
            if float(age) >= 65:
                return tr("Advanced age", "Idade avançada")
            return tr("Age below 65 years", "Idade abaixo de 65 anos")
    if base_feature in {"Pré-LVEF, %", "LVEF, %"}:
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


def explain_patient_risk(artifacts, input_features: pd.DataFrame, form_map: Dict[str, object], top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "LogisticRegression" not in artifacts.fitted_models:
        return pd.DataFrame(), pd.DataFrame()
    pipe = artifacts.fitted_models["LogisticRegression"]
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    transformed = prep.transform(input_features)
    arr = np.asarray(transformed)
    if arr.ndim == 2:
        arr = arr[0]
    feature_names = [str(n).replace("num__", "").replace("cat__onehot__", "").replace("cat__target_enc__", "") for n in prep.get_feature_names_out()]
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
    expl[tr("Factor", "Fator")] = expl["base"].map(lambda x: _patient_factor_label(x, form_map))
    expl[tr("Group", "Grupo")] = expl["base"].map(_feature_group)
    pos = expl[expl[tr("Contribution", "Contribuição")] > 0].sort_values(tr("Contribution", "Contribuição"), ascending=False)
    neg = expl[expl[tr("Contribution", "Contribuição")] < 0].sort_values(tr("Contribution", "Contribuição"), ascending=True)
    pos = pos[[tr("Factor", "Fator"), tr("Group", "Grupo"), tr("Contribution", "Contribuição")]].head(top_n)
    neg = neg[[tr("Factor", "Fator"), tr("Group", "Grupo"), tr("Contribution", "Contribuição")]].head(top_n)
    return pos, neg


def explain_table_column_config() -> dict:
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


def data_quality_alerts(form_map: Dict[str, object], prepared) -> list[tuple[str, str]]:
    alerts: list[tuple[str, str]] = []
    df_ref = prepared.data
    age = parse_number(form_map.get("Age (years)"))
    weight = parse_number(form_map.get("Weight (kg)"))
    height = parse_number(form_map.get("Height (cm)"))
    creatinine = parse_number(form_map.get("Creatinine (mg/dL)"))
    lvef = parse_number(form_map.get("LVEF, %"))
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


def prediction_uncertainty(patient_pred_df: pd.DataFrame, prob_col: str, imputed_features: int) -> tuple[str, str]:
    vals = patient_pred_df[prob_col].str.replace("%", "", regex=False).astype(float) / 100.0
    low = float(vals.quantile(0.25))
    high = float(vals.quantile(0.75))
    spread = high - low
    # Thresholds adjusted: the individual form always has ~15-20 imputed features
    # (detailed echo values, some lab values not in form), so confidence is based
    # primarily on model agreement (spread), not imputation count.
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


def model_table_column_config(kind: str) -> dict:
    if kind == "coefficient":
        return {
            "Variable": st.column_config.TextColumn(
                tr("Variable", "Variável"),
                help=tr(
                    "Clinical or encoded feature used by the model.",
                    "Variável clínica ou atributo codificado usado pelo modelo.",
                ),
            ),
            "Coefficient": st.column_config.NumberColumn(
                tr("Coefficient", "Coeficiente"),
                help=tr(
                    "Logistic coefficient after preprocessing. Positive values suggest higher predicted risk, negative values suggest lower predicted risk.",
                    "Coeficiente logístico após o pré-processamento. Valores positivos sugerem maior risco previsto e valores negativos sugerem menor risco previsto.",
                ),
                format="%.4f",
            ),
            "Absolute impact": st.column_config.NumberColumn(
                tr("Absolute impact", "Impacto absoluto"),
                help=tr(
                    "Absolute magnitude of the coefficient, useful for ranking the strongest effects.",
                    "Magnitude absoluta do coeficiente, útil para classificar os efeitos mais fortes.",
                ),
                format="%.4f",
            ),
        }
    if kind == "importance":
        return {
            tr("Variable", "Variável"): st.column_config.TextColumn(
                tr("Variable", "Variável"),
                help=tr(
                    "Clinical feature evaluated in the final selected model.",
                    "Variável clínica avaliada no modelo final selecionado.",
                ),
            ),
            tr("Group", "Grupo"): st.column_config.TextColumn(
                tr("Group", "Grupo"),
                help=tr(
                    "Clinical domain of the variable: clinical, laboratory, echocardiographic, or procedure-related.",
                    "Domínio clínico da variável: clínico, laboratorial, ecocardiográfico ou relacionado ao procedimento.",
                ),
            ),
            tr("Importance", "Importância"): st.column_config.NumberColumn(
                tr("Importance", "Importância"),
                help=tr(
                    "Permutation importance: estimated drop in model performance when the variable is randomly shuffled. Higher values mean greater relevance to the final model.",
                    "Importância por permutação: queda estimada no desempenho do modelo quando a variável é embaralhada aleatoriamente. Valores maiores significam maior relevância no modelo final.",
                ),
                format="%.5f",
            ),
            tr("Ranking", "Ranking"): st.column_config.NumberColumn(
                tr("Ranking", "Ranking"),
                help=tr(
                    "Position of the variable after ordering from most to least important.",
                    "Posição da variável após ordenar da mais importante para a menos importante.",
                ),
                format="%d",
            ),
        }
    if kind == "logistic_clinical":
        return {
            tr("Variable", "Variável"): st.column_config.TextColumn(
                tr("Variable", "Variável"),
                help=tr(
                    "Clinical variable represented in the logistic regression model.",
                    "Variável clínica representada no modelo de regressão logística.",
                ),
            ),
            tr("Group", "Grupo"): st.column_config.TextColumn(
                tr("Group", "Grupo"),
                help=tr(
                    "Clinical domain of the variable.",
                    "Domínio clínico da variável.",
                ),
            ),
            tr("Coefficient", "Coeficiente"): st.column_config.NumberColumn(
                tr("Coefficient", "Coeficiente"),
                help=tr(
                    "Representative logistic coefficient. Positive coefficients suggest higher predicted risk and negative coefficients suggest lower predicted risk.",
                    "Coeficiente logístico representativo. Valores positivos sugerem maior risco previsto e negativos sugerem menor risco previsto.",
                ),
                format="%.4f",
            ),
            tr("Direction", "Direção"): st.column_config.TextColumn(
                tr("Direction", "Direção"),
                help=tr(
                    "Clinical reading of the coefficient sign.",
                    "Leitura clínica do sinal do coeficiente.",
                ),
            ),
            tr("Absolute impact", "Impacto absoluto"): st.column_config.NumberColumn(
                tr("Absolute impact", "Impacto absoluto"),
                help=tr(
                    "Absolute magnitude of the coefficient, used to rank stronger effects regardless of sign.",
                    "Magnitude absoluta do coeficiente, usada para ranquear efeitos mais fortes independentemente do sinal.",
                ),
                format="%.4f",
            ),
            tr("Ranking", "Ranking"): st.column_config.NumberColumn(
                tr("Ranking", "Ranking"),
                help=tr(
                    "Order from strongest to weakest absolute effect.",
                    "Ordem do efeito absoluto mais forte para o mais fraco.",
                ),
                format="%d",
            ),
        }
    if kind == "stacking":
        return {
            "Model": st.column_config.TextColumn(
                tr("Model", "Modelo"),
                help=tr(
                    "Base model used inside the stacking ensemble.",
                    "Modelo base usado dentro do ensemble por stacking.",
                ),
            ),
            "Weight": st.column_config.NumberColumn(
                tr("Weight", "Peso"),
                help=tr(
                    "Weight assigned by the stacking meta-model to the prediction of each base model. This is not a direct clinical variable weight.",
                    "Peso atribuído pelo meta-modelo do stacking à predição de cada modelo base. Não é um peso direto de variável clínica.",
                ),
                format="%.4f",
            ),
            "Absolute impact": st.column_config.NumberColumn(
                tr("Absolute impact", "Impacto absoluto"),
                help=tr(
                    "Absolute magnitude of the base-model weight, useful for ranking which base model contributes most to the ensemble.",
                    "Magnitude absoluta do peso do modelo base, útil para ranquear qual modelo contribui mais para o ensemble.",
                ),
                format="%.4f",
            ),
        }
    if kind == "euroscore":
        return {
            tr("Factor", "Fator"): st.column_config.TextColumn(
                tr("Factor", "Fator"),
                help=tr(
                    "Variable included in the official EuroSCORE II formula.",
                    "Variável incluída na fórmula oficial do EuroSCORE II.",
                ),
            ),
            tr("Coefficient", "Coeficiente"): st.column_config.NumberColumn(
                tr("Coefficient", "Coeficiente"),
                help=tr(
                    "Published EuroSCORE II logistic coefficient. Higher positive coefficients generally indicate stronger contribution to risk.",
                    "Coeficiente logístico publicado do EuroSCORE II. Coeficientes positivos maiores geralmente indicam contribuição mais forte para o risco.",
                ),
                format="%.6f",
            ),
        }
    return {}


def stats_table_column_config(kind: str) -> dict:
    common = {
        "Score": st.column_config.TextColumn(
            tr("Score", "Escore"),
            help=tr(
                "Risk score or prediction model being evaluated.",
                "Escore de risco ou modelo preditivo em avaliação.",
            ),
        ),
        "n": st.column_config.NumberColumn(
            "n",
            help=tr(
                "Number of observations included in that analysis.",
                "Número de observações incluídas naquela análise.",
            ),
            format="%d",
        ),
        "AUC": st.column_config.NumberColumn(
            "AUC",
            help=tr(
                "Area under the ROC curve. Higher values indicate better overall discrimination.",
                "Área sob a curva ROC. Valores maiores indicam melhor discriminação global.",
            ),
            format="%.3f",
        ),
        "AUPRC": st.column_config.NumberColumn(
            "AUPRC",
            help=tr(
                "Area under the precision-recall curve. Especially useful when the event is relatively uncommon.",
                "Área sob a curva precisão-revocação. Especialmente útil quando o evento é relativamente incomum.",
            ),
            format="%.3f",
        ),
        "Brier": st.column_config.NumberColumn(
            "Brier",
            help=tr(
                "Brier score measures probabilistic accuracy. Lower values are better.",
                "O Brier score mede a acurácia probabilística. Valores menores são melhores.",
            ),
            format="%.4f",
        ),
        "Sensitivity": st.column_config.NumberColumn(
            tr("Sensitivity", "Sensibilidade"),
            help=tr(
                "Proportion of patients with the event correctly classified as positive at the selected threshold.",
                "Proporção de pacientes com evento corretamente classificados como positivos no limiar selecionado.",
            ),
            format="%.3f",
        ),
        "Specificity": st.column_config.NumberColumn(
            tr("Specificity", "Especificidade"),
            help=tr(
                "Proportion of patients without the event correctly classified as negative at the selected threshold.",
                "Proporção de pacientes sem evento corretamente classificados como negativos no limiar selecionado.",
            ),
            format="%.3f",
        ),
        "PPV": st.column_config.TextColumn(
            "PPV",
            help=tr(
                "Positive predictive value: probability that a patient classified as positive truly has the event. '—' means no positive predictions at this threshold.",
                "Valor preditivo positivo: probabilidade de um paciente classificado como positivo realmente apresentar o evento. '—' indica que não houve predições positivas neste limiar.",
            ),
        ),
        "NPV": st.column_config.TextColumn(
            "NPV",
            help=tr(
                "Negative predictive value: probability that a patient classified as negative truly does not have the event. '—' means no negative predictions at this threshold.",
                "Valor preditivo negativo: probabilidade de um paciente classificado como negativo realmente não apresentar o evento. '—' indica que não houve predições negativas neste limiar.",
            ),
        ),
        "AUC_IC95_inf": st.column_config.NumberColumn(tr("AUC CI low", "AUC IC95% inf"), format="%.3f"),
        "AUC_IC95_sup": st.column_config.NumberColumn(tr("AUC CI high", "AUC IC95% sup"), format="%.3f"),
        "AUPRC_IC95_inf": st.column_config.NumberColumn(tr("AUPRC CI low", "AUPRC IC95% inf"), format="%.3f"),
        "AUPRC_IC95_sup": st.column_config.NumberColumn(tr("AUPRC CI high", "AUPRC IC95% sup"), format="%.3f"),
        "Brier_IC95_inf": st.column_config.NumberColumn(tr("Brier CI low", "Brier IC95% inf"), format="%.4f"),
        "Brier_IC95_sup": st.column_config.NumberColumn(tr("Brier CI high", "Brier IC95% sup"), format="%.4f"),
    }

    if kind == "comparison":
        common.update(
            {
                tr("Comparison", "Comparação"): st.column_config.TextColumn(
                    tr("Comparison", "Comparação"),
                    help=tr(
                        "Pair of models or scores being compared.",
                        "Par de modelos ou escores sendo comparados.",
                    ),
                ),
                "Delta AUC (A-B)": st.column_config.NumberColumn(
                    "Delta AUC (A-B)",
                    help=tr(
                        "Difference in AUC between model A and model B. Positive values favor model A.",
                        "Diferença de AUC entre o modelo A e o modelo B. Valores positivos favorecem o modelo A.",
                    ),
                    format="%.3f",
                ),
                tr("95% CI low", "IC95% inf"): st.column_config.NumberColumn(tr("95% CI low", "IC95% inf"), format="%.3f"),
                tr("95% CI high", "IC95% sup"): st.column_config.NumberColumn(tr("95% CI high", "IC95% sup"), format="%.3f"),
                "p (bootstrap)": st.column_config.NumberColumn(
                    "p (bootstrap)",
                    help=tr(
                        "Approximate p-value from bootstrap comparison.",
                        "Valor de p aproximado obtido por comparação via bootstrap.",
                    ),
                    format="%.4f",
                ),
                "p (DeLong)": st.column_config.NumberColumn(
                    "p (DeLong)",
                    help=tr(
                        "P-value from DeLong test for correlated ROC curves.",
                        "Valor de p do teste de DeLong para curvas ROC correlacionadas.",
                    ),
                    format="%.4f",
                ),
            }
        )
    elif kind == "calibration":
        common.update(
            {
                "Calibration intercept": st.column_config.NumberColumn(
                    tr("Calibration-in-the-large", "Calibration-in-the-large"),
                    help=tr(
                        "Calibration intercept (calibration-in-the-large). Values closer to 0 indicate better average agreement between predicted and observed risk.",
                        "Intercepto de calibração (calibration-in-the-large). Valores mais próximos de 0 indicam melhor concordância média entre risco previsto e observado.",
                    ),
                    format="%.4f",
                ),
                "Calibration slope": st.column_config.NumberColumn(
                    tr("Calibration slope", "Slope de calibração"),
                    help=tr(
                        "Values closer to 1 indicate better calibration. Values below 1 may suggest overfitting.",
                        "Valores mais próximos de 1 indicam melhor calibração. Valores abaixo de 1 podem sugerir sobreajuste.",
                    ),
                    format="%.4f",
                ),
                "HL chi-square": st.column_config.NumberColumn(
                    tr("HL chi-square", "Qui-quadrado HL"),
                    help=tr(
                        "Hosmer-Lemeshow statistic, interpreted as complementary to visual calibration and Brier score.",
                        "Estatística de Hosmer-Lemeshow, interpretada como complementar à calibração visual e ao Brier score.",
                    ),
                    format="%.4f",
                ),
                "HL dof": st.column_config.NumberColumn(tr("HL dof", "GL HL"), format="%d"),
                "HL p-value": st.column_config.NumberColumn(
                    tr("HL p-value", "p do HL"),
                    help=tr(
                        "P-value of the Hosmer-Lemeshow test. It should not be used alone to define model adequacy.",
                        "Valor de p do teste de Hosmer-Lemeshow. Não deve ser usado isoladamente para definir a adequação do modelo.",
                    ),
                    format="%.4f",
                ),
            }
        )
    elif kind == "dca":
        common.update(
            {
                "Threshold": st.column_config.NumberColumn(
                    tr("Threshold", "Limiar"),
                    help=tr(
                        "Risk threshold at which a patient would be considered positive/high risk for decision-making.",
                        "Limiar de risco a partir do qual um paciente seria considerado positivo/alto risco para tomada de decisão.",
                    ),
                    format="%.2f",
                ),
                "Strategy": st.column_config.TextColumn(
                    tr("Strategy", "Estratégia"),
                    help=tr(
                        "Model or reference strategy (treat all / treat none) shown in decision curve analysis.",
                        "Modelo ou estratégia de referência (tratar todos / tratar ninguém) mostrada na decision curve analysis.",
                    ),
                ),
                "Net Benefit": st.column_config.NumberColumn(
                    tr("Net Benefit", "Benefício líquido"),
                    help=tr(
                        "Clinical utility measure in decision curve analysis. Higher values indicate greater usefulness at that threshold.",
                        "Medida de utilidade clínica na decision curve analysis. Valores mais altos indicam maior utilidade naquele limiar.",
                    ),
                    format="%.4f",
                ),
            }
        )
    elif kind == "reclass":
        common.update(
            {
                tr("Comparison", "Comparação"): st.column_config.TextColumn(tr("Comparison", "Comparação")),
                "NRI events": st.column_config.NumberColumn(
                    tr("NRI events", "NRI eventos"),
                    help=tr(
                        "Net reclassification improvement among patients with the event.",
                        "Net reclassification improvement entre pacientes com evento.",
                    ),
                    format="%.4f",
                ),
                "NRI non-events": st.column_config.NumberColumn(
                    tr("NRI non-events", "NRI não-eventos"),
                    help=tr(
                        "Net reclassification improvement among patients without the event.",
                        "Net reclassification improvement entre pacientes sem evento.",
                    ),
                    format="%.4f",
                ),
                "NRI total": st.column_config.NumberColumn(
                    tr("NRI total", "NRI total"),
                    help=tr(
                        "Overall net reclassification improvement. Positive values suggest better reclassification by the new model.",
                        "Melhora líquida global de reclassificação. Valores positivos sugerem melhor reclassificação pelo novo modelo.",
                    ),
                    format="%.4f",
                ),
                "IDI": st.column_config.NumberColumn(
                    "IDI",
                    help=tr(
                        "Integrated discrimination improvement. Positive values suggest improved average separation between events and non-events.",
                        "Integrated discrimination improvement. Valores positivos sugerem melhor separação média entre eventos e não eventos.",
                    ),
                    format="%.4f",
                ),
            }
        )
    elif kind == "subgroup":
        common.update(
            {
                "Subgroup": st.column_config.TextColumn(
                    tr("Subgroup", "Subgrupo"),
                    help=tr(
                        "Subgroup definition being evaluated.",
                        "Definição do subgrupo em avaliação.",
                    ),
                ),
                "Group": st.column_config.TextColumn(
                    tr("Group", "Grupo"),
                    help=tr(
                        "Specific category within the selected subgroup panel.",
                        "Categoria específica dentro do painel de subgrupos selecionado.",
                    ),
                ),
                "Deaths": st.column_config.NumberColumn(
                    tr("Deaths (primary outcome)", "Óbitos (desfecho primário)"),
                    help=tr(
                        "Number of 30-day deaths within that subgroup.",
                        "Número de óbitos em 30 dias dentro daquele subgrupo.",
                    ),
                    format="%d",
                ),
            }
        )
    return common


def general_table_column_config(kind: str) -> dict:
    if kind == "leaderboard":
        return {
            "Modelo": st.column_config.TextColumn(
                tr("Model", "Modelo"),
                help=tr("Machine-learning algorithm evaluated in cross-validation.", "Algoritmo de aprendizado de máquina avaliado na validação cruzada."),
            ),
            "AUC": st.column_config.NumberColumn("AUC", help=tr("Overall discrimination. Higher is better.", "Discriminação global. Quanto maior, melhor."), format="%.3f"),
            "AUPRC": st.column_config.NumberColumn("AUPRC", help=tr("Precision-recall performance. Useful when the event is uncommon.", "Desempenho precisão-revocação. Útil quando o evento é incomum."), format="%.3f"),
            "Brier": st.column_config.NumberColumn("Brier", help=tr("Probabilistic accuracy. Lower is better.", "Acurácia probabilística. Quanto menor, melhor."), format="%.4f"),
            "Sensibilidade": st.column_config.NumberColumn(tr("Sensitivity", "Sensibilidade"), help=tr("Out-of-fold sensitivity at the optimal threshold (Youden's J).", "Sensibilidade out-of-fold no limiar ótimo (Youden's J)."), format="%.3f"),
            "Especificidade": st.column_config.NumberColumn(tr("Specificity", "Especificidade"), help=tr("Out-of-fold specificity at the optimal threshold (Youden's J).", "Especificidade out-of-fold no limiar ótimo (Youden's J)."), format="%.3f"),
        }
    if kind == "eligibility":
        return {
            tr("Step", "Etapa"): st.column_config.TextColumn(
                tr("Step", "Etapa"),
                help=tr("Processing step in the eligibility flow from raw data to the final analytic dataset.", "Etapa do fluxo de elegibilidade desde os dados brutos até a base analítica final."),
            ),
            tr("Count", "Quantidade"): st.column_config.NumberColumn(
                tr("Count", "Quantidade"),
                help=tr("Number of records remaining or excluded at that step.", "Número de registros remanescentes ou excluídos naquela etapa."),
                format="%d",
            ),
        }
    if kind == "available_scores":
        return {
            tr("Score", "Escore"): st.column_config.TextColumn(
                tr("Score", "Escore"),
                help=tr("Score or model listed in the app summary.", "Escore ou modelo listado no resumo do aplicativo."),
            ),
            tr("Patients with value", "Pacientes com valor"): st.column_config.NumberColumn(
                tr("Patients with value", "Pacientes com valor"),
                help=tr("Number of patients with an available value for that score/model.", "Número de pacientes com valor disponível para aquele escore/modelo."),
                format="%d",
            ),
        }
    if kind == "patient_scores":
        return {
            tr("Score", "Escore"): st.column_config.TextColumn(
                tr("Score", "Escore"),
                help=tr("Model or score displayed for the current patient.", "Modelo ou escore exibido para o paciente atual."),
            ),
            tr("Probability", "Probabilidade"): st.column_config.TextColumn(
                tr("Probability", "Probabilidade"),
                help=tr("Predicted risk expressed as percentage.", "Risco predito expresso em porcentagem."),
            ),
            tr("Class", "Classe"): st.column_config.TextColumn(
                tr("Class", "Classe"),
                help=tr("Risk category derived from the predicted probability.", "Categoria de risco derivada da probabilidade predita."),
            ),
            tr("Model", "Modelo"): st.column_config.TextColumn(
                tr("Model", "Modelo"),
                help=tr("Machine-learning model used for the current patient prediction.", "Modelo de aprendizado de máquina usado para a predição do paciente atual."),
            ),
        }
    if kind == "export":
        return {
            "Name": st.column_config.TextColumn(
                tr("Patient name", "Nome do paciente"),
                help=tr(
                    "Patient identifier from the source spreadsheet.",
                    "Identificador do paciente vindo da planilha de origem.",
                ),
            ),
            "Surgery": st.column_config.TextColumn(
                tr("Surgery", "Cirurgia"),
                help=tr(
                    "Planned or recorded surgery description used in the analytic dataset.",
                    "Descrição da cirurgia planejada/registrada usada na base analítica.",
                ),
            ),
            "morte_30d": st.column_config.NumberColumn(
                tr("30-day death", "Óbito 30d"),
                help=tr(
                    "Observed 30-day outcome in the dataset (1 = death, 0 = no death).",
                    "Desfecho observado em 30 dias na base (1 = óbito, 0 = sem óbito).",
                ),
                format="%d",
            ),
            "ia_risk_oof": st.column_config.NumberColumn(
                "AI Risk",
                help=tr(
                    "AI Risk probability from out-of-fold validation predictions.",
                    "Probabilidade do AI Risk derivada das predições out-of-fold da validação.",
                ),
                format="%.4f",
            ),
            "euroscore_calc": st.column_config.NumberColumn(
                "EuroSCORE II",
                help=tr(
                    "EuroSCORE II probability calculated by the app from the published logistic equation (Nashef et al., 2012). Not read from the input file.",
                    "Probabilidade do EuroSCORE II calculada pelo app pela equação logística publicada (Nashef et al., 2012). Não lida do arquivo de entrada.",
                ),
                format="%.4f",
            ),
            "sts_score": st.column_config.NumberColumn(
                "STS",
                help=tr(
                    "STS Operative Mortality calculated by the app via automated query to the STS web calculator. Not read from the input file.",
                    "Mortalidade Operatória do STS calculada pelo app via consulta automatizada à calculadora web do STS. Não lida do arquivo de entrada.",
                ),
                format="%.4f",
            ),
            "classe_ia": st.column_config.TextColumn(tr("IA class", "Classe IA")),
            "classe_euro": st.column_config.TextColumn(tr("EuroSCORE class", "Classe EuroSCORE")),
            "classe_sts": st.column_config.TextColumn(tr("STS class", "Classe STS")),
        }
    return {}


def _fig_to_png_bytes(fig) -> bytes:
    """Convert a Matplotlib figure to PNG bytes (300 DPI)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def _chart_download_buttons(data_df: pd.DataFrame, png_bytes: bytes | None, chart_name: str):
    """Add XLSX + PNG download buttons below a chart."""
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        buf = BytesIO()
        data_df.to_excel(buf, index=False, engine="openpyxl")
        st.download_button(
            "XLSX", buf.getvalue(), f"{chart_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_xlsx_{chart_name}",
        )
    with c2:
        if png_bytes is not None:
            st.download_button(
                "PNG", png_bytes, f"{chart_name}.png",
                mime="image/png",
                key=f"dl_png_{chart_name}",
            )


def _make_line_chart_png(chart_df: pd.DataFrame, title: str, xlabel: str, ylabel: str, diagonal: bool = False) -> bytes:
    """Render a line chart DataFrame as PNG for export (not displayed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col in chart_df.columns:
        ax.plot(chart_df.index, chart_df[col], label=col)
    if diagonal:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = _fig_to_png_bytes(fig)
    plt.close(fig)
    return png


def _make_boxplot_png(chart_df: pd.DataFrame, x_col: str, y_col: str, group_col: str, title: str) -> bytes:
    """Render boxplot data as PNG for export (not displayed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = chart_df[group_col].unique()
    x_vals = chart_df[x_col].unique()
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.5), sharey=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, grp in zip(axes, groups):
        subset = chart_df[chart_df[group_col] == grp]
        data = [subset[subset[x_col] == v][y_col].dropna().values for v in x_vals]
        bp = ax.boxplot(data, tick_labels=x_vals, patch_artist=True, showfliers=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C78A8"); patch.set_alpha(0.6)
        ax.set_title(grp, fontsize=9)
        ax.set_ylabel(y_col if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    png = _fig_to_png_bytes(fig)
    plt.close(fig)
    return png


def _plot_roc(scores: Dict[str, np.ndarray], y: np.ndarray):
    roc_long = []
    for name, p in scores.items():
        fpr, tpr = roc_data(y, p)
        roc_long.append(pd.DataFrame({"FPR": fpr, "Score": name, "TPR": tpr}))
    merged = pd.concat(roc_long, ignore_index=True)
    chart = merged.pivot_table(index="FPR", columns="Score", values="TPR", aggfunc="mean").sort_index().interpolate(method="index").bfill().ffill()
    st.line_chart(chart, height=320)
    st.caption(tr("ROC curves (X axis: 1-specificity, Y axis: sensitivity)", "Curvas ROC (eixo X: 1-especificidade, eixo Y: sensibilidade)"))
    png = _make_line_chart_png(chart, "ROC Curves", "1 - Specificity (FPR)", "Sensitivity (TPR)", diagonal=True)
    _chart_download_buttons(merged, png, "roc_curves")


def _plot_calibration(scores: Dict[str, np.ndarray], y: np.ndarray):
    cal_long = []
    for name, p in scores.items():
        xp, yp = calibration_data(y, p)
        cal_long.append(pd.DataFrame({"Pred": xp, "Score": name, "Observed": yp}))
    merged = pd.concat(cal_long, ignore_index=True)
    chart = merged.pivot_table(index="Pred", columns="Score", values="Observed", aggfunc="mean").sort_index().interpolate(method="index").bfill().ffill()
    st.line_chart(chart, height=320)
    st.caption(tr("Calibration (X axis: predicted probability, Y axis: observed frequency)", "Calibração (eixo X: probabilidade predita, eixo Y: frequência observada)"))
    png = _make_line_chart_png(chart, "Calibration Curves", "Predicted probability", "Observed frequency", diagonal=True)
    _chart_download_buttons(merged, png, "calibration_curves")


def _plot_boxplots(df_plot: pd.DataFrame):
    if df_plot.empty:
        st.info(tr("No data available for boxplots.", "Sem dados disponíveis para boxplots."))
        return

    chart_df = df_plot.melt(id_vars=["Outcome"], var_name="Score", value_name="Probability").dropna()
    if chart_df.empty:
        st.info(tr("No data available for boxplots.", "Sem dados disponíveis para boxplots."))
        return

    st.vega_lite_chart(
        chart_df,
        {
            "mark": {"type": "boxplot", "extent": 1.5},
            "encoding": {
                "x": {"field": "Score", "type": "nominal", "title": tr("Score", "Escore")},
                "y": {
                    "field": "Probability",
                    "type": "quantitative",
                    "title": tr("Predicted probability", "Probabilidade predita"),
                },
                "color": {"field": "Outcome", "type": "nominal", "title": tr("Outcome", "Desfecho")},
                "column": {"field": "Outcome", "type": "nominal", "title": tr("Outcome", "Desfecho")},
            },
            "height": 320,
        },
        width="stretch",
    )
    png = _make_boxplot_png(chart_df, "Score", "Probability", "Outcome", tr("Predicted probabilities by outcome", "Probabilidades preditas por desfecho"))
    _chart_download_buttons(chart_df, png, "boxplots_scores")


def _plot_ia_model_boxplots(y_true: np.ndarray, oof_predictions: Dict[str, np.ndarray]):
    rows = []
    outcome_yes = tr("Death within 30 days", "Óbito em 30 dias")
    outcome_no = tr("No death within 30 days", "Sem óbito em 30 dias")
    for model_name, probs in oof_predictions.items():
        for y_val, prob in zip(y_true, probs):
            rows.append(
                {
                    tr("Model", "Modelo"): model_name,
                    tr("Outcome", "Desfecho"): outcome_yes if int(y_val) == 1 else outcome_no,
                    tr("Predicted probability", "Probabilidade predita"): float(prob),
                }
            )

    chart_df = pd.DataFrame(rows)
    if chart_df.empty:
        st.info(tr("No AI model data available for boxplots.", "Sem dados dos modelos de IA para boxplots."))
        return

    model_col = tr("Model", "Modelo")
    outcome_col = tr("Outcome", "Desfecho")
    prob_col = tr("Predicted probability", "Probabilidade predita")

    st.vega_lite_chart(
        chart_df,
        {
            "mark": {"type": "boxplot", "extent": 1.5},
            "encoding": {
                "x": {"field": model_col, "type": "nominal", "title": model_col},
                "y": {
                    "field": prob_col,
                    "type": "quantitative",
                    "title": prob_col,
                },
                "color": {"field": outcome_col, "type": "nominal", "title": outcome_col},
                "column": {"field": outcome_col, "type": "nominal", "title": outcome_col},
            },
            "height": 320,
        },
        width="stretch",
    )
    png = _make_boxplot_png(chart_df, model_col, prob_col, outcome_col, tr("AI model predictions by outcome", "Predições dos modelos IA por desfecho"))
    _chart_download_buttons(chart_df, png, "boxplots_ia_models")


def _plot_dca(curve_df: pd.DataFrame):
    if curve_df.empty:
        st.info(tr("No data available for decision curve analysis.", "Sem dados disponíveis para decision curve analysis."))
        return
    display_df = curve_df.copy()
    display_df["Strategy"] = display_df["Strategy"].replace(
        {
            "Treat all": tr("Treat all", "Tratar todos"),
            "Treat none": tr("Treat none", "Tratar ninguém"),
        }
    )
    chart = display_df.pivot(index="Threshold", columns="Strategy", values="Net Benefit").reset_index()
    st.line_chart(chart.set_index("Threshold"), height=320)
    st.caption(tr("Decision curve analysis: higher net benefit indicates greater clinical utility across thresholds.", "Decision curve analysis: maior benefício líquido indica maior utilidade clínica ao longo dos limiares."))
    png = _make_line_chart_png(chart.set_index("Threshold"), "Decision Curve Analysis", "Decision threshold", "Net benefit")
    _chart_download_buttons(display_df, png, "dca")


def _surgery_family(text: object) -> str:
    parts = set(split_surgery_procedures(text))
    coronary = bool(parts & {"cabg", "opcab"})
    valve = bool(parts & {"avr", "av repair", "mv repair", "mvr", "tv repair", "tvr", "ross"})
    if coronary and not valve:
        return tr("Coronary", "Coronária")
    if valve and not coronary:
        return tr("Valve", "Valvar")
    if coronary and valve:
        return tr("Mixed", "Mista")
    return tr("Other", "Outra")


def _surgery_type_group(text: object) -> str:
    parts = set(split_surgery_procedures(text))
    coronary = bool(parts & {"cabg", "opcab"})
    valve = bool(parts & {"avr", "av repair", "mv repair", "mvr", "tv repair", "tvr", "ross"})
    aorta = bool(parts & {"aortic aneurism repair", "aortic dissection repair", "bentall-de bono procedure", "valve sparing aortic root replacement (david procedure)"})
    transplant = bool(parts & {"heart transplant"})
    classes = [coronary, valve, aorta, transplant]
    n_active = sum(classes)
    if n_active > 1:
        return tr("Mixed", "Mista")
    if coronary:
        return tr("Myocardial revascularization", "Revascularização do miocárdio")
    if valve:
        return tr("Valve surgery", "Cirurgia valvar")
    if aorta:
        return tr("Aortic surgery", "Cirurgia da aorta")
    if transplant:
        return tr("Transplant / assist device", "Transplante / assist device")
    return tr("Other", "Outra")


def _lvef_group(value: object, fallback: object = None) -> str:
    v = parse_number(value)
    if pd.isna(v) and fallback is not None:
        v = parse_number(fallback)
    if pd.isna(v):
        return tr("Unknown", "Desconhecida")
    if v >= 50:
        return tr("Preserved", "Preservada")
    if v >= 41:
        return tr("Mildly reduced", "Levemente reduzida")
    return tr("Reduced", "Reduzida")

def _renal_group(clearance: object, dialysis: object,
                  creatinine: object = None, age: object = None,
                  weight: object = None, sex: object = None) -> str:
    if str(dialysis).strip().lower() in {"yes", "sim", "1", "true"}:
        return tr("Dialysis", "Diálise")

    v = parse_number(clearance)

    # Fallback: compute Cockcroft-Gault if clearance is missing but
    # creatinine, age, weight, and sex are available
    if pd.isna(v) and creatinine is not None:
        scr = parse_number(creatinine)
        a = parse_number(age)
        w = parse_number(weight)
        sx = str(sex).strip().upper()[:1] if pd.notna(sex) else ""
        if not pd.isna(scr) and scr > 0 and not pd.isna(a) and not pd.isna(w) and sx in {"M", "F"}:
            factor = 0.85 if sx == "F" else 1.0
            v = ((140.0 - a) * w * factor) / (72.0 * scr)

    if pd.isna(v):
        return tr("Unknown", "Desconhecida")
    if v > 85:
        return ">85"
    if v > 50:
        return "51-85"
    return "<=50"


def build_methods_text(mode: str) -> str:
    detailed = mode == tr("Detailed", "Detalhado")
    if language == "English":
        if detailed:
            return (
                "A retrospective analytical study was performed using matched data from the Preoperative, Pre-Echocardiogram, and Postoperative sheets. "
                "The primary endpoint was in-hospital or 30-day mortality, operationalized from the Postoperative Death field by considering Operative status and postoperative days 0 to 30 as events. "
                "Only records with Surgery and Procedure Date available were eligible, and matching across sheets was performed using patient identity and procedure date; these identifiers were used exclusively for linkage and not as predictors. "
                "AI Risk was trained exclusively with preoperative clinical, laboratory, echocardiographic, and procedural variables, excluding postoperative complications in order to avoid temporal leakage. Numeric variables were preprocessed with comma-decimal conversion (for Brazilian-format data), clinically impossible zeros treated as missing (e.g. BSA=0), median imputation, and StandardScaler normalization; valve severity variables were encoded ordinally (None < Trivial < Mild < Moderate < Severe); remaining categorical variables were encoded via TargetEncoder with automatic smoothing. Tree-based models (random forest, XGBoost, LightGBM, CatBoost) were post-hoc calibrated via Platt scaling (sigmoid method) to improve probability accuracy; calibration was applied inside each cross-validation fold so that calibrated out-of-fold predictions are free of information leakage. Candidate algorithms included logistic regression, random forest, multilayer perceptron, XGBoost, LightGBM, CatBoost, and a stacking ensemble. Model selection was based on stratified cross-validation grouped by patient, ensuring that the same patient never appeared simultaneously in training and testing folds. "
                "EuroSCORE II was calculated from the published logistic equation, with operationalization of variables according to the available dataset. STS was obtained via automated interaction with the official STS Risk Calculator web application (https://acsdriskcalc.research.sts.org/), which does not offer a documented public API; preoperative data were mapped from the dataset and submitted via the calculator's WebSocket interface, for both batch analysis and individual prediction. "
                "The primary comparative analysis was the triple-cohort analysis including patients with simultaneous availability of AI Risk, EuroSCORE II, and STS. Complementary analyses may be conducted in specific subsets according to score availability, with explicit acknowledgment of sample differences. Discrimination was assessed by AUC-ROC and AUPRC. In the model leaderboard (internal comparison), sensitivity and specificity were computed at the optimal threshold determined by Youden's J index (J = sensitivity + specificity − 1) for each model independently. In the triple-score comparison, a fixed clinical decision threshold (default 8%) was applied uniformly to all three scores. Calibration was assessed by visual calibration curve, calibration-in-the-large, calibration slope, Brier score, and the Hosmer-Lemeshow test (10 groups) as a complementary assessment. Formal ROC comparison included bootstrap-based delta AUC (2,000 resamples) with 95% confidence intervals and DeLong testing for correlated ROC curves. Clinical utility was assessed by decision curve analysis across risk thresholds from 5% to 20%, and reclassification was summarized by NRI (risk categories: low <5%, intermediate 5–15%, high >15%) and IDI as complementary analyses. Subgroup analyses were planned according to surgery type (valve, myocardial revascularization, aortic surgery, transplant/assist device, and mixed surgery) and clinical profile (age group, left ventricular ejection fraction category, renal function, and sex), with 95% confidence intervals by bootstrap."
            )
        return (
            "A retrospective comparison of AI Risk, EuroSCORE II, and STS was performed for in-hospital or 30-day mortality after cardiac surgery. "
            "AI Risk was validated by cross-validation grouped by patient, and model performance was evaluated through discrimination, calibration, formal ROC comparison, decision curve analysis, and reclassification metrics."
        )
    if detailed:
        return (
                "Foi realizada análise retrospectiva com dados pareados das abas Preoperative, Pre-Echocardiogram e Postoperative. "
                "O desfecho primário foi mortalidade hospitalar ou em 30 dias, operacionalizada a partir do campo Death da aba Postoperative, considerando como evento os casos classificados como Operative ou com ocorrência entre os dias 0 e 30 de pós-operatório. "
                "Foram elegíveis apenas registros com Surgery e Procedure Date preenchidos, sendo o pareamento entre as abas realizado por identidade do paciente e data do procedimento; esses identificadores foram utilizados exclusivamente para vinculação interna, sem participação como variáveis preditoras. "
                "O AI Risk foi treinado exclusivamente com variáveis pré-operatórias clínicas, laboratoriais, ecocardiográficas e de procedimento, sem inclusão de complicações pós-operatórias, a fim de evitar vazamento temporal. Variáveis numéricas foram pré-processadas com conversão de vírgula decimal (para dados em formato brasileiro), zeros clinicamente impossíveis tratados como ausentes (ex: BSA=0), imputação pela mediana e normalização StandardScaler; variáveis de gravidade valvar foram codificadas ordinalmente (None < Trivial < Mild < Moderate < Severe); demais variáveis categóricas foram codificadas via TargetEncoder com suavização automática. Modelos baseados em árvore (random forest, XGBoost, LightGBM, CatBoost) foram calibrados por Platt scaling (método sigmoid) para melhorar a acurácia das probabilidades; a calibração foi aplicada dentro de cada fold da validação cruzada para que as predições out-of-fold calibradas sejam livres de vazamento de informação. Foram avaliados regressão logística, random forest, multilayer perceptron, XGBoost, LightGBM, CatBoost e ensemble por stacking. A seleção do melhor modelo foi baseada em validação cruzada estratificada e agrupada por paciente, garantindo que o mesmo paciente não estivesse simultaneamente em treino e teste. "
                "O EuroSCORE II foi calculado pela equação logística publicada, com operacionalização das variáveis a partir da base disponível. O STS foi obtido via interação automatizada com a calculadora web oficial do STS Risk Calculator (https://acsdriskcalc.research.sts.org/), que não disponibiliza uma API pública documentada; os dados pré-operatórios foram mapeados da base e enviados pela interface WebSocket da calculadora, tanto para análise em lote quanto para predição individual. "
                "A análise comparativa principal foi a coorte tripla, composta pelos pacientes com disponibilidade simultânea de AI Risk, EuroSCORE II e STS. Análises complementares poderão ser conduzidas em subconjuntos específicos, de acordo com a disponibilidade de cada escore, com explicitação das diferenças amostrais. A discriminação foi avaliada por AUC-ROC e AUPRC. No leaderboard de modelos (comparação interna), sensibilidade e especificidade foram calculadas no limiar ótimo determinado pelo índice de Youden (J = sensibilidade + especificidade − 1) para cada modelo independentemente. Na comparação tripla de escores, um limiar clínico fixo de decisão (padrão 8%) foi aplicado uniformemente aos três escores. A calibração foi avaliada por curva de calibração visual, calibration-in-the-large, slope de calibração, Brier score e teste de Hosmer-Lemeshow (10 grupos) como avaliação complementar. A comparação formal entre curvas ROC utilizou delta AUC por bootstrap (2.000 reamostras) com intervalos de confiança de 95% e teste de DeLong para curvas correlacionadas. A utilidade clínica foi avaliada por decision curve analysis em limiares de 5% a 20%, e a reclassificação prognóstica foi resumida por NRI (categorias de risco: baixo <5%, intermediário 5–15%, alto >15%) e IDI como análises complementares. Foram ainda planejadas análises por subgrupos segundo tipo de cirurgia (valvar, revascularização do miocárdio, cirurgia da aorta, transplante/assist device e cirurgia mista) e perfil clínico (faixa etária, categoria de FEVE, função renal e sexo), com intervalos de confiança de 95% por bootstrap."
        )
    return (
        "Foi realizada comparação retrospectiva entre AI Risk, EuroSCORE II e STS para mortalidade hospitalar ou em 30 dias após cirurgia cardíaca. "
        "O AI Risk foi validado por validação cruzada agrupada por paciente, e o desempenho foi avaliado por métricas de discriminação, calibração, comparação formal entre curvas ROC, decision curve analysis e reclassificação."
    )


def build_results_text(mode: str, context: Dict[str, object]) -> str:
    detailed = mode == tr("Detailed", "Detalhado")
    if language == "English":
        if detailed:
            return (
                f"In the primary triple-cohort analysis including patients with simultaneous availability of AI Risk, EuroSCORE II, and STS (n={context['n_triple']}), the best overall discrimination was observed for {context['best_auc_model']} with AUC-ROC={context['best_auc']:.3f}. "
                f"At the selected decision threshold of {context['threshold']:.2f}, the highest sensitivity was observed for {context['best_sens_model']}, whereas the highest specificity was observed for {context['best_spec_model']}. The highest positive predictive value was observed for {context['best_ppv_model']}, and the highest negative predictive value for {context['best_npv_model']}. "
                f"Regarding calibration, the best overall Brier score was observed for {context['best_brier_model']}. Formal comparison of ROC curves indicated that {context['formal_summary']} In terms of clinical utility, the highest mean net benefit between 5% and 20% was observed for {context['best_dca_model']}. Regarding reclassification, {context['reclass_summary']} Overall, these findings indicate that model choice should not rely on discrimination alone, but also on calibration, threshold-dependent performance, and net clinical benefit."
            )
        return (
            f"In the primary triple cohort (n={context['n_triple']}), {context['best_auc_model']} showed the best discrimination, {context['best_brier_model']} the best calibration, and {context['best_dca_model']} the highest average net benefit between 5% and 20%."
        )
    if detailed:
        return (
            f"Na análise principal da coorte tripla, composta pelos pacientes com disponibilidade simultânea de AI Risk, EuroSCORE II e STS (n={context['n_triple']}), a melhor discriminação global foi observada em {context['best_auc_model']}, com AUC-ROC={context['best_auc']:.3f}. "
            f"No limiar de decisão selecionado de {context['threshold']:.2f}, a maior sensibilidade foi observada em {context['best_sens_model']}, enquanto a maior especificidade foi observada em {context['best_spec_model']}. O maior valor preditivo positivo foi observado em {context['best_ppv_model']}, e o maior valor preditivo negativo em {context['best_npv_model']}. "
            f"Em relação à calibração, o melhor desempenho global pelo Brier score foi observado em {context['best_brier_model']}. A comparação formal entre curvas ROC indicou que {context['formal_summary']} Em termos de utilidade clínica, o maior benefício líquido médio entre 5% e 20% foi observado em {context['best_dca_model']}. Em relação à reclassificação, {context['reclass_summary']} Em conjunto, esses achados sugerem que a escolha do melhor modelo não deve se apoiar apenas na discriminação, mas também na calibração, no desempenho dependente do limiar e no benefício clínico líquido."
        )
    return (
        f"Na coorte tripla principal (n={context['n_triple']}), {context['best_auc_model']} apresentou a melhor discriminação, {context['best_brier_model']} a melhor calibração e {context['best_dca_model']} o maior benefício líquido médio entre 5% e 20%."
    )


def evaluate_subgroup(df_in: pd.DataFrame, subgroup_col: str, score_cols: list[str], threshold: float) -> pd.DataFrame:
    rows = []
    for subgroup_name, sub in df_in.groupby(subgroup_col):
        for score in score_cols:
            s = sub[["morte_30d", score]].dropna()
            if len(s) < 20 or s["morte_30d"].nunique() < 2:
                continue
            metrics = evaluate_scores_with_threshold(s, "morte_30d", [score], threshold)
            if metrics.empty:
                continue
            rec = metrics.iloc[0].to_dict()
            rec["Subgroup"] = subgroup_col
            rec["Group"] = subgroup_name
            rec["Deaths"] = int(s["morte_30d"].sum())
            # Bootstrap CI for AUC, AUPRC, Brier (reduced resamples for speed in subgroups)
            ci = bootstrap_metrics_ci(s["morte_30d"].values, s[score].values, n_boot=500, seed=42)
            rec["AUC_IC95_inf"] = ci.get("AUC_IC95_inf", np.nan)
            rec["AUC_IC95_sup"] = ci.get("AUC_IC95_sup", np.nan)
            rec["AUPRC_IC95_inf"] = ci.get("AUPRC_IC95_inf", np.nan)
            rec["AUPRC_IC95_sup"] = ci.get("AUPRC_IC95_sup", np.nan)
            rec["Brier_IC95_inf"] = ci.get("Brier_IC95_inf", np.nan)
            rec["Brier_IC95_sup"] = ci.get("Brier_IC95_sup", np.nan)
            rows.append(rec)
    return pd.DataFrame(rows)


def _get_numeric_columns_from_pipeline(model_pipeline) -> set:
    """Extract the set of numeric column names from a trained sklearn Pipeline."""
    try:
        prep = model_pipeline.named_steps.get("prep")
        if prep and hasattr(prep, "transformers"):
            for name, _trans, cols in prep.transformers:
                if name == "num":
                    return set(cols)
    except Exception:
        pass
    return set()


def _safe_select_features(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Select feature columns from df, adding missing ones as NaN."""
    out = df.copy()
    for c in feature_columns:
        if c not in out.columns:
            out[c] = np.nan
    return out[feature_columns]


def _build_input_row(feature_columns, form: Dict[str, object]) -> pd.DataFrame:
    row = {c: np.nan for c in feature_columns}
    # Build normalized lookups to handle whitespace/truncation in Excel column names
    _norm = {c.strip(): c for c in feature_columns}
    # Also map by prefix to handle truncated names (e.g. "Surgical Priorit" matches "Surgical Priority")
    for k, v in form.items():
        if k in row:
            row[k] = v
        elif k.strip() in _norm:
            row[_norm[k.strip()]] = v
        else:
            # Check if any feature column is a prefix of form key (truncation)
            k_stripped = k.strip()
            for fc in feature_columns:
                fc_stripped = fc.strip()
                if fc_stripped != k_stripped and (k_stripped.startswith(fc_stripped) or fc_stripped.startswith(k_stripped)):
                    row[fc] = v
                    break
    surg = form.get("Surgery", "")
    row["cirurgia_combinada"] = is_combined_surgery(surg)
    row["peso_procedimento"] = procedure_weight(surg)
    row["thoracic_aorta_flag"] = thoracic_aorta_surgery(surg)

    # Clean numeric fields that may contain string values from CSV
    _susp = row.get("Suspension of Anticoagulation (day)")
    if isinstance(_susp, str):
        _susp_clean = _susp.strip().replace(">", "").strip()
        try:
            row["Suspension of Anticoagulation (day)"] = float(_susp_clean)
        except (ValueError, TypeError):
            row["Suspension of Anticoagulation (day)"] = 0

    out = pd.DataFrame([row])
    defaults = {
        "HF": "No",
        "Arrhythmia Remote": "No",
        "Arrhythmia Recent": "No",
        "Hypertension": "No",
        "Diabetes": "No",
        "Dyslipidemia": "No",
        "CVA": "No",
        "PVD": "No",
        "Alcohol": "No",
        "Smoking (Pack-year)": "Never",
        "Ex-Smoker (Pack-year)": "Never",
        "Cancer ≤ 5 yrs": "No",
        "Family Hx of CAD": "No",
        "Pneumonia": "No",
        "Anticoagulation/ Antiaggregation": "No",
        "Dialysis": "No",
        "IE": "No",
        "Aortic Stenosis": "None",
        "Aortic Regurgitation": "None",
        "Mitral Stenosis": "None",
        "Mitral Regurgitation": "None",
        "Tricuspid Regurgitation": "None",
        "Aortic Root Abscess": "No",
        "Suspension of Anticoagulation (day)": 0,
    }
    for c, v in defaults.items():
        if c in out.columns and (pd.isna(out.at[0, c]) or str(out.at[0, c]).strip() == ""):
            out.at[0, c] = v

    return out


def _align_input_to_training_schema(input_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    aligned = input_df.copy()
    for col in aligned.columns:
        if col in reference_df.columns:
            if pd.api.types.is_numeric_dtype(reference_df[col]):
                # Force numeric: strip symbols, fix comma decimals, coerce strings to NaN
                if aligned[col].dtype == object:
                    aligned[col] = (
                        aligned[col].astype(str)
                        .str.replace(r'[><~]', '', regex=True)
                        .str.strip()
                        .str.replace(',', '.', regex=False)
                    )
                aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
            elif reference_df[col].dtype == object and aligned[col].dtype != object:
                # Force categorical
                aligned[col] = aligned[col].astype(str)
    return aligned


def _risk_badge(p: float) -> str:
    if np.isnan(p):
        return "Not available"
    label = class_risk(float(p))
    label_map = {"Low": tr("Low", "Baixo"), "Intermediate": tr("Intermediate", "Intermediário"), "High": tr("High", "Alto")}
    return f"{label_map.get(label, label)} ({100*p:.1f}%)"


st.markdown(
    tr(
        "<h1 style='margin-bottom:0'>AI Risk</h1>"
        "<p style='color:gray; font-size:1.1em; margin-top:0'>"
        "Cardiac Surgery Risk Stratification</p>",
        "<h1 style='margin-bottom:0'>AI Risk</h1>"
        "<p style='color:gray; font-size:1.1em; margin-top:0'>"
        "Estratificação de Risco em Cirurgia Cardíaca</p>",
    ),
    unsafe_allow_html=True,
)

default_xlsx = Path("Tables.xlsx")

# ── Data source section ──
_src_options = {
    "local": tr("Local", "Local"),
    "upload": tr("Upload", "Upload"),
    "gsheets": tr("Google Sheets", "Google Sheets"),
}
data_source = st.sidebar.radio(
    tr("Data source", "Fonte de dados"),
    list(_src_options.values()),
    index=0,
    horizontal=True,
)

xlsx_path = str(default_xlsx)

if data_source == _src_options["local"]:
    local_candidates = _local_source_candidates()
    local_label_map = {p.name: str(p) for p in local_candidates}
    local_labels = list(local_label_map.keys())
    default_label = default_xlsx.name if default_xlsx.exists() and default_xlsx.name in local_label_map else (local_labels[0] if local_labels else "")
    if local_labels:
        selected_local = st.sidebar.selectbox(
            tr("Data file", "Arquivo de dados"),
            local_labels,
            index=local_labels.index(default_label) if default_label in local_labels else 0,
        )
    else:
        selected_local = ""
        st.sidebar.warning(tr(
            "No files in local_data/. Place .xlsx, .csv, or .parquet files there.",
            "Pasta local_data/ vazia. Coloque arquivos .xlsx, .csv ou .parquet nela.",
        ))
    xlsx_path = local_label_map.get(selected_local, str(default_xlsx))

elif data_source == _src_options["upload"]:
    up = st.sidebar.file_uploader(
        tr("Upload patient data", "Upload do arquivo de pacientes"),
        type=["xlsx", "xls", "csv", "parquet", "db", "sqlite", "sqlite3"],
    )
    if up is not None:
        suffix = Path(up.name).suffix or ".bin"
        upload_path = UPLOAD_CACHE_FILE.with_suffix(suffix)
        upload_path.write_bytes(up.getvalue())
        xlsx_path = str(upload_path)
    else:
        cached_uploads = sorted(TEMP_DATA_DIR.glob(f"{UPLOAD_CACHE_FILE.stem}.*"))
        if cached_uploads:
            xlsx_path = str(cached_uploads[-1])
            st.sidebar.caption(tr(f"Last upload: {Path(xlsx_path).name}", f"Último upload: {Path(xlsx_path).name}"))

else:  # Google Sheets
    g_url = st.sidebar.text_input(
        tr("Google Sheets URL", "URL do Google Sheets"),
        value="",
        placeholder="https://docs.google.com/spreadsheets/d/...",
    )
    refresh_g = st.sidebar.button(tr("Reload", "Recarregar"), width="stretch")

    g_clean = g_url.strip()
    auto_test_needed = False
    if g_clean:
        last_url = st.session_state.get("last_gsheet_url", "")
        auto_test_needed = refresh_g or (g_clean != last_url)

    if auto_test_needed:
        try:
            export_url = _google_sheet_export_url(g_clean)
            ok, msg = _download_to_file(export_url, GSHEETS_CACHE_FILE)
            if ok:
                v_ok, v_msg = _validate_source(str(GSHEETS_CACHE_FILE))
                if v_ok:
                    st.session_state["last_gsheet_url"] = g_clean
                    st.sidebar.success(tr("Loaded and validated", "Carregado e validado"))
                else:
                    st.sidebar.error(tr(f"Invalid data: {v_msg}", f"Dados inválidos: {v_msg}"))
            else:
                st.sidebar.error(tr(f"Load failed: {msg}", f"Falha: {msg}"))
        except Exception as e:
            st.sidebar.error(tr(f"Invalid URL: {e}", f"URL inválida: {e}"))

    if GSHEETS_CACHE_FILE.exists():
        xlsx_path = str(GSHEETS_CACHE_FILE)
        st.sidebar.caption(tr(
            "Using cached Google Sheets file. Clear temp files to reset.",
            "Usando cache do Google Sheets. Limpe temporários para resetar.",
        ))

st.sidebar.divider()

# ── Model section ──
force_retrain = st.sidebar.button(
    tr("Train / Retrain models", "Treinar / Retreinar modelos"),
    width="stretch",
    type="primary",
)

with st.sidebar.expander(tr("Advanced", "Avançado")):
    clear_temp = st.button(tr("Clear temporary files", "Limpar arquivos temporários"), width="stretch")
    if clear_temp:
        ok, msg = _clear_temp_data_dir()
        if ok:
            st.success(tr("Cleaned", "Limpo"))
            if "last_gsheet_url" in st.session_state:
                del st.session_state["last_gsheet_url"]
        else:
            st.error(msg)

if not Path(xlsx_path).exists():
    st.error(tr(f"File not found: {xlsx_path}", f"Arquivo não encontrado: {xlsx_path}"))
    st.stop()

valid_ok, valid_msg = _validate_source(xlsx_path)
if not valid_ok:
    st.error(
        tr(
            f"Data source validation failed: {valid_msg}",
            f"Falha na validação da fonte de dados: {valid_msg}",
        )
    )
    st.stop()

_retrained_bundle = None
if force_retrain:
    load_cached_bundle_only.clear()
    try:
        _train_progress = st.progress(0, text=tr(
            "Preparing data…", "Preparando dados…",
        ))
        def _train_progress_cb(phase, current, total, model_name):
            if phase == "cross_validation":
                _pct = current / max(total * 2, 1)  # CV = first half
                _train_progress.progress(_pct, text=tr(
                    f"Cross-validating: {model_name} ({current + 1}/{total})",
                    f"Validação cruzada: {model_name} ({current + 1}/{total})",
                ))
            elif phase == "final_fit":
                _pct = 0.5 + current / max(total * 2, 1)  # Final fit = second half
                _train_progress.progress(_pct, text=tr(
                    f"Final training: {model_name} ({current + 1}/{total})",
                    f"Treino final: {model_name} ({current + 1}/{total})",
                ))
            elif phase == "selecting_best":
                _train_progress.progress(0.95, text=tr(
                    "Selecting best model…", "Selecionando melhor modelo…",
                ))

        _retrained_bundle, bundle_source, _retrained_info = load_train_bundle(
            xlsx_path, force_retrain=True, progress_callback=_train_progress_cb,
        )
        _train_progress.progress(1.0, text=tr(
            "Training complete!", "Treinamento concluído!",
        ))
    except Exception as e:
        st.error(
            tr(
                f"Training failed: {e}",
                f"Falha no treino: {e}",
            )
        )
        try:
            st.markdown(tr("**Eligibility summary**", "**Resumo de elegibilidade**"))
            st.dataframe(_eligibility_summary(xlsx_path), width="stretch", column_config=general_table_column_config("eligibility"))
        except Exception:
            pass
        st.stop()
else:
    bundle_source = tr("No action", "Sem ação")

# Use the freshly trained bundle directly (avoids stale cache issues)
if _retrained_bundle is not None:
    bundle, cache_status, bundle_info = _retrained_bundle, "Recalculado", _retrained_info
else:
    bundle, cache_status, bundle_info = load_cached_bundle_only(xlsx_path)

_status_color = "green" if "local" in cache_status.lower() or "cache" in cache_status.lower() else "orange"
st.sidebar.markdown(
    f"<small style='color:gray'>{tr('Status', 'Status')}: "
    f"<span style='color:{_status_color}'>{cache_status}</span></small>",
    unsafe_allow_html=True,
)

if bundle is None:
    st.warning(
        tr(
            "No trained model found for this file. Click 'Train/Retrain models' in the sidebar.",
            "Não há modelo treinado para este arquivo. Clique em 'Treinar/Retreinar modelos' na barra lateral.",
        )
    )
    st.info(
        tr(
            "After training, the app saves the model and does not retrain automatically on next openings.",
            "Depois do treino, o app salva o modelo e não recalcula automaticamente nas próximas aberturas.",
        )
    )
    st.stop()

prepared = bundle["prepared"]
artifacts = bundle["artifacts"]
base_df = bundle["data"].copy()
best_model_name = artifacts.best_model_name

model_options = artifacts.leaderboard["Modelo"].tolist()
default_idx = model_options.index(best_model_name) if best_model_name in model_options else 0
forced_model = st.sidebar.selectbox(
    tr("Model", "Modelo"),
    model_options,
    index=default_idx,
    help=tr(
        f"Best model from training: {best_model_name}. You can override it here.",
        f"Melhor modelo do treino: {best_model_name}. Você pode forçar outro aqui.",
    ),
)

st.sidebar.divider()
st.sidebar.markdown(
    "<div style='text-align:center; color:gray; font-size:0.7em; line-height:1.5'>"
    "Dr. Michael Hikaru Mikami<br>CRM-PR 47.366"
    "</div>",
    unsafe_allow_html=True,
)

df = base_df.copy()
df["ia_risk_oof"] = artifacts.oof_predictions[forced_model]
df["ia_risk_fullfit"] = artifacts.fitted_models[forced_model].predict_proba(
    clean_features(_safe_select_features(df, artifacts.feature_columns))
)[:, 1]
raw_surgery_values = [
    x
    for x in prepared.data["Surgery"].dropna().astype(str).str.strip().unique().tolist()
    if x and x.lower() != "nan"
]

procedure_item_map: dict[str, str] = {}
for value in raw_surgery_values:
    normalized_parts = split_surgery_procedures(value)
    original_parts = [p.strip() for p in value.replace(";", ",").replace("+", ",").split(",") if p.strip()]
    for norm_part, original_part in zip(normalized_parts, original_parts):
        if norm_part not in procedure_item_map:
            procedure_item_map[norm_part] = original_part

surgery_component_options = sorted(procedure_item_map.values(), key=str.lower)
surgery_other_label = tr("Other / custom", "Outro / personalizado")
# Default threshold: 8% — a conservative choice for cardiac surgery where
# missing a high-risk patient (false negative) is worse than a false alarm.
# The user can adjust freely via the slider.
_default_threshold = 0.08

_tab_labels = [
    tr("Overview", "Visão Geral"),
    tr("Prediction", "Predição"),
    tr("Comparison", "Comparação"),
    tr("Guide", "Guia"),
    tr("Batch", "Lote"),
    tr("Models", "Modelos"),
    tr("Subgroups", "Subgrupos"),
    tr("Data Quality", "Qualidade"),
    tr("Dictionary", "Dicionário"),
    tr("Temporal Validation", "Validação Temporal"),
]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0
if st.session_state.active_tab >= len(_tab_labels):
    st.session_state.active_tab = 0

_selected_tab_label = st.segmented_control(
    "nav",
    _tab_labels,
    default=_tab_labels[st.session_state.active_tab],
    selection_mode="single",
    label_visibility="collapsed",
    key="_tab_nav",
)
_active_tab = _tab_labels.index(_selected_tab_label) if _selected_tab_label else st.session_state.active_tab
st.session_state.active_tab = _active_tab

if _active_tab == 0:  # Overview
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("Patients (matched)", "Pacientes (pareados)"), f"{prepared.info['n_rows']}")
    c2.metric(tr("In-hospital / 30-day outcome", "Desfecho hospitalar / 30d"), f"{prepared.info['positive_rate']*100:.1f}%")
    c3.metric(tr("Predictor variables", "Variáveis preditoras"), f"{prepared.info['n_features']}")
    c4.metric(tr("AI model in use", "Modelo IA em uso"), forced_model)
    st.caption(tr(f"Best training model: {best_model_name}", f"Melhor modelo automático no treino: {best_model_name}"))
    st.caption(tr(f"Last action: {bundle_source}", f"Última ação: {bundle_source}"))
    st.caption(tr(f"Model version: {MODEL_VERSION}", f"Versão do modelo: {MODEL_VERSION}"))

    # Model metadata panel
    _model_meta = build_model_metadata(
        prepared.info, artifacts.leaderboard, best_model_name,
        artifacts.feature_columns, xlsx_path, sts_available=HAS_STS,
        bundle_saved_at=bundle_info.get("saved_at"),
        training_source_file=bundle_info.get("training_source"),
        calibration_method=getattr(artifacts, "calibration_method", "sigmoid"),
        training_data=prepared.data,
    )
    with st.expander(tr("Model version details", "Detalhes da versão do modelo"), expanded=False):
        st.dataframe(format_metadata_for_display(_model_meta, language), width="stretch")
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            _meta_json = json.dumps(_model_meta, ensure_ascii=False, indent=2, default=str)
            _txt_download_btn(_meta_json, "model_metadata.json", tr("Export metadata (JSON)", "Exportar metadados (JSON)"))
        with col_meta2:
            st.caption(tr(
                "Export metadata for version tracking, bundle comparison, and external validation.",
                "Exporte os metadados para rastreamento de versões, comparação de bundles e validação externa.",
            ))

    st.subheader(tr("IA Model Performance", "Desempenho dos modelos de IA"))
    st.caption(
        tr(
            "Stratified cross-validation grouped by patient (same patient never appears in both train and test folds).",
            "Validação cruzada estratificada e agrupada por paciente (sem mistura do mesmo paciente entre treino e teste).",
        )
    )
    st.caption(
        tr(
            "Sensitivity and specificity are shown at the optimal threshold (Youden's J) for each model, not at a fixed 0.50 cutoff.",
            "Sensibilidade e especificidade são mostradas no limiar ótimo (Youden's J) de cada modelo, não em um corte fixo de 0,50.",
        )
    )
    st.caption(
        tr(
            "AUC summarizes overall discrimination. AUPRC is especially useful when the event is uncommon. Sensitivity and specificity in this table use the optimal threshold (Youden's J) from out-of-fold predictions.",
            "A AUC resume a discriminação global. A AUPRC é especialmente útil quando o evento é incomum. Sensibilidade e especificidade nesta tabela usam o limiar ótimo (Youden's J) das predições out-of-fold.",
        )
    )
    st.dataframe(artifacts.leaderboard, width="stretch", column_config=general_table_column_config("leaderboard"))

    _elig_info = _cached_eligibility_info(xlsx_path)
    if _elig_info.get("source_type") != "flat" and _elig_info.get("pre_rows_before_criteria", 0) > 0:
        st.subheader(tr("Eligibility flow", "Fluxo de elegibilidade"))
        st.dataframe(_eligibility_summary(xlsx_path), width="stretch", column_config=general_table_column_config("eligibility"))

    st.subheader(tr("Available scores summary", "Resumo dos escores disponíveis"))
    summary = pd.DataFrame(
        {
            tr("Score", "Escore"): ["AI Risk", "EuroSCORE II (app-calculated)", "STS (app-calculated)"],
            tr("Patients with value", "Pacientes com valor"): [
                int(df["ia_risk_oof"].notna().sum()),
                int(df["euroscore_calc"].notna().sum()),
                int(df["sts_score"].notna().sum()),
            ],
            tr("Source", "Origem"): [
                tr("Cross-validated out-of-fold predictions", "Predições out-of-fold por validação cruzada"),
                tr("Published logistic equation (Nashef et al., 2012)", "Equação logística publicada (Nashef et al., 2012)"),
                tr("Automated query to the STS web calculator", "Consulta automatizada à calculadora web do STS"),
            ],
        }
    )
    st.dataframe(summary, width="stretch", column_config=general_table_column_config("available_scores"))
    st.caption(tr(
        "All scores shown are computed by the app — not read from the input file. Sheet-derived values are retained only as optional reference in the Data Quality tab.",
        "Todos os escores exibidos são calculados pelo app — não lidos do arquivo de entrada. Valores derivados da planilha são mantidos apenas como referência opcional no painel de Qualidade da Base.",
    ))

elif _active_tab == 1:  # Individual Prediction
    st.subheader(tr("Individual calculation", "Cálculo individual"))
    st.caption(tr("Fill in fields in clinical order. The app calculates AI Risk, EuroSCORE II, and STS automatically.", "Preencha os campos em ordem clínica. O app calcula AI Risk, EuroSCORE II e STS automaticamente."))

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
            st.caption(tr("STS Score will be calculated automatically via the STS web calculator.", "O STS Score será calculado automaticamente via a calculadora web do STS."))

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
            hf = st.selectbox(tr("Heart failure (HF)", "Insuficiência cardíaca (HF)"), yn_options, index=0, help=hp("Clinical diagnosis of heart failure before surgery. Usually associated with worse functional reserve.", "Diagnóstico clínico de insuficiência cardíaca antes da cirurgia. Geralmente associa-se a pior reserva funcional."))
            htn = st.selectbox(tr("Hypertension", "Hipertensão"), yn_options, index=0, help=hp("History of systemic arterial hypertension.", "História de hipertensão arterial sistêmica."))
            dlp = st.selectbox(tr("Dyslipidemia", "Dislipidemia"), yn_options, index=0, help=hp("Presence of dyslipidemia or lipid-lowering treatment.", "Presença de dislipidemia ou uso de tratamento redutor de lipídios."))
            diabetes_pt = st.selectbox(tr("Diabetes", "Diabetes"), [tr("No", "Não"), "Oral", tr("Insulin", "Insulina"), tr("Diet", "Dieta"), tr("No Control Method", "Sem método de controle")], index=0, help=hp("Diabetes treatment category. Insulin usually indicates more severe metabolic disease in risk models.", "Categoria de tratamento do diabetes. Uso de insulina costuma indicar doença metabólica mais grave nos modelos de risco."))
        with cx2:
            cva = st.selectbox(tr("Cerebrovascular disease (CVA)", "Doença cerebrovascular (CVA)"), yn_options, index=0, help=hp("History of stroke or cerebrovascular disease.", "História de AVC ou doença cerebrovascular."))
            pvd2 = st.selectbox(tr("Peripheral vascular disease (PVD)", "Doença vascular periférica (PVD)"), yn_options, index=0, key="pvd_comorb", help=hp("Peripheral arterial disease. In EuroSCORE II it is used as an approximation of extracardiac arteriopathy.", "Doença arterial periférica. No EuroSCORE II é usada como aproximação de extracardiac arteriopathy."))
            cancer5 = st.selectbox(tr("Cancer <= 5 years", "Câncer <= 5 anos"), yn_options, index=0, help=hp("History of cancer diagnosed or treated within the last 5 years.", "História de câncer diagnosticado ou tratado nos últimos 5 anos."))
            dialysis = st.selectbox(tr("Dialysis", "Diálise"), yn_options, index=0, help=hp("Indicates established dialysis therapy. Strong marker of severe renal dysfunction.", "Indica terapia dialítica estabelecida. Marcador forte de disfunção renal grave."))
        with cx3:
            arr_rem = st.selectbox(tr("Remote arrhythmia", "Arritmia remota"), yn_options, index=0, help=hp("Past history of arrhythmia not active in the immediate perioperative period.", "História pregressa de arritmia sem atividade imediata no perioperatório."))
            arr_rec = st.selectbox(tr("Recent arrhythmia", "Arritmia recente"), yn_options, index=0, help=hp("Recent arrhythmia before surgery, suggesting greater electrical instability.", "Arritmia recente antes da cirurgia, sugerindo maior instabilidade elétrica."))
            fam_cad = st.selectbox(tr("Family history of CAD", "História familiar de DAC"), yn_options, index=0, help=hp("Family history of coronary artery disease.", "História familiar de doença arterial coronariana."))
            ie_pt = st.selectbox(tr("Active/probable endocarditis", "Endocardite ativa/provável"), [tr("No", "Não"), tr("Yes", "Sim"), tr("Possible", "Possível")], index=0, help=hp("Active or probable infective endocarditis at the time of surgery. In this app, 'Possible' is treated as positive for EuroSCORE II operationalization.", "Endocardite infecciosa ativa ou provável no momento da cirurgia. Neste app, 'Possível' é tratada como positiva na operacionalização do EuroSCORE II."))
        with cx4:
            smoker = st.selectbox(tr("Smoking", "Tabagismo"), [tr("Never", "Nunca"), tr("Current", "Atual"), tr("Former", "Ex-tabagista")], index=0, help=hp("Smoking status. Active or former smoking may reflect pulmonary and vascular risk burden.", "Situação tabágica. Tabagismo atual ou prévio pode refletir maior carga de risco pulmonar e vascular."))
            _smoker_en = str(smoker).strip().lower()
            _is_current = _smoker_en in {"atual", "current"}
            _is_former = _smoker_en in {"ex-tabagista", "former"}
            if _is_current or _is_former:
                smoke_pack_years = st.number_input(
                    tr("Pack-years", "Maços-ano"),
                    min_value=0.0, max_value=300.0, value=20.0, step=5.0, format="%.0f",
                    help=hp("Pack-years = (packs/day) × years of smoking. Leave default if unknown.", "Maços-ano = (maços/dia) × anos de tabagismo. Deixe o padrão se desconhecido."),
                )
            else:
                smoke_pack_years = None
            alcohol = st.selectbox(tr("Alcohol", "Álcool"), yn_options, index=0, help=hp("History of relevant alcohol use.", "História de uso relevante de álcool."))
            recent_pneum = st.selectbox(tr("Recent pneumonia", "Pneumonia recente"), yn_options, index=0, help=hp("Recent pneumonia before surgery, which may indicate active systemic stress or pulmonary compromise.", "Pneumonia recente antes da cirurgia, podendo indicar estresse sistêmico ou comprometimento pulmonar ativo."))
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
            "LVEF, %": lvef,
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
            "HF": yn_pt_to_en(hf),
            "Hypertension": yn_pt_to_en(htn),
            "Dyslipidemia": yn_pt_to_en(dlp),
            "CVA": yn_pt_to_en(cva),
            "PVD": yn_pt_to_en(pvd2),
            "Cancer ≤ 5 yrs": yn_pt_to_en(cancer5),
            "Arrhythmia Remote": yn_pt_to_en(arr_rem),
            "Arrhythmia Recent": yn_pt_to_en(arr_rec),
            "Family Hx of CAD": yn_pt_to_en(fam_cad),
            "Smoking (Pack-year)": str(int(smoke_pack_years)) if _is_current and smoke_pack_years else "Never",
            "Ex-Smoker (Pack-year)": str(int(smoke_pack_years)) if _is_former and smoke_pack_years else "Never",
            "Alcohol": yn_pt_to_en(alcohol),
            "Pneumonia": yn_pt_to_en(recent_pneum),
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

        input_row = _build_input_row(prepared.feature_columns, form_map)
        # Also build from model's feature_columns to catch features not in prepared data
        input_row_model = _build_input_row(artifacts.feature_columns, form_map)
        # Merge: add any columns from model features missing in prepared features
        for _mc in artifacts.feature_columns:
            if _mc not in input_row.columns:
                input_row[_mc] = input_row_model[_mc].values if _mc in input_row_model.columns else np.nan
        input_row = _align_input_to_training_schema(input_row, prepared.data[prepared.feature_columns])
        _num_cols = _get_numeric_columns_from_pipeline(artifacts.fitted_models[forced_model])
        model_input = clean_features(input_row[artifacts.feature_columns], numeric_columns=_num_cols)
        informed_features = int(model_input.notna().sum(axis=1).iloc[0])
        imputed_features = int(model_input.shape[1] - informed_features)
        ia_prob = float(artifacts.fitted_models[forced_model].predict_proba(model_input)[:, 1][0])
        euro_prob = float(euroscore_from_inputs(form_map))
        # Calculate STS via web calculator
        sts_result = {}
        if HAS_STS:
            with st.spinner(tr("Querying STS web calculator...", "Consultando calculadora web do STS...")):
                sts_result = calculate_sts(form_map)
        sts_prob = sts_result.get("predmort", np.nan)

        patient_pred = []
        for model_name in model_options:
            p = float(artifacts.fitted_models[model_name].predict_proba(model_input)[:, 1][0])
            patient_pred.append({tr("Model", "Modelo"): model_name, tr("Probability", "Probabilidade"): p})
        patient_pred_df = pd.DataFrame(patient_pred).sort_values(tr("Probability", "Probabilidade"), ascending=False)
        patient_pred_df[tr("Probability", "Probabilidade")] = patient_pred_df[tr("Probability", "Probabilidade")].map(lambda x: f"{x*100:.2f}%")
        quality_alerts = data_quality_alerts(form_map, prepared)
        likely_range_text, confidence_text = prediction_uncertainty(patient_pred_df, tr("Probability", "Probabilidade"), imputed_features)

        best_model_prob = float(artifacts.fitted_models[best_model_name].predict_proba(model_input)[:, 1][0])
        st.success(
            tr(
                f"Best recommended model for this patient (based on validation): {best_model_name}. Estimated risk: {best_model_prob*100:.2f}%",
                f"Melhor modelo recomendado para este paciente (com base na validação da base): {best_model_name}. Risco estimado por ele: {best_model_prob*100:.2f}%",
            )
        )

        st.markdown(tr("**Risk summary**", "**Resumo do risco**"))
        st.caption(tr(
            f"Primary prediction generated by: **{forced_model}**",
            f"Predição principal gerada por: **{forced_model}**",
        ))
        r1, r2, r3 = st.columns(3)
        r1.metric("\U0001f916 AI Risk", f"{ia_prob*100:.2f}%", _risk_badge(ia_prob), delta_color="off")
        r2.metric("\U0001f4ca EuroSCORE II", f"{euro_prob*100:.2f}%", _risk_badge(euro_prob), delta_color="off")
        r3.metric("\U0001f310 STS", "-" if np.isnan(sts_prob) else f"{sts_prob*100:.2f}%", _risk_badge(sts_prob), delta_color="off")
        st.caption(tr(
            "Individual predictions reflect the model's estimate for this specific combination of variables. Population-level performance (AUC, calibration) does not guarantee accuracy for any single patient. This tool is for research purposes and does not replace clinical judgment.",
            "Predições individuais refletem a estimativa do modelo para esta combinação específica de variáveis. O desempenho populacional (AUC, calibração) não garante acurácia para um paciente isolado. Esta ferramenta é para fins de pesquisa e não substitui o julgamento clínico.",
        ))
        if np.isnan(sts_prob):
            st.caption(tr(
                "STS score unavailable — the web calculator could not be reached or did not return a result for this patient. No dataset fallback is used.",
                "Score STS indisponível — a calculadora web não pôde ser acessada ou não retornou resultado para este paciente. Nenhum fallback do dataset é utilizado.",
            ))

        out = pd.DataFrame(
            {
                tr("Score", "Escore"): ["\U0001f916 AI Risk", "\U0001f4ca EuroSCORE II", "\U0001f310 STS"],
                tr("Probability", "Probabilidade"): [ia_prob, euro_prob, sts_prob],
                tr("Class", "Classe"): [_risk_badge(ia_prob), _risk_badge(euro_prob), _risk_badge(sts_prob)],
            }
        )
        out[tr("Probability", "Probabilidade")] = out[tr("Probability", "Probabilidade")].map(lambda x: "-" if np.isnan(x) else f"{x*100:.2f}%")
        with st.expander(tr("Detailed score table", "Tabela detalhada dos escores"), expanded=False):
            st.dataframe(out, width="stretch", column_config=general_table_column_config("patient_scores"))
            # Show all STS sub-scores if available
            if sts_result:
                st.markdown(tr("**STS Sub-scores (web calculator)**", "**Sub-escores STS (calculadora web)**"))
                sts_rows = []
                for key, label in STS_LABELS.items():
                    val = sts_result.get(key, np.nan)
                    sts_rows.append({
                        tr("STS Endpoint", "Desfecho STS"): label,
                        tr("Value", "Valor"): f"{val*100:.2f}%" if not (isinstance(val, float) and np.isnan(val)) else "-",
                    })
                st.dataframe(pd.DataFrame(sts_rows), width="stretch", column_config=general_table_column_config("sts_subscores"))

        with st.expander(tr("Prediction by each AI model", "Predição por cada modelo de IA"), expanded=False):
            st.info(tr(
                f"The risk summary uses **{forced_model}**, selected as the best-performing model in internal cross-validation (highest AUC). "
                "The table below shows predictions from all candidate models for comparison. Differences are expected — each algorithm learns patterns differently.",
                f"O resumo de risco usa **{forced_model}**, selecionado como o modelo de melhor desempenho na validação cruzada interna (maior AUC). "
                "A tabela abaixo mostra as predições de todos os modelos candidatos para comparação. Diferenças são esperadas — cada algoritmo aprende padrões de forma diferente.",
            ))
            # Highlight the selected model
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

        # --- Input completeness indicator ---
        # Use model_input (post clean_features) but restore values that
        # clean_features wrongly converted to NaN (e.g. categorical "No"→NaN).
        # A feature is "informed" if the user provided a value in form_map.
        _derived = {"cirurgia_combinada", "peso_procedimento", "thoracic_aorta_flag"}
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
        with st.expander(tr("Imputation detail per variable", "Detalhe de imputação por variável"), expanded=False):
            _imp_detail = format_imputation_detail(artifacts.feature_columns, model_input, language)
            st.dataframe(_imp_detail, width="stretch", hide_index=True)

        st.markdown(tr("**Prediction quality**", "**Qualidade da predição**"))
        q1, q2 = st.columns(2)
        with q1:
            st.info(
                tr(
                    f"Model input: {informed_features} features informed, {imputed_features} imputed (detailed echo/lab values not available in the form use training median).",
                    f"Entrada do modelo: {informed_features} variáveis informadas, {imputed_features} imputadas (valores detalhados de eco/lab não disponíveis no formulário usam a mediana do treino).",
                )
            )
        with q2:
            st.info(tr(f"{likely_range_text} {confidence_text}.", f"{likely_range_text} {confidence_text}."))
        with st.expander(tr("What does prediction quality mean?", "O que significa qualidade da predição?"), expanded=False):
            st.markdown(
                tr(
                    """
**Informed vs. imputed variables:** The model uses 61 predictor variables. In the individual form, some variables (detailed echocardiographic measurements, specific lab values) are not available as input fields — for these, the model uses the training dataset median. More imputed variables means less personalized prediction, but the core clinical variables (age, surgery type, renal function, LVEF) are always informed.

**Interquartile risk range (IQR):** Shows the range between the 25th and 75th percentile of predictions across all 8 AI models. A narrow IQR means models agree; a wide IQR means disagreement.

| Agreement | IQR spread | Interpretation |
|:--|:--|:--|
| **High** | < 4 percentage points | Models converge — prediction is robust |
| **Moderate** | 4–10 percentage points | Some disagreement — consider the range, not just the point estimate |
| **Low** | > 10 percentage points | Large disagreement — prediction is uncertain, clinical judgment should prevail |

**Why do models disagree?** Tree-based models (CatBoost, XGBoost, LightGBM) capture non-linear interactions and may predict differently from linear models (Logistic Regression) or neural networks (MLP). For atypical patients, disagreement is more likely.
""",
                    """
**Variáveis informadas vs. imputadas:** O modelo usa 61 variáveis preditoras. No formulário individual, algumas variáveis (medidas ecocardiográficas detalhadas, exames laboratoriais específicos) não estão disponíveis como campos — para estas, o modelo usa a mediana do dataset de treinamento. Mais variáveis imputadas significa predição menos personalizada, mas as variáveis clínicas centrais (idade, tipo de cirurgia, função renal, FEVE) são sempre informadas.

**Faixa interquartil de risco (IQR):** Mostra o intervalo entre o percentil 25 e o percentil 75 das predições dos 8 modelos de IA. IQR estreito significa que os modelos concordam; IQR largo significa discordância.

| Concordância | Amplitude do IQR | Interpretação |
|:--|:--|:--|
| **Alta** | < 4 pontos percentuais | Modelos convergem — predição robusta |
| **Moderada** | 4–10 pontos percentuais | Alguma discordância — considere a faixa, não apenas o valor pontual |
| **Baixa** | > 10 pontos percentuais | Grande discordância — predição incerta, o julgamento clínico deve prevalecer |

**Por que os modelos discordam?** Modelos baseados em árvore (CatBoost, XGBoost, LightGBM) capturam interações não-lineares e podem predizer diferente de modelos lineares (Regressão Logística) ou redes neurais (MLP). Para pacientes atípicos, a discordância é mais provável.
""",
                )
            )

        st.caption(
            tr(
                "Note: EuroSCORE II and STS are calculated automatically. For EuroSCORE II, variables not available in the spreadsheet (e.g., poor mobility, critical preoperative state) are entered manually in this form. STS is obtained via automated interaction with the STS web calculator.",
                "Observação: EuroSCORE II e STS são calculados automaticamente. No EuroSCORE II, variáveis não presentes na planilha (ex.: mobilidade, estado crítico) são informadas manualmente neste formulário. O STS é obtido via interação automatizada com a calculadora web do STS.",
            )
        )

        if quality_alerts:
            with st.expander(tr("Data quality alerts", "Alertas de qualidade do dado"), expanded=True):
                for level, message in quality_alerts:
                    if level == tr("Critical", "Crítico"):
                        st.error(f"{level}: {message}")
                    elif level == tr("Warning", "Atenção"):
                        st.warning(f"{level}: {message}")
                    else:
                        st.info(f"{level}: {message}")
        pos_factors, neg_factors = explain_patient_risk(artifacts, model_input, form_map, top_n=5)
        st.markdown(tr("**Clinical explanation for this patient**", "**Explicação clínica para este paciente**"))
        st.caption(
            tr(
                "This explanation uses the logistic regression reference model as an interpretable layer. It reflects estimated associations with risk, not causal relationships.",
                "Esta explicação usa o modelo de regressão logística como camada interpretável de referência. Ela reflete associações estimadas com o risco, e não relações causais.",
            )
        )
        c_pos, c_neg = st.columns(2)
        with c_pos:
            st.markdown(tr("**Factors associated with higher risk**", "**Fatores associados ao aumento do risco**"))
            if pos_factors.empty:
                st.info(tr("No strong increasing-risk factors were identified by the interpretable layer.", "Nenhum fator forte de aumento de risco foi identificado pela camada interpretável."))
            else:
                st.dataframe(pos_factors, width="stretch", column_config=explain_table_column_config())
        with c_neg:
            st.markdown(tr("**Factors associated with lower risk**", "**Fatores associados à redução do risco**"))
            if neg_factors.empty:
                st.info(tr("No strong lower-risk factors were identified by the interpretable layer.", "Nenhum fator forte de redução de risco foi identificado pela camada interpretável."))
            else:
                st.dataframe(neg_factors, width="stretch", column_config=explain_table_column_config())

        with st.expander(tr("Why this prediction? (SHAP — actual model)", "Por que essa predição? (SHAP — modelo real)"), expanded=False):
            st.caption(
                tr(
                    "Unlike the interpretable layer above (logistic regression proxy), this explanation uses SHAP values computed directly from the selected model. Each row shows how that variable pushed the predicted probability up or down for this specific patient.",
                    "Ao contrário da camada interpretável acima (proxy de regressão logística), esta explicação usa valores SHAP calculados diretamente do modelo selecionado. Cada linha mostra quanto aquela variável empurrou a probabilidade prevista para cima ou para baixo neste paciente específico.",
                )
            )
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

        # --- Clinical explanation text ---
        st.markdown(tr("**Clinical summary (auto-generated)**", "**Resumo clínico (gerado automaticamente)**"))
        _clinical_text = generate_clinical_explanation(pos_factors, neg_factors, ia_prob, language)
        st.info(_clinical_text)

        # --- Individual report export ---
        st.divider()
        st.markdown(tr("**Export individual report**", "**Exportar relatório individual**"))
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
            model_version=MODEL_VERSION,
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
                st.download_button(
                    tr("Download PDF", "Baixar PDF"),
                    _rpt_pdf,
                    f"report_{_patient_id}.pdf",
                    mime="application/pdf",
                    key="dl_report_pdf",
                )
        with _rpt_c3:
            _rpt_xlsx = statistical_summary_to_xlsx(_report_text)
            if _rpt_xlsx:
                st.download_button(
                    tr("Download XLSX", "Baixar XLSX"),
                    _rpt_xlsx,
                    f"report_{_patient_id}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_report_xlsx",
                )
        with _rpt_c4:
            _rpt_csv = statistical_summary_to_csv(_report_text)
            if _rpt_csv:
                st.download_button(
                    tr("Download CSV", "Baixar CSV"),
                    _rpt_csv,
                    f"report_{_patient_id}.csv",
                    mime="text/csv",
                    key="dl_report_csv",
                )

        # --- Audit trail logging ---
        log_analysis(
            analysis_type="individual_prediction",
            source_file="manual_form",
            model_version=MODEL_VERSION,
            n_patients=1,
            n_imputed=_completeness["n_imputed"],
            completeness_level=_completeness["level"],
            sts_method="websocket" if sts_result else "unavailable",
            extra={"patient_id": _patient_id, "ia_risk": round(ia_prob, 4)},
        )

elif _active_tab == 2:  # Statistical Comparison
    st.subheader(tr("Statistical performance comparison", "Comparação estatística de desempenho"))
    st.caption(tr("Report with 95% CI by bootstrap (2,000 resamples) and formal model comparison.", "Relatório com IC95% por bootstrap (2.000 reamostras) e comparação formal entre modelos."))

    decision_threshold = st.slider(
        tr("Decision threshold", "Limiar de decisão"),
        min_value=0.01,
        max_value=0.99,
        value=_default_threshold,
        step=0.01,
    )
    with st.expander(tr("What is the decision threshold?", "O que é o limiar de decisão?"), expanded=False):
        st.markdown(
            tr(
                f"""
The decision threshold converts a predicted probability into a binary decision: **positive** (high risk) or **negative** (low risk).

- If the predicted risk is **>= {decision_threshold:.0%}**, the patient is classified as **positive** (high risk).
- If the predicted risk is **< {decision_threshold:.0%}**, the patient is classified as **negative** (low risk).

**How it affects the metrics:**

| Lower threshold | Higher threshold |
|:-|:-|
| More patients classified as positive | Fewer patients classified as positive |
| Higher sensitivity (catches more events) | Lower sensitivity (misses more events) |
| Lower specificity (more false alarms) | Higher specificity (fewer false alarms) |

The default value ({_default_threshold:.0%}) is a **conservative** threshold for cardiac surgery — it favors higher sensitivity (detecting more at-risk patients) at the cost of more false positives. You can adjust it using the slider. AUC, AUPRC, and Brier score are **not affected** by the threshold — they evaluate the full probability distribution.

**Why {_default_threshold:.0%}?** In cardiac surgery, the cost of missing a high-risk patient (false negative) far outweighs the cost of an unnecessary alert (false positive). A missed at-risk patient may die without adequate preparation; an unnecessary alert only means the team prepares more carefully — causing no harm. The {_default_threshold:.0%} value sits just above the average mortality rate in cardiac surgery (3–8% globally), which means it does not classify most patients as high-risk, but is low enough to capture patients at real risk before it becomes clinically obvious. This is consistent with EuroSCORE II risk stratification thresholds (low <3%, intermediate 3–8%, high >8%).
""",
                f"""
O limiar de decisão converte uma probabilidade predita em uma decisão binária: **positivo** (alto risco) ou **negativo** (baixo risco).

- Se o risco predito é **>= {decision_threshold:.0%}**, o paciente é classificado como **positivo** (alto risco).
- Se o risco predito é **< {decision_threshold:.0%}**, o paciente é classificado como **negativo** (baixo risco).

**Como o limiar afeta as métricas:**

| Limiar mais baixo | Limiar mais alto |
|:-|:-|
| Mais pacientes classificados como positivos | Menos pacientes classificados como positivos |
| Maior sensibilidade (detecta mais eventos) | Menor sensibilidade (perde mais eventos) |
| Menor especificidade (mais falsos alarmes) | Maior especificidade (menos falsos alarmes) |

O valor padrão ({_default_threshold:.0%}) é um limiar **conservador** para cirurgia cardíaca — favorece maior sensibilidade (detectar mais pacientes em risco) ao custo de mais falsos positivos. AUC, AUPRC e Brier score **não são afetados** pelo limiar — eles avaliam a distribuição completa de probabilidades.

**Por que {_default_threshold:.0%}?** Em cirurgia cardíaca, o custo de não identificar um paciente de alto risco (falso negativo) é muito maior que o custo de um alerta desnecessário (falso positivo). Um paciente em risco não identificado pode evoluir a óbito sem preparo adequado da equipe; um alerta desnecessário apenas faz a equipe se preparar mais — sem causar dano. O valor de {_default_threshold:.0%} está logo acima da mortalidade média em cirurgia cardíaca (3–8% mundialmente), o que significa que não classifica a maioria dos pacientes como alto risco, mas é baixo o suficiente para capturar pacientes em risco real antes que isso se torne clinicamente óbvio. Isso é consistente com a estratificação do EuroSCORE II (baixo <3%, intermediário 3–8%, alto >8%).
""",
            )
        )
    with st.expander(tr("How to read this section", "Como ler esta seção"), expanded=False):
        st.write(
            tr(
                "The main analysis is the fair triple comparison, where AI Risk, EuroSCORE II, and STS are evaluated in the same patients. Pairwise comparisons are complementary and use larger samples when one of the three scores is unavailable. Threshold-dependent metrics (sensitivity, specificity, PPV, NPV) change when the decision threshold changes. Calibration metrics evaluate agreement between predicted and observed risk, whereas DCA evaluates clinical usefulness across thresholds.",
                "A análise principal é a comparação tripla justa, em que AI Risk, EuroSCORE II e STS são avaliados nos mesmos pacientes. As comparações pareadas são complementares e usam amostras maiores quando um dos três escores está ausente. Métricas dependentes do limiar (sensibilidade, especificidade, PPV, NPV) mudam quando o limiar de decisão muda. Métricas de calibração avaliam a concordância entre risco previsto e observado, enquanto a DCA avalia utilidade clínica ao longo dos limiares.",
            )
        )

    metrics_all = evaluate_scores(
        df,
        y_col="morte_30d",
        score_cols=["ia_risk_oof", "euroscore_calc", "sts_score"],
    )
    _score_rename = {"ia_risk_oof": "AI Risk", "euroscore_calc": "EuroSCORE II", "sts_score": "STS"}
    if not metrics_all.empty:
        metrics_all["Score"] = metrics_all["Score"].replace(_score_rename)
    st.markdown(tr("**Overall comparison (each score with all available patients)**", "**Comparação geral (cada escore com todos os pacientes disponíveis)**"))
    st.caption(
        tr(
            "This table uses all available observations for each score separately. It is useful for a broad overview, but the fairest head-to-head comparison is the triple analysis below.",
            "Esta tabela usa todas as observações disponíveis para cada escore separadamente. É útil para uma visão ampla, mas a comparação mais justa entre os três está na análise tripla abaixo.",
        )
    )
    st.dataframe(metrics_all, width="stretch", column_config=stats_table_column_config("overall"))

    triple = df[["morte_30d", "ia_risk_oof", "euroscore_calc", "sts_score"]].dropna()
    st.markdown(tr("**Fair triple comparison (same patients for all 3 scores)**", "**Comparação tripla justa (mesmos pacientes para os 3 escores)**"))
    st.write(f"n = {len(triple)}")
    st.caption(
        tr(
            "This is the main comparison because AI Risk, EuroSCORE II, and STS are evaluated in exactly the same patients.",
            "Esta é a comparação principal porque AI Risk, EuroSCORE II e STS são avaliados exatamente nos mesmos pacientes.",
        )
    )
    triple_ci = pd.DataFrame()
    threshold_metrics = pd.DataFrame()
    if len(triple) >= 30 and triple["morte_30d"].nunique() > 1:
        threshold_metrics = evaluate_scores_with_threshold(
            triple,
            y_col="morte_30d",
            score_cols=["ia_risk_oof", "euroscore_calc", "sts_score"],
            threshold=decision_threshold,
        )
        triple_metrics = evaluate_scores(
            triple,
            y_col="morte_30d",
            score_cols=["ia_risk_oof", "euroscore_calc", "sts_score"],
        )
        if not triple_metrics.empty:
            triple_metrics["Score"] = triple_metrics["Score"].replace(_score_rename)
        st.dataframe(triple_metrics, width="stretch", column_config=stats_table_column_config("overall"))
        if not threshold_metrics.empty:
            threshold_metrics["Score"] = threshold_metrics["Score"].map(
                {"ia_risk_oof": "AI Risk", "euroscore_calc": "EuroSCORE II", "sts_score": "STS"}
            )
            st.markdown(tr("**Threshold-based classification metrics**", "**Métricas de classificação por limiar**"))
            st.caption(
                tr(
                    "PPV and NPV depend strongly on event prevalence and on the selected decision threshold.",
                    "PPV e NPV dependem fortemente da prevalência do evento e do limiar de decisão selecionado.",
                )
            )
            st.dataframe(_format_ppv_npv(threshold_metrics), width="stretch", column_config=stats_table_column_config("overall"))

        st.markdown(tr("**95% CI report (triple comparison, same sample)**", "**Relatório com IC95% (comparação tripla, mesma amostra)**"))
        triple_ci = evaluate_scores_with_ci(
            triple,
            y_col="morte_30d",
            score_cols=["ia_risk_oof", "euroscore_calc", "sts_score"],
            n_boot=2000,
            seed=42,
        )
        score_label_ci = {
            "ia_risk_oof": "AI Risk",
            "euroscore_calc": "EuroSCORE II",
            "sts_score": "STS",
        }
        if not triple_ci.empty:
            triple_ci["Score"] = triple_ci["Score"].map(score_label_ci)
            st.dataframe(triple_ci, width="stretch", column_config=stats_table_column_config("overall"))

            _csv_download_btn(triple_ci, "relatorio_ic95_modelos.csv", tr("Download 95% CI report (CSV)", "Baixar relatório IC95% (CSV)"))

        calib_rows = []
        for label, col in [("AI Risk", "ia_risk_oof"), ("EuroSCORE II", "euroscore_calc"), ("STS", "sts_score")]:
            ci_vals = calibration_intercept_slope(triple["morte_30d"].values, triple[col].values)
            hl_vals = hosmer_lemeshow_test(triple["morte_30d"].values, triple[col].values)
            calib_rows.append({"Score": label, **ci_vals, **hl_vals})
        calib_df = pd.DataFrame(calib_rows)
        st.markdown(tr("**Advanced calibration metrics**", "**Métricas avançadas de calibração**"))
        st.caption(
            tr(
                "Brier measures probabilistic accuracy. Calibration-in-the-large (intercept) close to 0 and slope close to 1 are desirable. Hosmer-Lemeshow should be interpreted as complementary, not in isolation.",
                "O Brier mede a acurácia probabilística. Calibration-in-the-large (intercepto) próximo de 0 e slope próximo de 1 são desejáveis. O teste de Hosmer-Lemeshow deve ser interpretado como complementar, e não isoladamente.",
            )
        )
        st.dataframe(calib_df, width="stretch", column_config=stats_table_column_config("calibration"))

        scores_plot = {
            "AI Risk": triple["ia_risk_oof"].values,
            "EuroSCORE II": triple["euroscore_calc"].values,
            "STS": triple["sts_score"].values,
        }
        box_df = pd.DataFrame(
            {
                "Outcome": triple["morte_30d"].map(lambda x: tr("Death within 30 days", "Óbito em 30 dias") if x == 1 else tr("No death within 30 days", "Sem óbito em 30 dias")),
                "AI Risk": triple["ia_risk_oof"].values,
                "EuroSCORE II": triple["euroscore_calc"].values,
                "STS": triple["sts_score"].values,
            }
        )
        p1, p2 = st.columns(2)
        with p1:
            _plot_roc(scores_plot, triple["morte_30d"].values)
        with p2:
            _plot_calibration(scores_plot, triple["morte_30d"].values)
        st.markdown(tr("**Boxplots of predicted probabilities by outcome**", "**Boxplots das probabilidades preditas por desfecho**"))
        _plot_boxplots(box_df)

        st.markdown(tr("**Boxplots for each AI model**", "**Boxplots de cada modelo de IA**"))
        _plot_ia_model_boxplots(df["morte_30d"].values, artifacts.oof_predictions)
    else:
        st.warning(tr("Insufficient sample for complete triple comparison.", "Amostra insuficiente para comparação tripla completa."))

    st.markdown(tr("**Pairwise comparisons (larger sample)**", "**Comparações por pares (amostra maior)**"))
    st.caption(
        tr(
            "Pairwise analyses use more patients when one of the three scores is missing. They are complementary to the triple main analysis.",
            "As análises pareadas usam mais pacientes quando um dos três escores está ausente. Elas são complementares à análise tripla principal.",
        )
    )
    pair_rows = []
    score_label = {
        "ia_risk_oof": "AI Risk",
        "euroscore_calc": "EuroSCORE II",
        "sts_score": "STS",
    }
    for a, b in [("ia_risk_oof", "euroscore_calc"), ("ia_risk_oof", "sts_score"), ("euroscore_calc", "sts_score")]:
        sub = df[["morte_30d", a, b]].dropna()
        if len(sub) < 30 or sub["morte_30d"].nunique() < 2:
            continue
        boot = bootstrap_auc_diff(sub["morte_30d"].values, sub[a].values, sub[b].values)
        pair_rows.append(
            {
                tr("Comparison", "Comparação"): f"{score_label[a]} vs {score_label[b]}",
                "n": len(sub),
                "Delta AUC (A-B)": boot["delta_auc"],
                tr("95% CI low", "IC95% inf"): boot["ci_low"],
                tr("95% CI high", "IC95% sup"): boot["ci_high"],
                "p (bootstrap)": boot["p"],
            }
        )
    st.dataframe(pd.DataFrame(pair_rows), width="stretch", column_config=stats_table_column_config("comparison"))

    st.markdown(tr("**Formal pairwise comparison (triple sample only, same cohort)**", "**Comparação formal por pares (apenas amostra tripla, mesma coorte)**"))
    st.caption(
        tr(
            "These comparisons are restricted to the same triple cohort, which is the correct setting for direct statistical comparison between models.",
            "Essas comparações são restritas à mesma coorte tripla, que é o cenário correto para comparação estatística direta entre os modelos.",
        )
    )
    formal_rows = []
    if len(triple) >= 30 and triple["morte_30d"].nunique() > 1:
        pairs = [
            ("ia_risk_oof", "euroscore_calc"),
            ("ia_risk_oof", "sts_score"),
            ("euroscore_calc", "sts_score"),
        ]
        for a, b in pairs:
            boot = bootstrap_auc_diff(
                triple["morte_30d"].values,
                triple[a].values,
                triple[b].values,
                n_boot=2000,
                seed=42,
            )
            formal_rows.append(
                {
                    tr("Comparison", "Comparação"): f"{score_label[a]} vs {score_label[b]}",
                    "n": len(triple),
                    "Delta AUC (A-B)": boot["delta_auc"],
                    tr("95% CI low", "IC95% inf"): boot["ci_low"],
                    tr("95% CI high", "IC95% sup"): boot["ci_high"],
                    "p (bootstrap)": boot["p"],
                }
            )

    formal_df = pd.DataFrame(formal_rows)
    st.dataframe(formal_df, width="stretch", column_config=stats_table_column_config("comparison"))
    if not formal_df.empty:
        _csv_download_btn(formal_df, "comparacao_formal_modelos.csv", tr("Download formal comparison (CSV)", "Baixar comparação formal (CSV)"))

    st.markdown(tr("**DeLong test (same triple cohort)**", "**Teste de DeLong (mesma coorte tripla)**"))
    st.caption(
        tr(
            "DeLong formally compares correlated AUCs in the same patients. It complements the bootstrap-based delta AUC analysis.",
            "O teste de DeLong compara formalmente AUCs correlacionadas nos mesmos pacientes. Ele complementa a análise de delta AUC por bootstrap.",
        )
    )
    delong_rows = []
    if len(triple) >= 30 and triple["morte_30d"].nunique() > 1:
        for a, b in [("ia_risk_oof", "euroscore_calc"), ("ia_risk_oof", "sts_score"), ("euroscore_calc", "sts_score")]:
            dtest = delong_roc_test(triple["morte_30d"].values, triple[a].values, triple[b].values)
            delong_rows.append(
                {
                    tr("Comparison", "Comparação"): f"{score_label[a]} vs {score_label[b]}",
                    "AUC 1": dtest["AUC_1"],
                    "AUC 2": dtest["AUC_2"],
                    "Delta AUC": dtest["delta_auc"],
                    "z": dtest["z"],
                    "p (DeLong)": dtest["p"],
                }
            )
    delong_df = pd.DataFrame(delong_rows)
    st.dataframe(delong_df, width="stretch", column_config=stats_table_column_config("comparison"))

    st.markdown(tr("**Decision curve analysis (DCA)**", "**Decision curve analysis (DCA)**"))
    st.caption(
        tr(
            "DCA evaluates clinical usefulness. Higher net benefit means a model is more useful for decision-making at that risk threshold.",
            "A DCA avalia utilidade clínica. Benefício líquido mais alto significa que o modelo é mais útil para tomada de decisão naquele limiar de risco.",
        )
    )
    best_dca_model = None
    dca_label = tr("N/A", "N/D")
    if len(triple) >= 30 and triple["morte_30d"].nunique() > 1:
        thresholds = np.linspace(0.05, 0.20, 16)
        dca_df = decision_curve(
            triple["morte_30d"].values,
            {
                "AI Risk": triple["ia_risk_oof"].values,
                "EuroSCORE II": triple["euroscore_calc"].values,
                "STS": triple["sts_score"].values,
            },
            thresholds,
        )
        _plot_dca(dca_df)
        dca_summary = dca_df[dca_df["Threshold"].isin([0.05, 0.10, 0.15, 0.20])].copy()
        st.dataframe(dca_summary, width="stretch", column_config=stats_table_column_config("dca"))

        model_only = dca_df[dca_df["Strategy"].isin(["AI Risk", "EuroSCORE II", "STS"])].copy()
        avg_nb = (
            model_only.groupby("Strategy", observed=True)["Net Benefit"]
            .mean()
            .sort_values(ascending=False)
        )
        if not avg_nb.empty:
            best_dca_model = avg_nb.index[0]
            dca_label = str(best_dca_model)
            best_dca_value = float(avg_nb.iloc[0])
            st.info(
                tr(
                    f"Between 5% and 20% risk thresholds, the model with the highest average net benefit is {dca_label} (mean net benefit = {best_dca_value:.4f}).",
                    f"Entre os limiares de risco de 5% a 20%, o modelo com maior benefício líquido médio é o {dca_label} (benefício líquido médio = {best_dca_value:.4f}).",
                )
            )
    else:
        st.info(tr("DCA is unavailable because the triple comparison sample is insufficient.", "A DCA não está disponível porque a amostra da comparação tripla é insuficiente."))

    st.markdown(tr("**Reclassification (NRI / IDI)**", "**Reclassificação (NRI / IDI)**"))
    st.caption(
        tr(
            "NRI evaluates whether the new model moves patients to more appropriate risk categories (low <5%, intermediate 5–15%, high >15%). IDI evaluates average improvement in separation between events and non-events. Both are complementary metrics — they should not be used as the sole evidence of model superiority.",
            "O NRI avalia se o novo modelo move os pacientes para categorias de risco mais apropriadas (baixo <5%, intermediário 5–15%, alto >15%). O IDI avalia a melhora média da separação entre eventos e não eventos. Ambas são métricas complementares — não devem ser usadas como única evidência de superioridade de um modelo.",
        )
    )
    reclass_df = pd.DataFrame()
    if len(triple) >= 30 and triple["morte_30d"].nunique() > 1:
        reclass_rows = []
        for new_name, new_col, old_name, old_col in [
            ("AI Risk", "ia_risk_oof", "EuroSCORE II", "euroscore_calc"),
            ("AI Risk", "ia_risk_oof", "STS", "sts_score"),
        ]:
            nri = compute_nri(triple["morte_30d"].values, triple[old_col].values, triple[new_col].values, cutoffs=(0.05, 0.15))
            idi = compute_idi(triple["morte_30d"].values, triple[old_col].values, triple[new_col].values)
            reclass_rows.append(
                {
                    tr("Comparison", "Comparação"): f"{new_name} vs {old_name}",
                    "NRI events": nri["NRI events"],
                    "NRI non-events": nri["NRI non-events"],
                    "NRI total": nri["NRI total"],
                    "IDI": idi["IDI"],
                }
            )
        reclass_df = pd.DataFrame(reclass_rows)
        st.dataframe(reclass_df, width="stretch", column_config=stats_table_column_config("reclass"))
        if not reclass_df.empty:
            best_nri = reclass_df.sort_values("NRI total", ascending=False).iloc[0]
            best_idi = reclass_df.sort_values("IDI", ascending=False).iloc[0]
            st.info(
                tr(
                    f"The highest NRI was observed for {best_nri[tr('Comparison','Comparação')]} (NRI total = {best_nri['NRI total']:.3f}). The highest IDI was observed for {best_idi[tr('Comparison','Comparação')]} (IDI = {best_idi['IDI']:.3f}). These are complementary reclassification metrics and should be interpreted alongside discrimination and calibration results.",
                    f"O maior NRI foi observado em {best_nri[tr('Comparison','Comparação')]} (NRI total = {best_nri['NRI total']:.3f}). O maior IDI foi observado em {best_idi[tr('Comparison','Comparação')]} (IDI = {best_idi['IDI']:.3f}). Essas são métricas complementares de reclassificação e devem ser interpretadas em conjunto com os resultados de discriminação e calibração.",
                )
            )
    else:
        st.info(tr("NRI/IDI are unavailable because the triple comparison sample is insufficient.", "NRI/IDI não estão disponíveis porque a amostra da comparação tripla é insuficiente."))

    st.markdown(tr("**Clinical interpretation**", "**Interpretação clínica**"))
    if len(triple) >= 30 and triple["morte_30d"].nunique() > 1:
        same_sample_rows = []
        for label, col in [("AI Risk", "ia_risk_oof"), ("EuroSCORE II", "euroscore_calc"), ("STS", "sts_score")]:
            y = triple["morte_30d"].values
            p = triple[col].values
            pred = (p >= decision_threshold).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            sens = float(tp / (tp + fn)) if (tp + fn) else np.nan
            spec = float(tn / (tn + fp)) if (tn + fp) else np.nan
            same_sample_rows.append({"Score": label, "Sensitivity": sens, "Specificity": spec})

        same_sample_df = pd.DataFrame(same_sample_rows)
        best_auc = triple_ci.sort_values("AUC", ascending=False).iloc[0]["Score"] if not triple_ci.empty else None
        best_brier = triple_ci.sort_values("Brier", ascending=True).iloc[0]["Score"] if not triple_ci.empty else None
        best_sens = same_sample_df.sort_values("Sensitivity", ascending=False).iloc[0]["Score"]
        best_spec = same_sample_df.sort_values("Specificity", ascending=False).iloc[0]["Score"]
        best_ppv = threshold_metrics.sort_values("PPV", ascending=False).iloc[0]["Score"] if not threshold_metrics.empty else None
        best_npv = threshold_metrics.sort_values("NPV", ascending=False).iloc[0]["Score"] if not threshold_metrics.empty else None

        interp_text = tr(
            f"On the same comparable sample (triple cohort), the best discrimination (AUC) was observed for {best_auc}. "
            f"The best calibration (Brier score) was observed for {best_brier}. "
            f"At the selected threshold, the highest sensitivity was observed for {best_sens} and the highest specificity for {best_spec}. "
            f"The highest PPV was observed for {best_ppv}, the highest NPV for {best_npv}, and the highest average net benefit (5–20%) for {dca_label}.",
            f"Na mesma amostra comparável (coorte tripla), a melhor discriminação (AUC) foi observada em {best_auc}. "
            f"A melhor calibração (Brier score) foi observada em {best_brier}. "
            f"No limiar selecionado, a maior sensibilidade foi observada em {best_sens} e a maior especificidade em {best_spec}. "
            f"O maior VPP foi observado em {best_ppv}, o maior VPN em {best_npv}, e o maior benefício líquido médio (5–20%) em {dca_label}."
        )
        st.info(interp_text)
    else:
        st.info(tr("Clinical interpretation is unavailable because the triple comparison sample is insufficient.", "A interpretação clínica não está disponível porque a amostra da comparação tripla é insuficiente."))

    st.markdown(tr("**Results**", "**Resultados**"))
    if not triple_ci.empty:
        tri_sorted = triple_ci.sort_values("AUC", ascending=False).reset_index(drop=True)
        top = tri_sorted.iloc[0]
        ia_row = tri_sorted[tri_sorted["Score"] == "AI Risk"]
        euro_row = tri_sorted[tri_sorted["Score"] == "EuroSCORE II"]
        sts_row = tri_sorted[tri_sorted["Score"] == "STS"]

        def _fmt_auc(r):
            return f"{r['AUC']:.3f} (IC95% {r['AUC_IC95_inf']:.3f}-{r['AUC_IC95_sup']:.3f})"

        auc_ia = _fmt_auc(ia_row.iloc[0]) if not ia_row.empty else "N/A"
        auc_euro = _fmt_auc(euro_row.iloc[0]) if not euro_row.empty else "N/A"
        auc_sts = _fmt_auc(sts_row.iloc[0]) if not sts_row.empty else "N/A"

        sig_text = ""
        if not formal_df.empty:
            sig_parts = []
            for _, r in formal_df.iterrows():
                pval = r["p (bootstrap)"]
                sig = tr("statistically significant difference", "diferença estatisticamente significativa") if pd.notna(pval) and pval < 0.05 else tr("no statistically significant difference", "sem diferença estatisticamente significativa")
                comp_col = tr("Comparison", "Comparação")
                ci_lo_col = tr("95% CI low", "IC95% inf")
                ci_hi_col = tr("95% CI high", "IC95% sup")
                sig_parts.append(
                    tr(
                        f"{r[comp_col]} showed ΔAUC={r['Delta AUC (A-B)']:.3f} (95% CI {r[ci_lo_col]:.3f}-{r[ci_hi_col]:.3f}; p={pval:.3f}), {sig}",
                        f"{r[comp_col]} apresentou ΔAUC={r['Delta AUC (A-B)']:.3f} (IC95% {r[ci_lo_col]:.3f}-{r[ci_hi_col]:.3f}; p={pval:.3f}), {sig}",
                    )
                )
            sig_text = "; ".join(sig_parts) + "."

        formal_summary_text = sig_text if sig_text else tr("No statistically significant differences were observed in formal ROC comparison.", "Não foram observadas diferenças estatisticamente significativas na comparação formal das curvas ROC.")
        reclass_summary_text = (
            tr("Reclassification analyses were not available.", "As análises de reclassificação não estavam disponíveis.")
            if reclass_df.empty
            else tr(
                f"The highest total NRI was observed for {reclass_df.sort_values('NRI total', ascending=False).iloc[0][tr('Comparison','Comparação')]} and the highest IDI for {reclass_df.sort_values('IDI', ascending=False).iloc[0][tr('Comparison','Comparação')]}",
                f"O maior NRI total foi observado em {reclass_df.sort_values('NRI total', ascending=False).iloc[0][tr('Comparison','Comparação')]} e o maior IDI em {reclass_df.sort_values('IDI', ascending=False).iloc[0][tr('Comparison','Comparação')]}",
            )
        )

        methods_mode = st.radio(
            tr("Methods text format", "Formato do texto de métodos"),
            [tr("Short", "Curto"), tr("Detailed", "Detalhado")],
            horizontal=True,
            key="methods_mode",
        )
        st.markdown(tr("**Statistical Methods**", "**Métodos estatísticos**"))
        st.text_area(
            tr("Methods for manuscript", "Texto para artigo - Métodos"),
            value=build_methods_text(methods_mode),
            height=220,
        )

        results_mode = st.radio(
            tr("Results text format", "Formato do texto de resultados"),
            [tr("Short", "Curto"), tr("Detailed", "Detalhado")],
            horizontal=True,
            key="results_mode",
        )

        results_context = {
            "n_triple": len(triple),
            "threshold": decision_threshold,
            "best_auc_model": top["Score"],
            "best_auc": float(top["AUC"]),
            "best_brier_model": best_brier,
            "best_sens_model": best_sens,
            "best_spec_model": best_spec,
            "best_ppv_model": best_ppv,
            "best_npv_model": best_npv,
            "best_dca_model": dca_label,
            "formal_summary": formal_summary_text,
            "reclass_summary": reclass_summary_text,
        }

        resultados_txt = build_results_text(results_mode, results_context)
        st.text_area(tr("Manuscript-ready text", "Texto para manuscrito"), value=resultados_txt, height=220)
        _txt_download_btn(resultados_txt, "results_for_manuscript.txt", tr("Download Results text (.txt)", "Baixar texto de Resultados (.txt)"))
    else:
        st.info(tr("Triple sample size was insufficient to generate automatic results text with 95% CI.", "A amostra tripla foi insuficiente para gerar texto automático de resultados com IC95%."))

    # ── Statistical summary export (Task 10) ──
    st.divider()
    st.subheader(tr("Export full statistical summary", "Exportar resumo estatístico completo"))
    st.caption(tr(
        "Generates a single Markdown document with all statistical tables (discrimination, calibration, DeLong, NRI/IDI).",
        "Gera um documento Markdown único com todas as tabelas estatísticas (discriminação, calibração, DeLong, NRI/IDI).",
    ))
    _stat_summary = build_statistical_summary(
        triple_ci=triple_ci,
        calib_df=calib_df if 'calib_df' in locals() else pd.DataFrame(),
        formal_df=formal_df if 'formal_df' in locals() else pd.DataFrame(),
        delong_df=delong_df if 'delong_df' in locals() else pd.DataFrame(),
        reclass_df=reclass_df if 'reclass_df' in locals() else pd.DataFrame(),
        threshold=decision_threshold,
        threshold_metrics=threshold_metrics,
        n_triple=len(triple) if 'triple' in locals() else 0,
        model_version=MODEL_VERSION,
        language=language,
    )
    with st.expander(tr("Preview summary", "Pré-visualizar resumo"), expanded=False):
        st.markdown(_stat_summary)

    _exp_col1, _exp_col2, _exp_col3, _exp_col4 = st.columns(4)
    with _exp_col1:
        _pdf_bytes = statistical_summary_to_pdf(_stat_summary)
        if _pdf_bytes:
            st.download_button(
                label="📄 PDF",
                data=_pdf_bytes,
                file_name="statistical_summary.pdf",
                mime="application/pdf",
                width="stretch",
            )
        else:
            st.caption(tr("PDF unavailable (install fpdf2)", "PDF indisponível (instale fpdf2)"))
    with _exp_col2:
        st.download_button(
            label="📊 XLSX",
            data=statistical_summary_to_xlsx(_stat_summary),
            file_name="statistical_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    with _exp_col3:
        st.download_button(
            label="📋 CSV",
            data=statistical_summary_to_csv(_stat_summary),
            file_name="statistical_summary.csv",
            mime="text/csv",
            width="stretch",
        )
    with _exp_col4:
        st.download_button(
            label="📝 Markdown",
            data=_stat_summary,
            file_name="statistical_summary.md",
            mime="text/markdown",
            width="stretch",
        )

elif _active_tab == 3:  # Analysis Guide
    render_analysis_guide(prepared, artifacts, len(triple) if 'triple' in locals() else None)

elif _active_tab == 4:  # Batch & Export
    st.subheader(tr("Batch export with scores", "Exportação da base com escores"))

    export_df = df.copy()

    # Add OOF predictions from ALL models (for research)
    for _model_name, _oof_probs in artifacts.oof_predictions.items():
        export_df[f"oof_{_model_name}"] = _oof_probs

    export_df["classe_ia"] = export_df["ia_risk_oof"].map(class_risk)
    export_df["classe_euro"] = export_df["euroscore_calc"].map(class_risk)
    export_df["classe_sts"] = export_df["sts_score"].map(lambda x: class_risk(x) if pd.notna(x) else np.nan)

    _all_oof_cols = [f"oof_{m}" for m in sorted(artifacts.oof_predictions.keys())]
    _show_all_oof = st.checkbox(
        tr("Show OOF predictions from all AI models (research)", "Mostrar predições OOF de todos os modelos de IA (pesquisa)"),
        value=False,
        key="export_show_all_oof",
    )

    cols_show = [
        "Name",
        "Surgery",
        "morte_30d",
        "ia_risk_oof",
    ]
    if _show_all_oof:
        cols_show += _all_oof_cols
    cols_show += [
        "euroscore_calc",
        "sts_score",
        "classe_ia",
        "classe_euro",
        "classe_sts",
    ]
    cols_show = [c for c in cols_show if c in export_df.columns]
    st.dataframe(export_df[cols_show], width="stretch", column_config=general_table_column_config("export"))

    # Download always includes all models
    _csv_download_btn(export_df, "ia_risk_resultados.csv", tr("Download results (CSV)", "Baixar resultados (CSV)"))
    _xlsx_export_buf = BytesIO()
    export_df.to_excel(_xlsx_export_buf, index=False, engine="openpyxl")
    st.download_button(
        tr("Download results (XLSX)", "Baixar resultados (XLSX)"),
        _xlsx_export_buf.getvalue(),
        "ia_risk_resultados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_export_xlsx",
    )

    st.caption(
        tr(
            "Note: All scores are calculated by the app — not read from the input file. EuroSCORE II is computed from the published logistic equation (Nashef et al., 2012). STS is obtained via automated query to the STS web calculator.",
            "Nota: Todos os escores são calculados pelo app — não lidos do arquivo de entrada. EuroSCORE II é calculado pela equação logística publicada (Nashef et al., 2012). STS é obtido via consulta automatizada à calculadora web do STS.",
        )
    )

    # --- Batch prediction for NEW patients ---
    st.divider()
    st.subheader(tr("Predict new patients (batch)", "Predição de novos pacientes (lote)"))
    st.caption(
        tr(
            "Upload a CSV or Excel file with the same clinical variables used in training. "
            "Each row will receive AI Risk, EuroSCORE II, and STS predictions. "
            "Outcome column (morte_30d) is NOT required.",
            "Faça upload de um arquivo CSV ou Excel com as mesmas variáveis clínicas usadas no treinamento. "
            "Cada linha receberá predições de AI Risk, EuroSCORE II e STS. "
            "A coluna de desfecho (morte_30d) NÃO é necessária.",
        )
    )
    st.info(
        tr(
            "**Note:** This tab uses the final model (trained on all data) to predict new patients. "
            "If you upload patients that were already in the training dataset, the AI Risk values may differ slightly "
            "from those in the Statistical Analysis tab, which uses out-of-fold (OOF) predictions — where each patient "
            "is predicted by a model that never saw that patient. For patients in the training dataset, the OOF values "
            "(Statistical Analysis tab) are the methodologically correct reference. This tab is intended for **new patients**.",
            "**Nota:** Esta aba usa o modelo final (treinado com todos os dados) para predizer novos pacientes. "
            "Se você enviar pacientes que já estavam no dataset de treinamento, os valores de AI Risk podem diferir "
            "ligeiramente dos apresentados na aba de Análise Estatística, que usa predições out-of-fold (OOF) — onde cada "
            "paciente é predito por um modelo que nunca viu aquele paciente. Para pacientes do dataset de treinamento, os "
            "valores OOF (aba Análise Estatística) são a referência metodologicamente correta. Esta aba é destinada a **novos pacientes**.",
        )
    )
    batch_file = st.file_uploader(
        tr("Upload patient file", "Upload do arquivo de pacientes"),
        type=["csv", "xlsx", "xls"],
        key="batch_new_patients",
    )
    if batch_file is not None:
        try:
            if batch_file.name.endswith(".csv"):
                try:
                    new_df = pd.read_csv(batch_file, sep=None, engine="python")
                except pd.errors.ParserError:
                    batch_file.seek(0)
                    new_df = pd.read_csv(batch_file, sep=None, engine="python", on_bad_lines="skip")
            else:
                new_df = pd.read_excel(batch_file)
            # Rename snake_case columns to model feature names
            _rename_map = {c: FLAT_ALIAS_TO_APP_COLUMNS[c] for c in new_df.columns if c in FLAT_ALIAS_TO_APP_COLUMNS}
            if _rename_map:
                new_df = new_df.rename(columns=_rename_map)

            st.success(tr(f"Loaded {len(new_df)} rows × {len(new_df.columns)} columns.", f"Carregadas {len(new_df)} linhas × {len(new_df.columns)} colunas."))

            # Show column mapping status (exclude derived features computed by the app)
            _derived_features = {"cirurgia_combinada", "peso_procedimento", "thoracic_aorta_flag"}
            matched_cols = [c for c in artifacts.feature_columns if c in new_df.columns or c in _derived_features]
            missing_cols = [c for c in artifacts.feature_columns if c not in new_df.columns and c not in _derived_features]
            st.info(
                tr(
                    f"Matched {len(matched_cols)}/{len(artifacts.feature_columns)} model features. Missing features will be imputed by the model.",
                    f"Encontradas {len(matched_cols)}/{len(artifacts.feature_columns)} variáveis do modelo. Variáveis ausentes serão imputadas pelo modelo.",
                )
            )
            if missing_cols:
                with st.expander(tr("Show missing features", "Ver variáveis ausentes")):
                    st.write(", ".join(missing_cols))

            _show_all_models = st.checkbox(
                tr("Show predictions from all AI models", "Mostrar predições de todos os modelos de IA"),
                value=False,
                key="batch_show_all_models",
            )
            _include_sts = st.checkbox(
                tr("Include STS (requires internet, ~1 min per 50 patients)", "Incluir STS (requer internet, ~1 min a cada 50 pacientes)"),
                value=False,
                key="batch_include_sts",
            )

            if st.button(tr("Run batch prediction", "Executar predição em lote"), type="primary"):
                _all_model_names = sorted(artifacts.fitted_models.keys())
                _n_total = len(new_df)
                results = []
                ref_df = prepared.data
                batch_rows_for_sts = []

                # Pre-clean the entire uploaded DataFrame: force numeric dtypes
                # to match training data, converting stray strings to NaN
                for _fc in artifacts.feature_columns:
                    if _fc in new_df.columns and _fc in ref_df.columns:
                        if pd.api.types.is_numeric_dtype(ref_df[_fc]) and not pd.api.types.is_numeric_dtype(new_df[_fc]):
                            new_df[_fc] = pd.to_numeric(
                                new_df[_fc].astype(str).str.replace(',', '.', regex=False),
                                errors="coerce",
                            )

                # --- Phase 1: AI Risk + EuroSCORE (local, fast) ---
                _progress_bar = st.progress(0, text=tr(
                    f"Computing AI Risk + EuroSCORE: 0/{_n_total}",
                    f"Calculando AI Risk + EuroSCORE: 0/{_n_total}",
                ))
                _n_errors = 0
                _num_cols_batch = _get_numeric_columns_from_pipeline(artifacts.fitted_models[forced_model])
                for idx, row_data in new_df.iterrows():
                    _i = len(results)
                    try:
                        form_map = row_data.to_dict()
                        input_row = _build_input_row(artifacts.feature_columns, form_map)
                        input_row = _align_input_to_training_schema(input_row, ref_df)
                        model_input = clean_features(input_row[artifacts.feature_columns], numeric_columns=_num_cols_batch)

                        # Final safety: force numeric columns that are still object
                        for _c in model_input.columns:
                            if model_input[_c].dtype == object and _c in ref_df.columns and pd.api.types.is_numeric_dtype(ref_df[_c]):
                                model_input[_c] = pd.to_numeric(
                                    model_input[_c].astype(str).str.replace(',', '.', regex=False),
                                    errors="coerce",
                                )

                        ia_prob = float(artifacts.fitted_models[forced_model].predict_proba(model_input)[:, 1][0])
                        euro_prob = float(euroscore_from_inputs(form_map))

                        row_result = {
                            tr("Row", "Linha"): idx + 1,
                            tr("Name", "Nome"): form_map.get("Name", form_map.get("Nome", f"Patient {idx+1}")),
                            tr("Surgery", "Cirurgia"): form_map.get("Surgery", form_map.get("Cirurgia", "")),
                            f"AI Risk - {forced_model} (%)": round(ia_prob * 100, 2),
                        }
                        # All AI models
                        for _mn in _all_model_names:
                            _p = float(artifacts.fitted_models[_mn].predict_proba(model_input)[:, 1][0])
                            row_result[f"IA-{_mn} (%)"] = round(_p * 100, 2)

                        row_result["EuroSCORE II (%)"] = round(euro_prob * 100, 2)
                        row_result[tr("Risk class", "Classe de risco")] = class_risk(ia_prob)
                        results.append(row_result)
                        batch_rows_for_sts.append(form_map)
                    except Exception as _row_err:
                        _n_errors += 1
                        results.append({
                            tr("Row", "Linha"): idx + 1,
                            tr("Name", "Nome"): form_map.get("Name", form_map.get("Nome", f"Patient {idx+1}")),
                            tr("Error", "Erro"): str(_row_err),
                        })
                        batch_rows_for_sts.append({})

                    _pct = (_i + 1) / _n_total
                    _progress_bar.progress(_pct, text=tr(
                        f"Computing AI Risk + EuroSCORE: {_i + 1}/{_n_total}",
                        f"Calculando AI Risk + EuroSCORE: {_i + 1}/{_n_total}",
                    ))

                _progress_bar.progress(1.0, text=tr(
                    f"AI Risk + EuroSCORE complete: {_n_total - _n_errors} OK, {_n_errors} errors",
                    f"AI Risk + EuroSCORE completo: {_n_total - _n_errors} OK, {_n_errors} erros",
                ))

                # --- Phase 2: STS (optional, slow — WebSocket per patient) ---
                sts_probs = [np.nan] * len(results)
                if HAS_STS and _include_sts:
                    _sts_progress = st.progress(0, text=tr(
                        f"Querying STS web calculator: 0/{_n_total}",
                        f"Consultando calculadora web do STS: 0/{_n_total}",
                    ))
                    try:
                        def _sts_progress_cb(done, total):
                            try:
                                _sts_progress.progress(
                                    done / max(total, 1),
                                    text=tr(
                                        f"Querying STS web calculator: {done}/{total}",
                                        f"Consultando calculadora web do STS: {done}/{total}",
                                    ),
                                )
                            except Exception:
                                pass
                        sts_results = calculate_sts_batch(batch_rows_for_sts, progress_callback=_sts_progress_cb)
                        if sts_results:
                            for _ri, _sr in enumerate(sts_results):
                                if isinstance(_sr, dict) and "predmort" in _sr:
                                    sts_probs[_ri] = _sr["predmort"]
                        _n_sts_ok = sum(1 for p in sts_probs if pd.notna(p))
                        _sts_progress.progress(1.0, text=tr(
                            f"STS complete: {_n_sts_ok}/{_n_total} calculated",
                            f"STS completo: {_n_sts_ok}/{_n_total} calculados",
                        ))
                        if _n_sts_ok < _n_total:
                            _fail_log = getattr(calculate_sts_batch, 'failure_log', [])
                            if _fail_log:
                                _fail_details = "\n".join(
                                    f"- **{f.get('name', '?')}** ({f.get('surgery', '?')}): {f.get('reason', '?')}"
                                    for f in _fail_log
                                )
                                st.warning(tr(
                                    f"STS calculated for {_n_sts_ok}/{_n_total} patients. Failed patients:",
                                    f"STS calculado para {_n_sts_ok}/{_n_total} pacientes. Pacientes que falharam:",
                                ))
                                st.markdown(_fail_details)
                            else:
                                st.warning(tr(
                                    f"STS calculated for {_n_sts_ok}/{_n_total} patients.",
                                    f"STS calculado para {_n_sts_ok}/{_n_total} pacientes.",
                                ))
                    except Exception as _sts_err:
                        _sts_progress.progress(1.0, text=tr(
                            f"STS failed: {_sts_err}", f"STS falhou: {_sts_err}",
                        ))
                        st.warning(tr(
                            f"STS calculation failed: {_sts_err}. Results shown without STS.",
                            f"Cálculo STS falhou: {_sts_err}. Resultados mostrados sem STS.",
                        ))

                for i, sp in enumerate(sts_probs):
                    results[i]["STS (%)"] = round(sp * 100, 2) if pd.notna(sp) else np.nan

                result_df = pd.DataFrame(results)

                # Column visibility: hide individual model columns unless checkbox is checked
                _ia_detail_cols = [f"IA-{_mn} (%)" for _mn in _all_model_names]
                if not _show_all_models:
                    display_df = result_df.drop(columns=_ia_detail_cols, errors="ignore")
                else:
                    display_df = result_df

                st.dataframe(display_df, width="stretch")

                # --- Summary statistics per score ---
                _ia_col = f"AI Risk - {forced_model} (%)"
                _euro_col = "EuroSCORE II (%)"
                _sts_col = "STS (%)"
                _summary_rows = []
                for _scol, _slabel in [(_ia_col, f"AI Risk ({forced_model})"), (_euro_col, "EuroSCORE II"), (_sts_col, "STS PROM")]:
                    if _scol in result_df.columns:
                        _vals = pd.to_numeric(result_df[_scol], errors="coerce").dropna()
                        if len(_vals) > 0:
                            _summary_rows.append({
                                tr("Score", "Escore"): _slabel,
                                "n": int(len(_vals)),
                                tr("Mean", "Média"): f"{_vals.mean():.2f}%",
                                tr("Median", "Mediana"): f"{_vals.median():.2f}%",
                                "Min": f"{_vals.min():.2f}%",
                                "Max": f"{_vals.max():.2f}%",
                                "IQR": f"{_vals.quantile(0.25):.2f}–{_vals.quantile(0.75):.2f}%",
                                f"> {_default_threshold:.0%}": int((_vals / 100 >= _default_threshold).sum()),
                            })
                if _summary_rows:
                    st.markdown(tr("**Summary by score**", "**Resumo por escore**"))
                    st.dataframe(pd.DataFrame(_summary_rows), width="stretch", hide_index=True)

                # Build Markdown version for MD/PDF export
                _batch_md_lines = [
                    f"# {tr('Batch Prediction Report', 'Relatório de Predição em Lote')}",
                    "",
                    f"**{tr('Date', 'Data')}:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
                    f"**{tr('Source file', 'Arquivo fonte')}:** {batch_file.name}",
                    f"**{tr('Primary model', 'Modelo principal')}:** {forced_model}",
                    f"**{tr('Model version', 'Versão do modelo')}:** {MODEL_VERSION}",
                    f"**{tr('Patients', 'Pacientes')}:** {len(result_df)}",
                    "",
                    f"## {tr('Predictions', 'Predições')}",
                    "",
                    result_df.to_markdown(index=False),
                    "",
                ]
                _batch_md = "\n".join(_batch_md_lines)

                # Downloads: CSV + XLSX + MD + PDF (full data always includes all models)
                _dl1, _dl2, _dl3, _dl4 = st.columns(4)
                with _dl1:
                    _csv_download_btn(result_df, "ia_risk_batch_predictions.csv", tr("CSV", "CSV"))
                with _dl2:
                    _xlsx_buf = BytesIO()
                    result_df.to_excel(_xlsx_buf, index=False, engine="openpyxl")
                    st.download_button(
                        "XLSX",
                        _xlsx_buf.getvalue(),
                        "ia_risk_batch_predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_batch_xlsx",
                    )
                with _dl3:
                    _txt_download_btn(_batch_md, "ia_risk_batch_predictions.md", "MD")
                with _dl4:
                    _batch_pdf = statistical_summary_to_pdf(_batch_md)
                    if _batch_pdf:
                        st.download_button(
                            "PDF",
                            _batch_pdf,
                            "ia_risk_batch_predictions.pdf",
                            mime="application/pdf",
                            key="dl_batch_pdf",
                        )

                # Audit trail for batch prediction
                _n_sts_ok = sum(1 for sp in sts_probs if pd.notna(sp))
                log_analysis(
                    analysis_type="batch_prediction",
                    source_file=batch_file.name,
                    model_version=MODEL_VERSION,
                    n_patients=len(results),
                    n_imputed=len(missing_cols),
                    completeness_level=f"{len(matched_cols)}/{len(artifacts.feature_columns)} features matched",
                    sts_method="websocket" if _n_sts_ok > 0 else "unavailable",
                    extra={"n_sts_calculated": _n_sts_ok, "n_rows": len(results)},
                )
        except Exception as e:
            st.error(tr(f"Error processing file: {e}", f"Erro ao processar arquivo: {e}"))

elif _active_tab == 5:  # Model Guide
    st.subheader(tr("Understand the selected model", "Entenda o modelo selecionado"))

    model_docs_en = {
        "LogisticRegression": {
            "como_funciona": (
                "Assigns a weight to each clinical variable, sums them, and converts the result into a 30-day mortality probability."
            ),
            "comportamento": (
                "Stable and transparent. Each factor changes risk in a predictable way. "
                "Works well when risk increases gradually across variables."
            ),
            "forcas": "High interpretability, good calibration, and easy clinical communication.",
            "limitacoes": "May miss complex nonlinear interactions.",
            "quando_erra": (
                "Can underperform when risk depends on very specific variable combinations."
            ),
        },
        "RandomForest": {
            "como_funciona": (
                "Builds many decision trees on different samples and averages their predictions."
            ),
            "comportamento": (
                "Captures nonlinear relationships and interactions automatically. Robust to noise, "
                "but can smooth extreme probabilities."
            ),
            "forcas": "Strong tabular performance with limited feature engineering.",
            "limitacoes": "Less explainable than logistic regression; probabilities may need calibration.",
            "quando_erra": (
                "May underperform for rare patient profiles and very extreme risk estimates."
            ),
        },
        "XGBoost": {
            "como_funciona": (
                "Trains trees sequentially; each new tree corrects previous errors."
            ),
            "comportamento": (
                "Very strong for complex tabular patterns and imbalanced outcomes. "
                "Can achieve high discrimination but needs careful overfitting control."
            ),
            "forcas": "High predictive performance in many clinical datasets.",
            "limitacoes": "Harder to explain; often requires tuning and calibration monitoring.",
            "quando_erra": (
                "May learn spurious patterns in small subgroups and become miscalibrated over time."
            ),
        },
        "LightGBM": {
            "como_funciona": (
                "A gradient-boosted tree model optimized for speed and scalability."
            ),
            "comportamento": (
                "Finds nonlinear interactions efficiently; on smaller datasets it may be parameter-sensitive."
            ),
            "forcas": "Fast training and strong tabular performance.",
            "limitacoes": "Indirect interpretability; performance depends on proper configuration.",
            "quando_erra": (
                "Can be unstable with small samples and many categories, especially in underrepresented subgroups."
            ),
        },
        "CatBoost": {
            "como_funciona": (
                "Boosted trees with native handling of categorical variables and built-in overfitting control."
            ),
            "comportamento": (
                "Often stable when there are many categorical and missing values, with less manual preprocessing."
            ),
            "forcas": "Excellent for heterogeneous clinical tabular data.",
            "limitacoes": "Still complex for individual-case explanation.",
            "quando_erra": (
                "Can underperform when key variables are missing or when case-mix shifts over time."
            ),
        },
        "MLP": {
            "como_funciona": (
                "A multilayer neural network that learns complex combinations of variables."
            ),
            "comportamento": (
                "Captures sophisticated patterns, but can be unstable and overfit in small datasets."
            ),
            "forcas": "Highly flexible for complex relationships.",
            "limitacoes": "Lower transparency and sensitive to dataset size/quality.",
            "quando_erra": (
                "Can overfit in small/noisy datasets, especially for rare profiles."
            ),
        },
        "StackingEnsemble": {
            "como_funciona": (
                "Combines predictions from multiple base models and uses a meta-model for final probability."
            ),
            "comportamento": (
                "Can improve robustness when base models make different errors, at the cost of complexity."
            ),
            "forcas": "Integrates strengths of multiple algorithms.",
            "limitacoes": "Harder to explain, validate, and maintain in routine care.",
            "quando_erra": (
                "Can fail when base models share the same bias or when epidemiologic profile shifts."
            ),
        },
    }

    model_docs_pt = {
        "LogisticRegression": {
            "como_funciona": "Atribui pesos às variáveis e estima a probabilidade de óbito em 30 dias.",
            "comportamento": "É estável e transparente, com efeito previsível de cada variável.",
            "forcas": "Alta interpretabilidade e boa comunicação clínica.",
            "limitacoes": "Pode perder interações não lineares complexas.",
            "quando_erra": "Pode falhar quando o risco depende de combinações muito específicas de fatores.",
        },
        "RandomForest": {
            "como_funciona": "Combina muitas árvores de decisão e faz média das previsões.",
            "comportamento": "Captura não linearidades, mas tende a suavizar probabilidades extremas.",
            "forcas": "Bom desempenho tabular sem muita engenharia manual.",
            "limitacoes": "Menos explicável que regressão logística.",
            "quando_erra": "Pode piorar em perfis raros e riscos muito extremos.",
        },
        "XGBoost": {
            "como_funciona": "Treina árvores em sequência corrigindo erros anteriores.",
            "comportamento": "Alto poder discriminativo, com risco de sobreajuste sem controle.",
            "forcas": "Alta performance preditiva.",
            "limitacoes": "Exige ajuste fino e monitoramento de calibração.",
            "quando_erra": "Pode aprender padrões espúrios em subgrupos pequenos.",
        },
        "LightGBM": {
            "como_funciona": "Boosting de árvores otimizado para velocidade e escala.",
            "comportamento": "Eficiente em interações; pode ser sensível em amostras pequenas.",
            "forcas": "Rápido e eficaz em dados tabulares.",
            "limitacoes": "Interpretação indireta.",
            "quando_erra": "Pode ser instável em subgrupos pouco representados.",
        },
        "CatBoost": {
            "como_funciona": "Boosting com tratamento nativo de variáveis categóricas.",
            "comportamento": "Tende a ser estável com muitos dados categóricos e faltantes.",
            "forcas": "Muito bom para dados clínicos heterogêneos.",
            "limitacoes": "Ainda é complexo para explicar caso a caso.",
            "quando_erra": "Perde desempenho quando faltam variáveis-chave.",
        },
        "MLP": {
            "como_funciona": "Rede neural em camadas para padrões complexos.",
            "comportamento": "Flexível, porém sensível a base pequena e ruído.",
            "forcas": "Capta relações complexas.",
            "limitacoes": "Menor transparência clínica.",
            "quando_erra": "Pode sobreajustar em bases pequenas.",
        },
        "StackingEnsemble": {
            "como_funciona": "Combina previsões de vários modelos em um meta-modelo final.",
            "comportamento": "Pode aumentar robustez ao combinar erros diferentes.",
            "forcas": "Integra forças de múltiplos algoritmos.",
            "limitacoes": "Mais difícil de validar e explicar.",
            "quando_erra": "Falha quando os modelos base compartilham o mesmo viés.",
        },
    }

    model_docs = model_docs_en if language == "English" else model_docs_pt

    d = model_docs.get(
        forced_model,
        {
            "como_funciona": tr("Model not documented.", "Modelo não documentado."),
            "comportamento": "",
            "forcas": "",
            "limitacoes": "",
            "quando_erra": "",
        },
    )

    st.markdown(tr(f"### Selected model: `{forced_model}`", f"### Modelo selecionado: `{forced_model}`"))
    st.markdown(tr("**How it works**", "**Como funciona**"))
    st.write(d["como_funciona"])
    st.markdown(tr("**How it behaves in practice**", "**Como ele se comporta na prática**"))
    st.write(d["comportamento"])
    st.markdown(tr("**Strengths**", "**Pontos fortes**"))
    st.write(d["forcas"])
    st.markdown(tr("**Limitations**", "**Limitações**"))
    st.write(d["limitacoes"])
    st.markdown(tr("**When this model usually fails**", "**Quando este modelo costuma errar**"))
    st.write(d["quando_erra"])

    st.markdown(tr("**How to interpret in practice**", "**Como interpretar na prática**"))
    st.markdown(
        tr(
            "- Risk is reported as a probability (0% to 100%).\n"
            "- The app classifies risk as Low (<5%), Intermediate (5-15%), and High (>15%).\n"
            "- Changes in patient profile may shift probability even without class change.\n"
            "- Changes in case-mix over time may change the best-performing model.",
            "- O risco mostrado é uma probabilidade (0% a 100%).\n"
            "- O app classifica em Baixo (<5%), Intermediário (5-15%) e Alto (>15%).\n"
            "- Mudança no perfil do paciente pode deslocar a probabilidade mesmo sem mudar a classe.\n"
            "- Mudança de base (novos pacientes) pode alterar qual modelo fica melhor.",
        )
    )

    st.markdown(tr("**Model explanation table**", "**Tabela explicativa do modelo**"))
    w, w_kind = model_weight_table(artifacts, prepared, forced_model, top_n=20)
    if w_kind == "coefficient":
        st.caption(
            tr(
                "These are model coefficients after preprocessing. For categorical variables, the values refer to encoded categories rather than the raw clinical field as a whole.",
                "Estes são coeficientes do modelo após o pré-processamento. Para variáveis categóricas, os valores se referem às categorias codificadas, e não ao campo clínico bruto como um todo.",
            )
        )
    elif w_kind == "importance":
        st.caption(
            tr(
                "These values represent variable importance, not direct clinical effect size. They indicate how much the variable helps the model, but not whether it increases or decreases risk.",
                "Esses valores representam importância de variável, e não efeito clínico direto. Eles indicam o quanto a variável ajuda o modelo, mas não se aumenta ou reduz o risco.",
            )
        )
    elif w_kind == "stacking":
        st.caption(
            tr(
                "These are weights of the base models inside the stacking ensemble, not direct weights of the clinical variables.",
                "Esses são pesos dos modelos base dentro do ensemble por stacking, e não pesos diretos das variáveis clínicas.",
            )
        )

    if w.empty:
        st.info(
            tr(
                "This model does not expose clinically interpretable per-variable weights in a simple way. Use global importance, calibration, discrimination, and sensitivity analysis.",
                "Este modelo não expõe pesos por variável clinicamente interpretáveis de forma simples. Nesses casos, a interpretação deve se apoiar em importância global, calibração, discriminação e análise de sensibilidade.",
            )
        )
    else:
        st.dataframe(w, width="stretch", column_config=model_table_column_config(w_kind))

    st.markdown(tr("**Clinical variables and importance**", "**Variáveis clínicas e importância**"))
    show_all_final = st.checkbox(
        tr("Show all variables for final model", "Ver todas as variáveis do modelo final"),
        value=False,
        key="show_all_final_importance",
    )
    X_perm = clean_features(_safe_select_features(prepared.data, artifacts.feature_columns))
    y_perm = prepared.data["morte_30d"].astype(int).values
    perm_table = cached_permutation_importance_table(
        xlsx_path,
        forced_model,
        20,
        show_all_final,
        artifacts.fitted_models[forced_model],
        X_perm,
        y_perm,
        artifacts.feature_columns,
    )
    st.caption(
        tr(
            "Permutation importance estimates how much model performance worsens when a variable is randomly shuffled. This is the recommended summary for the selected final model, but it should be interpreted as exploratory because it is computed on the fitted dataset.",
            "A importância por permutação estima o quanto o desempenho do modelo piora quando uma variável é embaralhada aleatoriamente. Este é o resumo mais indicado para o modelo final selecionado, mas deve ser interpretado como exploratório porque é calculado na base ajustada.",
        )
    )
    st.dataframe(perm_table, width="stretch", column_config=model_table_column_config("importance"))

    st.markdown(tr("**SHAP global importance (selected model)**", "**Importância global SHAP (modelo selecionado)**"))
    _shap_model_estimator = artifacts.fitted_models[forced_model].named_steps["model"]
    if not hasattr(_shap_model_estimator, "feature_importances_"):
        st.info(
            tr(
                "SHAP global visualization is available only for tree-based models (RandomForest, XGBoost, LightGBM, CatBoost). The currently selected model does not expose tree structure.",
                "A visualização global SHAP está disponível apenas para modelos baseados em árvore (RandomForest, XGBoost, LightGBM, CatBoost). O modelo selecionado não expõe estrutura de árvore.",
            )
        )
    else:
        st.caption(
            tr(
                "Unlike permutation importance (which only shows magnitude), SHAP values also show direction: positive mean SHAP pushes predicted risk up, negative pushes it down. Computed on the full fitted dataset using TreeExplainer.",
                "Ao contrário da importância por permutação (que mostra apenas magnitude), os valores SHAP também mostram direção: SHAP médio positivo aumenta o risco previsto, negativo reduz. Calculado na base ajustada completa com TreeExplainer.",
            )
        )
        with st.spinner(tr("Computing SHAP global importance…", "Calculando importância global SHAP…")):
            shap_global_df = _cached_shap_global(
                xlsx_path,
                forced_model,
                20,
                artifacts.fitted_models[forced_model],
                X_perm,
            )
        if shap_global_df.empty:
            st.info(tr("SHAP global importance could not be computed.", "Importância global SHAP não pôde ser calculada."))
        else:
            shap_global_df.columns = [
                tr("Feature", "Variável"),
                tr("Mean |SHAP|", "SHAP médio |absoluto|"),
                tr("Mean SHAP (direction)", "SHAP médio (direção)"),
            ]
            st.dataframe(shap_global_df, width="stretch")

        # --- SHAP Beeswarm Plot ---
        st.markdown(tr("**SHAP Beeswarm plot**", "**Gráfico Beeswarm SHAP**"))
        st.caption(
            tr(
                "Each dot is one patient-feature pair. Position on x-axis shows the SHAP value (impact on prediction). Color shows the feature value (red = high, blue = low). This reveals both importance and direction of each variable's effect.",
                "Cada ponto é um par paciente-variável. A posição no eixo x mostra o valor SHAP (impacto na predição). A cor indica o valor da variável (vermelho = alto, azul = baixo). Isso revela importância e direção do efeito de cada variável.",
            )
        )
        with st.spinner(tr("Generating beeswarm plot…", "Gerando gráfico beeswarm…")):
            beeswarm_fig = _cached_shap_beeswarm(
                xlsx_path,
                forced_model,
                15,
                artifacts.fitted_models[forced_model],
                X_perm,
            )
        if beeswarm_fig is not None:
            st.pyplot(beeswarm_fig)
            _chart_download_buttons(shap_global_df, _fig_to_png_bytes(beeswarm_fig), "shap_beeswarm")
        else:
            st.info(tr("Beeswarm plot could not be generated.", "Gráfico beeswarm não pôde ser gerado."))

        # --- SHAP Dependence Plot ---
        st.markdown(tr("**SHAP Feature dependence**", "**Dependência de variável SHAP**"))
        st.caption(
            tr(
                "Shows how a single variable's value affects the model prediction. Each dot is a patient; the y-axis shows that variable's SHAP contribution to the predicted risk.",
                "Mostra como o valor de uma variável individual afeta a predição do modelo. Cada ponto é um paciente; o eixo y mostra a contribuição SHAP daquela variável ao risco previsto.",
            )
        )
        _shap_dep_features = shap_global_df[tr("Feature", "Variável")].tolist() if not shap_global_df.empty else []
        if _shap_dep_features:
            selected_feature = st.selectbox(
                tr("Select feature for dependence plot", "Selecione variável para gráfico de dependência"),
                _shap_dep_features,
                key="shap_dep_feature",
            )
            with st.spinner(tr("Generating dependence plot…", "Gerando gráfico de dependência…")):
                dep_fig = _cached_shap_dependence(
                    xlsx_path,
                    forced_model,
                    selected_feature,
                    artifacts.fitted_models[forced_model],
                    X_perm,
                )
            if dep_fig is not None:
                st.pyplot(dep_fig)
                _dep_data = pd.DataFrame({"Feature": [selected_feature]})
                _chart_download_buttons(_dep_data, _fig_to_png_bytes(dep_fig), f"shap_dependence_{selected_feature}")
            else:
                st.info(tr("Dependence plot could not be generated for this feature.", "Gráfico de dependência não pôde ser gerado para esta variável."))

    st.markdown(tr("**Logistic regression clinical coefficients**", "**Coeficientes clínicos da regressão logística**"))
    show_all_lr = st.checkbox(
        tr("Show all logistic regression variables", "Ver todas as variáveis da regressão logística"),
        value=False,
        key="show_all_lr_coeffs",
    )
    lr_table = logistic_clinical_coefficients_table(artifacts, prepared, top_n=20, show_all=show_all_lr)
    st.caption(
        tr(
            "These coefficients come from the fitted logistic regression model. Positive coefficients suggest a tendency toward higher risk, whereas negative coefficients suggest a tendency toward lower risk. For categorical variables, the table keeps the category with the strongest absolute coefficient as the representative effect.",
            "Esses coeficientes vêm do modelo de regressão logística ajustado. Coeficientes positivos sugerem tendência a maior risco, enquanto coeficientes negativos sugerem tendência a menor risco. Para variáveis categóricas, a tabela mantém a categoria com maior coeficiente absoluto como efeito representativo.",
        )
    )
    st.dataframe(lr_table, width="stretch", column_config=model_table_column_config("logistic_clinical"))

    st.markdown(tr("**EuroSCORE II coefficients (official)**", "**Coeficientes do EuroSCORE II (oficial)**"))
    euro_coef_df = pd.DataFrame(
        {
            tr("Factor", "Fator"): [tr("Constant", "Constante")] + list(EURO_COEF.keys()),
            tr("Coefficient", "Coeficiente"): [EURO_CONST] + [EURO_COEF[k] for k in EURO_COEF.keys()],
        }
    )
    st.dataframe(euro_coef_df, width="stretch", column_config=model_table_column_config("euroscore"))

elif _active_tab == 6:  # Subgroups
    st.subheader(tr("Subgroup analysis", "Análise por subgrupos"))
    st.caption(
        tr(
            "This panel evaluates whether model performance changes across clinically relevant groups. Results in small groups should be interpreted with caution.",
            "Este painel avalia se o desempenho do modelo muda em grupos clinicamente relevantes. Resultados em grupos pequenos devem ser interpretados com cautela.",
        )
    )
    subgroup_threshold = st.slider(
        tr("Subgroup decision threshold", "Limiar de decisão dos subgrupos"),
        min_value=0.01,
        max_value=0.99,
        value=_default_threshold,
        step=0.01,
    )
    st.caption(
        tr(
            f"Default: {_default_threshold:.0%} (dataset prevalence). Sensitivity, specificity, PPV, and NPV change with this threshold; AUC, AUPRC, and Brier do not.",
            f"Padrão: {_default_threshold:.0%} (prevalência do dataset). Sensibilidade, especificidade, PPV e NPV mudam com este limiar; AUC, AUPRC e Brier não.",
        )
    )
    subgroup_df = df.copy()
    subgroup_df["Surgery type"] = subgroup_df["Surgery"].map(_surgery_type_group)
    subgroup_df["Sex group"] = subgroup_df["Sex"].fillna(tr("Unknown", "Desconhecido"))
    subgroup_df["Age group"] = np.where(pd.to_numeric(subgroup_df["Age (years)"], errors="coerce") < 65, "<65", ">=65")
    _nan_f = pd.Series(np.nan, index=subgroup_df.index)
    _nan_o = pd.Series(np.nan, index=subgroup_df.index, dtype=object)
    subgroup_df["LVEF group"] = [
        _lvef_group(eco, pre) for eco, pre in zip(
            subgroup_df["LVEF, %"] if "LVEF, %" in subgroup_df.columns else _nan_f,
            subgroup_df["Pré-LVEF, %"] if "Pré-LVEF, %" in subgroup_df.columns else _nan_f,
        )
    ]
    subgroup_df["Renal function group"] = [
        _renal_group(cc, d, cr, a, w, s) for cc, d, cr, a, w, s in zip(
            subgroup_df["Cr clearance, ml/min *"] if "Cr clearance, ml/min *" in subgroup_df.columns else _nan_f,
            subgroup_df["Dialysis"] if "Dialysis" in subgroup_df.columns else _nan_o,
            subgroup_df["Creatinine (mg/dL)"] if "Creatinine (mg/dL)" in subgroup_df.columns else _nan_f,
            subgroup_df["Age (years)"] if "Age (years)" in subgroup_df.columns else _nan_f,
            subgroup_df["Weight (kg)"] if "Weight (kg)" in subgroup_df.columns else _nan_f,
            subgroup_df["Sex"] if "Sex" in subgroup_df.columns else _nan_o,
        )
    ]

    subgroup_choice = st.selectbox(
        tr("Subgroup panel", "Painel de subgrupos"),
        [
            tr("Surgery type", "Tipo de cirurgia"),
            tr("Age", "Idade"),
            tr("LVEF", "FEVE"),
            tr("Renal function", "Função renal"),
            tr("Sex", "Sexo"),
        ],
    )
    subgroup_map = {
        tr("Surgery type", "Tipo de cirurgia"): "Surgery type",
        tr("Age", "Idade"): "Age group",
        tr("LVEF", "FEVE"): "LVEF group",
        tr("Renal function", "Função renal"): "Renal function group",
        tr("Sex", "Sexo"): "Sex group",
    }
    subgroup_col = subgroup_map[subgroup_choice]
    subgroup_metrics = evaluate_subgroup(
        subgroup_df,
        subgroup_col,
        ["ia_risk_oof", "euroscore_calc", "sts_score"],
        subgroup_threshold,
    )
    if subgroup_metrics.empty:
        st.info(tr("No subgroup results are available for the current selection.", "Não há resultados de subgrupos disponíveis para a seleção atual."))
    else:
        subgroup_metrics["Score"] = subgroup_metrics["Score"].replace(
            {"ia_risk_oof": "AI Risk", "euroscore_calc": "EuroSCORE II", "sts_score": "STS"}
        )
        small_n = subgroup_metrics[subgroup_metrics["n"] < 50][["Group", "Score", "n"]]
        low_events = subgroup_metrics[subgroup_metrics["Deaths"] < 10][["Group", "Score", "Deaths"]]
        if not small_n.empty or not low_events.empty:
            warn_parts = []
            if not small_n.empty:
                groups_small_n = ", ".join(sorted(set(small_n["Group"].astype(str).tolist())))
                warn_parts.append(
                    tr(
                        f"small sample size in: {groups_small_n}",
                        f"tamanho amostral pequeno em: {groups_small_n}",
                    )
                )
            if not low_events.empty:
                groups_low_events = ", ".join(sorted(set(low_events["Group"].astype(str).tolist())))
                warn_parts.append(
                    tr(
                        f"low event count in: {groups_low_events}",
                        f"baixo número de eventos em: {groups_low_events}",
                    )
                )
            st.warning(
                tr(
                    f"Subgroup results should be interpreted with caution due to {'; '.join(warn_parts)}.",
                    f"Os resultados por subgrupos devem ser interpretados com cautela devido a {'; '.join(warn_parts)}.",
                )
            )
        # Reorder columns: identifiers first, then metrics, then CIs
        _sub_col_order = [
            "Score", "Subgroup", "Group", "Deaths", "n",
            "AUC", "AUC_IC95_inf", "AUC_IC95_sup",
            "AUPRC", "AUPRC_IC95_inf", "AUPRC_IC95_sup",
            "Brier", "Brier_IC95_inf", "Brier_IC95_sup",
            "Sensitivity", "Specificity", "PPV", "NPV",
        ]
        _sub_col_order = [c for c in _sub_col_order if c in subgroup_metrics.columns]
        subgroup_metrics = subgroup_metrics[_sub_col_order]
        st.dataframe(_format_ppv_npv(subgroup_metrics), width="stretch", column_config=stats_table_column_config("subgroup"))
        _csv_download_btn(subgroup_metrics, "subgroup_results.csv", tr("Download subgroup results (CSV)", "Baixar resultados dos subgrupos (CSV)"))
        best_sub = subgroup_metrics.sort_values("AUC", ascending=False).iloc[0]
        _ci_lo = best_sub.get("AUC_IC95_inf", np.nan)
        _ci_hi = best_sub.get("AUC_IC95_sup", np.nan)
        _ci_str = f" (95% CI: {_ci_lo:.3f}–{_ci_hi:.3f})" if pd.notna(_ci_lo) and pd.notna(_ci_hi) else ""
        st.info(
            tr(
                f"In the selected subgroup panel, the best discriminative performance was observed for {best_sub['Score']} in group {best_sub['Group']} (AUC={best_sub['AUC']:.3f}{_ci_str}).",
                f"No painel de subgrupos selecionado, a melhor discriminação foi observada em {best_sub['Score']} no grupo {best_sub['Group']} (AUC={best_sub['AUC']:.3f}{_ci_str}).",
            )
        )

elif _active_tab == 7:  # Data Quality
    st.subheader(tr("Data Quality Panel", "Painel de Qualidade da Base"))
    st.caption(tr(
        "Overview of dataset completeness, score availability, and surgical case-mix.",
        "Visão geral de completude do dataset, disponibilidade de escores e case-mix cirúrgico.",
    ))

    _dq = compute_data_quality_summary(df, prepared.feature_columns, language)

    # Key metrics
    dq1, dq2, dq3, dq4 = st.columns(4)
    dq1.metric(tr("Eligible surgeries", "Cirurgias elegíveis"), _dq["n_total"])
    dq2.metric(tr("Deaths (primary outcome)", "Óbitos (desfecho primário)"), _dq["n_events"])
    dq3.metric(tr("Event rate", "Taxa de eventos"), f"{_dq['event_rate']:.1%}")
    dq4.metric(tr("Triple cohort", "Coorte tripla"), _dq["n_triple"])

    # Score availability — app-calculated scores (primary)
    st.markdown(tr("**App-calculated scores (primary)**", "**Escores calculados pelo app (primários)**"))
    _score_primary = pd.DataFrame([
        {tr("Score", "Escore"): "AI Risk (OOF)", tr("Patients", "Pacientes"): int(df["ia_risk_oof"].notna().sum()) if "ia_risk_oof" in df.columns else 0},
        {tr("Score", "Escore"): tr("EuroSCORE II (app-calculated)", "EuroSCORE II (calculado pelo app)"), tr("Patients", "Pacientes"): _dq["n_euro_calc"]},
        {tr("Score", "Escore"): tr("STS (app-calculated)", "STS (calculado pelo app)"), tr("Patients", "Pacientes"): _dq["n_sts"]},
        {tr("Score", "Escore"): tr("Triple cohort (all 3 scores)", "Coorte tripla (3 escores)"), tr("Patients", "Pacientes"): _dq["n_triple"]},
    ])
    st.dataframe(_score_primary, width="stretch")

    # Sheet-derived scores (reference only)
    with st.expander(tr("Sheet-derived scores (reference only)", "Escores derivados da planilha (apenas referência)")):
        st.caption(tr(
            "These values were read from the original input file and are shown for comparison/validation purposes only. They are NOT used in the primary analysis.",
            "Estes valores foram lidos do arquivo de entrada original e são mostrados apenas para fins de comparação/validação. NÃO são usados na análise principal.",
        ))
        _score_ref = pd.DataFrame([
            {tr("Score", "Escore"): "EuroSCORE II (sheet)", tr("Patients", "Pacientes"): _dq["n_euro_sheet"]},
            {tr("Score", "Escore"): "EuroSCORE II Auto (sheet)", tr("Patients", "Pacientes"): _dq["n_euro_auto"]},
            {tr("Score", "Escore"): "STS (sheet)", tr("Patients", "Pacientes"): _dq["n_sts_sheet"]},
        ])
        st.dataframe(_score_ref, width="stretch")

    # Missing rates per variable
    st.markdown(tr("**Missing rate per variable**", "**Taxa de missing por variável**"))
    st.caption(tr(
        "Proportion of missing values in the analytical dataset for each predictor variable.",
        "Proporção de valores ausentes no dataset analítico para cada variável preditora.",
    ))
    _miss_df = pd.DataFrame([
        {tr("Variable", "Variável"): var, tr("Missing rate", "Taxa de missing"): rate, tr("Missing %", "Missing %"): f"{rate*100:.1f}%"}
        for var, rate in sorted(_dq["missing_rates"].items(), key=lambda x: x[1], reverse=True)
    ])
    if not _miss_df.empty:
        _miss_high = _miss_df[_miss_df[tr("Missing rate", "Taxa de missing")] > 0.3]
        if not _miss_high.empty:
            st.warning(tr(
                f"{len(_miss_high)} variables have >30% missing data. This may affect prediction reliability.",
                f"{len(_miss_high)} variáveis têm >30% de dados faltantes. Isso pode afetar a confiabilidade das predições.",
            ))
        st.dataframe(_miss_df, width="stretch")

    # Surgery type distribution
    if _dq["surgery_dist"]:
        st.markdown(tr("**Surgical procedure distribution**", "**Distribuição de procedimentos cirúrgicos**"))
        _surg_df = pd.DataFrame([
            {tr("Procedure", "Procedimento"): proc, tr("Count", "Contagem"): count}
            for proc, count in _dq["surgery_dist"].items()
        ])
        st.dataframe(_surg_df, width="stretch")

    # Validation readiness
    st.markdown(tr("**Validation readiness**", "**Prontidão para validação**"))
    _model_meta_dq = build_model_metadata(
        prepared.info, artifacts.leaderboard, best_model_name,
        artifacts.feature_columns, xlsx_path, sts_available=HAS_STS,
        bundle_saved_at=bundle_info.get("saved_at"),
        training_source_file=bundle_info.get("training_source"),
        calibration_method=getattr(artifacts, "calibration_method", "sigmoid"),
        training_data=prepared.data,
    )
    _val_checks = check_validation_readiness(_model_meta_dq, language)
    for vc in _val_checks:
        st.markdown(f"- **{vc['check']}**: {vc['status']} — {vc['note']}")

    # Audit trail
    st.divider()
    st.markdown(tr("**Analysis audit trail**", "**Trilha de auditoria**"))
    st.caption(tr(
        "Recent analysis events logged by the application.",
        "Eventos de análise recentes registrados pelo aplicativo.",
    ))
    _audit_entries = read_audit_log(20)
    if _audit_entries:
        _audit_df = pd.DataFrame(_audit_entries)
        _audit_cols = [c for c in ["timestamp", "analysis_type", "source_file", "model_version", "n_patients", "n_imputed", "completeness_level", "sts_method"] if c in _audit_df.columns]
        st.dataframe(_audit_df[_audit_cols], width="stretch")
    else:
        st.info(tr("No audit entries yet. They will appear as you use the app.", "Nenhum registro de auditoria ainda. Eles aparecerão conforme você usar o app."))

elif _active_tab == 8:  # Variable Dictionary
    st.subheader(tr("Variable Dictionary", "Dicionário de Variáveis"))
    st.caption(tr(
        "Formal reference table with clinical definitions, origins, units, and model usage for all variables.",
        "Tabela de referência formal com definições clínicas, origens, unidades e uso no modelo para todas as variáveis.",
    ))

    _dict_df = get_dictionary_dataframe(language)
    _domain_col = "Domínio" if language != "English" else "Domain"

    _dict_filter = st.multiselect(
        tr("Filter by domain", "Filtrar por domínio"),
        _dict_df[_domain_col].unique().tolist(),
        default=[],
    )
    if _dict_filter:
        _dict_display = _dict_df[_dict_df[_domain_col].isin(_dict_filter)]
    else:
        _dict_display = _dict_df

    st.dataframe(_dict_display, width="stretch")
    _csv_download_btn(_dict_df, "variable_dictionary.csv", tr("Download dictionary (CSV)", "Baixar dicionário (CSV)"))

elif _active_tab == 9:  # Temporal Validation
    st.subheader(tr("Temporal Validation", "Validação Temporal"))
    st.caption(tr(
        "This module applies a previously locked model to a later independent cohort. "
        "No retraining, recalibration, or model reselection is performed.",
        "Este módulo aplica um modelo previamente congelado em uma coorte posterior e independente. "
        "Não há retreinamento, recalibração ou nova seleção de modelo.",
    ))
    st.info(tr(
        "The frozen model pipeline (preprocessing + fitted estimator + calibration) is applied "
        "exactly as saved. The locked clinical threshold from training is used for all classification metrics.",
        "O pipeline congelado do modelo (pré-processamento + estimador ajustado + calibração) é aplicado "
        "exatamente como salvo. O limiar clínico bloqueado do treinamento é usado para todas as métricas de classificação.",
    ))

    # ── 1. Locked model info ──
    _tv_meta = build_model_metadata(
        prepared.info, artifacts.leaderboard, best_model_name,
        artifacts.feature_columns, xlsx_path, sts_available=HAS_STS,
        bundle_saved_at=bundle_info.get("saved_at"),
        training_source_file=bundle_info.get("training_source"),
        calibration_method=getattr(artifacts, "calibration_method", "sigmoid"),
        training_data=prepared.data,
    )
    _tv_locked_threshold = _tv_meta.get("locked_threshold", 0.08)

    with st.expander(tr("Locked model details", "Detalhes do modelo congelado"), expanded=True):
        st.dataframe(
            format_locked_model_for_display(_tv_meta, language),
            width="stretch",
            hide_index=True,
        )

    # ── 2. Upload temporal cohort ──
    st.divider()
    st.markdown(tr("### Upload temporal cohort", "### Upload da coorte temporal"))
    st.caption(tr(
        "Upload a dataset with the same structure as the training data. "
        "Accepted formats: .xlsx, .csv, .parquet, .db, .sqlite, .sqlite3",
        "Faça upload de um dataset com a mesma estrutura dos dados de treinamento. "
        "Formatos aceitos: .xlsx, .csv, .parquet, .db, .sqlite, .sqlite3",
    ))

    _tv_file = st.file_uploader(
        tr("Temporal validation dataset", "Dataset de validação temporal"),
        type=["xlsx", "csv", "parquet", "db", "sqlite", "sqlite3"],
        key="temporal_validation_upload",
    )

    if _tv_file is not None:
        # Save uploaded file to temp location
        _tv_ext = Path(_tv_file.name).suffix.lower()
        _tv_temp_path = TEMP_DATA_DIR / f"temporal_validation{_tv_ext}"
        TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        _tv_temp_path.write_bytes(_tv_file.getvalue())

        # ── 3. Prepare temporal dataset ──
        _tv_error = None
        _tv_prepared = None
        try:
            _tv_prepared = prepare_master_dataset(str(_tv_temp_path))
        except Exception as _tv_prep_err:
            _tv_error = str(_tv_prep_err)

        if _tv_error:
            st.error(tr(
                f"Error preparing temporal dataset: {_tv_error}",
                f"Erro ao preparar dataset temporal: {_tv_error}",
            ))
        elif _tv_prepared is not None:
            _tv_data = _tv_prepared.data.copy()

            # Validate outcome column exists
            if "morte_30d" not in _tv_data.columns or _tv_data["morte_30d"].isna().all():
                st.error(tr(
                    "The temporal dataset does not contain an observed outcome (Death / morte_30d). "
                    "Temporal validation requires known outcomes to compute performance metrics.",
                    "O dataset temporal não contém desfecho observado (Death / morte_30d). "
                    "A validação temporal requer desfechos conhecidos para calcular métricas de desempenho.",
                ))
            elif _tv_data["morte_30d"].nunique() < 2:
                st.error(tr(
                    "The temporal cohort has only one outcome class (all survivors or all deaths). "
                    "Discrimination metrics (AUC) require both events and non-events.",
                    "A coorte temporal tem apenas uma classe de desfecho (todos sobreviventes ou todos óbitos). "
                    "Métricas de discriminação (AUC) requerem tanto eventos quanto não-eventos.",
                ))
            else:
                _tv_n = len(_tv_data)
                _tv_events = int(_tv_data["morte_30d"].sum())
                _tv_rate = _tv_events / _tv_n if _tv_n > 0 else 0

                # Display basic cohort info
                _tv_c1, _tv_c2, _tv_c3 = st.columns(3)
                _tv_c1.metric(tr("Patients", "Pacientes"), _tv_n)
                _tv_c2.metric(tr("Events", "Eventos"), _tv_events)
                _tv_c3.metric(tr("Event rate", "Taxa de eventos"), f"{_tv_rate:.1%}")

                # ── 4. Chronological check ──
                from model_metadata import _extract_year_quarter_range
                _tv_val_start, _tv_val_end = _extract_year_quarter_range(_tv_data)

                _tv_overlap = check_temporal_overlap(
                    _tv_meta.get("training_start_date", "Unknown"),
                    _tv_meta.get("training_end_date", "Unknown"),
                    _tv_val_start,
                    _tv_val_end,
                )
                with st.expander(tr("Chronological check", "Verificação cronológica"), expanded=True):
                    _ov_c1, _ov_c2 = st.columns(2)
                    _ov_c1.markdown(tr("**Training cohort:**", "**Coorte de treinamento:**") +
                        f" {_tv_meta.get('training_start_date', 'Unknown')} — {_tv_meta.get('training_end_date', 'Unknown')}")
                    _ov_c2.markdown(tr("**Validation cohort:**", "**Coorte de validação:**") +
                        f" {_tv_val_start} — {_tv_val_end}")

                    sev = _tv_overlap["severity"]
                    msg = _tv_overlap[f"message_{'en' if language == 'English' else 'pt'}"]
                    if sev == "success":
                        st.success(msg)
                    elif sev == "warning":
                        st.warning(msg)
                    elif sev == "error":
                        st.error(msg)
                    else:
                        st.info(msg)

                # ── 5. Run button ──
                st.divider()
                _tv_run = st.button(
                    tr("Run temporal validation", "Executar validação temporal"),
                    type="primary",
                    use_container_width=True,
                )

                if _tv_run:
                    _tv_progress = st.progress(0, text=tr("Preparing...", "Preparando..."))

                    # ── 5.1 Apply frozen AI Risk model ──
                    _tv_progress.progress(0.05, text=tr("Applying frozen AI Risk model...", "Aplicando modelo AI Risk congelado..."))

                    _tv_ref_df = prepared.data[prepared.feature_columns]
                    _tv_num_cols = _get_numeric_columns_from_pipeline(artifacts.fitted_models[forced_model])
                    _tv_results = []

                    for _tv_idx in range(_tv_n):
                        _tv_row_data = _tv_data.iloc[[_tv_idx]]
                        try:
                            _tv_form = _tv_row_data.to_dict(orient="records")[0]
                            _tv_input = _build_input_row(artifacts.feature_columns, _tv_form)
                            _tv_input = _align_input_to_training_schema(_tv_input, _tv_ref_df)
                            _tv_model_input = clean_features(_tv_input[artifacts.feature_columns], numeric_columns=_tv_num_cols)

                            # Force numeric columns that are still object
                            for _c in _tv_model_input.columns:
                                if _tv_model_input[_c].dtype == object and _c in _tv_ref_df.columns and pd.api.types.is_numeric_dtype(_tv_ref_df[_c]):
                                    _tv_model_input[_c] = pd.to_numeric(
                                        _tv_model_input[_c].astype(str).str.replace(',', '.', regex=False),
                                        errors="coerce",
                                    )

                            _tv_ia_prob = float(artifacts.fitted_models[forced_model].predict_proba(_tv_model_input)[:, 1][0])

                            # Completeness
                            _tv_comp = assess_input_completeness(artifacts.feature_columns, _tv_input, language)

                            _tv_results.append({
                                "ia_risk": _tv_ia_prob,
                                "completeness": _tv_comp["level"],
                            })
                        except Exception:
                            _tv_results.append({"ia_risk": np.nan, "completeness": "error"})

                        if (_tv_idx + 1) % max(1, _tv_n // 20) == 0:
                            _tv_progress.progress(
                                0.05 + 0.35 * (_tv_idx + 1) / _tv_n,
                                text=tr(
                                    f"AI Risk: {_tv_idx + 1}/{_tv_n}",
                                    f"AI Risk: {_tv_idx + 1}/{_tv_n}",
                                ),
                            )

                    _tv_data["ia_risk"] = [r["ia_risk"] for r in _tv_results]
                    _tv_data["_completeness"] = [r["completeness"] for r in _tv_results]

                    # ── 5.2 EuroSCORE II ──
                    _tv_progress.progress(0.42, text=tr("Computing EuroSCORE II...", "Calculando EuroSCORE II..."))
                    _tv_data["euroscore_calc"] = _tv_data.apply(euroscore_from_row, axis=1)

                    # ── 5.3 STS ──
                    _tv_data["sts_score"] = np.nan
                    _tv_sts_ok = False
                    if HAS_STS:
                        _tv_progress.progress(0.50, text=tr("Querying STS web calculator...", "Consultando calculadora web do STS..."))
                        try:
                            _tv_sts_rows = _tv_data.to_dict(orient="records")
                            _tv_sts_results = calculate_sts_batch(_tv_sts_rows)
                            if _tv_sts_results:
                                _tv_data["sts_score"] = [r.get("predmort", np.nan) for r in _tv_sts_results]
                                _tv_sts_ok = _tv_data["sts_score"].notna().sum() > 0
                        except Exception as _tv_sts_err:
                            st.warning(tr(
                                f"STS calculation failed: {_tv_sts_err}. Continuing without STS.",
                                f"Cálculo do STS falhou: {_tv_sts_err}. Continuando sem STS.",
                            ))

                    if not _tv_sts_ok:
                        st.info(tr(
                            "STS scores are not available. Analysis will proceed with AI Risk and EuroSCORE II only.",
                            "Escores STS não estão disponíveis. A análise prosseguirá apenas com AI Risk e EuroSCORE II.",
                        ))

                    # ── 5.4 Risk classes ──
                    _tv_progress.progress(0.80, text=tr("Computing metrics...", "Calculando métricas..."))
                    _tv_data["class_ia"] = _tv_data["ia_risk"].map(class_risk)
                    _tv_data["class_euro"] = _tv_data["euroscore_calc"].map(class_risk)
                    _tv_data["class_sts"] = _tv_data["sts_score"].map(lambda x: class_risk(x) if pd.notna(x) else np.nan)

                    # Score columns and rename map
                    _tv_score_cols = ["ia_risk", "euroscore_calc"]
                    _tv_rename = {"ia_risk": "AI Risk", "euroscore_calc": "EuroSCORE II"}
                    if _tv_sts_ok:
                        _tv_score_cols.append("sts_score")
                        _tv_rename["sts_score"] = "STS"

                    # ── 6. Metrics ──
                    # 6.1 Performance table
                    _tv_perf = evaluate_scores_temporal(
                        _tv_data, "morte_30d", _tv_score_cols, _tv_locked_threshold,
                        n_boot=AppConfig.N_BOOTSTRAP_SAMPLES, seed=AppConfig.BOOTSTRAP_SEED,
                    )
                    if not _tv_perf.empty:
                        _tv_perf["Score"] = _tv_perf["Score"].replace(_tv_rename)

                    # 6.2 Pairwise comparison
                    _tv_pairs = [("ia_risk", "euroscore_calc")]
                    if _tv_sts_ok:
                        _tv_pairs.append(("ia_risk", "sts_score"))
                        _tv_pairs.append(("sts_score", "euroscore_calc"))
                    _tv_pairwise = pairwise_score_comparison(
                        _tv_data, "morte_30d", _tv_pairs,
                        n_boot=AppConfig.N_BOOTSTRAP_SAMPLES, seed=AppConfig.BOOTSTRAP_SEED,
                    )
                    if not _tv_pairwise.empty:
                        for _old, _new in _tv_rename.items():
                            _tv_pairwise["Comparison"] = _tv_pairwise["Comparison"].str.replace(_old, _new)

                    # 6.3 Risk categories
                    _tv_risk_cat = risk_category_table(_tv_data, "morte_30d", _tv_score_cols)
                    if not _tv_risk_cat.empty:
                        _tv_risk_cat["Score"] = _tv_risk_cat["Score"].replace(_tv_rename)

                    # 6.4 Calibration data
                    _tv_calib_rows = []
                    for _sc in _tv_score_cols:
                        _sub = _tv_data[["morte_30d", _sc]].dropna()
                        if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                            _cal = calibration_intercept_slope(_sub["morte_30d"].values, _sub[_sc].values)
                            _hl = hosmer_lemeshow_test(_sub["morte_30d"].values, _sub[_sc].values)
                            _tv_calib_rows.append({
                                "Score": _tv_rename.get(_sc, _sc),
                                "Calibration_Intercept": _cal["Calibration intercept"],
                                "Calibration_Slope": _cal["Calibration slope"],
                                "HL_chi2": _hl["HL chi-square"],
                                "HL_p": _hl["HL p-value"],
                            })
                    _tv_calib_df = pd.DataFrame(_tv_calib_rows) if _tv_calib_rows else pd.DataFrame()

                    # Cohort summary dict
                    _comp_counts = _tv_data["_completeness"].value_counts()
                    _tv_cohort_summary = {
                        "n_total": _tv_n,
                        "n_events": _tv_events,
                        "event_rate": _tv_rate,
                        "date_range": f"{_tv_val_start} — {_tv_val_end}",
                        "n_complete": int(_comp_counts.get("complete", 0)),
                        "n_adequate": int(_comp_counts.get("adequate", 0)),
                        "n_partial": int(_comp_counts.get("partial", 0)),
                        "n_low": int(_comp_counts.get("low", 0)),
                    }

                    _tv_progress.progress(1.0, text=tr("Done.", "Concluído."))

                    # ── 7. Display results ──
                    st.divider()
                    st.markdown(tr("### Results", "### Resultados"))

                    # 7.1 Cohort summary
                    with st.expander(tr("Cohort summary", "Resumo da coorte"), expanded=True):
                        _cs = _tv_cohort_summary
                        _cs_c1, _cs_c2, _cs_c3, _cs_c4 = st.columns(4)
                        _cs_c1.metric(tr("Total", "Total"), _cs["n_total"])
                        _cs_c2.metric(tr("Events", "Eventos"), _cs["n_events"])
                        _cs_c3.metric(tr("Event rate", "Taxa de eventos"), f"{_cs['event_rate']:.1%}")
                        _cs_c4.metric(tr("Date range", "Período"), _cs["date_range"])

                        _comp_df = pd.DataFrame([
                            {tr("Level", "Nível"): tr("Complete", "Completo"), "n": _cs["n_complete"], "%": f"{_cs['n_complete']/_cs['n_total']*100:.1f}" if _cs["n_total"] else "0"},
                            {tr("Level", "Nível"): tr("Adequate", "Adequado"), "n": _cs["n_adequate"], "%": f"{_cs['n_adequate']/_cs['n_total']*100:.1f}" if _cs["n_total"] else "0"},
                            {tr("Level", "Nível"): tr("Partially imputed", "Parcialmente imputado"), "n": _cs["n_partial"], "%": f"{_cs['n_partial']/_cs['n_total']*100:.1f}" if _cs["n_total"] else "0"},
                            {tr("Level", "Nível"): tr("Heavily imputed", "Muito imputado"), "n": _cs["n_low"], "%": f"{_cs['n_low']/_cs['n_total']*100:.1f}" if _cs["n_total"] else "0"},
                        ])
                        st.dataframe(_comp_df, width="stretch", hide_index=True)

                    # 7.2 Performance table
                    if not _tv_perf.empty:
                        with st.expander(tr("Discrimination and calibration", "Discriminação e calibração"), expanded=True):
                            st.caption(tr(
                                f"95% CI by bootstrap ({AppConfig.N_BOOTSTRAP_SAMPLES} resamples). "
                                f"Classification metrics at locked threshold = {_tv_locked_threshold:.0%}.",
                                f"IC 95% por bootstrap ({AppConfig.N_BOOTSTRAP_SAMPLES} reamostras). "
                                f"Métricas de classificação no limiar bloqueado = {_tv_locked_threshold:.0%}.",
                            ))
                            _tv_perf_display = _tv_perf.copy()
                            # Format for display
                            for _fc in ["AUC", "AUPRC", "Brier", "Calibration_Intercept", "Calibration_Slope",
                                        "Sensitivity", "Specificity", "PPV", "NPV"]:
                                if _fc in _tv_perf_display.columns:
                                    _tv_perf_display[_fc] = _tv_perf_display[_fc].map(
                                        lambda v: f"{v:.3f}" if pd.notna(v) else "—"
                                    )
                            for _fc in ["AUC_IC95_inf", "AUC_IC95_sup", "AUPRC_IC95_inf", "AUPRC_IC95_sup"]:
                                if _fc in _tv_perf_display.columns:
                                    _tv_perf_display[_fc] = _tv_perf_display[_fc].map(
                                        lambda v: f"{v:.3f}" if pd.notna(v) else "—"
                                    )
                            if "HL_p" in _tv_perf_display.columns:
                                _tv_perf_display["HL_p"] = _tv_perf_display["HL_p"].map(
                                    lambda v: f"{v:.4f}" if pd.notna(v) else "—"
                                )
                            st.dataframe(_tv_perf_display, width="stretch", hide_index=True)

                    # 7.3 Pairwise comparison
                    if not _tv_pairwise.empty:
                        with st.expander(tr("Pairwise comparison", "Comparação pareada"), expanded=True):
                            _tv_pw_display = _tv_pairwise.copy()
                            for _fc in ["Delta_AUC", "Delta_AUC_IC95_inf", "Delta_AUC_IC95_sup", "NRI", "IDI"]:
                                if _fc in _tv_pw_display.columns:
                                    _tv_pw_display[_fc] = _tv_pw_display[_fc].map(
                                        lambda v: f"{v:.3f}" if pd.notna(v) else "—"
                                    )
                            for _fc in ["Bootstrap_p", "DeLong_p"]:
                                if _fc in _tv_pw_display.columns:
                                    _tv_pw_display[_fc] = _tv_pw_display[_fc].map(
                                        lambda v: f"{v:.4f}" if pd.notna(v) else "—"
                                    )
                            st.dataframe(_tv_pw_display, width="stretch", hide_index=True)

                    # 7.4 Risk categories
                    if not _tv_risk_cat.empty:
                        with st.expander(tr("Risk category distribution", "Distribuição por classe de risco"), expanded=True):
                            _tv_rc_display = _tv_risk_cat.copy()
                            _tv_rc_display["Observed_mortality"] = _tv_rc_display["Observed_mortality"].map(
                                lambda v: f"{v:.1%}" if pd.notna(v) else "—"
                            )
                            st.dataframe(_tv_rc_display, width="stretch", hide_index=True)

                    # ── 8. Graphs ──
                    st.divider()
                    st.markdown(tr("### Graphs", "### Gráficos"))

                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    from sklearn.metrics import (
                        precision_recall_curve as _pr_curve_fn,
                        average_precision_score as _auprc_fn,
                        roc_auc_score as _roc_auc_fn,
                    )

                    _tv_y = _tv_data["morte_30d"].values

                    # 8.1 ROC curves
                    _fig_roc, _ax_roc = plt.subplots(figsize=(7, 6))
                    _ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
                    for _sc in _tv_score_cols:
                        _sub = _tv_data[["morte_30d", _sc]].dropna()
                        if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                            fpr, tpr = roc_data(_sub["morte_30d"].values, _sub[_sc].values)
                            _auc_val = float(_roc_auc_fn(_sub["morte_30d"].values, _sub[_sc].values))
                            _ax_roc.plot(fpr, tpr, label=f"{_tv_rename.get(_sc, _sc)} (AUC={_auc_val:.3f})")
                    _ax_roc.set_xlabel("1 - Specificity (FPR)")
                    _ax_roc.set_ylabel("Sensitivity (TPR)")
                    _ax_roc.set_title(tr("ROC Curves — Temporal Validation", "Curvas ROC — Validação Temporal"))
                    _ax_roc.legend(loc="lower right")
                    plt.tight_layout()
                    st.pyplot(_fig_roc)
                    plt.close(_fig_roc)

                    # 8.2 Precision-Recall curves
                    _fig_pr, _ax_pr = plt.subplots(figsize=(7, 6))
                    for _sc in _tv_score_cols:
                        _sub = _tv_data[["morte_30d", _sc]].dropna()
                        if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                            precision, recall, _ = _pr_curve_fn(_sub["morte_30d"].values, _sub[_sc].values)
                            _auprc_val = float(_auprc_fn(_sub["morte_30d"].values, _sub[_sc].values))
                            _ax_pr.plot(recall, precision, label=f"{_tv_rename.get(_sc, _sc)} (AUPRC={_auprc_val:.3f})")
                    _ax_pr.set_xlabel("Recall (Sensitivity)")
                    _ax_pr.set_ylabel("Precision (PPV)")
                    _ax_pr.set_title(tr("Precision-Recall Curves — Temporal Validation", "Curvas Precisão-Recall — Validação Temporal"))
                    _ax_pr.legend(loc="upper right")
                    plt.tight_layout()
                    st.pyplot(_fig_pr)
                    plt.close(_fig_pr)

                    # 8.3 Calibration curves
                    _fig_cal, _ax_cal = plt.subplots(figsize=(7, 6))
                    _ax_cal.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
                    for _sc in _tv_score_cols:
                        _sub = _tv_data[["morte_30d", _sc]].dropna()
                        if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                            prob_pred, prob_true = calibration_data(_sub["morte_30d"].values, _sub[_sc].values)
                            _ax_cal.plot(prob_pred, prob_true, "o-", label=_tv_rename.get(_sc, _sc))
                    _ax_cal.set_xlabel(tr("Predicted probability", "Probabilidade predita"))
                    _ax_cal.set_ylabel(tr("Observed frequency", "Frequência observada"))
                    _ax_cal.set_title(tr("Calibration Curves — Temporal Validation", "Curvas de Calibração — Validação Temporal"))
                    _ax_cal.legend()
                    plt.tight_layout()
                    st.pyplot(_fig_cal)
                    plt.close(_fig_cal)

                    # 8.4 Distribution of predicted probabilities
                    _fig_dist, _ax_dist = plt.subplots(figsize=(7, 5))
                    for _sc in _tv_score_cols:
                        _vals = _tv_data[_sc].dropna().values
                        if len(_vals) > 0:
                            _ax_dist.hist(_vals, bins=30, alpha=0.5, label=_tv_rename.get(_sc, _sc), density=True)
                    _ax_dist.set_xlabel(tr("Predicted probability", "Probabilidade predita"))
                    _ax_dist.set_ylabel(tr("Density", "Densidade"))
                    _ax_dist.set_title(tr("Distribution of Predicted Probabilities", "Distribuição das Probabilidades Preditas"))
                    _ax_dist.legend()
                    plt.tight_layout()
                    st.pyplot(_fig_dist)
                    plt.close(_fig_dist)

                    # 8.5 Decision Curve Analysis
                    _tv_dca_scores = {}
                    for _sc in _tv_score_cols:
                        _sub = _tv_data[["morte_30d", _sc]].dropna()
                        if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                            _tv_dca_scores[_tv_rename.get(_sc, _sc)] = _sub[_sc].values
                    if _tv_dca_scores:
                        _tv_dca_y = _tv_data.loc[_tv_data[_tv_score_cols[0]].notna(), "morte_30d"].values
                        _tv_thresholds = np.linspace(0.01, 0.30, 30)
                        _tv_dca_df = decision_curve(_tv_dca_y, _tv_dca_scores, _tv_thresholds)

                        _fig_dca, _ax_dca = plt.subplots(figsize=(8, 6))
                        for _strat in _tv_dca_df["Strategy"].unique():
                            _dca_sub = _tv_dca_df[_tv_dca_df["Strategy"] == _strat]
                            _style = "--" if _strat in ("Treat all", "Treat none") else "-"
                            _alpha = 0.5 if _strat in ("Treat all", "Treat none") else 1.0
                            _ax_dca.plot(_dca_sub["Threshold"], _dca_sub["Net Benefit"], _style, alpha=_alpha, label=_strat)
                        _ax_dca.set_xlabel(tr("Threshold probability", "Probabilidade limiar"))
                        _ax_dca.set_ylabel(tr("Net benefit", "Benefício líquido"))
                        _ax_dca.set_title(tr("Decision Curve Analysis — Temporal Validation", "Análise de Curva de Decisão — Validação Temporal"))
                        _ax_dca.legend(loc="upper right", fontsize=8)
                        _ax_dca.set_ylim(bottom=-0.05)
                        plt.tight_layout()
                        st.pyplot(_fig_dca)
                        plt.close(_fig_dca)

                    # 8.6 Observed vs Predicted by risk decile
                    _fig_decile, _ax_decile = plt.subplots(figsize=(7, 5))
                    _has_decile = False
                    for _sc_i, _sc in enumerate(_tv_score_cols):
                        _sub = _tv_data[["morte_30d", _sc]].dropna()
                        if len(_sub) >= 20:
                            _sub = _sub.copy()
                            try:
                                _sub["_decile"] = pd.qcut(_sub[_sc], q=10, labels=False, duplicates="drop")
                            except ValueError:
                                _sub["_decile"] = pd.cut(_sub[_sc], bins=10, labels=False, duplicates="drop")
                            _grp = _sub.groupby("_decile", observed=True).agg(
                                predicted=(_sc, "mean"),
                                observed=("morte_30d", "mean"),
                            ).reset_index()
                            _offset = (_sc_i - 1) * 0.003
                            _ax_decile.scatter(_grp["predicted"] + _offset, _grp["observed"], label=_tv_rename.get(_sc, _sc), s=40, zorder=3)
                            _has_decile = True
                    if _has_decile:
                        _ax_decile.plot([0, max(0.3, _ax_decile.get_xlim()[1])], [0, max(0.3, _ax_decile.get_xlim()[1])], "k--", lw=0.8, alpha=0.5)
                        _ax_decile.set_xlabel(tr("Mean predicted probability", "Probabilidade predita média"))
                        _ax_decile.set_ylabel(tr("Observed mortality", "Mortalidade observada"))
                        _ax_decile.set_title(tr("Observed vs Predicted by Decile", "Observado vs Predito por Decil"))
                        _ax_decile.legend()
                        plt.tight_layout()
                        st.pyplot(_fig_decile)
                    plt.close(_fig_decile)

                    # 8.7 Confusion matrix at locked threshold
                    from stats_compare import classification_metrics_at_threshold
                    _fig_cm, _axes_cm = plt.subplots(1, len(_tv_score_cols), figsize=(5 * len(_tv_score_cols), 4))
                    if len(_tv_score_cols) == 1:
                        _axes_cm = [_axes_cm]
                    for _ax_i, _sc in enumerate(_tv_score_cols):
                        _sub = _tv_data[["morte_30d", _sc]].dropna()
                        if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                            _cls = classification_metrics_at_threshold(_sub["morte_30d"].values, _sub[_sc].values, _tv_locked_threshold)
                            _cm = np.array([[_cls["TN"], _cls["FP"]], [_cls["FN"], _cls["TP"]]])
                            _ax = _axes_cm[_ax_i]
                            _im = _ax.imshow(_cm, cmap="Blues", interpolation="nearest")
                            for (_r, _cc), _val in np.ndenumerate(_cm):
                                _ax.text(_cc, _r, str(_val), ha="center", va="center", fontsize=14, fontweight="bold")
                            _ax.set_xticks([0, 1])
                            _ax.set_xticklabels([tr("Predicted -", "Predito -"), tr("Predicted +", "Predito +")])
                            _ax.set_yticks([0, 1])
                            _ax.set_yticklabels([tr("Actual -", "Real -"), tr("Actual +", "Real +")])
                            _ax.set_title(f"{_tv_rename.get(_sc, _sc)}\n(threshold={_tv_locked_threshold:.0%})")
                    plt.tight_layout()
                    st.pyplot(_fig_cm)
                    plt.close(_fig_cm)

                    # 7.5 Case-level predictions table
                    st.divider()
                    st.markdown(tr("### Case-level predictions", "### Predições por paciente"))

                    _tv_case_cols = []
                    for _cand in ("Name", "Nome", "_patient_key"):
                        if _cand in _tv_data.columns:
                            _tv_case_cols.append(_cand)
                            break
                    for _date_cand in ("surgery_year", "surgery_quarter"):
                        if _date_cand in _tv_data.columns:
                            _tv_case_cols.append(_date_cand)
                    _tv_case_cols.append("morte_30d")
                    _tv_case_cols.extend(["ia_risk", "euroscore_calc"])
                    if _tv_sts_ok:
                        _tv_case_cols.append("sts_score")
                    _tv_case_cols.extend(["class_ia", "class_euro"])
                    if _tv_sts_ok:
                        _tv_case_cols.append("class_sts")
                    _tv_case_cols.append("_completeness")
                    _tv_case_cols = [c for c in _tv_case_cols if c in _tv_data.columns]

                    _tv_case_df = _tv_data[_tv_case_cols].copy()
                    # Rename for display
                    _tv_case_rename = {
                        "morte_30d": tr("Outcome", "Desfecho"),
                        "ia_risk": "AI Risk",
                        "euroscore_calc": "EuroSCORE II",
                        "sts_score": "STS",
                        "class_ia": tr("AI Risk class", "Classe AI Risk"),
                        "class_euro": tr("EuroSCORE II class", "Classe EuroSCORE II"),
                        "class_sts": tr("STS class", "Classe STS"),
                        "_completeness": tr("Completeness", "Completude"),
                    }
                    _tv_case_df = _tv_case_df.rename(columns=_tv_case_rename)

                    # Format probabilities as percentages
                    for _pc in ["AI Risk", "EuroSCORE II", "STS"]:
                        if _pc in _tv_case_df.columns:
                            _tv_case_df[_pc] = _tv_case_df[_pc].map(
                                lambda v: f"{v*100:.2f}%" if pd.notna(v) else "—"
                            )

                    # Outcome filter
                    _tv_filter_outcome = st.selectbox(
                        tr("Filter by outcome", "Filtrar por desfecho"),
                        [tr("All", "Todos"), tr("Events only", "Apenas eventos"), tr("Non-events only", "Apenas não-eventos")],
                    )
                    _tv_display_case = _tv_case_df.copy()
                    _out_col = tr("Outcome", "Desfecho")
                    if _tv_filter_outcome == tr("Events only", "Apenas eventos"):
                        _tv_display_case = _tv_display_case[_tv_display_case[_out_col] == 1]
                    elif _tv_filter_outcome == tr("Non-events only", "Apenas não-eventos"):
                        _tv_display_case = _tv_display_case[_tv_display_case[_out_col] == 0]

                    st.dataframe(_tv_display_case, width="stretch", hide_index=True)

                    # ── 9. Exports ──
                    st.divider()
                    st.markdown(tr("### Export results", "### Exportar resultados"))

                    # Build markdown report
                    _tv_md = build_temporal_validation_summary(
                        _tv_cohort_summary, _tv_perf, _tv_pairwise, _tv_calib_df,
                        _tv_risk_cat, _tv_meta, _tv_locked_threshold, language,
                    )

                    # 9.1 XLSX
                    _tv_xlsx_buf = BytesIO()
                    with pd.ExcelWriter(_tv_xlsx_buf, engine="openpyxl") as _tv_writer:
                        # Cohort summary sheet
                        _cs_export = pd.DataFrame([
                            {"Property": k, "Value": v} for k, v in _tv_cohort_summary.items()
                        ])
                        _cs_export.to_excel(_tv_writer, sheet_name="cohort_summary", index=False)
                        # Performance sheet
                        if not _tv_perf.empty:
                            _tv_perf.to_excel(_tv_writer, sheet_name="performance", index=False)
                        # Pairwise sheet
                        if not _tv_pairwise.empty:
                            _tv_pairwise.to_excel(_tv_writer, sheet_name="pairwise_comparison", index=False)
                        # Risk categories sheet
                        if not _tv_risk_cat.empty:
                            _tv_risk_cat.to_excel(_tv_writer, sheet_name="risk_categories", index=False)
                        # Calibration sheet
                        if not _tv_calib_df.empty:
                            _tv_calib_df.to_excel(_tv_writer, sheet_name="calibration", index=False)
                        # Case-level predictions sheet
                        _tv_case_export = _tv_data[_tv_case_cols].copy()
                        _tv_case_export.to_excel(_tv_writer, sheet_name="case_level_predictions", index=False)
                    _tv_xlsx_bytes = _tv_xlsx_buf.getvalue()

                    # 9.2 CSV
                    _tv_csv_export = _tv_data[_tv_case_cols].copy()
                    _tv_csv_bytes = _tv_csv_export.to_csv(index=False).encode("utf-8")

                    # 9.4 PDF
                    _tv_pdf_bytes = statistical_summary_to_pdf(_tv_md)

                    # 9.5 Markdown
                    _tv_md_bytes = _tv_md.encode("utf-8")

                    _ex_c1, _ex_c2, _ex_c3, _ex_c4 = st.columns(4)
                    with _ex_c1:
                        st.download_button(
                            tr("Download XLSX", "Baixar XLSX"),
                            data=_tv_xlsx_bytes,
                            file_name="temporal_validation_summary.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    with _ex_c2:
                        st.download_button(
                            tr("Download CSV", "Baixar CSV"),
                            data=_tv_csv_bytes,
                            file_name="temporal_validation_predictions.csv",
                            mime="text/csv",
                        )
                    with _ex_c3:
                        if _tv_pdf_bytes:
                            st.download_button(
                                tr("Download PDF", "Baixar PDF"),
                                data=_tv_pdf_bytes,
                                file_name="temporal_validation_report.pdf",
                                mime="application/pdf",
                            )
                        else:
                            st.caption(tr("PDF export requires fpdf library.", "Exportação PDF requer biblioteca fpdf."))
                    with _ex_c4:
                        st.download_button(
                            tr("Download Markdown", "Baixar Markdown"),
                            data=_tv_md_bytes,
                            file_name="temporal_validation_report.md",
                            mime="text/markdown",
                        )

                    # Log audit
                    log_analysis(
                        analysis_type="temporal_validation",
                        source_file=_tv_file.name,
                        model_version=MODEL_VERSION,
                        n_patients=_tv_n,
                        extra={
                            "n_events": _tv_events,
                            "event_rate": round(_tv_rate, 4),
                            "validation_date_range": f"{_tv_val_start} — {_tv_val_end}",
                            "sts_available": _tv_sts_ok,
                        },
                    )

# ── Footer ──
st.divider()
st.caption(
    tr(
        "\u26a0\ufe0f This is a research tool developed as part of a master's dissertation. "
        "It is not intended for autonomous clinical decision-making.",
        "\u26a0\ufe0f Esta é uma ferramenta de pesquisa desenvolvida como parte de uma dissertação de mestrado. "
        "Não se destina à tomada de decisão clínica autônoma.",
    )
)
