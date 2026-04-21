# force-reload-v5: ensure modeling.py / sts_calculator.py / sts_cache.py changes are picked up
import warnings as _warnings

_SKLEARN_PARALLEL_DELAYED_WARNING = (
    r"`sklearn\.utils\.parallel\.delayed` should be used with "
    r"`sklearn\.utils\.parallel\.Parallel`.*"
)
_warnings.filterwarnings(
    "ignore",
    message=_SKLEARN_PARALLEL_DELAYED_WARNING,
    category=UserWarning,
)
_warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)

import importlib as _il
import modeling as _modeling_mod
try:
    _il.reload(_modeling_mod)
except Exception:
    pass
try:
    import sts_cache as _sts_cache_mod
    _il.reload(_sts_cache_mod)
except Exception:
    _sts_cache_mod = None  # type: ignore[assignment]
import sts_calculator as _sts_mod
try:
    _il.reload(_sts_mod)
except Exception:
    pass

import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List
import re
import shutil
import sqlite3
import time
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
    normalize_arrhythmia_recent_value,
    normalize_arrhythmia_remote_value,
    normalize_cva_value,
    normalize_hf_value,
    normalize_pneumonia_value,
    parse_number,
    prepare_master_dataset,
    procedure_weight,
    split_surgery_procedures,
    thoracic_aorta_surgery,
)
from sts_calculator import (
    HAS_WEBSOCKETS as HAS_STS,
    STS_LABELS,
    STS_PER_PATIENT_TIMEOUT_S,
    calculate_sts,
    calculate_sts_batch,
    classify_sts_eligibility,
)
from stats_compare import (
    bootstrap_auc_diff,
    bootstrap_metrics_ci,
    calibration_data,
    calibration_intercept_slope,
    class_risk,
    classification_metrics_at_threshold,
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
    is_surrogate_timeline,        # surrogate-year detection for temporal-validation UI
    build_surrogate_timeline_note,
)
from app_data_dictionary import (
    build_dictionary_xlsx_bytes,
    get_app_reading_dictionary_dataframe,
    get_reading_aliases_dataframe,
    get_reading_rules_dataframe,
)
from ai_risk_inference import (
    _get_numeric_columns_from_pipeline,
    _safe_select_features,
    _build_input_row,
    _align_input_to_training_schema,
    _patient_identifier_from_row,
    _run_ai_risk_inference_row,
    apply_frozen_model_to_temporal_cohort,
)
from tabs import TabContext
from tabs import comparison as _tab_comparison
from tabs import batch_export as _tab_batch_export
from tabs import temporal_validation as _tab_temporal_validation


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
            tr("Step", "Etapa"): tr("Preoperative rows read", "Linhas lidas na planilha 'Preoperative'"),
            tr("Count", "Quantidade"): int(info.get("pre_rows_before_criteria", 0)),
        },
        {
            tr("Step", "Etapa"): tr("Excluded: missing Surgery or Procedure Date", "Excluídos por ausência de 'Surgery' ou 'Procedure Date'"),
            tr("Count", "Quantidade"): int(info.get("excluded_missing_surgery_or_date", 0)),
        },
        {
            tr("Step", "Etapa"): tr("Eligible after inclusion criteria", "Elegíveis após critérios de inclusão"),
            tr("Count", "Quantidade"): int(info.get("pre_rows_after_criteria", 0)),
        },
        {
            tr("Step", "Etapa"): tr("Unique patient-procedure keys in Postoperative", "Chaves únicas na planilha 'Postoperative'"),
            tr("Count", "Quantidade"): int(info.get("post_unique_patient_date", 0)),
        },
        {
            tr("Step", "Etapa"): tr("Matched Preoperative-Postoperative rows", "Linhas pareadas ('Preoperative' ↔ 'Postoperative')"),
            tr("Count", "Quantidade"): int(info.get("matched_pre_post_rows", 0)),
        },
        {
            tr("Step", "Etapa"): tr("Excluded: no Preoperative-Postoperative match", "Excluídos sem correspondência em 'Postoperative'"),
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


@st.fragment
def _bytes_download_btn(data: bytes, filename: str, label: str, mime: str, key: str = None) -> None:
    """Generic bytes download button isolated in a fragment — clicking it won't rerun the whole page."""
    st.download_button(label, data=data, file_name=filename, mime=mime, key=key)


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


def _sts_score_patient_id(row: dict) -> "str | None":
    """Extract a stable STS Score patient identifier from a row dict.

    Used by the STS Score cache layer to key the cross-hash stale
    fallback index. Returns None when no stable identifier is available.
    """
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


def _sts_score_patient_ids(rows) -> list:
    return [_sts_score_patient_id(r) for r in rows]


def _format_training_sts_phase(label: object, phase_num: object = None, phase_total: object = None) -> str:
    """Translate STS Score operational phase labels without changing worker logic."""
    raw = str(label or "").strip().lower()
    labels = {
        "checking cache": tr("checking cache", "verificando cache"),
        "identifying cache misses": tr("identifying cache misses", "identificando itens fora do cache"),
        "querying web calculator": tr("querying web calculator", "consultando a calculadora web"),
        "validating and consolidating": tr("validating and consolidating", "validando e consolidando resultados"),
        "STS Score processing": tr("STS Score processing", "processamento do STS Score"),
    }
    text = labels.get(raw, str(label or tr("STS Score processing", "processamento do STS Score")))
    if phase_num and phase_total:
        return f"{tr('Phase', 'Fase')} {phase_num}/{phase_total}: {text}"
    return text


def _format_training_sts_detail(detail: object) -> str:
    """Make known STS Score progress details readable in the selected language."""
    raw = str(detail or "").strip()
    if not raw or language == "English":
        return raw
    m = re.match(r"^(\d+)/(\d+) checked$", raw)
    if m:
        return f"{m.group(1)}/{m.group(2)} verificados"
    m = re.match(r"^(\d+) cache hits?, (\d+) misses$", raw)
    if m:
        return f"{m.group(1)} em cache, {m.group(2)} fora do cache"
    m = re.match(r"^(\d+) patients? to fetch$", raw)
    if m:
        return f"{m.group(1)} pacientes para consultar"
    return raw


def _format_training_sts_status(status: object) -> str:
    raw = str(status or "").strip()
    if language == "English":
        return raw
    return {
        "cached": "em cache",
        "fresh": "novo",
        "refreshed": "cache atualizado",
        "stale_fallback": "fallback de cache",
        "failed": "falha",
        "unknown": "desconhecido",
    }.get(raw, raw)


def _format_training_sts_error(message: object) -> str:
    raw = str(message or "").strip()
    if language == "English" or not raw:
        return raw
    prefix = "STS Score query returned no usable result for "
    if raw.startswith(prefix):
        patient = raw[len(prefix):].strip()
        return f"A consulta do STS Score não retornou resultado utilizável para {patient}"
    return raw


def _update_phase(slot, phase_num: int, phase_total: int, label: str) -> None:
    """Update a st.empty() phase-label slot. No-op if slot is None or on error."""
    try:
        if slot is not None:
            slot.caption(f"{tr('Phase', 'Fase')} {phase_num}/{phase_total}: {label}")
    except Exception:
        pass


def _compute_bundle(xlsx_path: str, progress_callback=None) -> Dict[str, object]:
    # Force-reload modeling module to pick up any code changes without server restart
    import importlib
    import config.model_config as _cfg_mod
    importlib.reload(_cfg_mod)
    import modeling as _mod
    importlib.reload(_mod)
    _fresh_train = _mod.train_and_select_model

    # Phase 3: observability — build a RunReport as each phase completes.
    import observability as _obs
    importlib.reload(_obs)
    run_report = _obs.RunReport()

    def _emit_progress(phase: str, current: int = 0, total: int = 1, detail=None) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(phase, current, total, detail or "")
        except Exception:
            pass

    if progress_callback is not None:
        try:
            progress_callback("loading_data", 0, 1, "")
        except Exception:
            pass
    prepared = prepare_master_dataset(xlsx_path)

    # Phase 3: ingestion + cohort eligibility steps. A required-column
    # failure is a hard stop; the report is attached to the raised error
    # so the caller can render it before aborting.
    _ingest_step = _obs.build_step_ingestion(
        getattr(prepared, "ingestion_report", None),
        feature_columns=getattr(prepared, "feature_columns", None),
    )
    run_report.add(_ingest_step)
    if _ingest_step is not None and _ingest_step.status == _obs.STATUS_ERROR:
        err = RuntimeError(
            "Ingestion halted: required columns are missing. "
            "See the observability report for details."
        )
        err.run_report = run_report  # type: ignore[attr-defined]
        err.source_label = Path(xlsx_path).name  # type: ignore[attr-defined]
        raise err
    run_report.add(_obs.build_step_eligibility(prepared.info))
    if progress_callback is not None:
        try:
            progress_callback("eligibility_done", 0, 1, "")
        except Exception:
            pass

    artifacts = _fresh_train(prepared.data, prepared.feature_columns, progress_callback=progress_callback)

    # Phase 3: training step.
    run_report.add(_obs.build_step_training(
        leaderboard=artifacts.leaderboard,
        best_model_name=artifacts.best_model_name,
        n_features=len(prepared.feature_columns),
        prevalence=float(prepared.info.get("positive_rate", 0.0) or 0.0),
    ))

    if progress_callback is not None:
        try:
            progress_callback("euroscore_calc", 0, 1, "")
        except Exception:
            pass
    df = prepared.data.copy()
    df["euroscore_calc"] = df.apply(euroscore_from_row, axis=1)
    df["euroscore_sheet_clean"] = pd.to_numeric(df["euroscore_sheet"], errors="coerce")
    df["euroscore_auto_sheet_clean"] = pd.to_numeric(df["euroscore_auto_sheet"], errors="coerce")

    # STS Score: query the web calculator for all patients, routed
    # through the Phase 2 persistent STS Score cache (14-day TTL,
    # patient-keyed stale fallback, structured execution log).
    sts_ws_results = []
    if HAS_STS:
        rows_as_dicts = df.to_dict(orient="records")
        _sts_pids = _sts_score_patient_ids(rows_as_dicts)
        _sts_total = len(rows_as_dicts)
        _sts_state = {
            "total": _sts_total,
            "processed": 0,
            "net_success": 0,
            "net_failed": 0,
            "pending": _sts_total,
            "current_patient": "",
            "current_batch_size": 0,
            "current_position": 0,
            "last_completed": "",
            "last_error": "",
            "phase_label": "",
            "phase_detail": "",
        }
        _sts_chunk_size = 5

        _emit_progress("sts_score_calc", 0, _sts_total, dict(_sts_state))

        def _sts_progress_cb(done, total):
            _sts_state["processed"] = int(done or 0)
            _sts_state["total"] = int(total or _sts_total or 0)
            _sts_state["pending"] = max(int(total or 0) - int(done or 0), 0)
            _emit_progress("sts_score_progress", int(done or 0), int(total or 0), dict(_sts_state))

        def _sts_phase_cb(phase_num, phase_total, label, detail=""):
            _sts_state["phase_label"] = label or ""
            _sts_state["phase_num"] = int(phase_num or 0)
            _sts_state["phase_total"] = int(phase_total or 0)
            _sts_state["phase_detail"] = detail or ""
            _emit_progress("sts_score_phase", int(phase_num or 0), int(phase_total or 0), dict(_sts_state))

        def _sts_chunk_start_cb(patient_idx, total_pending, patient_id=None):
            _idx = int(patient_idx or 0)
            _total_pending = int(total_pending or 0)
            _sts_state["current_position"] = _idx + 1
            _sts_state["current_patient"] = patient_id or ""
            _sts_state["current_batch_size"] = max(min(_sts_chunk_size, _total_pending - _idx), 0)
            _sts_state["pending_network"] = _total_pending
            _emit_progress("sts_score_patient", _idx, _total_pending, dict(_sts_state))

        def _sts_chunk_done_cb(patient_idx, total_pending, success):
            patient_label = _sts_state.get("current_patient") or f"row_{int(patient_idx or 0) + 1}"
            _sts_state["last_completed"] = patient_label
            if success:
                _sts_state["net_success"] = int(_sts_state.get("net_success", 0)) + 1
            else:
                _sts_state["net_failed"] = int(_sts_state.get("net_failed", 0)) + 1
                _sts_state["last_error"] = f"STS Score query returned no usable result for {patient_label}"
            _emit_progress("sts_score_patient_done", int(patient_idx or 0) + 1, int(total_pending or 0), dict(_sts_state))

        sts_ws_results = calculate_sts_batch(
            rows_as_dicts,
            patient_ids=_sts_pids,
            progress_callback=_sts_progress_cb,
            phase_callback=_sts_phase_cb,
            chunk_start_callback=_sts_chunk_start_cb,
            chunk_done_callback=_sts_chunk_done_cb,
            chunk_size=_sts_chunk_size,
        )

        _exec_log = list(getattr(calculate_sts_batch, "last_execution_log", []) or [])
        _fail_log = list(getattr(calculate_sts_batch, "failure_log", []) or [])
        _status_counts = {}
        for _rec in _exec_log:
            _status = getattr(_rec, "status", None) or "unknown"
            _status_counts[_status] = _status_counts.get(_status, 0) + 1
        _last_fail = _fail_log[-1] if _fail_log else {}
        _sts_done_detail = dict(_sts_state)
        _sts_done_detail.update({
            "processed": _sts_total,
            "pending": 0,
            "failures": len(_fail_log),
            "status_counts": _status_counts,
            "last_error": (_last_fail.get("reason") or _sts_state.get("last_error") or ""),
            "last_error_patient": (_last_fail.get("patient_id") or _last_fail.get("name") or ""),
        })
        _emit_progress("sts_score_done", _sts_total, _sts_total, _sts_done_detail)

    if sts_ws_results:
        df["sts_score"] = [r.get("predmort", np.nan) for r in sts_ws_results]
        # Store all STS sub-scores for later display
        for key in ["predmort", "predmm", "predstro", "predrenf", "predreop",
                     "predvent", "preddeep", "pred14d", "pred6d"]:
            df[f"sts_{key}"] = [r.get(key, np.nan) for r in sts_ws_results]
    else:
        # No fallback — STS is only available via the web calculator
        df["sts_score"] = np.nan

    # Phase 3: STS Score step (falls back gracefully if no log attached).
    run_report.add(_obs.build_step_sts_score(
        getattr(calculate_sts_batch, "last_execution_log", None)
    ))

    if progress_callback is not None:
        try:
            progress_callback("building_reports", 0, 1, "")
        except Exception:
            pass

    return {
        "prepared": prepared,
        "artifacts": artifacts,
        "data": df,
        "run_report": run_report,
    }


# Phase 4: bundle I/O helpers extracted to bundle_io.py.  The underscore-
# prefixed aliases are kept so internal call sites inside this module
# continue to work without touching their bodies.
from bundle_io import (
    bundle_signature as _bundle_signature,
    serialize_bundle as _serialize_bundle,
    deserialize_bundle as _deserialize_bundle,
    normalize_payload as _normalize_payload,
    BundleSchemaError as _BundleSchemaError,
    BUNDLE_SCHEMA_VERSION as _BUNDLE_SCHEMA_VERSION,
)


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

    def _has_flat_outcome(columns) -> bool:
        mapped = {
            FLAT_ALIAS_TO_APP_COLUMNS.get(str(c).strip(), str(c).strip())
            for c in columns
        }
        return bool({"morte_30d", "Death"} & mapped)

    if ext in {".csv", ".parquet"}:
        try:
            if ext == ".csv":
                df = pd.read_csv(path, sep=None, engine="python", nrows=5)
            else:
                df = pd.read_parquet(path)
        except Exception as e:
            return False, f"invalid_dataset: {e}"
        if not _has_flat_outcome(df.columns):
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
        if len(xls.sheet_names) == 1:
            sheet_name = xls.sheet_names[0]
            try:
                df = pd.read_excel(path, sheet_name=sheet_name, nrows=5)
            except Exception as e:
                return False, f"invalid_xlsx_flat_sheet: {e}"
            if _has_flat_outcome(df.columns):
                return True, "ok"
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
| **STS Score** | STS Risk Calculator (automated web query) | Obtained via automated queries to the official web calculator |

The app evaluates **discrimination and calibration** jointly, together with clinical utility and reclassification — all from preoperative data only. The operational clinical threshold is fixed at **8%**; per-model Youden thresholds are shown in the leaderboard as a complementary reference, not as the default.
""",
                """
Este app compara **três escores de risco** para predição de mortalidade hospitalar ou em 30 dias após cirurgia cardíaca:

| Escore | Como é calculado | Fonte |
|:--|:--|:--|
| **AI Risk** | Modelo de machine learning treinado nos seus dados locais | Este app |
| **EuroSCORE II** | Equação logística publicada (Nashef et al., 2012) | Calculado localmente |
| **STS Score** | STS Risk Calculator (consulta web automatizada) | Obtido via consultas automatizadas à calculadora web oficial |

O app avalia **discriminação e calibração** conjuntamente, juntamente com utilidade clínica e reclassificação — tudo a partir de dados exclusivamente pré-operatórios. O limiar clínico operacional é fixo em **8%**; os limiares de Youden por modelo são exibidos no leaderboard como referência complementar, e não como limiar padrão.
""",
            )
        )

    # ── 2. Data ──
    with st.expander(tr("Data and eligibility", "Dados e elegibilidade")):
        st.markdown(
            tr(
                """
**Data sources:** The app accepts multi-sheet Excel files (.xlsx with Preoperative, Pre-Echocardiogram, and Postoperative sheets), single-sheet flat Excel files, or flat CSV/Parquet files. CSV files with either `,` or `;` as field separator and comma or dot as decimal separator are handled automatically.

**Inclusion criteria (Excel):** Only records with 'Surgery' and 'Procedure Date' fields are included. Records are matched across the three sheets using patient identity and procedure date. Name and date are used only for internal linkage and **never as predictors**.

**Inclusion criteria (CSV):** All rows with a valid 'Surgery' field are included. Column names are mapped from snake_case (e.g. `lvef_pre_pct`) to the internal format automatically.

**Primary outcome:** 30-day or in-hospital mortality, extracted from the `Death` column. Values "Operative" or numeric ≤30 (days to death, including 0 = immediate postoperative death) are coded as events; ">30" or "-" are coded as survivors.
""",
                """
**Fontes de dados:** O app aceita arquivos Excel multi-abas (.xlsx com abas Preoperative, Pre-Echocardiogram e Postoperative), Excel plano de aba única, ou arquivos CSV/Parquet. Arquivos CSV com separador `,` ou `;` e decimal com vírgula ou ponto são tratados automaticamente.

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

**Calibration:** Post-hoc calibration is applied inside each CV fold using a per-model strategy — RandomForest uses sigmoid (Platt scaling) with inner cv≤5; LightGBM and CatBoost use isotonic with inner cv≤5; XGBoost uses isotonic with inner cv≤3; LogisticRegression and StackingEnsemble are used uncalibrated. The calibrated probability is used directly as the clinical output, with only a minimal numerical-stability bound (1e-6).

**Candidate models:** {', '.join(artifacts.leaderboard['Modelo'].tolist())}

**Validation:** StratifiedGroupKFold with {AppConfig.CV_SPLITS} folds, grouped by patient key (`_patient_key`). This ensures:
- The same patient **never** appears in both training and test folds
- Class balance (mortality rate) is preserved across folds
- Out-of-fold (OOF) predictions — including calibration applied inside each fold — are used for all performance metrics

**Model selection:** Candidate models are compared by cross-validated, calibrated OOF performance (discrimination and calibration). Automatic selection applies explicit clinical-usability guardrails to the calibrated OOF distribution — the auto-selected default must produce at least some predictions below the 8% clinical threshold, have AUC above a minimum floor, have a Brier score lower than the prevalence baseline, and have a non-degenerate dynamic range. Models that fail any guardrail stay visible in the leaderboard and can still be force-selected manually. Current operational default: **{artifacts.best_model_name}**.

**Leaderboard thresholds vs. operational threshold:** The leaderboard reports sensitivity and specificity at each model's own Youden's J threshold (OOF-optimal), shown as a complementary per-model reference. The **operational clinical threshold is fixed at 8%** and is the default used in the Statistical Comparison and temporal validation tabs — Youden is not the app's default operational threshold.
""",
                f"""
**Variáveis preditoras:** Apenas dados pré-operatórios (clínicos, laboratoriais, ecocardiográficos). Complicações pós-operatórias **nunca** são usadas como preditores, evitando vazamento temporal. Total: {prepared.info['n_features']} variáveis.

**Pré-processamento:**
- Variáveis numéricas: valores com vírgula decimal (ex: "64,7") são convertidos automaticamente; zeros clinicamente impossíveis (ex: BSA=0,00 quando altura/peso ausentes) são tratados como ausentes. Imputação pela mediana + normalização StandardScaler
- Variáveis de gravidade valvar: OrdinalEncoder com categorias clinicamente ordenadas (None < Trivial < Mild < Moderate < Severe) + imputação pela mediana + StandardScaler
- Demais variáveis categóricas: imputação pela moda + TargetEncoder (smooth="auto") + imputação pela mediana pós-codificação

**Calibração:** A calibração pós-hoc é aplicada dentro de cada fold de CV com estratégia por modelo — RandomForest usa sigmoid (Platt scaling) com cv interno ≤5; LightGBM e CatBoost usam isotonic com cv interno ≤5; XGBoost usa isotonic com cv interno ≤3; LogisticRegression e StackingEnsemble são usados sem calibração. A probabilidade calibrada é utilizada diretamente como saída clínica, com apenas um limite mínimo de estabilidade numérica (1e-6).

**Modelos candidatos:** {', '.join(artifacts.leaderboard['Modelo'].tolist())}

**Validação:** StratifiedGroupKFold com {AppConfig.CV_SPLITS} folds, agrupado por chave do paciente (`_patient_key`). Isso garante:
- O mesmo paciente **nunca** aparece em treino e teste simultaneamente
- O balanceamento de classes (taxa de mortalidade) é preservado entre os folds
- Predições out-of-fold (OOF) — incluindo calibração aplicada dentro de cada fold — são usadas para todas as métricas de desempenho

**Seleção do modelo:** Os modelos candidatos são comparados por desempenho OOF calibrado por validação cruzada (discriminação e calibração). A seleção automática aplica guardrails explícitos de usabilidade clínica à distribuição OOF calibrada — o modelo padrão automaticamente selecionado precisa produzir ao menos algumas predições abaixo do limiar clínico de 8%, ter AUC acima de um piso mínimo, ter Brier menor que o baseline da prevalência e apresentar amplitude dinâmica não-degenerada. Modelos que falham em qualquer guardrail permanecem visíveis no leaderboard e ainda podem ser forçados manualmente. Padrão operacional atual: **{artifacts.best_model_name}**.

**Limiares do leaderboard vs. limiar operacional:** O leaderboard reporta sensibilidade e especificidade no limiar de Youden (J) de cada modelo (ótimo OOF), exibido como referência complementar por modelo. O **limiar clínico operacional é fixo em 8%** e é o padrão usado nas abas de Comparação Estatística e de validação temporal — Youden não é o limiar operacional padrão do app.
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
The **STS Score** (STS Predicted Risk of Mortality) is obtained via **automated interaction with the official STS Risk Calculator web application** (Society of Thoracic Surgeons) hosted at `acsdriskcalc.research.sts.org`. The STS does not publish a documented public API; this implementation automates the same web calculator that clinicians use manually, via its WebSocket interface.

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
O **STS Score** (STS Predicted Risk of Mortality) é obtido via **interação automatizada com a calculadora web oficial do STS Risk Calculator** (Society of Thoracic Surgeons), hospedada em `acsdriskcalc.research.sts.org`. O STS não publica uma API pública documentada; esta implementação automatiza a mesma calculadora web que os clínicos usam manualmente, via sua interface WebSocket.

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
- **STS Score web calculator dependency:** STS Score calculation requires internet access and depends on the availability of the STS web calculator at acsdriskcalc.research.sts.org. The interface may change without notice. ~1–3% of patients may fail due to procedure mapping ambiguity.
- **Small subgroups:** Results in subgroups with <50 patients or <10 events should be interpreted with caution — confidence intervals may be wide.
- **Calibration slope <1 for OOF predictions:** Can occur with cross-validated predictions, especially in small samples. Calibration is applied inside each CV fold using a per-model strategy (sigmoid for RandomForest, isotonic for XGBoost/LightGBM/CatBoost, none for LogisticRegression and StackingEnsemble), so the OOF calibration metrics reflect the same calibration strategy used in the final model. The final model (used for individual predictions) is refitted on all data and may show slightly different calibration.
- **Missing data and imputation:** Missing variables are replaced by the training dataset median (numeric) or mode (categorical). The input completeness indicator classifies each prediction as complete, adequate, partially imputed, or heavily imputed — considering both the number and clinical relevance of missing variables. Predictions with heavily imputed data should be interpreted with greater caution.
- **TRIPOD/PROBAST:** Methodological transparency follows TRIPOD/TRIPOD-AI principles. Risk of bias should be assessed across PROBAST domains (participants, predictors, outcome, analysis).
""",
                """
- **Dados de centro único:** O AI Risk é treinado em dados locais e pode não generalizar para outras populações sem validação externa.
- **Validação interna apenas:** A validação cruzada OOF reduz o sobreajuste, mas não substitui a validação em uma coorte independente.
- **Aproximações do EuroSCORE II:** Algumas variáveis (mobilidade reduzida, estado crítico pré-operatório) podem ser aproximadas a partir dos campos disponíveis, e não capturadas exatamente como no formulário original.
- **Dependência da calculadora web do STS Score:** O cálculo do STS Score requer acesso à internet e depende da disponibilidade da calculadora web do STS em acsdriskcalc.research.sts.org. A interface pode mudar sem aviso. ~1–3% dos pacientes podem falhar por ambiguidade no mapeamento de procedimentos.
- **Subgrupos pequenos:** Resultados em subgrupos com <50 pacientes ou <10 eventos devem ser interpretados com cautela — os intervalos de confiança podem ser amplos.
- **Slope de calibração <1 nas predições OOF:** Pode ocorrer em predições de validação cruzada, especialmente em amostras pequenas. A calibração é aplicada dentro de cada fold do CV com estratégia por modelo (sigmoid para RandomForest, isotonic para XGBoost/LightGBM/CatBoost, nenhuma para LogisticRegression e StackingEnsemble), portanto as métricas de calibração OOF refletem a mesma estratégia de calibração do modelo final. O modelo final (usado para predições individuais) é reajustado em todos os dados e pode ter calibração ligeiramente diferente.
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


# _serialize_bundle and _deserialize_bundle were extracted to bundle_io.py
# in Phase 4; they are re-imported at the top of this module.


def load_train_bundle(xlsx_path: str, force_retrain: bool = False, progress_callback=None) -> tuple[Dict[str, object], str, dict]:
    sig = _bundle_signature(xlsx_path)

    if not force_retrain and MODEL_CACHE_FILE.exists():
        try:
            payload = joblib.load(MODEL_CACHE_FILE)
            if payload.get("signature") == sig:
                payload = _normalize_payload(payload)
                bundle_info = {
                    "saved_at": payload.get("saved_at", "Unknown"),
                    "training_source": payload.get("training_source", Path(xlsx_path).name),
                    "schema_version": payload["bundle_schema_version"],
                    "loaded_schema_version": payload["_loaded_schema_version"],
                }
                return _deserialize_bundle(payload["bundle"]), "Cache local", bundle_info
        except (_BundleSchemaError, Exception):
            pass

    from datetime import datetime, timezone
    bundle = _compute_bundle(xlsx_path, progress_callback=progress_callback)
    saved_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "bundle_schema_version": _BUNDLE_SCHEMA_VERSION,
        "signature": sig,
        "bundle": _serialize_bundle(bundle),
        "saved_at": saved_at,
        "training_source": Path(xlsx_path).name,
    }
    joblib.dump(payload, MODEL_CACHE_FILE)
    return bundle, "Recalculado", {
        "saved_at": saved_at,
        "training_source": Path(xlsx_path).name,
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "loaded_schema_version": _BUNDLE_SCHEMA_VERSION,
    }


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

    try:
        payload = _normalize_payload(payload)
    except _BundleSchemaError:
        return None, "Cache corrompido", empty_info

    raw = payload["bundle"]  # normalize_payload guarantees this is a dict
    bundle_info = {
        "saved_at": payload.get("saved_at", "Unknown"),
        "training_source": payload.get("training_source", Path(xlsx_path).name),
        "schema_version": payload["bundle_schema_version"],
        "loaded_schema_version": payload["_loaded_schema_version"],
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

    feature_names = [_feat_display_name(n) for n in prep.get_feature_names_out()]

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


# ---------------------------------------------------------------------------
# Feature display-name helper
# ---------------------------------------------------------------------------
# ColumnTransformer.get_feature_names_out() prefixes each feature with the
# transformer name (e.g. "valve__", "cat__target_enc__", "num__").  These are
# correct internally but must not appear in user-facing tables.
# Longer/more-specific prefixes are checked first so that "cat__onehot__"
# wins over the shorter "cat__" fallback.

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
    """Return a human-friendly label for a preprocessor output feature name.

    Strips sklearn ColumnTransformer / Pipeline prefixes while leaving the
    underlying column name unchanged.  Falls back to the raw name if no known
    prefix is found.
    """
    s = str(name)
    for prefix in _FEAT_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def _feature_group(base_feature: str) -> str:
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
    feat_names = [_feat_display_name(n) for n in prep.get_feature_names_out()]

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
    feat_names = [_feat_display_name(n) for n in prep.get_feature_names_out()]

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
    feat_names = [_feat_display_name(n) for n in prep.get_feature_names_out()]

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
    feature_names = [_feat_display_name(n) for n in prep.get_feature_names_out()]
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


_MODEL_DISAGREEMENT_RANGE_THRESHOLD = 0.10


def candidate_model_disagreement_summary(model_probs: Dict[str, float]) -> dict:
    """Summarize individual-level disagreement across candidate AI models."""
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
            "Limiar_Youden": st.column_config.NumberColumn(tr("Youden threshold", "Limiar de Youden"), help=tr("Optimal out-of-fold threshold maximizing Youden's J.", "Limiar ótimo out-of-fold que maximiza o índice J de Youden."), format="%.3f"),
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
    if kind == "surgery_profile":
        return {
            tr("Surgery group", "Grupo cirúrgico"): st.column_config.TextColumn(
                tr("Surgery group", "Grupo cirúrgico"),
                help=tr("Grouped surgery category (descriptive, not model-oriented).", "Categoria cirúrgica agrupada (descritiva, não orientada ao modelo)."),
            ),
            tr("N", "N"): st.column_config.NumberColumn(
                tr("N", "N"),
                help=tr("Number of patients in the group.", "Número de pacientes no grupo."),
                format="%d",
            ),
            tr("Deaths", "Óbitos"): st.column_config.NumberColumn(
                tr("Deaths", "Óbitos"),
                help=tr("Number of deaths (primary outcome: morte_30d).", "Número de óbitos (desfecho primário: morte_30d)."),
                format="%d",
            ),
            tr("Mortality rate (%)", "Mortalidade (%)"): st.column_config.NumberColumn(
                tr("Mortality rate (%)", "Mortalidade (%)"),
                help=tr("Observed mortality rate in the group.", "Taxa de mortalidade observada no grupo."),
                format="%.1f",
            ),
        }
    if kind == "surgery_profile_raw":
        return {
            tr("Surgery (raw)", "Cirurgia (bruta)"): st.column_config.TextColumn(
                tr("Surgery (raw)", "Cirurgia (bruta)"),
                help=tr("Raw free-text surgery description from the source dataset.", "Descrição livre da cirurgia, conforme o arquivo fonte."),
            ),
            tr("N", "N"): st.column_config.NumberColumn(
                tr("N", "N"),
                help=tr("Number of patients with this exact surgery string.", "Número de pacientes com esta descrição exata."),
                format="%d",
            ),
            tr("Deaths", "Óbitos"): st.column_config.NumberColumn(
                tr("Deaths", "Óbitos"),
                help=tr("Number of deaths (primary outcome: morte_30d).", "Número de óbitos (desfecho primário: morte_30d)."),
                format="%d",
            ),
            tr("Mortality rate (%)", "Mortalidade (%)"): st.column_config.NumberColumn(
                tr("Mortality rate (%)", "Mortalidade (%)"),
                help=tr("Observed mortality rate for this raw surgery string.", "Taxa de mortalidade observada para esta descrição bruta."),
                format="%.1f",
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
                    "STS Score Operative Mortality calculated by the app via automated query to the STS Score web calculator. Not read from the input file.",
                    "Mortalidade Operatória do STS Score calculada pelo app via consulta automatizada à calculadora web do STS Score. Não lida do arquivo de entrada.",
                ),
                format="%.4f",
            ),
            "classe_ia": st.column_config.TextColumn(tr("IA class", "Classe IA")),
            "classe_euro": st.column_config.TextColumn(tr("EuroSCORE class", "Classe EuroSCORE")),
            "classe_sts": st.column_config.TextColumn(tr("STS Score class", "Classe STS Score")),
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


def _plot_roc(scores: Dict[str, np.ndarray], y: np.ndarray) -> bytes | None:
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
    return png


def _plot_calibration(scores: Dict[str, np.ndarray], y: np.ndarray) -> bytes | None:
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
    return png


def _plot_boxplots(df_plot: pd.DataFrame) -> bytes | None:
    if df_plot.empty:
        st.info(tr("No data available for boxplots.", "Sem dados disponíveis para boxplots."))
        return None

    chart_df = df_plot.melt(id_vars=["Outcome"], var_name="Score", value_name="Probability").dropna()
    if chart_df.empty:
        st.info(tr("No data available for boxplots.", "Sem dados disponíveis para boxplots."))
        return None

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
    return png


def _plot_ia_model_boxplots(y_true: np.ndarray, oof_predictions: Dict[str, np.ndarray]) -> bytes | None:
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
        return None

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
    return png


def _plot_dca(curve_df: pd.DataFrame) -> bytes | None:
    if curve_df.empty:
        st.info(tr("No data available for decision curve analysis.", "Sem dados disponíveis para decision curve analysis."))
        return None
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
    return png


# Phase 4: subgroup assignment helpers extracted to subgroups.py.  Thin
# wrappers preserve the original call signatures (no ``tr`` argument) so
# every existing call site keeps working unchanged.
from subgroups import (
    surgery_family as _surgery_family_impl,
    surgery_type_group as _surgery_type_group_impl,
    surgery_descriptive_group as _surgery_descriptive_group_impl,
    lvef_group as _lvef_group_impl,
    renal_group as _renal_group_impl,
)


def _surgery_family(text: object) -> str:
    return _surgery_family_impl(text, tr)


def _surgery_type_group(text: object) -> str:
    return _surgery_type_group_impl(text, tr)


def _surgery_descriptive_group(text: object) -> str:
    return _surgery_descriptive_group_impl(text, tr)


def _lvef_group(value: object, fallback: object = None) -> str:
    return _lvef_group_impl(value, fallback, tr)


def _renal_group(clearance: object, dialysis: object,
                  creatinine: object = None, age: object = None,
                  weight: object = None, sex: object = None) -> str:
    return _renal_group_impl(clearance, dialysis, creatinine, age, weight, sex, tr)


# Phase 4: text builders extracted to report_text.py; evaluate_subgroup
# extracted to subgroups.py.  Thin wrappers preserve the original call
# signatures (no ``language`` / ``tr`` argument) so every call site in
# this module keeps working unchanged.
from report_text import (
    build_methods_text as _build_methods_text_impl,
    build_results_text as _build_results_text_impl,
)
from subgroups import evaluate_subgroup


def _subgroup_add_caution_flags(metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach the same exploratory caution semantics shown in the Subgroups UI."""
    out = metrics.copy()
    if out.empty:
        return out
    out["small_n_flag"] = pd.to_numeric(out.get("n"), errors="coerce") < 50
    out["low_events_flag"] = pd.to_numeric(out.get("Deaths"), errors="coerce") < 10
    out["caution_flag"] = out["small_n_flag"] | out["low_events_flag"]

    def _reason(row) -> str:
        reasons = []
        if bool(row.get("small_n_flag", False)):
            reasons.append("n < 50")
        if bool(row.get("low_events_flag", False)):
            reasons.append("deaths < 10")
        return "; ".join(reasons)

    out["caution_reason"] = out.apply(_reason, axis=1)
    return out


def _subgroup_compact_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    out = metrics.copy()
    if {"AUC_IC95_inf", "AUC_IC95_sup"}.issubset(out.columns):
        out["AUC CI"] = out.apply(
            lambda r: (
                f"{r['AUC_IC95_inf']:.3f}-{r['AUC_IC95_sup']:.3f}"
                if pd.notna(r["AUC_IC95_inf"]) and pd.notna(r["AUC_IC95_sup"])
                else ""
            ),
            axis=1,
        )
    cols = [
        "Subgroup panel", "Subgroup", "Group", "Score", "n", "Deaths",
        "AUC", "AUC CI", "caution_flag", "caution_reason",
    ]
    return out[[c for c in cols if c in out.columns]]


@st.cache_data(show_spinner=False)
def _build_all_subgroup_metrics_cached(
    subgroup_df: pd.DataFrame,
    subgroup_panels: tuple,
    score_cols: tuple,
    threshold: float,
) -> pd.DataFrame:
    """Build all subgroup panels with the same evaluator used by the UI."""
    frames = []
    score_labels = {
        "ia_risk_oof": "AI Risk",
        "euroscore_calc": "EuroSCORE II",
        "sts_score": "STS Score",
    }
    for panel_label, subgroup_col in subgroup_panels:
        metrics = evaluate_subgroup(
            subgroup_df,
            subgroup_col,
            list(score_cols),
            float(threshold),
        )
        if metrics.empty:
            continue
        metrics = metrics.copy()
        metrics["Subgroup panel"] = panel_label
        metrics["Score"] = metrics["Score"].replace(score_labels)
        frames.append(metrics)
    if not frames:
        return pd.DataFrame()
    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics = _subgroup_add_caution_flags(all_metrics)
    preferred = [
        "Subgroup panel", "Score", "Subgroup", "Group", "Deaths", "n",
        "AUC", "AUC_IC95_inf", "AUC_IC95_sup",
        "AUPRC", "AUPRC_IC95_inf", "AUPRC_IC95_sup",
        "Brier", "Brier_IC95_inf", "Brier_IC95_sup",
        "Sensitivity", "Specificity", "PPV", "NPV",
        "small_n_flag", "low_events_flag", "caution_flag", "caution_reason",
    ]
    ordered = [c for c in preferred if c in all_metrics.columns]
    ordered += [c for c in all_metrics.columns if c not in ordered]
    return all_metrics[ordered]


def _build_subgroup_summary_table(all_metrics: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if all_metrics.empty:
        return pd.DataFrame()
    rows = []
    for panel, panel_df in all_metrics.groupby("Subgroup panel", dropna=False):
        best = panel_df.sort_values("AUC", ascending=False).iloc[0]
        rows.append({
            "Subgroup panel": panel,
            "Rows in export": int(len(panel_df)),
            "Groups evaluated": int(panel_df["Group"].nunique()) if "Group" in panel_df.columns else np.nan,
            "Scores evaluated": int(panel_df["Score"].nunique()) if "Score" in panel_df.columns else np.nan,
            "Caution-flagged rows": int(panel_df.get("caution_flag", pd.Series(False, index=panel_df.index)).sum()),
            "Best score": best.get("Score", ""),
            "Best group": best.get("Group", ""),
            "Best AUC": best.get("AUC", np.nan),
            "Best AUC CI lower": best.get("AUC_IC95_inf", np.nan),
            "Best AUC CI upper": best.get("AUC_IC95_sup", np.nan),
            "Decision threshold": float(threshold),
        })
    return pd.DataFrame(rows)


def _build_subgroup_caution_table(all_metrics: pd.DataFrame) -> pd.DataFrame:
    if all_metrics.empty or "caution_flag" not in all_metrics.columns:
        return pd.DataFrame()
    cols = [
        "Subgroup panel", "Subgroup", "Group", "Score", "n", "Deaths",
        "small_n_flag", "low_events_flag", "caution_reason",
    ]
    return all_metrics.loc[all_metrics["caution_flag"], [c for c in cols if c in all_metrics.columns]].copy()


def _build_subgroup_xlsx_bytes(all_metrics: pd.DataFrame, threshold: float, language: str) -> bytes:
    readme = pd.DataFrame(
        [
            {
                "Field": "Purpose",
                "Value": (
                    "Consolidated subgroup export across all panels and available scores/models."
                    if language == "English"
                    else "Export consolidado de subgrupos em todos os painéis e escores/modelos disponíveis."
                ),
            },
            {"Field": "Decision threshold", "Value": float(threshold)},
            {
                "Field": "Method",
                "Value": (
                    "Uses the same evaluate_subgroup() routine and metrics shown in the Subgroups tab."
                    if language == "English"
                    else "Usa a mesma rotina evaluate_subgroup() e as mesmas métricas exibidas na aba Subgroups."
                ),
            },
            {
                "Field": "Caution flags",
                "Value": "n < 50 and/or deaths < 10",
            },
        ]
    )
    summary = _build_subgroup_summary_table(all_metrics, threshold)
    compact = _subgroup_compact_table(all_metrics)
    cautions = _build_subgroup_caution_table(all_metrics)

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="00_README", index=False)
        summary.to_excel(writer, sheet_name="01_SUMMARY", index=False)
        compact.to_excel(writer, sheet_name="02_SUBGROUP_COMPACT", index=False)
        all_metrics.to_excel(writer, sheet_name="03_SUBGROUP_FULL", index=False)
        cautions.to_excel(writer, sheet_name="04_CAUTION_FLAGS", index=False)
    return buf.getvalue()


def _markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int = 12) -> str:
    view = df[[c for c in cols if c in df.columns]].head(max_rows).copy()
    if view.empty:
        return ""
    for col in view.select_dtypes(include=[np.number]).columns:
        view[col] = view[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in headers) + " |")
    return "\n".join(lines)


def _build_subgroup_summary_pdf_bytes(
    subgroup_metrics: pd.DataFrame,
    subgroup_choice: str,
    threshold: float,
    language: str,
) -> bytes:
    if subgroup_metrics.empty:
        return b""
    metrics = _subgroup_add_caution_flags(subgroup_metrics)
    best = metrics.sort_values("AUC", ascending=False).iloc[0]
    compact = _subgroup_compact_table(metrics)
    caution_count = int(metrics.get("caution_flag", pd.Series(False, index=metrics.index)).sum())
    title = "Subgroup Analysis Summary" if language == "English" else "Resumo da Analise por Subgrupos"
    caution_note = (
        f"{caution_count} row(s) flagged for n < 50 and/or deaths < 10."
        if language == "English"
        else f"{caution_count} linha(s) sinalizadas por n < 50 e/ou mortes < 10."
    )
    method_note = (
        "Metrics use the same subgroup evaluator shown in the app. AUC, AUPRC and Brier are threshold-independent; sensitivity, specificity, PPV and NPV use the selected decision threshold."
        if language == "English"
        else "As metricas usam o mesmo avaliador de subgrupos exibido no app. AUC, AUPRC e Brier independem do limiar; sensibilidade, especificidade, PPV e NPV usam o limiar selecionado."
    )
    md = f"""# {title}

**Panel:** {subgroup_choice}  
**Decision threshold:** {threshold:.1%}

## Best Subgroup

Best discriminative performance: **{best.get('Score', '')}** in **{best.get('Group', '')}** with AUC = **{best.get('AUC', np.nan):.3f}**.

## Compact Table

{_markdown_table(compact, ['Group', 'Score', 'n', 'Deaths', 'AUC', 'AUC CI', 'caution_reason'])}

## Caution

{caution_note}

## Method Note

{method_note}
"""
    return statistical_summary_to_pdf(md)


def build_methods_text(mode: str) -> str:
    return _build_methods_text_impl(mode, language, tr)


def build_results_text(mode: str, context: Dict[str, object]) -> str:
    return _build_results_text_impl(mode, context, language, tr)


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
                    st.sidebar.success(tr("Loaded and validated", "Arquivo carregado e validado"))
                else:
                    st.sidebar.error(tr(f"Invalid data: {v_msg}", f"Dados inválidos: {v_msg}"))
            else:
                st.sidebar.error(tr(f"Load failed: {msg}", f"Falha ao carregar: {msg}"))
        except Exception as e:
            st.sidebar.error(tr(f"Invalid URL: {e}", f"URL inválida: {e}"))

    if GSHEETS_CACHE_FILE.exists():
        xlsx_path = str(GSHEETS_CACHE_FILE)
        st.sidebar.caption(tr(
            "Using cached Google Sheets file. Clear temp files to reset.",
            "Usando arquivo do Google Sheets em cache. Limpe os temporários para redefinir.",
        ))

st.sidebar.divider()

# ── Model section ──
force_retrain = st.sidebar.button(
    tr("Train / Retrain models", "Treinar / retreinar modelos"),
    width="stretch",
    type="primary",
)

with st.sidebar.expander(tr("Advanced", "Avançado")):
    clear_temp = st.button(tr("Clear temporary files", "Limpar temporários"), width="stretch")
    if clear_temp:
        ok, msg = _clear_temp_data_dir()
        if ok:
            st.success(tr("Cleaned", "Arquivos temporários limpos"))
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
        _train_phase_slot = st.empty()
        _train_progress = st.progress(0, text=tr(
            "Preparing data…", "Preparando dados…",
        ))
        _train_ops_slot = st.empty()
        _train_last_phase: list = [""]  # [label] — updated at each phase transition
        _train_t0 = time.monotonic()
        _train_sts_ops = {
            "total": 0,
            "processed": 0,
            "net_success": 0,
            "net_failed": 0,
            "pending": 0,
            "current_patient": "",
            "last_completed": "",
            "last_error": "",
            "last_error_patient": "",
            "phase_label": "",
            "phase_detail": "",
            "status_counts": {},
            "failures": 0,
        }

        def _train_elapsed_label() -> str:
            _seconds = max(int(time.monotonic() - _train_t0), 0)
            _minutes, _secs = divmod(_seconds, 60)
            if _minutes:
                return tr(f"{_minutes}m {_secs}s", f"{_minutes}min {_secs}s")
            return tr(f"{_secs}s", f"{_secs}s")

        def _merge_train_sts_ops(detail) -> None:
            if not isinstance(detail, dict):
                return
            for _k, _v in detail.items():
                if _v is not None:
                    _train_sts_ops[_k] = _v

        def _sts_success_count() -> int:
            _counts = _train_sts_ops.get("status_counts") or {}
            if _counts:
                return int(sum(_counts.get(_k, 0) for _k in ("cached", "fresh", "refreshed", "stale_fallback")))
            return int(_train_sts_ops.get("net_success", 0) or 0)

        def _sts_error_count() -> int:
            _failures = int(_train_sts_ops.get("failures", 0) or 0)
            if _failures:
                return _failures
            _counts = _train_sts_ops.get("status_counts") or {}
            if _counts:
                return int(_counts.get("failed", 0) or 0)
            return int(_train_sts_ops.get("net_failed", 0) or 0)

        def _render_train_sts_ops(final: bool = False) -> None:
            _total = int(_train_sts_ops.get("total", 0) or 0)
            if _total <= 0:
                return
            _processed = min(int(_train_sts_ops.get("processed", 0) or 0), _total)
            _pending = max(_total - _processed, 0)
            _pct = _processed / max(_total, 1)
            _current = str(_train_sts_ops.get("current_patient") or tr("not available", "não informado"))
            _current_batch_size = int(_train_sts_ops.get("current_batch_size", 0) or 0)
            _batch_suffix = ""
            if _current_batch_size > 1:
                _batch_suffix = tr(
                    f" (+{_current_batch_size - 1} in the same request)",
                    f" (+{_current_batch_size - 1} no mesmo lote)",
                )
            _phase = _format_training_sts_phase(
                _train_sts_ops.get("phase_label") or "STS Score processing",
                _train_sts_ops.get("phase_num"),
                _train_sts_ops.get("phase_total"),
            )
            _detail = _format_training_sts_detail(_train_sts_ops.get("phase_detail") or "")
            _last_done = str(_train_sts_ops.get("last_completed") or tr("none yet", "nenhum até agora"))
            _last_err_patient = str(_train_sts_ops.get("last_error_patient") or "")
            _last_err = str(_train_sts_ops.get("last_error") or "")
            _success = _sts_success_count()
            _errors = _sts_error_count()
            _counts = _train_sts_ops.get("status_counts") or {}
            _cache_line = ""
            if _counts:
                _cache_bits = []
                for _k in ("cached", "fresh", "refreshed", "stale_fallback", "failed"):
                    if _counts.get(_k, 0):
                        _cache_bits.append(f"{_format_training_sts_status(_k)}: {_counts[_k]}")
                _cache_line = " · ".join(_cache_bits)
            with _train_ops_slot.container():
                st.markdown(tr("**Operational processing status**", "**Status operacional do treino**"))
                st.progress(
                    _pct,
                    text=tr(
                        f"{_processed}/{_total} patients processed ({_pct:.0%})",
                        f"{_processed}/{_total} pacientes processados ({_pct:.0%})",
                    ),
                )
                st.caption(tr(
                    f"Current patient/request: {_current}{_batch_suffix} · elapsed: {_train_elapsed_label()} · phase: {_phase}",
                    f"Paciente/consulta atual: {_current}{_batch_suffix} · tempo decorrido: {_train_elapsed_label()} · etapa: {_phase}",
                ))
                if _detail:
                    st.caption(_detail)
                _m1, _m2, _m3, _m4 = st.columns(4)
                _m1.metric(tr("Success", "Concluídos"), _success)
                _m2.metric(tr("Errors", "Erros"), _errors)
                _m3.metric(tr("Pending", "Pendentes"), _pending)
                _m4.metric(tr("Skipped/incompatible", "Ignorados/incompatíveis"), tr("n/a", "não aplicável"))
                _op_bits = [
                    tr(f"Last completed: {_last_done}", f"Último concluído: {_last_done}"),
                ]
                if _last_err:
                    _err_prefix = (
                        f"{_last_err_patient}: " if _last_err_patient else ""
                    )
                    _op_bits.append(tr(
                        f"Last relevant error: {_err_prefix}{_last_err}",
                        f"Último erro relevante: {_err_prefix}{_format_training_sts_error(_last_err)}",
                    ))
                if _cache_line:
                    _op_bits.append(tr(
                        f"STS Score/cache/fallback: {_cache_line}",
                        f"STS Score, cache e fallback: {_cache_line}",
                    ))
                if final:
                    _op_bits.append(tr("Final operational summary.", "Resumo operacional final."))
                st.caption(" · ".join(_op_bits))

        def _render_train_sts_summary() -> None:
            _total = int(_train_sts_ops.get("total", 0) or 0)
            if _total <= 0:
                return
            _success = _sts_success_count()
            _errors = _sts_error_count()
            _success_rate = _success / max(_total, 1)
            _s1, _s2, _s3, _s4 = st.columns(4)
            _s1.metric(tr("Processed", "Processados"), _total)
            _s2.metric(tr("Success", "Concluídos"), _success)
            _s3.metric(tr("Errors", "Erros"), _errors)
            _s4.metric(tr("Success rate", "Taxa de sucesso"), f"{_success_rate:.1%}")
            _counts = _train_sts_ops.get("status_counts") or {}
            if _counts:
                st.caption(
                    tr("STS Score status counts: ", "Contagens de status do STS Score: ")
                    + " · ".join(f"{_format_training_sts_status(_k)}: {_v}" for _k, _v in _counts.items() if _v)
                )

        def _train_progress_cb(phase, current, total, model_name):
            if phase == "loading_data":
                _train_last_phase[0] = tr("loading and preparing dataset", "carregando e preparando a base")
                _update_phase(_train_phase_slot, 1, 5, tr(
                    "loading and preparing dataset",
                    "carregando e preparando a base",
                ))
            elif phase == "eligibility_done":
                _train_last_phase[0] = tr("cohort eligibility", "verificando elegibilidade da coorte")
                _update_phase(_train_phase_slot, 2, 5, tr(
                    "cohort eligibility",
                    "verificando elegibilidade da coorte",
                ))
            elif phase == "cross_validation":
                _train_last_phase[0] = tr("training candidate models", "treinando modelos candidatos")
                _update_phase(_train_phase_slot, 3, 5, tr(
                    "training candidate models",
                    "treinando modelos candidatos",
                ))
                _pct = current / max(total * 2, 1)  # CV = first half
                _train_progress.progress(_pct, text=tr(
                    f"Cross-validating: {model_name} ({current + 1}/{total})",
                    f"Validação cruzada: {model_name} ({current + 1}/{total})",
                ))
            elif phase == "final_fit":
                _train_last_phase[0] = tr("training candidate models", "treinando modelos candidatos")
                _update_phase(_train_phase_slot, 3, 5, tr(
                    "training candidate models",
                    "treinando modelos candidatos",
                ))
                _pct = 0.5 + current / max(total * 2, 1)  # Final fit = second half
                _train_progress.progress(_pct, text=tr(
                    f"Final training: {model_name} ({current + 1}/{total})",
                    f"Treino final: {model_name} ({current + 1}/{total})",
                ))
            elif phase == "selecting_best":
                _train_last_phase[0] = tr("selecting best model", "selecionando o melhor modelo")
                _update_phase(_train_phase_slot, 3, 5, tr(
                    "selecting best model",
                    "selecionando o melhor modelo",
                ))
                _train_progress.progress(0.95, text=tr(
                    "Selecting best model…", "Selecionando o melhor modelo…",
                ))
            elif phase == "euroscore_calc":
                _train_last_phase[0] = tr("computing scores", "calculando escores")
                _update_phase(_train_phase_slot, 4, 5, tr(
                    "computing scores",
                    "calculando escores",
                ))
                _train_progress.progress(0.97, text=tr(
                    "Computing EuroSCORE II…", "Calculando EuroSCORE II…",
                ))
            elif phase == "sts_score_calc":
                _train_last_phase[0] = tr("querying STS Score web calculator", "consultando a calculadora web do STS Score")
                _update_phase(_train_phase_slot, 4, 5, tr(
                    "querying STS Score web calculator",
                    "consultando a calculadora web do STS Score",
                ))
                _train_progress.progress(0.97, text=tr(
                    "Querying STS Score…", "Consultando o STS Score…",
                ))
                _merge_train_sts_ops(model_name)
                _render_train_sts_ops()
            elif phase in {"sts_score_phase", "sts_score_progress", "sts_score_patient", "sts_score_patient_done"}:
                _train_last_phase[0] = tr("querying STS Score web calculator", "consultando a calculadora web do STS Score")
                _update_phase(_train_phase_slot, 4, 5, tr(
                    "querying STS Score web calculator",
                    "consultando a calculadora web do STS Score",
                ))
                _merge_train_sts_ops(model_name)
                _processed = int(_train_sts_ops.get("processed", 0) or 0)
                _total = int(_train_sts_ops.get("total", total or 0) or 0)
                _sts_frac = _processed / max(_total, 1) if _total else 0
                _train_progress.progress(
                    min(0.97 + 0.02 * _sts_frac, 0.99),
                    text=tr(
                        f"STS Score: {_processed}/{_total}",
                        f"STS Score: {_processed}/{_total}",
                    ),
                )
                _render_train_sts_ops()
            elif phase == "sts_score_done":
                _train_last_phase[0] = tr("querying STS Score web calculator", "consultando a calculadora web do STS Score")
                _merge_train_sts_ops(model_name)
                _train_progress.progress(0.99, text=tr(
                    "STS Score complete.", "STS Score concluído.",
                ))
                _render_train_sts_ops(final=True)
            elif phase == "building_reports":
                _train_last_phase[0] = tr("building reports and bundle", "gerando relatórios e pacote")
                _update_phase(_train_phase_slot, 5, 5, tr(
                    "building reports and bundle",
                    "gerando relatórios e pacote",
                ))
                _train_progress.progress(0.99, text=tr(
                    "Building reports…", "Gerando relatórios…",
                ))

        _retrained_bundle, bundle_source, _retrained_info = load_train_bundle(
            xlsx_path, force_retrain=True, progress_callback=_train_progress_cb,
        )
        _train_phase_slot.empty()
        _train_progress.progress(1.0, text=tr(
            "Training complete!", "Treinamento concluído!",
        ))
        _train_ops_slot.empty()
        with st.expander(tr("View training execution details", "Ver detalhes de execução do treinamento"), expanded=False):
            st.caption(tr(
                f"Last phase: {_train_last_phase[0]} | Source: {_retrained_info.get('training_source', '?')}",
                f"Última etapa: {_train_last_phase[0]} | Arquivo de origem: {_retrained_info.get('training_source', '?')}",
            ))
            _render_train_sts_summary()
    except Exception as e:
        # Phase 3: if ingestion halted on a required-column error, the
        # exception carries a RunReport — render it so the user sees
        # exactly which columns were missing before the app stops.
        _err_report = getattr(e, "run_report", None)
        _err_source = getattr(e, "source_label", None)
        if _err_source:
            st.error(
                tr(
                    f"Training halted while processing '{_err_source}': {e}",
                    f"Treinamento interrompido ao processar '{_err_source}': {e}",
                )
            )
        else:
            st.error(
                tr(
                    f"Training failed: {e}",
                    f"Falha no treinamento: {e}",
                )
            )
        if _err_report is not None:
            try:
                import observability as _obs
                _obs.render_run_report(_err_report, tr=tr)
            except Exception:
                pass
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
            "Não há modelo treinado para este arquivo. Clique em 'Treinar / retreinar modelos' na barra lateral.",
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

# Guard the single-patient inference simplification: the individual prediction
# flow calls _run_ai_risk_inference_row with artifacts.feature_columns only
# (no prepared/artifacts merge).  If the schemas ever diverge the input will
# be silently wrong, so we surface the discrepancy here at load time.
_prepared_cols = set(getattr(prepared, "feature_columns", []))
_artifact_cols = set(getattr(artifacts, "feature_columns", []))
if _prepared_cols != _artifact_cols:
    _only_prepared = _prepared_cols - _artifact_cols
    _only_artifact = _artifact_cols - _prepared_cols
    st.warning(
        tr(
            f"⚠️ Feature schema mismatch between training data and model artifacts "
            f"({len(_only_prepared)} column(s) only in prepared data, "
            f"{len(_only_artifact)} only in model artifacts). "
            f"Individual prediction uses the model artifact schema. "
            f"Consider retraining to resolve.",
            f"⚠️ Divergência de esquema entre dados de treino e artefatos do modelo "
            f"({len(_only_prepared)} coluna(s) apenas nos dados preparados, "
            f"{len(_only_artifact)} apenas nos artefatos). "
            f"A predição individual usa o esquema dos artefatos. "
            f"Considere retreinar para resolver.",
        )
    )

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

# Phase 3 — compact execution status near the top of the page.
# The full expandable report is rendered at the bottom of the Overview
# tab so it does not visually compete with the leaderboard/results.
# Blocking errors (if any) still surface prominently here via st.error.
_run_report = bundle.get("run_report")
if _run_report is not None and getattr(_run_report, "steps", None):
    try:
        import observability as _obs
        _obs.render_run_report_compact(_run_report, tr=tr)
    except Exception as _obs_err:
        st.caption(tr(
            f"Execution status unavailable: {_obs_err}",
            f"Status de execução indisponível: {_obs_err}",
        ))

# ── Build shared tab context ────────────────────────────────────────────
_tab_ctx = TabContext(
    tr=tr,
    hp=hp,
    language=language,
    prepared=prepared,
    artifacts=artifacts,
    df=df,
    forced_model=forced_model,
    best_model_name=best_model_name,
    bundle_info=bundle_info,
    xlsx_path=xlsx_path,
    default_threshold=_default_threshold,
    model_version=MODEL_VERSION,
    has_sts=HAS_STS,
    csv_download_btn=_csv_download_btn,
    txt_download_btn=_txt_download_btn,
    bytes_download_btn=_bytes_download_btn,
    update_phase=_update_phase,
    sts_score_patient_ids=_sts_score_patient_ids,
    general_table_column_config=general_table_column_config,
    stats_table_column_config=stats_table_column_config,
    format_ppv_npv=_format_ppv_npv,
    to_csv_bytes=_to_csv_bytes,
    safe_prob=_safe_prob,
    plot_roc=_plot_roc,
    plot_calibration=_plot_calibration,
    plot_boxplots=_plot_boxplots,
    plot_ia_model_boxplots=_plot_ia_model_boxplots,
    plot_dca=_plot_dca,
    build_methods_text=build_methods_text,
    build_results_text=build_results_text,
)

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
# Visual order shown in the segmented control. The dispatch below still
# uses `_tab_labels` indices, so reordering the display list never moves
# any tab body. Label strings must match `_tab_labels` exactly so the
# round-trip `_tab_labels.index(...)` below resolves to the correct
# canonical dispatch index.
_tab_display_order = [
    tr("Overview", "Visão Geral"),
    tr("Prediction", "Predição"),
    tr("Batch", "Lote"),
    tr("Comparison", "Comparação"),
    tr("Temporal Validation", "Validação Temporal"),
    tr("Data Quality", "Qualidade"),
    tr("Models", "Modelos"),
    tr("Subgroups", "Subgrupos"),
    tr("Guide", "Guia"),
    tr("Dictionary", "Dicionário"),
]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0
if st.session_state.active_tab >= len(_tab_labels):
    st.session_state.active_tab = 0

_canonical_active = st.session_state.active_tab
_current_label = _tab_labels[_canonical_active]
_default_display_label = (
    _current_label if _current_label in _tab_display_order else _tab_display_order[0]
)

_selected_tab_label = st.segmented_control(
    "nav",
    _tab_display_order,
    default=_default_display_label,
    selection_mode="single",
    label_visibility="collapsed",
    key="_tab_nav",
)
_active_tab = _tab_labels.index(_selected_tab_label) if _selected_tab_label else st.session_state.active_tab
st.session_state.active_tab = _active_tab

if _active_tab == 0:  # Overview

    st.subheader(tr("Overview", "Visão Geral"))
    st.caption(tr(
        "Executive summary of the current cohort, active model, performance, operational readiness, and audit trail.",
        "Resumo executivo da coorte atual, modelo ativo, desempenho, operação e trilha de auditoria.",
    ))

    _overview_events = int(pd.to_numeric(df["morte_30d"], errors="coerce").fillna(0).sum())
    _overview_event_rate = float(prepared.info.get("positive_rate", np.nan))
    _leaderboard_for_overview = artifacts.leaderboard.copy()
    _active_model_perf = pd.DataFrame()
    if "Modelo" in _leaderboard_for_overview.columns:
        _active_model_perf = _leaderboard_for_overview[_leaderboard_for_overview["Modelo"] == forced_model]
    if _active_model_perf.empty and "Modelo" in _leaderboard_for_overview.columns:
        _active_model_perf = _leaderboard_for_overview[_leaderboard_for_overview["Modelo"] == best_model_name]
    if _active_model_perf.empty and "AUC" in _leaderboard_for_overview.columns:
        _active_model_perf = _leaderboard_for_overview.sort_values("AUC", ascending=False).head(1)
    _active_auc = (
        float(_active_model_perf.iloc[0].get("AUC", np.nan))
        if not _active_model_perf.empty else np.nan
    )
    _active_brier = (
        float(_active_model_perf.iloc[0].get("Brier", np.nan))
        if not _active_model_perf.empty else np.nan
    )

    _k1, _k2, _k3, _k4 = st.columns(4)
    _k1.metric(tr("Patients", "Pacientes"), f"{prepared.info['n_rows']}", border=True)
    _k2.metric(tr("Events", "Eventos"), f"{_overview_events}", border=True)
    _k3.metric(
        tr("Event rate", "Taxa de eventos"),
        "N/A" if not np.isfinite(_overview_event_rate) else f"{_overview_event_rate*100:.1f}%",
        border=True,
    )
    _k4.metric(tr("Active AI model", "Modelo IA ativo"), forced_model, border=True)

    _k5, _k6, _k7, _k8 = st.columns(4)
    _k5.metric(tr("Operational threshold", "Limiar operacional"), f"{_default_threshold:.0%}", border=True)
    _k6.metric(
        "AUC",
        "N/A" if not np.isfinite(_active_auc) else f"{_active_auc:.3f}",
        border=True,
    )
    _k7.metric(
        "Brier",
        "N/A" if not np.isfinite(_active_brier) else f"{_active_brier:.4f}",
        border=True,
    )
    _k8.metric(tr("Predictors", "Preditores"), f"{prepared.info['n_features']}", border=True)

    # ── 1. Cohort Snapshot ──────────────────────────────────────────────
    st.divider()
    st.subheader(tr("Cohort Snapshot", "Coorte"))
    st.caption(tr(
        "Cohort composition and descriptive surgical profile for the active dataset.",
        "Composição da coorte e perfil cirúrgico descritivo da base ativa.",
    ))
    st.caption(tr(
        "Training data are drawn exclusively from this institution. Performance on external cohorts has not been assessed.",
        "Os dados de treinamento são exclusivamente desta instituição. O desempenho em coortes externas não foi avaliado.",
    ))

    # ── Surgery profile (descriptive cohort breakdown) ──
    st.markdown(tr("**Surgery profile**", "**Perfil cirúrgico**"))
    _surg_group_col_label = tr("Surgery group", "Grupo cirúrgico")
    _n_col_label = tr("N", "N")
    _deaths_col_label = tr("Deaths", "Óbitos")
    _mort_col_label = tr("Mortality rate (%)", "Mortalidade (%)")

    _category_order = [
        tr("Isolated CABG", "CABG isolada"),
        tr("Isolated AVR", "Troca valvar aórtica isolada"),
        tr("Isolated MVR", "Troca valvar mitral isolada"),
        tr("Isolated MV Repair", "Plastia mitral isolada"),
        tr("AVR + CABG", "Troca valvar aórtica + CABG"),
        tr("MVR + CABG", "Troca valvar mitral + CABG"),
        tr("MV Repair + CABG", "Plastia mitral + CABG"),
        tr("Thoracic aorta surgery", "Cirurgia de aorta torácica"),
        tr("Heart transplant", "Transplante cardíaco"),
        tr("Ross procedure", "Cirurgia de Ross"),
        tr("Pulmonary homograft", "Homoenxerto pulmonar"),
        tr("Aortic homograft", "Homoenxerto aórtico"),
        tr("Other combined surgeries", "Outras cirurgias combinadas"),
        tr("Other", "Outras"),
    ]

    _surg_profile_src = pd.DataFrame({
        _surg_group_col_label: df["Surgery"].map(_surgery_descriptive_group),
        "_death": pd.to_numeric(df["morte_30d"], errors="coerce").fillna(0).astype(int),
    })
    _surg_profile = (
        _surg_profile_src.groupby(_surg_group_col_label, dropna=False)
        .agg(**{_n_col_label: ("_death", "size"), _deaths_col_label: ("_death", "sum")})
        .reset_index()
    )
    _surg_profile = _surg_profile[_surg_profile[_n_col_label] > 0].copy()
    _surg_profile[_mort_col_label] = 100.0 * _surg_profile[_deaths_col_label] / _surg_profile[_n_col_label]
    _surg_profile = _surg_profile[[
        _surg_group_col_label, _n_col_label, _deaths_col_label, _mort_col_label,
    ]]
    _surg_profile[_surg_group_col_label] = pd.Categorical(
        _surg_profile[_surg_group_col_label],
        categories=_category_order,
        ordered=True,
    )
    _surg_profile = _surg_profile.sort_values(_surg_group_col_label).reset_index(drop=True)

    st.dataframe(
        _surg_profile,
        width="stretch",
        hide_index=True,
        column_config=general_table_column_config("surgery_profile"),
    )
    st.caption(tr(
        "Descriptive breakdown by grouped surgery category on the current cohort. "
        "Mortality uses the app's primary outcome (`morte_30d`). No inferential tests.",
        "Resumo descritivo por grupo cirúrgico na coorte atual. "
        "A mortalidade usa o desfecho primário do app (`morte_30d`). Sem testes inferenciais.",
    ))

    with st.expander(
        tr("Show raw surgery strings (optional)", "Mostrar descrições brutas de cirurgia (opcional)"),
        expanded=False,
    ):
        _raw_col_label = tr("Surgery (raw)", "Cirurgia (bruta)")
        _raw_profile_src = pd.DataFrame({
            _raw_col_label: df["Surgery"].astype(str),
            "_death": pd.to_numeric(df["morte_30d"], errors="coerce").fillna(0).astype(int),
        })
        _raw_profile = (
            _raw_profile_src.groupby(_raw_col_label, dropna=False)
            .agg(**{_n_col_label: ("_death", "size"), _deaths_col_label: ("_death", "sum")})
            .reset_index()
        )
        _raw_profile[_mort_col_label] = 100.0 * _raw_profile[_deaths_col_label] / _raw_profile[_n_col_label]
        _raw_profile = _raw_profile[[
            _raw_col_label, _n_col_label, _deaths_col_label, _mort_col_label,
        ]]
        _raw_profile = _raw_profile.sort_values(_n_col_label, ascending=False).reset_index(drop=True)
        st.dataframe(
            _raw_profile,
            width="stretch",
            hide_index=True,
            column_config=general_table_column_config("surgery_profile_raw"),
        )

    # ── 2. Model Snapshot ───────────────────────────────────────────────

    def _display_calib_method(raw) -> str:
        """Safe display for calibration_method — handles None and legacy bundles."""
        if raw is None or str(raw).strip().lower() in ("", "none"):
            return tr("Legacy bundle", "Bundle legado")
        return str(raw)

    st.divider()
    st.subheader(tr("Model Snapshot", "Modelo"))
    st.caption(tr(
        "Active model configuration and bundle metadata for reproducibility.",
        "Configuração do modelo ativo e metadados do bundle para reprodutibilidade.",
    ))
    _ms1, _ms2, _ms3, _ms4 = st.columns(4)
    _ms1.metric(tr("Selected model", "Modelo selecionado"), forced_model, border=True)
    _ms2.metric(tr("Model version", "Versão do modelo"), MODEL_VERSION, border=True)
    _ms3.metric(tr("Decision threshold", "Limiar de decisão"), f"{_default_threshold:.0%}", border=True)
    _ms4.metric(
        tr("Calibration method", "Método de calibração"),
        _display_calib_method(getattr(artifacts, "calibration_method", None)),
        border=True,
    )
    st.caption(tr(
        f"Best training model: {best_model_name} · Last action: {bundle_source}",
        f"Melhor modelo no treino: {best_model_name} · Última ação: {bundle_source}",
    ))

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

    # ── 3. Performance Snapshot ─────────────────────────────────────────
    st.divider()
    st.subheader(tr("Performance Snapshot", "Desempenho"))
    st.caption(tr(
        "Cross-validated, calibrated out-of-fold performance for candidate AI models.",
        "Desempenho out-of-fold calibrado por validação cruzada para os modelos candidatos de IA.",
    ))
    st.markdown(tr("**AI model leaderboard**", "**Leaderboard dos modelos de IA**"))
    st.caption(
        tr(
            "AUC/AUPRC summarize discrimination; Brier summarizes calibration. Youden thresholds are complementary references, not the operational 8% default.",
            "AUC/AUPRC resumem discriminação; Brier resume calibração. Limiares de Youden são referências complementares, não o padrão operacional de 8%.",
        )
    )
    st.dataframe(artifacts.leaderboard, width="stretch", column_config=general_table_column_config("leaderboard"))

    with st.expander(tr("How to read the leaderboard", "Como ler o leaderboard"), expanded=False):
        st.caption(
            tr(
                "Stratified cross-validation grouped by patient (same patient never appears in both train and test folds).",
                "Validação cruzada estratificada e agrupada por paciente (sem mistura do mesmo paciente entre treino e teste).",
            )
        )
        st.caption(
            tr(
                "Leaderboard sensitivity and specificity are shown at each model's own Youden's J threshold (a per-model reference), not at a fixed 0.50 cutoff and not at the 8% clinical default. The operational clinical threshold remains fixed at 8% in the Statistical Comparison and temporal validation tabs.",
                "A sensibilidade e a especificidade do leaderboard são mostradas no limiar de Youden (J) de cada modelo (referência por modelo), não em um corte fixo de 0,50 e não no padrão clínico de 8%. O limiar clínico operacional permanece fixo em 8% nas abas de Comparação Estatística e de validação temporal.",
            )
        )
        st.caption(
            tr(
                "Models are compared by cross-validated, calibrated OOF performance. The per-model Youden threshold shown here is a complementary reference — it is not the app's default operational threshold.",
                "Os modelos são comparados por desempenho OOF calibrado via validação cruzada. O limiar de Youden por modelo mostrado aqui é uma referência complementar — não é o limiar operacional padrão do app.",
            )
        )
        if np.isfinite(_overview_event_rate) and _overview_event_rate > 0:
            st.caption(tr(
                f"Event rate in this cohort: {_overview_event_rate*100:.1f}%. "
                "When events are uncommon, AUPRC is a more sensitive discriminator than AUC-ROC: "
                "a random classifier's AUPRC baseline equals the prevalence, so gains above that baseline carry greater weight.",
                f"Taxa de eventos nesta coorte: {_overview_event_rate*100:.1f}%. "
                "Quando os eventos são pouco frequentes, a AUPRC é um discriminador mais sensível do que a AUC-ROC: "
                "a AUPRC de um classificador aleatório equivale à prevalência, portanto ganhos acima dessa linha de base têm maior relevância.",
            ))

    if forced_model != best_model_name:
        _sel_rationale_en = (
            f"The active model (**{forced_model}**) was selected manually. "
            f"The best cross-validated performance during training was observed for **{best_model_name}**. "
            "Model selection weighs discrimination (AUC, AUPRC), calibration (Brier), and net clinical benefit jointly — not AUC alone."
        )
        _sel_rationale_pt = (
            f"O modelo ativo (**{forced_model}**) foi selecionado manualmente. "
            f"O melhor desempenho cross-validado no treinamento foi observado em **{best_model_name}**. "
            "A seleção considera discriminação (AUC, AUPRC), calibração (Brier) e benefício clínico líquido em conjunto — não apenas a AUC."
        )
    else:
        _sel_rationale_en = (
            f"**{best_model_name}** was selected as the active model based on the best overall cross-validated performance, "
            "weighing discrimination (AUC, AUPRC), calibration (Brier), and net clinical benefit jointly — not AUC alone."
        )
        _sel_rationale_pt = (
            f"**{best_model_name}** foi selecionado como modelo ativo com base no melhor desempenho global na validação cruzada, "
            "considerando discriminação (AUC, AUPRC), calibração (Brier) e benefício clínico líquido em conjunto — não apenas a AUC."
        )
    st.caption(tr(_sel_rationale_en, _sel_rationale_pt))

    # ── 4. Operational Snapshot ─────────────────────────────────────────
    st.divider()
    st.subheader(tr("Operational Snapshot", "Operacional"))
    st.caption(tr(
        "Availability of app-computed scores and cohort-processing flow for the active dataset.",
        "Disponibilidade dos escores calculados pelo app e fluxo de processamento da coorte ativa.",
    ))
    st.markdown(tr("**Score availability**", "**Disponibilidade dos escores**"))
    summary = pd.DataFrame(
        {
            tr("Score", "Escore"): ["AI Risk", "EuroSCORE II (app-calculated)", "STS Score (app-calculated)"],
            tr("Patients with value", "Pacientes com valor"): [
                int(df["ia_risk_oof"].notna().sum()),
                int(df["euroscore_calc"].notna().sum()),
                int(df["sts_score"].notna().sum()),
            ],
            tr("Source", "Origem"): [
                tr("Cross-validated out-of-fold predictions", "Predições out-of-fold por validação cruzada"),
                tr("Published logistic equation (Nashef et al., 2012)", "Equação logística publicada (Nashef et al., 2012)"),
                tr("Automated query to the STS Score web calculator", "Consulta automatizada à calculadora web do STS Score"),
            ],
        }
    )
    st.dataframe(summary, width="stretch", column_config=general_table_column_config("available_scores"))
    st.caption(tr(
        "All scores shown are computed by the app — not read from the input file. Sheet-derived values are retained only as optional reference in the Data Quality tab.",
        "Todos os escores exibidos são calculados pelo app — não lidos do arquivo de entrada. Valores derivados da planilha são mantidos apenas como referência opcional no painel de Qualidade da Base.",
    ))

    _elig_info = _cached_eligibility_info(xlsx_path)
    if _elig_info.get("source_type") != "flat" and _elig_info.get("pre_rows_before_criteria", 0) > 0:
        st.markdown(tr("**Eligibility flow**", "**Fluxo de elegibilidade**"))
        st.dataframe(_eligibility_summary(xlsx_path), width="stretch", column_config=general_table_column_config("eligibility"))

    # ── 5. Audit Snapshot ───────────────────────────────────────────────
    st.divider()
    st.subheader(tr("Audit Snapshot", "Auditoria"))
    st.caption(tr(
        "Execution log, eligibility flow, and pipeline incidents. "
        "See Statistical Comparison for performance details.",
        "Log de execução, fluxo de elegibilidade e incidentes do pipeline. "
        "Ver Comparação Estatística para detalhes de desempenho.",
    ))
    if _run_report is not None and getattr(_run_report, "steps", None):
        with st.expander(tr("Detailed execution report", "Relatório de execução detalhado"), expanded=False):
            try:
                import observability as _obs
                _obs.render_run_report(_run_report, tr=tr)
            except Exception as _obs_err:
                st.caption(tr(
                    f"Execution report unavailable: {_obs_err}",
                    f"Relatório de execução indisponível: {_obs_err}",
                ))

elif _active_tab == 1:  # Individual Prediction
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
        _model_disagreement = candidate_model_disagreement_summary(_patient_model_probs)
        patient_pred_df = pd.DataFrame(patient_pred).sort_values(tr("Probability", "Probabilidade"), ascending=False)
        patient_pred_df[tr("Probability", "Probabilidade")] = patient_pred_df[tr("Probability", "Probabilidade")].map(lambda x: f"{x*100:.2f}%")
        quality_alerts = data_quality_alerts(form_map, prepared)
        likely_range_text, confidence_text = prediction_uncertainty(patient_pred_df, tr("Probability", "Probabilidade"), imputed_features)

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

        best_model_prob = float(artifacts.fitted_models[best_model_name].predict_proba(model_input)[:, 1][0])

        # ── PREDICTED RISK — dominant primary block ───────────────────────────
        st.divider()
        st.markdown(tr("### Predicted Risk", "### Risco Predito"))
        st.caption(tr(
            f"Primary prediction generated by: **{forced_model}**",
            f"Predição principal gerada por: **{forced_model}**",
        ))
        r1, r2, r3 = st.columns(3)
        r1.metric("\U0001f916 AI Risk", f"{ia_prob*100:.2f}%", _risk_badge(ia_prob), delta_color="off")
        r2.metric("\U0001f4ca EuroSCORE II", f"{euro_prob*100:.2f}%", _risk_badge(euro_prob), delta_color="off")
        r3.metric("\U0001f310 STS Score", "-" if np.isnan(sts_prob) else f"{sts_prob*100:.2f}%", _risk_badge(sts_prob), delta_color="off")

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
                tr("Class", "Classe"): [_risk_badge(ia_prob), _risk_badge(euro_prob), _risk_badge(sts_prob)],
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
        pos_factors, neg_factors = explain_patient_risk(artifacts, model_input, form_map, top_n=5)
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
                st.dataframe(pos_factors, width="stretch", column_config=explain_table_column_config())
        with c_neg:
            st.markdown(tr("**Factors associated with lower risk**", "**Fatores associados à redução do risco**"))
            if neg_factors.empty:
                st.info(tr("No strong lower-risk factors were identified by the interpretable layer.", "Nenhum fator forte de redução de risco foi identificado pela camada interpretável."))
            else:
                st.dataframe(neg_factors, width="stretch", column_config=explain_table_column_config())

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

        # ── AUDIT TRAIL ───────────────────────────────────────────────────────
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
    _tab_comparison.render(_tab_ctx)

elif _active_tab == 3:  # Analysis Guide
    render_analysis_guide(prepared, artifacts, len(triple) if 'triple' in locals() else None)

elif _active_tab == 4:  # Batch & Export
    _tab_batch_export.render(_tab_ctx)

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

    # ── Top predictors summary ──────────────────────────────────────────
    w, w_kind = model_weight_table(artifacts, prepared, forced_model, top_n=20)
    if not w.empty and w_kind in ("importance", "coefficient"):
        _name_col = w.columns[0]
        _top3_names = w.head(3)[_name_col].tolist()
        _top3_str = ", ".join(f"**{n}**" for n in _top3_names)
        if w_kind == "importance":
            _top_narrative_en = (
                f"The three variables with the greatest relative importance for this model are {_top3_str}. "
                "These rankings reflect contribution to model performance, not direct clinical effect size."
            )
            _top_narrative_pt = (
                f"As três variáveis com maior importância relativa neste modelo são {_top3_str}. "
                "Esses rankings refletem contribuição ao desempenho do modelo, não efeito clínico direto."
            )
        else:
            _top_narrative_en = (
                f"The three variables with the largest absolute coefficients in this model are {_top3_str}. "
                "Positive coefficients suggest higher risk; negative suggest lower risk."
            )
            _top_narrative_pt = (
                f"As três variáveis com maiores coeficientes absolutos neste modelo são {_top3_str}. "
                "Coeficientes positivos sugerem maior risco; negativos sugerem menor risco."
            )
        st.markdown(tr("**Top predictors**", "**Principais preditores**"))
        st.caption(tr(_top_narrative_en, _top_narrative_pt))
        st.dataframe(w.head(5), width="stretch", column_config=model_table_column_config(w_kind))
        st.divider()

    st.markdown(tr("**Model explanation table**", "**Tabela explicativa do modelo**"))
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
    st.subheader(tr("Subgroup Analysis", "Análise por Subgrupos"))
    st.caption(tr(
        "Model performance across clinically relevant strata. Small subgroups (n < 50 or < 10 events) are flagged — treat those results as exploratory.",
        "Desempenho do modelo em estratos clinicamente relevantes. Subgrupos pequenos (n < 50 ou < 10 eventos) são sinalizados — trate esses resultados como exploratórios.",
    ))

    # ── CONTROLS ─────────────────────────────────────────────────────────────
    _ctrl1, _ctrl2 = st.columns([1, 1])
    with _ctrl1:
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
    with _ctrl2:
        subgroup_threshold = st.slider(
            tr("Decision threshold", "Limiar de decisão"),
            min_value=0.01,
            max_value=0.99,
            value=_default_threshold,
            step=0.01,
            help=tr(
                f"Default: {_default_threshold:.0%} (dataset prevalence). Sensitivity, specificity, PPV, and NPV change with this threshold; AUC, AUPRC, and Brier do not.",
                f"Padrão: {_default_threshold:.0%} (prevalência do dataset). Sensibilidade, especificidade, PPV e NPV mudam com este limiar; AUC, AUPRC e Brier não.",
            ),
        )

    subgroup_df = df.copy()
    subgroup_df["Surgery type"] = subgroup_df["Surgery"].map(_surgery_type_group)
    subgroup_df["Sex group"] = subgroup_df["Sex"].fillna(tr("Unknown", "Desconhecido"))
    subgroup_df["Age group"] = np.where(pd.to_numeric(subgroup_df["Age (years)"], errors="coerce") < 65, "<65", ">=65")
    _nan_f = pd.Series(np.nan, index=subgroup_df.index)
    _nan_o = pd.Series(np.nan, index=subgroup_df.index, dtype=object)
    subgroup_df["LVEF group"] = [
        _lvef_group(np.nan, pre) for pre in (
            subgroup_df["Pré-LVEF, %"] if "Pré-LVEF, %" in subgroup_df.columns else _nan_f
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
    subgroup_map = {
        tr("Surgery type", "Tipo de cirurgia"): "Surgery type",
        tr("Age", "Idade"): "Age group",
        tr("LVEF", "FEVE"): "LVEF group",
        tr("Renal function", "Função renal"): "Renal function group",
        tr("Sex", "Sexo"): "Sex group",
    }
    subgroup_panel_specs = tuple((str(label), str(col)) for label, col in subgroup_map.items())
    subgroup_score_cols = tuple(c for c in ["ia_risk_oof", "euroscore_calc", "sts_score"] if c in subgroup_df.columns)
    subgroup_col = subgroup_map[subgroup_choice]
    subgroup_metrics = evaluate_subgroup(
        subgroup_df,
        subgroup_col,
        list(subgroup_score_cols),
        subgroup_threshold,
    )

    # ── RESULTS ───────────────────────────────────────────────────────────────
    st.divider()
    if subgroup_metrics.empty:
        st.info(tr("No subgroup results are available for the current selection.", "Não há resultados de subgrupos disponíveis para a seleção atual."))
    else:
        subgroup_metrics["Score"] = subgroup_metrics["Score"].replace(
            {"ia_risk_oof": "AI Risk", "euroscore_calc": "EuroSCORE II", "sts_score": "STS Score"}
        )
        subgroup_metrics["Subgroup panel"] = str(subgroup_choice)
        subgroup_metrics = _subgroup_add_caution_flags(subgroup_metrics)

        # Reorder columns: identifiers first, then metrics, then CIs
        _sub_col_order = [
            "Subgroup panel", "Score", "Subgroup", "Group", "Deaths", "n",
            "AUC", "AUC_IC95_inf", "AUC_IC95_sup",
            "AUPRC", "AUPRC_IC95_inf", "AUPRC_IC95_sup",
            "Brier", "Brier_IC95_inf", "Brier_IC95_sup",
            "Sensitivity", "Specificity", "PPV", "NPV",
            "small_n_flag", "low_events_flag", "caution_flag", "caution_reason",
        ]
        _sub_col_order = [c for c in _sub_col_order if c in subgroup_metrics.columns]
        subgroup_metrics = subgroup_metrics[_sub_col_order]

        # Best performer insight — shown first so the key result is immediately visible.
        best_sub = subgroup_metrics.sort_values("AUC", ascending=False).iloc[0]
        _ci_lo = best_sub.get("AUC_IC95_inf", np.nan)
        _ci_hi = best_sub.get("AUC_IC95_sup", np.nan)
        _ci_str = f" (95% CI: {_ci_lo:.3f}–{_ci_hi:.3f})" if pd.notna(_ci_lo) and pd.notna(_ci_hi) else ""
        st.info(tr(
            f"Best discriminative performance: **{best_sub['Score']}** in group **{best_sub['Group']}** — AUC = {best_sub['AUC']:.3f}{_ci_str}.",
            f"Melhor discriminação: **{best_sub['Score']}** no grupo **{best_sub['Group']}** — AUC = {best_sub['AUC']:.3f}{_ci_str}.",
        ))

        # Underpowered subgroup warnings.
        small_n = subgroup_metrics[subgroup_metrics["n"] < 50][["Group", "Score", "n"]]
        low_events = subgroup_metrics[subgroup_metrics["Deaths"] < 10][["Group", "Score", "Deaths"]]
        if not small_n.empty or not low_events.empty:
            warn_parts = []
            if not small_n.empty:
                groups_small_n = ", ".join(sorted(set(small_n["Group"].astype(str).tolist())))
                warn_parts.append(tr(
                    f"small sample size in: {groups_small_n}",
                    f"tamanho amostral pequeno em: {groups_small_n}",
                ))
            if not low_events.empty:
                groups_low_events = ", ".join(sorted(set(low_events["Group"].astype(str).tolist())))
                warn_parts.append(tr(
                    f"low event count in: {groups_low_events}",
                    f"baixo número de eventos em: {groups_low_events}",
                ))
            st.warning(tr(
                f"Interpret with caution — {'; '.join(warn_parts)}.",
                f"Interprete com cautela — {'; '.join(warn_parts)}.",
            ))

        # Compact summary: Group, Score, n, Deaths, AUC with CI — primary reading surface.
        _compact_cols = [c for c in ["Group", "Score", "n", "Deaths", "AUC", "AUC_IC95_inf", "AUC_IC95_sup"] if c in subgroup_metrics.columns]
        st.dataframe(
            subgroup_metrics[_compact_cols],
            width="stretch",
            column_config=stats_table_column_config("subgroup"),
            hide_index=True,
        )

        # Full metrics table (all CI columns, Sensitivity, Specificity, PPV, NPV) in expander.
        with st.expander(tr("Full metrics table (all columns)", "Tabela completa de métricas (todas as colunas)"), expanded=False):
            st.dataframe(_format_ppv_npv(subgroup_metrics), width="stretch", column_config=stats_table_column_config("subgroup"))

        st.markdown(tr("**Exports**", "**Exportações**"))
        _sub_dl1, _sub_dl2, _sub_dl3 = st.columns(3)
        with _sub_dl1:
            _csv_download_btn(
                subgroup_metrics,
                "subgroup_results.csv",
                tr("Download current CSV", "Baixar CSV atual"),
            )
        with _sub_dl2:
            _sub_pdf_bytes = _build_subgroup_summary_pdf_bytes(
                subgroup_metrics,
                str(subgroup_choice),
                subgroup_threshold,
                language,
            )
            if _sub_pdf_bytes:
                _bytes_download_btn(
                    _sub_pdf_bytes,
                    "subgroup_summary.pdf",
                    tr("Download summary PDF", "Baixar PDF resumido"),
                    "application/pdf",
                    key="subgroup_summary_pdf",
                )
            else:
                st.caption(tr("Summary PDF unavailable.", "PDF resumido indisponível."))
        with _sub_dl3:
            _sub_sig = json.dumps(_bundle_signature(xlsx_path), sort_keys=True)
            _sub_export_key = f"_subgroup_full_xlsx_{abs(hash((_sub_sig, round(float(subgroup_threshold), 6), language)))}"
            if st.button(
                tr("Prepare full XLSX", "Preparar XLSX completo"),
                key=f"{_sub_export_key}_prepare",
                width="stretch",
            ):
                with st.spinner(tr("Building consolidated subgroup export...", "Gerando export consolidado de subgrupos...")):
                    _all_subgroups = _build_all_subgroup_metrics_cached(
                        subgroup_df,
                        subgroup_panel_specs,
                        subgroup_score_cols,
                        float(subgroup_threshold),
                    )
                    st.session_state[_sub_export_key] = {
                        "xlsx": _build_subgroup_xlsx_bytes(_all_subgroups, float(subgroup_threshold), language),
                        "n_rows": int(len(_all_subgroups)),
                    }
            _sub_export_payload = st.session_state.get(_sub_export_key)
            if _sub_export_payload:
                if int(_sub_export_payload.get("n_rows", 0) or 0) == 0:
                    st.warning(tr(
                        "No consolidated subgroup rows were available.",
                        "Nenhuma linha consolidada de subgrupo ficou disponível.",
                    ))
                else:
                    _bytes_download_btn(
                        _sub_export_payload["xlsx"],
                        "subgroup_all_panels.xlsx",
                        tr("Download full XLSX", "Baixar XLSX completo"),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="subgroup_full_xlsx",
                    )

elif _active_tab == 7:  # Data Quality
    st.subheader(tr("Data Quality", "Qualidade da Base"))
    st.caption(tr(
        "Completeness, imputation exposure, score availability, and validation readiness for the current dataset.",
        "Completude, exposição à imputação, disponibilidade de escores e prontidão para validação do dataset atual.",
    ))

    _dq = compute_data_quality_summary(df, prepared.feature_columns, language)
    _model_meta_dq = build_model_metadata(
        prepared.info, artifacts.leaderboard, best_model_name,
        artifacts.feature_columns, xlsx_path, sts_available=HAS_STS,
        bundle_saved_at=bundle_info.get("saved_at"),
        training_source_file=bundle_info.get("training_source"),
        calibration_method=getattr(artifacts, "calibration_method", "sigmoid"),
        training_data=prepared.data,
    )
    _val_checks = check_validation_readiness(_model_meta_dq, language)

    # Pre-compute missing rate table once — reused in Issues block and expander.
    _miss_rate_col = tr("Missing rate", "Taxa de missing")
    _miss_var_col = tr("Variable", "Variável")
    _miss_pct_col = tr("Missing %", "Missing %")
    _miss_df = pd.DataFrame([
        {_miss_var_col: var, _miss_rate_col: rate, _miss_pct_col: f"{rate*100:.1f}%"}
        for var, rate in sorted(_dq["missing_rates"].items(), key=lambda x: x[1], reverse=True)
    ])
    _miss_high = (
        _miss_df[_miss_df[_miss_rate_col] > 0.3] if not _miss_df.empty else pd.DataFrame()
    )
    _warn_tokens = {"needs more", "precisa de"}

    # ── COVERAGE OVERVIEW ─────────────────────────────────────────────────────
    dq1, dq2, dq3, dq4 = st.columns(4)
    dq1.metric(tr("Eligible surgeries", "Cirurgias elegíveis"), _dq["n_total"])
    dq2.metric(tr("Deaths (primary outcome)", "Óbitos (desfecho primário)"), _dq["n_events"])
    dq3.metric(tr("Event rate", "Taxa de eventos"), f"{_dq['event_rate']:.1%}")
    dq4.metric(tr("Triple cohort", "Coorte tripla"), _dq["n_triple"])

    # ── ISSUES ────────────────────────────────────────────────────────────────
    _val_warn_checks = [vc for vc in _val_checks if any(t in str(vc.get("status", "")).lower() for t in _warn_tokens)]
    _has_issues = not _miss_high.empty or bool(_val_warn_checks)
    st.divider()
    if _has_issues:
        st.markdown(tr("### Issues", "### Problemas"))
        if not _miss_high.empty:
            st.error(tr(
                f"**{len(_miss_high)} variable(s) with >30% missing data** — may reduce prediction reliability.",
                f"**{len(_miss_high)} variável(is) com >30% de dados faltantes** — pode reduzir a confiabilidade das predições.",
            ))
            with st.expander(tr(f"Variables with >30% missing ({len(_miss_high)})", f"Variáveis com >30% missing ({len(_miss_high)})"), expanded=True):
                st.dataframe(_miss_high[[_miss_var_col, _miss_pct_col]], width="stretch", hide_index=True)
        for vc in _val_warn_checks:
            st.warning(f"**{vc['check']}**: {vc['status']} — {vc['note']}")
    else:
        st.success(tr(
            "No critical data quality issues detected.",
            "Nenhum problema crítico de qualidade detectado.",
        ))

    # ── SCORE AVAILABILITY ────────────────────────────────────────────────────
    st.divider()
    st.markdown(tr("### Score Availability", "### Disponibilidade de Escores"))
    _score_primary = pd.DataFrame([
        {tr("Score", "Escore"): "AI Risk (OOF)", tr("Patients", "Pacientes"): int(df["ia_risk_oof"].notna().sum()) if "ia_risk_oof" in df.columns else 0},
        {tr("Score", "Escore"): tr("EuroSCORE II (app-calculated)", "EuroSCORE II (calculado pelo app)"), tr("Patients", "Pacientes"): _dq["n_euro_calc"]},
        {tr("Score", "Escore"): tr("STS (app-calculated)", "STS (calculado pelo app)"), tr("Patients", "Pacientes"): _dq["n_sts"]},
        {tr("Score", "Escore"): tr("Triple cohort (all 3 scores)", "Coorte tripla (3 escores)"), tr("Patients", "Pacientes"): _dq["n_triple"]},
    ])
    st.dataframe(_score_primary, width="stretch", hide_index=True)

    with st.expander(tr("Sheet-derived scores (reference only)", "Escores derivados da planilha (apenas referência)"), expanded=False):
        st.caption(tr(
            "These values were read from the original input file and are shown for comparison/validation purposes only. They are NOT used in the primary analysis.",
            "Estes valores foram lidos do arquivo de entrada original e são mostrados apenas para fins de comparação/validação. NÃO são usados na análise principal.",
        ))
        _score_ref = pd.DataFrame([
            {tr("Score", "Escore"): "EuroSCORE II (sheet)", tr("Patients", "Pacientes"): _dq["n_euro_sheet"]},
            {tr("Score", "Escore"): "EuroSCORE II Auto (sheet)", tr("Patients", "Pacientes"): _dq["n_euro_auto"]},
            {tr("Score", "Escore"): "STS (sheet)", tr("Patients", "Pacientes"): _dq["n_sts_sheet"]},
        ])
        st.dataframe(_score_ref, width="stretch", hide_index=True)

    # ── VALIDATION READINESS ──────────────────────────────────────────────────
    st.divider()
    st.markdown(tr("### Validation Readiness", "### Prontidão para Validação"))
    for vc in _val_checks:
        _vc_status = str(vc.get("status", ""))
        _vc_is_warn = any(t in _vc_status.lower() for t in _warn_tokens)
        if _vc_is_warn:
            st.warning(f"**{vc['check']}**: {_vc_status} — {vc['note']}")
        else:
            st.success(f"**{vc['check']}**: {_vc_status} — {vc['note']}")

    # ── DETAILED TABLES ───────────────────────────────────────────────────────
    st.divider()
    st.markdown(tr("### Detailed Data", "### Dados Detalhados"))

    with st.expander(tr("Missing rate per variable (all predictors)", "Taxa de missing por variável (todos os preditores)"), expanded=False):
        st.caption(tr(
            "Proportion of missing values in the analytical dataset for each predictor variable, sorted by missing rate descending.",
            "Proporção de valores ausentes no dataset analítico para cada variável preditora, ordenada por taxa de missing decrescente.",
        ))
        if not _miss_df.empty:
            st.dataframe(_miss_df, width="stretch", hide_index=True)
        else:
            st.info(tr("No missing data in the analytical dataset.", "Nenhum dado faltante no dataset analítico."))

    if _dq["surgery_dist"]:
        with st.expander(tr("Surgical procedure distribution", "Distribuição de procedimentos cirúrgicos"), expanded=False):
            _surg_df = pd.DataFrame([
                {tr("Procedure", "Procedimento"): proc, tr("Count", "Contagem"): count}
                for proc, count in _dq["surgery_dist"].items()
            ])
            st.dataframe(_surg_df, width="stretch", hide_index=True)

    with st.expander(tr("Analysis audit trail", "Trilha de auditoria"), expanded=False):
        st.caption(tr(
            "Recent analysis events logged by the application.",
            "Eventos de análise recentes registrados pelo aplicativo.",
        ))
        _audit_entries = read_audit_log(20)
        if _audit_entries:
            _audit_df = pd.DataFrame(_audit_entries)
            _audit_cols = [c for c in ["timestamp", "analysis_type", "source_file", "model_version", "n_patients", "n_imputed", "completeness_level", "sts_method"] if c in _audit_df.columns]
            st.dataframe(_audit_df[_audit_cols], width="stretch", hide_index=True)
        else:
            st.info(tr("No audit entries yet. They will appear as you use the app.", "Nenhum registro de auditoria ainda. Eles aparecerão conforme você usar o app."))

elif _active_tab == 8:  # Variable Dictionary
    st.subheader(tr("Variable Dictionary", "Dicionário de Variáveis"))
    st.caption(tr(
        "Live reference table generated from the current ingestion code: source columns, accepted aliases, missing-value rules, derived fields, and active model usage.",
        "Tabela de referencia viva gerada a partir do codigo atual de leitura: colunas de origem, aliases aceitos, regras de ausentes, variaveis derivadas e uso no modelo ativo.",
    ))

    _dict_df = get_app_reading_dictionary_dataframe(
        language,
        model_feature_columns=getattr(artifacts, "feature_columns", None),
    )
    _alias_df = get_reading_aliases_dataframe(language)
    _rules_df = get_reading_rules_dataframe(language)
    _domain_col = "Dominio" if language != "English" else "Domain"

    _dict_filter = st.multiselect(
        tr("Filter by domain", "Filtrar por domínio"),
        sorted(_dict_df[_domain_col].dropna().unique().tolist()),
        default=[],
    )
    if _dict_filter:
        _dict_display = _dict_df[_dict_df[_domain_col].isin(_dict_filter)]
    else:
        _dict_display = _dict_df

    st.dataframe(_dict_display, width="stretch", hide_index=True)

    _dict_xlsx = build_dictionary_xlsx_bytes(_dict_df, _alias_df, _rules_df)
    _dl_cols = st.columns([1, 1, 4])
    with _dl_cols[0]:
        _bytes_download_btn(
            _dict_xlsx,
            "data_dictionary.xlsx",
            tr("Download XLSX", "Baixar XLSX"),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_live_data_dictionary_xlsx",
        )
    with _dl_cols[1]:
        _csv_download_btn(_dict_df, "data_dictionary.csv", tr("Download CSV", "Baixar CSV"))

    with st.expander(tr("Reading rules used by the app", "Regras de leitura usadas pelo app"), expanded=False):
        st.dataframe(_rules_df, width="stretch", hide_index=True)

    with st.expander(tr("Accepted flat-file aliases", "Aliases aceitos em arquivos planos"), expanded=False):
        st.dataframe(_alias_df, width="stretch", hide_index=True)

elif _active_tab == 9:  # Temporal Validation
    # Delegated to tabs/temporal_validation.py.  Every dependency the tab
    # needs is threaded through TabContext; no TV-specific state remains
    # inline here.  See tabs/temporal_validation.py:render().
    _tab_temporal_validation.render(_tab_ctx)

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
