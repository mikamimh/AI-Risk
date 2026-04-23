# -*- coding: utf-8 -*-
"""Temporal Validation tab — extracted from app.py (tab index 9).

Conservative extraction: behaviour, text, i18n, threading model, session-state
keys, scientific formulas, STS integration, and export formats are identical
to the original inline code.  The only structural change is that shared
Streamlit state is accessed through ``ctx`` (:class:`tabs.TabContext`) instead
of bare local variables captured by ``app.py``.

Every helper that was originally defined at module scope in ``app.py`` and
used only by this tab is kept here locally (see ``_sts_score_status_caption``),
so ``app.py`` no longer needs to own TV-specific code.

Caveats / invariants preserved verbatim:
  * The STS worker thread and the ``@st.fragment(run_every=1.0)`` polling
    fragment keep their original session-state keys (``_tv_sts_state``,
    ``_tv_sts_ctx``, ``_tv_sts_prog``, ``_tv_sts_results``, …).
  * The context-signature hash uses the same tuple of fields and the same
    24/16-char SHA-256 truncation.
  * ``_extract_year_quarter_range`` is imported from ``temporal_validation``
    (its canonical location) — ``model_metadata`` re-exports it for compat.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING


import numpy as np
import pandas as pd
import streamlit as st

from config import AppConfig
from euroscore import euroscore_from_row
from risk_data import (
    prepare_master_dataset,
    normalize_external_dataset,
    read_external_table_with_fallback,
)
from sts_calculator import (
    HAS_WEBSOCKETS as HAS_STS,
    STS_PER_PATIENT_TIMEOUT_S,
    calculate_sts_batch,
    classify_sts_eligibility,
)
from stats_compare import (
    calibration_bins_detail,
    calibration_in_the_large,
    calibration_intercept_slope,
    class_risk,
    classification_metrics_at_threshold,
    decision_curve,
    evaluate_scores_temporal,
    hosmer_lemeshow_test,
    integrated_calibration_index,
    pairwise_score_comparison,
    recalibrate_intercept_only,
    recalibrate_isotonic,
    recalibrate_logistic,
    risk_category_table,
    roc_data,
    threshold_analysis_table,
    youden_threshold,
)
from model_metadata import (
    build_model_metadata,
    format_locked_model_for_display,
    build_temporal_validation_summary,
    build_exploratory_recalibration_summary,
    build_exploratory_threshold_summary,
    build_exploratory_temporal_validation_section,
    build_sts_accounting_table,
    check_temporal_overlap,
    is_surrogate_timeline,
    build_surrogate_timeline_note,
    log_analysis,
    statistical_summary_to_pdf,
)
from tv_helpers import (
    chronological_state_label,
    classify_sts_availability,
    build_sts_availability_summary,
    STS_AVAILABILITY_PARTIAL,
    STS_AVAILABILITY_UNAVAILABLE,
    CHRONO_STATE_UNKNOWN,
)
from ai_risk_inference import apply_frozen_model_to_temporal_cohort
from export_helpers import build_export_manifest

if TYPE_CHECKING:
    from tabs import TabContext


# Convenience constants mirroring those app.py used via AppConfig.
# NOTE: ``MODEL_VERSION`` is intentionally NOT imported here.  The previous
# module-level binding (``MODEL_VERSION = AppConfig.MODEL_VERSION``) created
# a second source of truth that could silently drift from the loaded
# bundle's signature.  ``render()`` now reads it from
# ``ctx.model_version`` (which comes from ``bundle_info``), and any code
# outside ``render()`` that accidentally references ``MODEL_VERSION`` will
# fail at import time — by design.
TEMP_DATA_DIR = AppConfig.TEMP_DATA_DIR

_TV_MODEL_COLORS = {
    "AI Risk": "#1f77b4",
    "EuroSCORE II": "#ff7f0e",
    "STS Score": "#2ca02c",
}


def _sts_score_status_caption(execution_log) -> str:
    """Return a one-line summary like 'cached 431 · fresh 20 · stale_fallback 3'.

    Moved from ``app.py`` — only the Temporal Validation tab uses this helper.
    """
    try:
        import sts_cache as _sc
        summary = _sc.summarise_execution_log(execution_log)
    except Exception:
        return ""
    parts = []
    for k in ("fresh", "cached", "refreshed", "stale_fallback", "failed"):
        if summary.get(k, 0):
            parts.append(f"{k} {summary[k]}")
    return " · ".join(parts)


def _sts_availability_details_caption(
    availability: str,
    n_score: int,
    n_eligible: int,
    language: str = "English",
) -> str:
    """Human-readable STS availability line for execution-details blocks."""
    if availability == "complete":
        return (
            f"STS availability: complete ({n_score}/{n_eligible} eligible)"
            if language == "English"
            else f"Disponibilidade do STS: completa ({n_score}/{n_eligible} elegíveis)"
        )
    if availability == "partial":
        return (
            f"STS availability: partial ({n_score}/{n_eligible} eligible)"
            if language == "English"
            else f"Disponibilidade do STS: parcial ({n_score}/{n_eligible} elegíveis)"
        )
    if availability == "unavailable":
        return (
            f"STS availability: unavailable ({n_score}/{n_eligible} eligible)"
            if language == "English"
            else f"Disponibilidade do STS: indisponível ({n_score}/{n_eligible} elegíveis)"
        )
    if availability == "no_eligible":
        return (
            "STS availability: no STS-eligible rows"
            if language == "English"
            else "Disponibilidade do STS: sem linhas elegíveis para STS"
        )
    return (
        "STS availability: not requested"
        if language == "English"
        else "Disponibilidade do STS: não solicitada"
    )


def _compute_sts_availability_status(n_eligible: int, n_score: int) -> str:
    """Compatibility wrapper for the shared temporal-validation helper."""
    return classify_sts_availability(n_eligible, n_score)


def _build_sts_patient_audit(
    eligibility_log: list,
    eligible_idx: list,
    raw_results: list,
    exec_log: list,
    fail_log: list,
    sts_score_col,
) -> list:
    """Build a per-eligible-patient audit table for end-to-end STS traceability.

    Each row covers one patient classified as ``supported`` by
    ``classify_sts_eligibility``.  The table makes explicit exactly where a
    patient was lost in the pipeline so that a headline like "22 eligible → 6
    final scores" can be decomposed into specific loss categories.

    Parameters
    ----------
    eligibility_log:
        List of ``{row_index, patient_id, eligibility, reason}`` dicts (ALL
        patients, from the pre-classification step).
    eligible_idx:
        List mapping eligible-position → cohort row index.
        ``eligible_idx[eli_pos] == cohort_idx``.
    raw_results:
        List of result dicts (length = n_eligible).  ``raw_results[eli_pos]``
        is the raw STS result for the patient at that eligible position.
    exec_log:
        List of ``ExecutionRecord`` objects (length = n_eligible), from
        ``calculate_sts_batch.last_execution_log``.
    fail_log:
        List of failure dicts from ``calculate_sts_batch.failure_log``.
        Each entry has ``idx`` = 0-based eligible position.
    sts_score_col:
        ``pd.Series`` (the ``sts_score`` column of ``_tv_data``) or ``None``.
        Used to check whether a score is present in the final output.

    Returns
    -------
    list of dict with 15 fields per eligible patient:
      row_index, patient_id, sts_eligibility_status, sts_supported_class,
      sts_input_ready, sts_score_from_cache, sts_query_attempted,
      sts_query_success, sts_parse_success, sts_score_present_final,
      sts_batch_aborted_before_query, sts_failure_stage, sts_failure_reason,
      sts_chunk_index, sts_retry_attempted

    Consistency invariants
    ----------------------
    * ``sts_score_from_cache=True``  → ``sts_query_attempted=False``,
      ``sts_score_present_final=True`` (score came from Phase A cache hit).
    * ``sts_query_attempted=True, sts_query_success=True``
      → ``sts_score_present_final=True`` (live query returned predmort).
    * ``sts_score_present_final=False, sts_score_from_cache=False,
      sts_query_attempted=False`` → either build_input failed or batch_aborted.
    """
    _elig_by_cohort: dict = {e["row_index"]: e for e in eligibility_log}
    _fail_by_pos: dict = {f["idx"]: f for f in fail_log}

    rows = []
    for eli_pos, cohort_idx in enumerate(eligible_idx):
        elig = _elig_by_cohort.get(cohort_idx, {})
        fail = _fail_by_pos.get(eli_pos)
        exec_rec = exec_log[eli_pos] if eli_pos < len(exec_log) else None
        raw_res = (raw_results[eli_pos] if eli_pos < len(raw_results) else None) or {}

        _fail_stage  = fail["stage"]  if fail else None
        _exec_status = getattr(exec_rec, "status", None) if exec_rec else None

        # ── Derived audit flags ───────────────────────────────────────────
        # Was the STS input dict built without error?
        input_ready = _fail_stage != "build_input"

        # Did this patient get its score from the Phase-A memory/disk cache?
        # When True, no live network query was issued; score_present will also be True.
        from_cache = _exec_status == "cached"

        # Was a live network query sent to the STS endpoint?
        # True for: fetch failures (stage="fetch"), fresh/refreshed/stale_fallback.
        # False for: cached (Phase A short-circuit), build_input failures, batch_abort.
        query_attempted = (
            _fail_stage == "fetch"
            or _exec_status in ("fresh", "refreshed", "stale_fallback")
        )

        # Did the live query return a usable response?
        # Only True for fresh/refreshed; stale_fallback means the live query failed.
        query_success = _exec_status in ("fresh", "refreshed")

        # Is the primary STS endpoint (predmort) present in the raw result dict?
        parse_success = "predmort" in raw_res

        # Was this patient skipped because an abort triggered before it was queried?
        batch_aborted = _fail_stage == "batch_abort"

        # Is a final STS score present in the output DataFrame column?
        try:
            if sts_score_col is not None:
                _sv = sts_score_col.iloc[cohort_idx]
                score_present = bool(_sv is not None and not pd.isna(_sv))
            else:
                score_present = False
        except Exception:
            score_present = False

        rows.append({
            "row_index":                      cohort_idx,
            "patient_id":                     elig.get("patient_id", ""),
            "sts_eligibility_status":         elig.get("eligibility", "supported"),
            "sts_supported_class":            elig.get("reason", ""),
            "sts_input_ready":                input_ready,
            "sts_score_from_cache":           from_cache,
            "sts_query_attempted":            query_attempted,
            "sts_query_success":              query_success,
            "sts_parse_success":              parse_success,
            "sts_score_present_final":        score_present,
            "sts_batch_aborted_before_query": batch_aborted,
            "sts_failure_stage":              _fail_stage or "",
            "sts_failure_reason":             (fail.get("reason") or "") if fail else "",
            "sts_chunk_index":                eli_pos,
            "sts_retry_attempted":            (
                fail.get("retry_attempted", False) if fail
                else getattr(exec_rec, "retry_attempted", False) if exec_rec
                else False
            ),
        })
    return rows


def render(ctx: "TabContext") -> None:  # noqa: C901 — extracted verbatim; complexity matches original
    """Render the Temporal Validation tab (tab index 9)."""
    # ── Context aliases ─────────────────────────────────────────────────
    # Bind the names used by the original inline code to their ctx sources.
    # These aliases keep the body below byte-for-byte identical to the
    # version that lived inside ``app.py`` — the extraction is purely
    # structural.
    tr = ctx.tr
    language = ctx.language
    prepared = ctx.prepared
    artifacts = ctx.artifacts
    forced_model = ctx.forced_model
    best_model_name = ctx.best_model_name
    bundle_info = ctx.bundle_info
    xlsx_path = ctx.xlsx_path

    # Bundle-sourced version — single source of truth for filename, MD/PDF
    # headers, manifest, and audit log within this tab.  No module-level
    # ``MODEL_VERSION`` exists in this file by design (see header note),
    # so this is a fresh local binding, not a shadow.
    MODEL_VERSION = ctx.model_version

    _update_phase = ctx.update_phase
    _sts_score_patient_ids = ctx.sts_score_patient_ids

    # ── Begin original tab body (verbatim from app.py) ──────────────────

    st.subheader(tr("Temporal Validation", "Validação Temporal"))
    st.caption(tr(
        "Prospective validation: a previously locked model is applied to an independent later cohort. "
        "No retraining, recalibration, or model reselection is performed. "
        "Primary analysis uses the locked model with the fixed operational threshold. "
        "Recalibration and exploratory threshold analyses are supplementary.",
        "Validação prospectiva: um modelo previamente congelado é aplicado a uma coorte posterior independente. "
        "Não há retreinamento, recalibração ou nova seleção de modelo. "
        "A análise principal usa o modelo congelado com o limiar operacional fixo. "
        "Recalibração e análises exploratórias de limiar são suplementares.",
    ))

    # ── Block 2: Locked Model ──────────────────────────────────────────────
    st.markdown(tr("### Locked Model", "### Modelo Congelado"))
    st.caption(tr(
        "The frozen pipeline (preprocessing + fitted estimator + calibration) is applied exactly as saved. "
        "The locked threshold is used for all primary classification metrics.",
        "O pipeline congelado (pré-processamento + estimador ajustado + calibração) é aplicado exatamente como salvo. "
        "O limiar bloqueado é usado para todas as métricas de classificação primárias.",
    ))

    # ── 1. Locked model info ──
    _tv_meta = build_model_metadata(
        prepared.info, artifacts.leaderboard, best_model_name,
        artifacts.feature_columns, xlsx_path, sts_available=HAS_STS,
        bundle_saved_at=bundle_info.get("saved_at"),
        training_source_file=bundle_info.get("training_source"),
        calibration_method=getattr(artifacts, "calibration_method", "sigmoid"),
        training_data=prepared.data,
        model_version=MODEL_VERSION,
    )
    _tv_locked_threshold_default = _tv_meta.get("locked_threshold", 0.08)

    # ── Threshold mode selector ──
    _tv_best_youden = getattr(artifacts, "best_youden_threshold", None)
    _tv_youden_avail = _tv_best_youden is not None
    _tv_mode_fixed = tr(
        f"Locked clinical threshold ({_tv_locked_threshold_default*100:.0f}%)",
        f"Limiar clínico bloqueado ({_tv_locked_threshold_default*100:.0f}%)",
    )
    _tv_mode_youden = (
        tr(
            f"Training Youden threshold: {_tv_best_youden*100:.1f}%",
            f"Limiar de Youden do treino: {_tv_best_youden*100:.1f}%",
        )
        if _tv_youden_avail
        else tr("Youden threshold (not available)", "Limiar de Youden (indisponível)")
    )
    _tv_threshold_mode = st.radio(
        tr("Threshold mode", "Modo de limiar"),
        options=[_tv_mode_fixed, _tv_mode_youden],
        index=0,
        horizontal=True,
        disabled=not _tv_youden_avail,
        help=tr(
            "Choose the threshold for classification metrics. The Youden threshold was learned during training — it is NOT recomputed on the validation data.",
            "Escolha o limiar para métricas de classificação. O limiar de Youden foi aprendido durante o treino — NÃO é recalculado nos dados de validação.",
        ),
        key="tv_threshold_mode",
    )
    _tv_use_youden = _tv_youden_avail and _tv_threshold_mode == _tv_mode_youden
    _tv_locked_threshold = _tv_best_youden if _tv_use_youden else _tv_locked_threshold_default

    if _tv_use_youden:
        st.info(tr(
            f"**Active threshold: Training Youden = {_tv_locked_threshold*100:.1f}%** (probability {_tv_locked_threshold:.4f}). "
            f"Learned during model development — not recomputed on temporal data.",
            f"**Limiar ativo: Youden do treino = {_tv_locked_threshold*100:.1f}%** (probabilidade {_tv_locked_threshold:.4f}). "
            f"Aprendido durante o desenvolvimento — não recalculado nos dados temporais.",
        ))
    else:
        st.info(tr(
            f"**Active threshold: Locked clinical = {_tv_locked_threshold*100:.1f}%** (probability {_tv_locked_threshold:.4f})  \n"
            f"Locked at training time, aligned with the EuroSCORE II high-risk boundary (>8%). "
            f"Not recomputed on temporal data — this preserves the prospective validation integrity.",
            f"**Limiar ativo: Clínico bloqueado = {_tv_locked_threshold*100:.1f}%** (probabilidade {_tv_locked_threshold:.4f})  \n"
            f"Bloqueado no treinamento, alinhado com a fronteira de alto risco do EuroSCORE II (>8%). "
            f"Não recalculado nos dados temporais — isso preserva a integridade da validação prospectiva.",
        ))

    with st.expander(tr("Locked model details", "Detalhes do modelo congelado"), expanded=True):
        st.dataframe(
            format_locked_model_for_display(_tv_meta, language),
            width="stretch",
            hide_index=True,
        )

    # ── Block 1: Cohort Integrity ──────────────────────────────────────────
    st.divider()
    st.markdown(tr("### Cohort Integrity", "### Integridade da Coorte"))
    st.caption(tr(
        "Upload the validation dataset to verify temporal separation, event rate, "
        "STS availability, and normalization compatibility before running the model.",
        "Faça upload do dataset de validação para verificar separação temporal, taxa de eventos, "
        "disponibilidade do STS e compatibilidade de normalização antes de executar o modelo.",
    ))

    # ── 2. Upload temporal cohort ──
    st.markdown(tr("**Upload temporal cohort**", "**Upload da coorte temporal**"))
    st.caption(tr(
        "Accepted formats: .xlsx, .csv, .parquet, .db, .sqlite, .sqlite3",
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
        _tv_file_bytes = _tv_file.getvalue()          # captured once; reused for content hash
        _tv_temp_path.write_bytes(_tv_file_bytes)

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

                # ── 3a-bis. Cohort drift analysis ─────────────────────────
                try:
                    from stats_compare import compute_cohort_drift as _ccd
                    _tv_drift = _ccd(
                        df_train=prepared.data,
                        df_temporal=_tv_data,
                        y_col="morte_30d",
                        feature_columns=artifacts.feature_columns,
                    )
                    with st.expander(
                        tr("Cohort Drift Analysis", "Análise de Drift da Coorte"),
                        expanded=False,
                    ):
                        _dc1, _dc2 = st.columns(2)
                        _dc1.metric(
                            tr("Prevalence (training)", "Prevalência (treino)"),
                            f"{_tv_drift['prevalence_train']:.1%}",
                        )
                        _dc2.metric(
                            tr("Prevalence (temporal)", "Prevalência (temporal)"),
                            f"{_tv_drift['prevalence_temporal']:.1%}",
                            delta=f"{_tv_drift['prevalence_delta']:+.1%}" if not np.isnan(_tv_drift.get('prevalence_delta', float('nan'))) else None,
                        )
                        st.markdown(tr(
                            "**Top 10 variables with largest missingness shift:**",
                            "**Top 10 variáveis com maior mudança de missingness:**",
                        ))
                        _miss_df = [
                            {
                                tr("Variable", "Variável"): r["variable"],
                                tr("Missing (train)", "Missing (treino)"): f"{r['missing_train']:.1%}",
                                tr("Missing (temporal)", "Missing (temporal)"): f"{r['missing_temporal']:.1%}",
                                tr("Δ", "Δ"): f"{r['delta']:+.1%}" if r.get("delta") is not None and not np.isnan(r.get("delta", float("nan"))) else "—",
                            }
                            for r in _tv_drift["missingness_shift"][:10]
                        ]
                        if _miss_df:
                            st.dataframe(pd.DataFrame(_miss_df), width="stretch", hide_index=True)
                        st.markdown(tr(
                            "**Top 10 numeric variables with largest relative shift (|Δmedian| / IQR_train):**",
                            "**Top 10 variáveis numéricas com maior shift relativo (|Δmediana| / IQR_treino):**",
                        ))
                        _num_df = [
                            {
                                tr("Variable", "Variável"): r["variable"],
                                tr("Median (train)", "Mediana (treino)"): f"{r['median_train']:.2f}",
                                tr("Median (temporal)", "Mediana (temporal)"): f"{r['median_temporal']:.2f}",
                                "IQR (train)": f"{r['p75_train'] - r['p25_train']:.2f}",
                                tr("|Δ|/IQR", "|Δ|/IQR"): f"{r['rel_shift_over_iqr']:.2f}",
                            }
                            for r in _tv_drift["numeric_shift"][:10]
                        ]
                        if _num_df:
                            st.dataframe(pd.DataFrame(_num_df), width="stretch", hide_index=True)
                except Exception:
                    pass

                # ── 3b. External-dataset normalization summary ──
                # Only for CSV/Parquet uploads (not multi-sheet XLSX).
                # Cached per file-content hash so re-renders are instant.
                _tv_norm_report = None
                if _tv_ext in {".csv", ".parquet"}:
                    import hashlib as _tv_hl_norm
                    _tv_norm_hash = _tv_hl_norm.sha256(_tv_file_bytes).hexdigest()[:16]
                    _tv_norm_key = f"_tv_norm_{_tv_norm_hash}"
                    if _tv_norm_key not in st.session_state:
                        try:
                            _tv_read_meta = None
                            if _tv_ext == ".csv":
                                try:
                                    _, _tv_read_meta = read_external_table_with_fallback(
                                        str(_tv_temp_path)
                                    )
                                except Exception:
                                    pass
                            _, _tv_norm_report = normalize_external_dataset(
                                _tv_data,
                                source_name=_tv_file.name,
                                read_meta=_tv_read_meta,
                            )
                            st.session_state[_tv_norm_key] = _tv_norm_report
                        except Exception:
                            st.session_state[_tv_norm_key] = None
                    _tv_norm_report = st.session_state.get(_tv_norm_key)

                if _tv_norm_report is not None:
                    # ── Unit-conversion visible warning (shown outside expander) ──
                    _tv_us = _tv_norm_report.unit_summary or {}
                    _tv_unit_msgs = []
                    if _tv_us.get("height_converted"):
                        _tv_unit_msgs.append(tr(
                            f"Height auto-converted from inches to cm"
                            f" ({_tv_us['n_height_converted']} row(s),"
                            f" original median {_tv_us['height_original_median']:.1f} in).",
                            f"Altura convertida automaticamente de polegadas para cm"
                            f" ({_tv_us['n_height_converted']} linha(s),"
                            f" mediana original {_tv_us['height_original_median']:.1f} in).",
                        ))
                    if _tv_us.get("weight_converted"):
                        _tv_unit_msgs.append(tr(
                            f"Weight auto-converted from lb to kg"
                            f" ({_tv_us['n_weight_converted']} row(s),"
                            f" original median {_tv_us['weight_original_median']:.1f} lb).",
                            f"Peso convertido automaticamente de lb para kg"
                            f" ({_tv_us['n_weight_converted']} linha(s),"
                            f" mediana original {_tv_us['weight_original_median']:.1f} lb).",
                        ))
                    for _tv_unit_msg in _tv_unit_msgs:
                        st.warning(
                            tr("Unit auto-conversion applied: ", "Conversão de unidade automática: ")
                            + _tv_unit_msg
                        )
                    with st.expander(
                        tr(
                            "Dataset normalization summary",
                            "Resumo de normalização do dataset",
                        ),
                        expanded=False,
                    ):
                        _tv_norm_lines = _tv_norm_report.summary_lines()
                        if _tv_norm_lines:
                            for _ln in _tv_norm_lines:
                                if _ln.startswith("[WARNING]"):
                                    st.warning(_ln[9:].strip())
                                else:
                                    st.markdown(f"- {_ln}")
                        else:
                            st.info(tr(
                                "No normalization actions were applied to this dataset.",
                                "Nenhuma ação de normalização foi aplicada a este dataset.",
                            ))
                        _tv_sr = _tv_norm_report.sts_readiness_summary
                        if _tv_sr and _tv_sr.get("n_total", 0) > 0:
                            _tv_sr_parts = [
                                f"{_tv_sr['n_ready']}/{_tv_sr['n_total']}"
                                f" {tr('STS-ready', 'STS-prontos')}"
                            ]
                            if _tv_sr.get("n_pediatric_excluded", 0):
                                _tv_sr_parts.append(
                                    f"{_tv_sr['n_pediatric_excluded']}"
                                    f" {tr('pediatric excluded', 'pediátricos excluídos')}"
                                )
                            if _tv_sr.get("n_scope_excluded", 0):
                                _tv_sr_parts.append(
                                    f"{_tv_sr['n_scope_excluded']}"
                                    f" {tr('scope excluded', 'fora do escopo')}"
                                )
                            if _tv_sr.get("n_missing_fields", 0):
                                _tv_sr_parts.append(
                                    f"{_tv_sr['n_missing_fields']}"
                                    f" {tr('missing required fields', 'campos obrigatórios ausentes')}"
                                )
                            st.caption(
                                tr("STS preflight", "Pré-voo STS")
                                + ": "
                                + " \u00b7 ".join(_tv_sr_parts)
                            )

                # ── 4. Chronological check ──
                from model_metadata import _extract_year_quarter_range
                _tv_val_start, _tv_val_end = _extract_year_quarter_range(_tv_data)

                _tv_overlap = check_temporal_overlap(
                    _tv_meta.get("training_start_date", "Unknown"),
                    _tv_meta.get("training_end_date", "Unknown"),
                    _tv_val_start,
                    _tv_val_end,
                )
                _tv_is_surrogate = (
                    _tv_overlap.get("surrogate_timeline", False)
                    or is_surrogate_timeline(_tv_data)
                )
                with st.expander(tr("Chronological check", "Verificação cronológica"), expanded=True):
                    # Surrogate-timeline disclaimer — shown before range labels so
                    # readers understand the shifted years immediately.
                    if _tv_is_surrogate:
                        st.info(build_surrogate_timeline_note(language))

                    _ov_c1, _ov_c2 = st.columns(2)
                    _ov_range_label = (
                        tr("Surrogate range", "Intervalo substituto")
                        if _tv_is_surrogate
                        else tr("Date range", "Período")
                    )
                    _ov_c1.markdown(
                        tr("**Training cohort**", "**Coorte de treinamento**")
                        + f" ({_ov_range_label}): "
                        + f"{_tv_meta.get('training_start_date', 'Unknown')} \u2014 "
                        + f"{_tv_meta.get('training_end_date', 'Unknown')}"
                    )
                    _ov_c2.markdown(
                        tr("**Validation cohort**", "**Coorte de validação**")
                        + f" ({_ov_range_label}): "
                        + f"{_tv_val_start} \u2014 {_tv_val_end}"
                    )

                    sev = _tv_overlap["severity"]
                    msg = _tv_overlap[f"message_{'en' if language == 'English' else 'pt'}"]
                    # Explicit four-way chronological state label — surfaces the
                    # canonical status (no overlap / overlap / retrograde / unknown)
                    # so the UI never relies only on severity colour.
                    _chrono_state = _tv_overlap.get("status", CHRONO_STATE_UNKNOWN)
                    _chrono_label = chronological_state_label(_chrono_state, language)
                    st.markdown(
                        tr("**Chronological state:** ", "**Estado cronológico:** ")
                        + f"{_chrono_label}"
                    )
                    if sev == "success":
                        st.success(msg)
                    elif sev == "warning":
                        st.warning(msg)
                    elif sev == "error":
                        st.error(msg)
                    else:
                        st.info(msg)

                # ── 4b. Methodological hard-stop ──
                # Phase: temporal-validation hardening.  When the validation
                # cohort is chronologically *before* the training cohort, the
                # exercise is no longer temporal validation — it is retrograde
                # validation and must not be allowed to run.  Previously this
                # was only a visible warning; now we block the Run button so
                # the user cannot accidentally publish retrograde results.
                _tv_chrono_blocked = (sev == "error")

                # ── Session-cache context signature ──
                # Used to detect when the inputs have changed so cached results
                # from a prior run are NOT reused for a different file/model/threshold.
                # STS mode is included so that toggling the checkbox invalidates the
                # cache automatically — prevents showing STS results when STS is off
                # or vice versa.
                import hashlib as _tv_hl
                # Content-addressed signature: immune to filename, size, or Streamlit
                # file_id changes.  If the bytes change the sig changes; if not, it
                # doesn't matter — the result is correctly reusable.
                _tv_file_content_hash = _tv_hl.sha256(_tv_file_bytes).hexdigest()[:24]
                _tv_file_sig = _tv_file_content_hash
                # Read STS checkbox state from session before the widget renders
                # (session_state already holds the current value at this rerun).
                _tv_include_sts_sig = st.session_state.get("tv_include_sts", HAS_STS)
                _tv_context_sig = _tv_hl.sha256(
                    (
                        f"{_tv_file_sig}|"
                        f"{_tv_meta.get('bundle_saved_at', '')}|"
                        f"{forced_model}|"
                        f"{_tv_locked_threshold:.6f}|"
                        f"sts={'1' if (HAS_STS and bool(_tv_include_sts_sig)) else '0'}"
                    ).encode()
                ).hexdigest()[:16]
                _tv_has_cached = (
                    not _tv_chrono_blocked
                    and "_tv_result" in st.session_state
                    and st.session_state.get("_tv_result_sig") == _tv_context_sig
                )

                # ── Debug: cache-key audit ──────────────────────────────────────────
                # Temporary — lets the user confirm that a changed file produces a
                # new signature.  Remove once the cache behaviour is verified.
                with st.expander(
                    tr("Cache key audit (debug)", "Auditoria de chave de cache (debug)"),
                    expanded=False,
                ):
                    _cache_status_label = (
                        tr("HIT — restoring previous result", "HIT — restaurando resultado anterior")
                        if _tv_has_cached
                        else tr("MISS — fresh run required", "MISS — nova execução necessária")
                    )
                    st.caption(
                        tr(
                            f"Upload content hash: `{_tv_file_content_hash}` | "
                            f"Context sig: `{_tv_context_sig}` | "
                            f"STS mode: `{'on' if (HAS_STS and bool(_tv_include_sts_sig)) else 'off'}` | "
                            f"Cache: **{_cache_status_label}**",
                            f"Hash do conteúdo: `{_tv_file_content_hash}` | "
                            f"Assinatura de contexto: `{_tv_context_sig}` | "
                            f"Modo STS: `{'ligado' if (HAS_STS and bool(_tv_include_sts_sig)) else 'desligado'}` | "
                            f"Cache: **{_cache_status_label}**",
                        )
                    )

                # ── 4c. STS mode selector ──
                st.divider()
                st.markdown(tr("#### Scoring options", "#### Opções de pontuação"))
                _tv_include_sts = st.checkbox(
                    tr(
                        "Include STS Score (eligible cases only)",
                        "Incluir STS Score (apenas casos elegíveis)",
                    ),
                    value=HAS_STS,
                    disabled=not HAS_STS,
                    key="tv_include_sts",
                    help=tr(
                        "When enabled, AI Risk and EuroSCORE II are supplemented with the STS ACSD "
                        "web calculator for procedures it supports (CABG, AVR, MVR, MV Repair, and "
                        "combinations). Aortic dissection / aneurysm repair, Bentall procedure, and "
                        "similar out-of-scope operations are automatically skipped. STS queries the "
                        "official web calculator and may be slow or temporarily unavailable.",
                        "Quando ativado, AI Risk e EuroSCORE II são complementados pela calculadora "
                        "web STS ACSD para procedimentos suportados (CABG, AVR, MVR, Plastia Mitral "
                        "e combinações). Dissecção aórtica / reparo de aneurisma, procedimento Bentall "
                        "e operações semelhantes fora do escopo são automaticamente ignoradas. As "
                        "consultas STS usam a calculadora web oficial e podem ser lentas ou "
                        "temporariamente indisponíveis.",
                    ),
                )
                if not HAS_STS:
                    st.caption(tr(
                        "STS Score is not available in this environment (websockets package not installed).",
                        "STS Score não está disponível neste ambiente (pacote websockets não instalado).",
                    ))
                elif _tv_include_sts:
                    st.caption(tr(
                        "STS Score will be queried for supported procedures only. "
                        "Unsupported or uncertain cases will be skipped automatically and logged.",
                        "STS Score será consultado apenas para procedimentos suportados. "
                        "Casos não suportados ou incertos serão ignorados automaticamente e registrados.",
                    ))
                else:
                    st.caption(tr(
                        "Analysis will run with AI Risk + EuroSCORE II only (STS Score disabled).",
                        "A análise usará apenas AI Risk + EuroSCORE II (STS Score desativado).",
                    ))

                # ── 5. Run button ──
                # Read current STS thread state early (before key-constant block)
                # so we can disable the button when a background STS query is running.
                _tv_sts_running_early = st.session_state.get("_tv_sts_state", "idle") in (
                    "running", "cancelling"
                )
                st.divider()
                if _tv_chrono_blocked:
                    st.error(tr(
                        "Temporal validation execution is blocked: the validation cohort is "
                        "chronologically before the training cohort. Upload a cohort that is "
                        "after the training period.",
                        "Execução da validação temporal bloqueada: a coorte de validação é "
                        "cronologicamente anterior à coorte de treinamento. Faça upload de uma "
                        "coorte posterior ao período de treinamento.",
                    ))
                elif _tv_sts_running_early:
                    st.info(tr(
                        "STS Score batch is running — the Run button is disabled until "
                        "STS completes or is cancelled.",
                        "Lote STS em execução — o botão Run está desativado até o STS "
                        "ser concluído ou cancelado.",
                    ))
                _tv_run = st.button(
                    tr("Run temporal validation", "Executar validação temporal"),
                    type="primary",
                    width="stretch",
                    disabled=_tv_chrono_blocked or _tv_sts_running_early,
                )

                # ── Defaults (prevent NameErrors on first page load) ──────────────
                _tv_sts_eligibility_log = []
                _tv_is_surrogate = _tv_is_surrogate  # already set above from overlap check
                _tv_sts_availability: str = "no_sts"   # "complete"|"partial"|"unavailable"|"no_sts"|"no_eligible"
                _tv_sts_n_score: int = 0               # eligible rows with a final STS score
                _tv_sts_n_eligible: int = 0            # eligible rows (supported by STS ACSD)
                _tv_sts_availability_note: dict | None = None
                _tv_sts_score_label = "STS Score"

                # ── STS thread state — session_state keys ──────────────────────────
                # These constants are shared by the run block (which starts the
                # thread) and the polling handler (which polls it on reruns).
                _TV_STS_ST   = "_tv_sts_state"   # "idle"|"running"|"cancelling"
                _TV_STS_CTX  = "_tv_sts_ctx"     # dict: all context for Phase 5
                _TV_STS_CSIG = "_tv_sts_ctx_sig" # context signature string
                _TV_STS_DONE = "_tv_sts_done_evt"
                _TV_STS_ABRT = "_tv_sts_abort_evt"
                _TV_STS_RES  = "_tv_sts_results" # list, filled by worker thread
                _TV_STS_PROG = "_tv_sts_prog"    # dict, updated by phase_callback

                # Is the STS thread from a previous run still in progress?
                _tv_sts_state = st.session_state.get(_TV_STS_ST, "idle")
                _tv_sts_ctx_valid = (
                    st.session_state.get(_TV_STS_CSIG) == _tv_context_sig
                    and _tv_sts_state in ("running", "cancelling")
                    and not _tv_chrono_blocked
                    and not _tv_run   # Fresh button click takes priority
                )
                _tv_sts_just_completed = False

                # ── Stale-state purge ───────────────────────────────────────────────
                # When the upload content changes (new _tv_context_sig) any previously
                # cached result or in-progress STS ctx is stale.  Purge proactively so
                # it can NEVER bleed into the current run — even if the user navigates
                # away and back without pressing Run.
                # Guard: only purge when there IS a saved sig that differs (avoids
                # wiping state on the very first upload where nothing was saved yet).
                _stale_saved_sig = st.session_state.get("_tv_result_sig")
                if (
                    _stale_saved_sig is not None
                    and _stale_saved_sig != _tv_context_sig
                    and not _tv_sts_ctx_valid   # don't purge while a valid run is live
                ):
                    for _sk in (
                        "_tv_result", "_tv_result_sig",
                        _TV_STS_CTX, _TV_STS_CSIG,
                        _TV_STS_ST, _TV_STS_DONE, _TV_STS_ABRT,
                        _TV_STS_RES, _TV_STS_PROG,
                    ):
                        st.session_state.pop(_sk, None)

                # ── STS polling handler ─────────────────────────────────────────────
                # Runs on every Streamlit rerun while the STS worker thread is active.
                # The main STS UI (eligibility display) already ran in the prior rerun
                # that started the thread; here we only show live progress + cancel.
                if _tv_sts_ctx_valid:
                    import time as _tv_time_mod

                    # Check completion in main execution context first.
                    # When the thread signals done the fragment fires st.rerun(scope="app"),
                    # which brings us here; we collect results and rerun once more into
                    # the display/restore section — no further sleep or polling needed.
                    _sts_done_precheck = st.session_state.get(_TV_STS_DONE)
                    _thread_finished = (
                        _sts_done_precheck is not None and _sts_done_precheck.is_set()
                    )

                    if _thread_finished:
                        # ── Thread done: collect results ──────────────────────────────
                        _sts_ctx   = st.session_state.get(_TV_STS_CTX, {})
                        _sts_abort = st.session_state.get(_TV_STS_ABRT)
                        _sts_t0    = _sts_ctx.get("t0", _tv_time_mod.monotonic())

                        _sts_raw_results = list(st.session_state.get(_TV_STS_RES, []))
                        _sts_eligible_idx = _sts_ctx.get("eligible_idx", [])
                        _sts_n_elig_int   = _sts_ctx.get("n_eligible", 0)
                        _sts_n_total_int  = _sts_ctx.get("n_total", 0)
                        _sts_n_ns         = _sts_ctx.get("n_not_supported", 0)
                        _sts_n_unc        = _sts_ctx.get("n_uncertain", 0)
                        _sts_cancelled_r  = _sts_ctx.get("n_total", 0) - _sts_n_elig_int - _sts_n_ns - _sts_n_unc

                        # Re-hydrate _tv_data with the STS scores
                        _tv_data = _sts_ctx["data"].copy()
                        _tv_sts_scores = [np.nan] * _sts_n_total_int
                        for _ri, _oi in enumerate(_sts_eligible_idx):
                            _r = _sts_raw_results[_ri] if _ri < len(_sts_raw_results) else {}
                            _tv_sts_scores[_oi] = (_r or {}).get("predmort", np.nan)

                        _tv_data["sts_score"] = _tv_sts_scores
                        _tv_sts_ok = _tv_data["sts_score"].notna().sum() > 0
                        # ── STS availability classification ───────────────────────────────
                        _tv_sts_n_score    = int(_tv_data["sts_score"].notna().sum())
                        _tv_sts_n_eligible = _sts_n_elig_int
                        _tv_sts_availability = _compute_sts_availability_status(
                            _tv_sts_n_eligible, _tv_sts_n_score
                        )
                        _tv_sts_availability_note = build_sts_availability_summary(
                            _tv_sts_n_eligible,
                            _tv_sts_n_score,
                            language=language,
                        )
                        _tv_sts_score_label = _tv_sts_availability_note["score_label"]

                        _tv_exec_log = getattr(calculate_sts_batch, "last_execution_log", [])
                        _tv_fail_log = getattr(calculate_sts_batch, "failure_log", [])
                        _batch_aborted_ctx = getattr(calculate_sts_batch, "_batch_aborted", False)
                        _abort_before_query_count = getattr(calculate_sts_batch, "_abort_before_query_count", 0)
                        _tv_chunk_log = getattr(calculate_sts_batch, "chunk_log", [])
                        # eligibility_log must be read from ctx BEFORE the audit call
                        # so that patient_id, eligibility_status, and supported_class
                        # are populated correctly. It was previously assigned AFTER the
                        # call, causing the audit to be built with an empty log.
                        _tv_sts_eligibility_log = _sts_ctx.get("eligibility_log", [])
                        # ── Build per-patient STS audit table ────────────────────────────
                        _tv_sts_audit_rows = _build_sts_patient_audit(
                            _tv_sts_eligibility_log,
                            list(_sts_eligible_idx),
                            _sts_raw_results,
                            _tv_exec_log,
                            _tv_fail_log,
                            _tv_data["sts_score"],
                        )
                        _sts_cancelled_r = (
                            _sts_ctx.get("n_total", 0)
                            - sum(1 for r in _sts_raw_results if r and "predmort" in r)
                            - len(_tv_fail_log)
                            - _sts_n_ns
                            - _sts_n_unc
                        )
                        _sts_cancelled_r = max(0, _sts_cancelled_r)

                        _n_completed = sum(1 for r in _sts_raw_results if r and "predmort" in r)
                        _n_failed    = len(_tv_fail_log)
                        _tv_sts_status = _sts_score_status_caption(_tv_exec_log)
                        _tv_ai_incidents = _sts_ctx.get("ai_incidents", [])
                        _tv_is_surrogate = _sts_ctx.get("is_surrogate", False)
                        _tv_locked_threshold = _sts_ctx.get("locked_threshold", _tv_locked_threshold_default)

                        _elapsed_ctx = int(_tv_time_mod.monotonic() - _sts_t0)
                        _user_cancelled = (_tv_sts_state == "cancelling") or (
                            _sts_abort is not None and _sts_abort.is_set()
                        )

                        # Determine exclusive STS outcome type — stored in result so
                        # the display/restore section shows exactly one message.
                        if _user_cancelled:
                            _sts_outcome = "cancelled"
                        elif _batch_aborted_ctx:
                            _sts_outcome = "batch_aborted"
                        else:
                            _sts_outcome = "completed"

                        # Build the compact summary line once; stored in result for
                        # persistent display (not shown here — would flash before rerun).
                        _sts_summary_line = tr(
                            f"STS Score (temporal validation): "
                            f"eligible {_sts_n_elig_int}/{_sts_n_total_int} · "
                            f"completed {_n_completed} · "
                            f"failed {_n_failed} · "
                            f"skipped (not supported) {_sts_n_ns} · "
                            f"skipped (uncertain) {_sts_n_unc} · "
                            f"cancelled remaining {_sts_cancelled_r} · "
                            f"elapsed {_elapsed_ctx}s"
                            + (" · **aborted (endpoint unresponsive)**" if _batch_aborted_ctx else ""),
                            f"STS Score (validação temporal): "
                            f"elegíveis {_sts_n_elig_int}/{_sts_n_total_int} · "
                            f"concluídos {_n_completed} · "
                            f"falhas {_n_failed} · "
                            f"ignorados (não suportados) {_sts_n_ns} · "
                            f"ignorados (incertos) {_sts_n_unc} · "
                            f"cancelados restantes {_sts_cancelled_r} · "
                            f"decorrido {_elapsed_ctx}s"
                            + (" · **lote abortado (endpoint sem resposta)**" if _batch_aborted_ctx else ""),
                        )

                        # Build per-patient failure detail string for storage
                        _fail_details_str = "\n".join(
                            f"- **patient={f.get('patient_id') or '?'}** | "
                            f"status={f.get('status','?')} | reason={f.get('reason','?')}"
                            for f in _tv_fail_log
                        ) if _tv_fail_log else ""

                        # ── Phase 5 from polling path ─────────────────────────────────
                        _tv_n             = _sts_ctx["n"]
                        _tv_events        = _sts_ctx["events"]
                        _tv_rate          = _sts_ctx["rate"]
                        _tv_val_start     = _sts_ctx["val_start"]
                        _tv_val_end       = _sts_ctx["val_end"]
                        _tv_context_sig_r = _sts_ctx.get("ctx_sig", _tv_context_sig)
                        _tv_prepare_info  = _sts_ctx.get("prepare_info", {})

                        _tv_data["class_ia"]  = _tv_data["ia_risk"].map(class_risk)
                        _tv_data["class_euro"] = _tv_data["euroscore_calc"].map(class_risk)
                        _tv_data["class_sts"]  = _tv_data["sts_score"].map(
                            lambda x: class_risk(x) if pd.notna(x) else np.nan
                        )
                        _tv_score_cols = ["ia_risk", "euroscore_calc"]
                        _tv_rename     = {"ia_risk": "AI Risk", "euroscore_calc": "EuroSCORE II"}
                        if _tv_sts_ok:
                            _tv_score_cols.append("sts_score")
                            _tv_rename["sts_score"] = _tv_sts_score_label

                        _tv_perf = evaluate_scores_temporal(
                            _tv_data, "morte_30d", _tv_score_cols, _tv_locked_threshold,
                            n_boot=AppConfig.N_BOOTSTRAP_SAMPLES, seed=AppConfig.BOOTSTRAP_SEED,
                        )
                        if not _tv_perf.empty:
                            _tv_perf["Score"] = _tv_perf["Score"].replace(_tv_rename)

                        _tv_pairs = [("ia_risk", "euroscore_calc")]
                        if _tv_sts_ok:
                            _tv_pairs += [("ia_risk", "sts_score"), ("sts_score", "euroscore_calc")]
                        _tv_pairwise = pairwise_score_comparison(
                            _tv_data, "morte_30d", _tv_pairs,
                            n_boot=AppConfig.N_BOOTSTRAP_SAMPLES, seed=AppConfig.BOOTSTRAP_SEED,
                        )
                        if not _tv_pairwise.empty:
                            for _old, _new in _tv_rename.items():
                                _tv_pairwise["Comparison"] = _tv_pairwise["Comparison"].str.replace(_old, _new)

                        _tv_risk_cat = risk_category_table(_tv_data, "morte_30d", _tv_score_cols)
                        if not _tv_risk_cat.empty:
                            _tv_risk_cat["Score"] = _tv_risk_cat["Score"].replace(_tv_rename)

                        _tv_calib_rows = []
                        for _sc in _tv_score_cols:
                            _sub = _tv_data[["morte_30d", _sc]].dropna()
                            if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                                _cal = calibration_intercept_slope(_sub["morte_30d"].values, _sub[_sc].values)
                                _hl  = hosmer_lemeshow_test(_sub["morte_30d"].values, _sub[_sc].values)
                                _cil = calibration_in_the_large(_sub["morte_30d"].values, _sub[_sc].values)
                                _ici = integrated_calibration_index(_sub["morte_30d"].values, _sub[_sc].values)
                                _tv_calib_rows.append({
                                    "Score": _tv_rename.get(_sc, _sc),
                                    "Calibration_Intercept": _cal["Calibration intercept"],
                                    "Calibration_Slope":     _cal["Calibration slope"],
                                    "CIL": _cil["CIL"],
                                    "ICI": _ici,
                                    "HL_chi2": _hl["HL chi-square"],
                                    "HL_p":    _hl["HL p-value"],
                                })
                        _tv_calib_df = pd.DataFrame(_tv_calib_rows) if _tv_calib_rows else pd.DataFrame()

                        _comp_counts = _tv_data["_completeness"].value_counts()
                        _tv_cohort_summary = {
                            "n_total":   _tv_n,
                            "n_events":  _tv_events,
                            "event_rate": _tv_rate,
                            "date_range": (
                                f"{_tv_val_start} \u2014 {_tv_val_end} "
                                + (tr(
                                    "(de-identified surrogate timeline — not real clinical dates)",
                                    "(linha do tempo substituta desidentificada — não são datas reais)",
                                ) if _tv_is_surrogate else "")
                            ),
                            "n_complete": int(_comp_counts.get("complete", 0)),
                            "n_adequate": int(_comp_counts.get("adequate", 0)),
                            "n_partial":  int(_comp_counts.get("partial", 0)),
                            "n_low":      int(_comp_counts.get("low", 0)),
                        }

                        # ── Extra analytics (STS thread-done path) ───────────────────
                        _tv_thresholds_fixed = [0.02, 0.05, 0.08, 0.10]
                        _tv_youden: dict = {}
                        _tv_thresh_tables: dict = {}
                        for _sc in _tv_score_cols:
                            _sub = _tv_data[["morte_30d", _sc]].dropna()
                            if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                                _tv_youden[_sc] = youden_threshold(
                                    _sub["morte_30d"].values, _sub[_sc].values
                                )
                                _tv_thresh_tables[_sc] = threshold_analysis_table(
                                    _sub["morte_30d"].values,
                                    _sub[_sc].values,
                                    _tv_thresholds_fixed,
                                )

                        _tv_exploratory_recal: dict = {}
                        for _sc in _tv_score_cols:
                            _sub = _tv_data[["morte_30d", _sc]].dropna()
                            if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                                _ery = _sub["morte_30d"].values
                                _erp = _sub[_sc].values
                                _tv_exploratory_recal[_sc] = {}
                                for _mkey, _mfn in (
                                    ("intercept_only", recalibrate_intercept_only),
                                    ("logistic",        recalibrate_logistic),
                                    ("isotonic",        recalibrate_isotonic),
                                ):
                                    try:
                                        _tv_exploratory_recal[_sc][_mkey] = _mfn(_ery, _erp)
                                    except Exception:
                                        pass

                        _tv_exploratory_thresh_tables: dict = {}
                        for _sc in _tv_score_cols:
                            _sub = _tv_data[["morte_30d", _sc]].dropna()
                            if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                                _ethr_set = set(_tv_thresholds_fixed) | {_tv_locked_threshold}
                                if _sc in _tv_youden:
                                    _ethr_set.add(_tv_youden[_sc][0])
                                _tv_exploratory_thresh_tables[_sc] = threshold_analysis_table(
                                    _sub["morte_30d"].values,
                                    _sub[_sc].values,
                                    sorted(_ethr_set),
                                )

                        # ── Common-cohort (STS-available subset) ─────────────────────
                        _tv_common_perf = None
                        _tv_n_common = 0
                        if "sts_score" in _tv_data.columns and len(_tv_score_cols) >= 2:
                            _common_sub = _tv_data[_tv_score_cols + ["morte_30d"]].dropna()
                            _tv_n_common = len(_common_sub)
                            if _tv_n_common >= 10 and _common_sub["morte_30d"].nunique() >= 2:
                                _tv_common_perf = evaluate_scores_temporal(
                                    _common_sub, "morte_30d", _tv_score_cols, _tv_locked_threshold
                                )
                                if _tv_common_perf is not None and not _tv_common_perf.empty:
                                    _tv_common_perf["Score"] = _tv_common_perf["Score"].replace(_tv_rename)

                        st.session_state["_tv_result"] = {
                            "data": _tv_data.copy(),
                            "perf": _tv_perf,
                            "pairwise": _tv_pairwise,
                            "risk_cat": _tv_risk_cat,
                            "calib_df": _tv_calib_df,
                            "cohort_summary": dict(_tv_cohort_summary),
                            "score_cols": list(_tv_score_cols),
                            "rename": dict(_tv_rename),
                            "sts_ok": bool(_tv_sts_ok),
                            "ai_incidents": list(_tv_ai_incidents) if _tv_ai_incidents else [],
                            "exec_log": _tv_exec_log,
                            "sts_status": str(_tv_sts_status),
                            "fail_log": list(_tv_fail_log) if _tv_fail_log else [],
                            "locked_threshold": float(_tv_locked_threshold),
                            "sts_eligibility_log": list(_tv_sts_eligibility_log),
                            "surrogate_timeline": bool(_tv_is_surrogate),
                            "prepare_info": dict(_tv_prepare_info),
                            # ── STS outcome fields (for exclusive message display) ──────
                            "sts_outcome": _sts_outcome,          # "completed"|"cancelled"|"batch_aborted"|"no_sts"|"no_eligible"
                            "sts_summary_line": _sts_summary_line,
                            "sts_n_completed": _n_completed,
                            "sts_n_failed": _n_failed,
                            "sts_n_total_ctx": _sts_n_total_int,
                            "sts_n_cancelled_r": _sts_cancelled_r,
                            "sts_fail_details": _fail_details_str,
                            # ── STS audit fields ──────────────────────────────────────
                            "batch_aborted": bool(_batch_aborted_ctx),
                            "abort_before_query_count": int(_abort_before_query_count),
                            "chunk_log": list(_tv_chunk_log),
                            "sts_patient_audit": list(_tv_sts_audit_rows),
                            "endpoint_health_summary": dict(
                                getattr(calculate_sts_batch, "endpoint_health_summary", {})
                            ),
                            # ── STS availability fields ───────────────────────────────
                            "sts_availability": _tv_sts_availability,
                            "sts_n_score":      _tv_sts_n_score,
                            "sts_n_eligible":   _tv_sts_n_eligible,
                            "sts_n_not_supported": int(_sts_n_ns),
                            "sts_n_uncertain":     int(_sts_n_unc),
                            # ── Common cohort ──────────────────────────────────────────
                            "common_perf": (
                                _tv_common_perf.to_dict(orient="list")
                                if _tv_common_perf is not None else None
                            ),
                            "n_common": int(_tv_n_common),
                            # ── Exploratory analytics ──────────────────────────────────
                            "youden":                  dict(_tv_youden),
                            "exploratory_recal":       dict(_tv_exploratory_recal),
                            "exploratory_thresh_tables": dict(_tv_exploratory_thresh_tables),
                        }
                        st.session_state["_tv_result_sig"]   = _tv_context_sig_r
                        st.session_state[_TV_STS_ST]         = "idle"  # clear running state
                        # No messages here — they would flash before disappearing.
                        # All outcome messages are stored above and shown in the
                        # display/restore section which runs on the next rerun.
                        st.rerun()
                    else:
                        # ── Thread still running — show live progress via fragment ──────
                        # @st.fragment(run_every=1.0) reruns only this component every
                        # second, so the rest of the page is not re-executed on each tick.
                        # When the worker thread signals done, the fragment fires
                        # st.rerun(scope="app") to trigger result collection above.
                        @st.fragment(run_every=1.0)
                        def _tv_sts_polling_fragment():
                            import time as _frag_time
                            _sts_ctx_f   = st.session_state.get(_TV_STS_CTX, {})
                            _sts_done_f  = st.session_state.get(_TV_STS_DONE)
                            _sts_abort_f = st.session_state.get(_TV_STS_ABRT)
                            _sts_prog_f  = st.session_state.get(_TV_STS_PROG, {})
                            _frag_state  = st.session_state.get(_TV_STS_ST, "idle")
                            _sts_t0_f    = _sts_ctx_f.get("t0", _frag_time.monotonic())
                            _batch_elapsed_f   = int(_frag_time.monotonic() - _sts_t0_f)
                            _sts_n_elig_f      = _sts_ctx_f.get("n_eligible", 0) or 0
                            _sts_n_ns_f        = _sts_ctx_f.get("n_not_supported", 0) or 0
                            _sts_n_unc_f       = _sts_ctx_f.get("n_uncertain", 0) or 0

                            _pat_start_f   = _sts_prog_f.get("patient_start_ts")
                            _pat_n_f       = _sts_prog_f.get("patient_n", 0) or 0
                            _total_pend_f  = _sts_prog_f.get("total_pending", 0) or 0
                            _pat_id_f      = _sts_prog_f.get("patient_id", "") or ""
                            _net_comp_f    = _sts_prog_f.get("net_completed", 0) or 0
                            _net_fail_f    = _sts_prog_f.get("net_failed", 0) or 0
                            _pat_elapsed_f = (
                                int(_frag_time.monotonic() - _pat_start_f)
                                if _pat_start_f is not None else None
                            )

                            _cache_hits_f  = max(0, _sts_n_elig_f - _total_pend_f) if _total_pend_f > 0 else 0
                            # finalized = every patient with a definitive outcome
                            #             (cache hit  OR  network success  OR  network failure)
                            _finalized_f   = _cache_hits_f + _net_comp_f + _net_fail_f
                            # in_flight = 1 while chunk_start has fired but chunk_done has not yet
                            # Formula: dispatched_so_far - finalized_so_far
                            #   dispatched = cache_hits + (pat_n + 1)   [pat_n is 0-based index]
                            #   in window between chunk_done and next chunk_start this equals 0
                            _in_flight_f   = (
                                max(0, (_cache_hits_f + _pat_n_f + 1) - _finalized_f)
                                if _total_pend_f > 0 else 0
                            )
                            # remaining = patients not yet started (neither finalized nor in-flight)
                            _remaining_f   = max(0, _sts_n_elig_f - _finalized_f - _in_flight_f)
                            _skipped_f     = _sts_n_ns_f + _sts_n_unc_f
                            # cur_num = 1-based position of the patient currently being processed
                            _cur_num_f     = _finalized_f + _in_flight_f if _total_pend_f > 0 else "?"

                            # Progress bar — numerator = finalized (not completed-only)
                            if _sts_n_elig_f > 0 and _total_pend_f > 0:
                                _prog_frac_f = min(1.0, _finalized_f / _sts_n_elig_f)
                                _prog_text_f = tr(
                                    f"STS Score — {_finalized_f} of {_sts_n_elig_f} finalized ({_prog_frac_f:.1%})",
                                    f"STS Score — {_finalized_f} de {_sts_n_elig_f} finalizados ({_prog_frac_f:.1%})",
                                )
                            elif _sts_n_elig_f > 0 and _total_pend_f == 0 and _pat_start_f is None:
                                _prog_frac_f = 0.10
                                _prog_text_f = tr(
                                    f"STS Score — checking cache ({_sts_n_elig_f} eligible patients)…",
                                    f"STS Score — verificando cache ({_sts_n_elig_f} pacientes elegíveis)…",
                                )
                            else:
                                _prog_frac_f = 0.05
                                _prog_text_f = tr("STS Score — starting…", "STS Score — iniciando…")
                            st.progress(_prog_frac_f, text=_prog_text_f)

                            # Cancel button
                            _can_col_f, _inf_col_f = st.columns([2, 5])
                            _cancel_clicked_f = _can_col_f.button(
                                tr("Cancel STS run", "Cancelar consulta STS"),
                                key="tv_cancel_sts_btn",
                                type="secondary",
                                help=tr(
                                    f"Stop after the current patient finishes or reaches the "
                                    f"{STS_PER_PATIENT_TIMEOUT_S} s per-patient limit "
                                    f"(total across all retry attempts). "
                                    "Partial results and all AI Risk / EuroSCORE II results are preserved.",
                                    f"Interromper após o paciente atual concluir ou atingir o limite "
                                    f"de {STS_PER_PATIENT_TIMEOUT_S} s por paciente "
                                    f"(total incluindo todas as tentativas). "
                                    "Resultados parciais e todos os resultados de AI Risk / EuroSCORE II são preservados.",
                                ),
                                disabled=(_frag_state == "cancelling"),
                            )
                            if _cancel_clicked_f and _sts_abort_f is not None:
                                _sts_abort_f.set()
                                st.session_state[_TV_STS_ST] = "cancelling"
                                _frag_state = "cancelling"

                            # Status line
                            _batch_str_f = tr(
                                f"batch: {_batch_elapsed_f} s",
                                f"lote: {_batch_elapsed_f} s",
                            )
                            if _pat_elapsed_f is not None and _total_pend_f > 0:
                                # Show wall-clock time and the per-patient limit as two separate
                                # values (· not /) so the display is honest when setup/teardown
                                # overhead makes the elapsed counter exceed the asyncio timeout.
                                _time_disp_f = tr(
                                    f"current query: {_pat_elapsed_f} s · limit: {STS_PER_PATIENT_TIMEOUT_S} s · {_batch_str_f}",
                                    f"consulta atual: {_pat_elapsed_f} s · limite: {STS_PER_PATIENT_TIMEOUT_S} s · {_batch_str_f}",
                                )
                                _pat_num_f = tr(
                                    f"patient {_cur_num_f}/{_sts_n_elig_f}"
                                    + (f" · id: {_pat_id_f}" if _pat_id_f else ""),
                                    f"paciente {_cur_num_f}/{_sts_n_elig_f}"
                                    + (f" · id: {_pat_id_f}" if _pat_id_f else ""),
                                )
                            else:
                                _time_disp_f = _batch_str_f
                                _pat_num_f = _sts_prog_f.get("detail", tr("initialising…", "iniciando…"))

                            _state_lbl_f = (
                                tr("Cancelling", "Cancelando")
                                if _frag_state == "cancelling"
                                else tr("Running", "Em execução")
                            )
                            # All counters close: finalized + in_flight + remaining = eligible_total
                            _inf_col_f.caption(tr(
                                f"STS — {_state_lbl_f}: "
                                f"{_pat_num_f} · "
                                f"finalized {_finalized_f} · completed {_cache_hits_f + _net_comp_f} · "
                                f"failed {_net_fail_f} · in-flight {_in_flight_f} · "
                                f"remaining {_remaining_f} · skipped {_skipped_f} · "
                                f"{_time_disp_f} · "
                                f"phase: {_sts_prog_f.get('phase', '…')}",
                                f"STS — {_state_lbl_f}: "
                                f"{_pat_num_f} · "
                                f"finalizados {_finalized_f} · concluídos {_cache_hits_f + _net_comp_f} · "
                                f"falhas {_net_fail_f} · em consulta {_in_flight_f} · "
                                f"restantes {_remaining_f} · ignorados {_skipped_f} · "
                                f"{_time_disp_f} · "
                                f"fase: {_sts_prog_f.get('phase', '…')}",
                            ))

                            # Cancellation warning
                            if _frag_state == "cancelling":
                                st.warning(tr(
                                    f"Cancellation requested — waiting for the current in-flight "
                                    f"STS query to finish or reach the {STS_PER_PATIENT_TIMEOUT_S} s "
                                    f"per-patient limit (total across all retry attempts). "
                                    f"No new queries will start. Partial results will be preserved. "
                                    f"· {_pat_num_f} · {_time_disp_f}",
                                    f"Cancelamento solicitado — aguardando a consulta STS atual "
                                    f"terminar ou atingir o limite de {STS_PER_PATIENT_TIMEOUT_S} s "
                                    f"por paciente (total incluindo todas as tentativas). "
                                    f"Nenhuma nova consulta será iniciada. Resultados parciais preservados. "
                                    f"· {_pat_num_f} · {_time_disp_f}",
                                ))

                            # When done, trigger a full-page rerun so the main code
                            # can collect and store results (above, _thread_finished branch).
                            if _sts_done_f is not None and _sts_done_f.is_set():
                                st.rerun(scope="app")

                        _tv_sts_polling_fragment()

                if _tv_run:
                    # ── Exploratory analytics — initialised here so the thread-done ──────
                    # save block (which runs before the computation block) always has
                    # a defined value to store even if computation is skipped/fails.
                    _tv_youden: dict = {}
                    _tv_thresh_tables: dict = {}
                    _tv_thresholds_fixed: list = [0.02, 0.05, 0.08, 0.10]
                    _tv_exploratory_recal: dict = {}
                    _tv_exploratory_thresh_tables: dict = {}

                    # ── Reset residual STS thread state from any previous run ──────────
                    # A fresh run must start with a clean slate; stale events/dicts from
                    # an earlier (possibly cancelled) run must not bleed into this one.
                    for _k in (
                        "_tv_sts_state", "_tv_sts_ctx", "_tv_sts_ctx_sig",
                        "_tv_sts_done_evt", "_tv_sts_abort_evt",
                        "_tv_sts_results", "_tv_sts_prog",
                    ):
                        st.session_state.pop(_k, None)

                    _tv_phase_slot = st.empty()
                    _tv_progress = st.progress(0, text=tr("Preparing...", "Preparando..."))
                    _update_phase(_tv_phase_slot, 1, 5, tr(
                        "loading cohort",
                        "carregando coorte",
                    ))

                    # ── 5.1 Apply frozen AI Risk model ──
                    # Phase: temporal-validation hardening.  The per-row
                    # inference loop now lives in a single reusable helper
                    # (``apply_frozen_model_to_temporal_cohort``).  Behaviour
                    # is identical — the helper still routes inputs through
                    # ``_build_input_row`` + ``_align_input_to_training_schema``
                    # + ``clean_features`` and applies the frozen pipeline as
                    # saved.  The residual ad hoc numeric coercion that used
                    # to live inline (``str.replace(',', '.')`` per column)
                    # was removed because ``_align_input_to_training_schema``
                    # already performs that exact normalization on the
                    # training-schema reference frame.
                    _update_phase(_tv_phase_slot, 2, 5, tr(
                        "applying AI Risk model",
                        "aplicando modelo AI Risk",
                    ))
                    _tv_progress.progress(0.05, text=tr("Applying frozen AI Risk model...", "Aplicando modelo AI Risk congelado..."))

                    _tv_ref_df = prepared.data[prepared.feature_columns]

                    def _tv_progress_cb(_done, _total):
                        # Throttle updates to ~5% increments to avoid
                        # hammering the Streamlit progress widget.
                        if _total <= 0:
                            return
                        _step = max(1, _total // 20)
                        if _done % _step == 0 or _done == _total:
                            _tv_progress.progress(
                                0.05 + 0.35 * _done / _total,
                                text=tr(
                                    f"AI Risk: {_done}/{_total}",
                                    f"AI Risk: {_done}/{_total}",
                                ),
                            )

                    _tv_inference = apply_frozen_model_to_temporal_cohort(
                        model_pipeline=artifacts.fitted_models[forced_model],
                        feature_columns=artifacts.feature_columns,
                        reference_df=_tv_ref_df,
                        temporal_data=_tv_data,
                        language=language,
                        progress_callback=_tv_progress_cb,
                    )

                    _tv_data["ia_risk"] = _tv_inference["probabilities"]
                    _tv_data["_completeness"] = _tv_inference["completeness"]
                    _tv_ai_incidents = _tv_inference["incidents"]

                    # Phase: AI Risk incident transparency.  Mirror the STS
                    # Score incident UI so per-patient inference failures
                    # are not silently reduced to NaN.
                    if _tv_ai_incidents:
                        st.warning(tr(
                            f"AI Risk inference incidents (temporal validation): {len(_tv_ai_incidents)} patient(s) failed.",
                            f"Incidentes de inferência do AI Risk (validação temporal): {len(_tv_ai_incidents)} paciente(s) falharam.",
                        ))
                        with st.expander(
                            tr(
                                f"AI Risk per-patient incidents (temporal validation) ({len(_tv_ai_incidents)})",
                                f"Incidentes por paciente do AI Risk (validação temporal) ({len(_tv_ai_incidents)})",
                            ),
                            expanded=False,
                        ):
                            st.dataframe(
                                pd.DataFrame(_tv_ai_incidents),
                                width="stretch",
                                hide_index=True,
                            )

                    # ── 5.2 EuroSCORE II ──
                    _update_phase(_tv_phase_slot, 3, 5, tr(
                        "computing EuroSCORE II",
                        "calculando EuroSCORE II",
                    ))
                    _tv_progress.progress(0.42, text=tr("Computing EuroSCORE II...", "Calculando EuroSCORE II..."))
                    _tv_data["euroscore_calc"] = _tv_data.apply(euroscore_from_row, axis=1)

                    # ── 5.3 STS Score — pre-classify eligibility ──────────────────
                    _tv_data["sts_score"] = np.nan
                    _tv_sts_ok = False
                    _tv_exec_log: list = []
                    _tv_sts_status: str = ""
                    _tv_fail_log: list = []
                    _tv_sts_eligibility_log = []

                    if HAS_STS and _tv_include_sts:
                        import threading as _tv_threading
                        import time as _tv_time_mod

                        # Eligibility pre-classification (synchronous — fast)
                        _tv_all_rows = _tv_data.to_dict(orient="records")
                        _tv_all_pids = _sts_score_patient_ids(_tv_all_rows)

                        _tv_sts_eligible_idx:  list = []
                        _tv_sts_eligible_rows: list = []
                        _tv_sts_eligible_pids: list = []
                        _tv_n_not_supported = 0
                        _tv_n_uncertain     = 0

                        for _eidx, (_erow, _epid) in enumerate(zip(_tv_all_rows, _tv_all_pids)):
                            _estatus, _ereason = classify_sts_eligibility(_erow)
                            _tv_sts_eligibility_log.append({
                                "row_index":   _eidx,
                                "patient_id":  _epid,
                                "eligibility": _estatus,
                                "reason":      _ereason,
                            })
                            if _estatus == "supported":
                                _tv_sts_eligible_idx.append(_eidx)
                                _tv_sts_eligible_rows.append(_erow)
                                _tv_sts_eligible_pids.append(_epid)
                            elif _estatus == "not_supported":
                                _tv_n_not_supported += 1
                            else:  # uncertain
                                _tv_n_uncertain += 1

                        _tv_n_eligible = len(_tv_sts_eligible_rows)
                        _tv_n_total    = len(_tv_all_rows)

                        # Eligibility summary (always visible before querying)
                        st.caption(tr(
                            f"STS eligibility: "
                            f"{_tv_n_eligible} supported · "
                            f"{_tv_n_not_supported} not supported (skipped) · "
                            f"{_tv_n_uncertain} uncertain — OBSERVATION ADMIT or unmapped priority (skipped)",
                            f"Elegibilidade STS: "
                            f"{_tv_n_eligible} suportados · "
                            f"{_tv_n_not_supported} não suportados (ignorados) · "
                            f"{_tv_n_uncertain} incertos — OBSERVATION ADMIT ou prioridade não mapeável (ignorados)",
                        ))
                        if _tv_n_not_supported > 0 or _tv_n_uncertain > 0:
                            with st.expander(tr(
                                f"STS eligibility details "
                                f"({_tv_n_not_supported} not supported, "
                                f"{_tv_n_uncertain} uncertain — both skipped)",
                                f"Detalhes de elegibilidade STS "
                                f"({_tv_n_not_supported} não suportados, "
                                f"{_tv_n_uncertain} incertos — ambos ignorados)",
                            ), expanded=False):
                                _non_sup = [e for e in _tv_sts_eligibility_log
                                            if e["eligibility"] != "supported"]
                                st.dataframe(pd.DataFrame(_non_sup),
                                             width="stretch", hide_index=True)

                        # ── Eligibility export ──────────────────────────────────────────
                        # Available immediately after eligibility is classified — before
                        # the STS batch starts, while it runs, after it completes/cancels.
                        if _tv_sts_eligibility_log:
                            _elig_df_export = pd.DataFrame(_tv_sts_eligibility_log)
                            _elig_dl_c1, _elig_dl_c2 = st.columns(2)
                            with _elig_dl_c1:
                                st.download_button(
                                    label=tr(
                                        "Download STS eligibility (CSV)",
                                        "Baixar elegibilidade STS (CSV)",
                                    ),
                                    data=_elig_df_export.to_csv(index=False).encode("utf-8"),
                                    file_name="sts_eligibility.csv",
                                    mime="text/csv",
                                    key="tv_elig_dl_csv_run",
                                )
                            with _elig_dl_c2:
                                _elig_xlsx_buf = BytesIO()
                                with pd.ExcelWriter(_elig_xlsx_buf, engine="openpyxl") as _ew:
                                    _elig_df_export.to_excel(_ew, sheet_name="eligibility", index=False)
                                st.download_button(
                                    label=tr(
                                        "Download STS eligibility (XLSX)",
                                        "Baixar elegibilidade STS (XLSX)",
                                    ),
                                    data=_elig_xlsx_buf.getvalue(),
                                    file_name="sts_eligibility.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="tv_elig_dl_xlsx_run",
                                )

                        if _tv_n_eligible == 0:
                            _tv_sts_availability = "no_eligible"
                            _tv_sts_n_eligible = 0
                            _tv_sts_n_score = 0
                            st.info(tr(
                                "No patients have STS-supported procedures in this cohort. "
                                "Analysis will proceed with AI Risk and EuroSCORE II only.",
                                "Nenhum paciente possui procedimento suportado pelo STS. "
                                "A análise prosseguirá apenas com AI Risk e EuroSCORE II.",
                            ))
                        else:
                            # ── Start STS worker thread ────────────────────────────────
                            # The batch is long-running; we hand it off to a daemon
                            # thread and immediately rerun so the UI remains responsive
                            # and a cancel button can be shown.
                            _tv_abort_evt  = _tv_threading.Event()
                            _tv_done_evt   = _tv_threading.Event()
                            _tv_res_store  = [None] * _tv_n_eligible
                            _tv_prog_dict  = {
                                "phase": "starting",
                                "detail": "",
                                # Per-patient timing (set by chunk_start_callback)
                                "patient_start_ts": None,  # monotonic timestamp
                                "patient_n": 0,            # 0-based pending index
                                "total_pending": 0,        # non-cache patients
                                "patient_id": "",
                                # Running success/failure counters (set by chunk_done_callback)
                                "net_completed": 0,        # network queries → valid result
                                "net_failed": 0,           # network queries → no result
                            }

                            # Phase callback: ONLY updates the shared dict — no
                            # Streamlit calls from the worker thread.
                            def _make_sts_phase_cb(prog_d):
                                def _cb(phase_num, phase_total, label, detail=""):
                                    prog_d["phase"]  = f"{phase_num}/{phase_total}: {label}"
                                    prog_d["detail"] = detail
                                return _cb

                            _sts_phase_cb = _make_sts_phase_cb(_tv_prog_dict)

                            # chunk_start_callback — fires just BEFORE each patient's
                            # network query.  With chunk_size=1 this is once per patient.
                            # Records the per-patient start timestamp so the polling
                            # handler can display "current: Xs / 90s" separately from
                            # the batch elapsed.
                            def _make_sts_chunk_start_cb(prog_d):
                                import time as _t
                                def _cb(patient_idx, total_pending, patient_id=None):
                                    prog_d["patient_start_ts"] = _t.monotonic()
                                    prog_d["patient_n"]        = patient_idx
                                    prog_d["total_pending"]    = total_pending
                                    prog_d["patient_id"]       = patient_id or ""
                                return _cb

                            _sts_chunk_cb = _make_sts_chunk_start_cb(_tv_prog_dict)

                            # chunk_done_callback — fires just AFTER each patient's
                            # query completes (success or failure).  Increments the
                            # running completed/failed counters in the shared dict so
                            # the polling handler can display them live.
                            def _make_sts_chunk_done_cb(prog_d):
                                def _cb(patient_idx, total_pending, success):
                                    if success:
                                        prog_d["net_completed"] = prog_d.get("net_completed", 0) + 1
                                    else:
                                        prog_d["net_failed"] = prog_d.get("net_failed", 0) + 1
                                return _cb

                            _sts_done_cb = _make_sts_chunk_done_cb(_tv_prog_dict)

                            # Worker function — runs in background thread.
                            def _make_sts_worker(eligible_rows, eligible_pids, res_store,
                                                  phase_cb, chunk_start_cb, chunk_done_cb,
                                                  abort_e, done_e):
                                def _worker():
                                    try:
                                        results = calculate_sts_batch(
                                            eligible_rows,
                                            patient_ids=eligible_pids,
                                            phase_callback=phase_cb,
                                            chunk_start_callback=chunk_start_cb,
                                            chunk_done_callback=chunk_done_cb,
                                            abort_event=abort_e,
                                            # chunk_size=1: abort checked after every patient;
                                            # combined with the per-patient global timeout in
                                            # _calculate_sts_chunk_async, cancellation is
                                            # strictly bounded by STS_PER_PATIENT_TIMEOUT_S.
                                            chunk_size=1,
                                        )
                                        for _wi, _wr in enumerate(results):
                                            res_store[_wi] = _wr
                                    except Exception:
                                        pass
                                    finally:
                                        done_e.set()
                                return _worker

                            _sts_thread = _tv_threading.Thread(
                                target=_make_sts_worker(
                                    _tv_sts_eligible_rows,
                                    _tv_sts_eligible_pids,
                                    _tv_res_store,
                                    _sts_phase_cb,
                                    _sts_chunk_cb,
                                    _sts_done_cb,
                                    _tv_abort_evt,
                                    _tv_done_evt,
                                ),
                                daemon=True,
                            )

                            # Persist everything the polling handler will need.
                            st.session_state[_TV_STS_CTX] = {
                                "ctx_sig":       _tv_context_sig,
                                "data":          _tv_data.copy(),
                                "eligible_idx":  list(_tv_sts_eligible_idx),
                                "n_eligible":    _tv_n_eligible,
                                "n_not_supported": _tv_n_not_supported,
                                "n_uncertain":   _tv_n_uncertain,
                                "n_total":       _tv_n_total,
                                "eligibility_log": list(_tv_sts_eligibility_log),
                                "n":             _tv_n,
                                "events":        _tv_events,
                                "rate":          _tv_rate,
                                "val_start":     _tv_val_start,
                                "val_end":       _tv_val_end,
                                "is_surrogate":  _tv_is_surrogate,
                                "locked_threshold": _tv_locked_threshold,
                                "ai_incidents":  list(_tv_ai_incidents) if _tv_ai_incidents else [],
                                "prepare_info":  dict(_tv_prepared.info) if _tv_prepared is not None else {},
                                "t0":            _tv_time_mod.monotonic(),
                            }
                            st.session_state[_TV_STS_CSIG]  = _tv_context_sig
                            st.session_state[_TV_STS_DONE]  = _tv_done_evt
                            st.session_state[_TV_STS_ABRT]  = _tv_abort_evt
                            st.session_state[_TV_STS_RES]   = _tv_res_store
                            st.session_state[_TV_STS_PROG]  = _tv_prog_dict
                            st.session_state[_TV_STS_ST]    = "running"

                            _sts_thread.start()

                            _tv_progress.progress(
                                0.50,
                                text=tr(
                                    f"STS Score started ({_tv_n_eligible} eligible patients queued)…",
                                    f"STS Score iniciado ({_tv_n_eligible} pacientes elegíveis na fila)…",
                                ),
                            )
                            _tv_phase_slot.empty()
                            st.rerun()  # Hand control to the polling handler

                    if not (HAS_STS and _tv_include_sts) or _tv_n_eligible == 0:
                        # STS disabled or no eligible patients — proceed to Phase 5 inline.
                        # Only show "STS not available" when STS is actually disabled;
                        # the _tv_n_eligible == 0 case already showed its own message above.
                        if not (HAS_STS and _tv_include_sts):
                            st.info(tr(
                                "STS scores are not available. "
                                "Analysis will proceed with AI Risk and EuroSCORE II only.",
                                "Escores STS não disponíveis. "
                                "A análise prosseguirá apenas com AI Risk e EuroSCORE II.",
                            ))

                    # ── 5.4 Risk classes ──
                    _update_phase(_tv_phase_slot, 5, 5, tr(
                        "computing metrics and consolidating",
                        "calculando métricas e consolidando",
                    ))
                    _tv_progress.progress(0.80, text=tr("Computing metrics...", "Calculando métricas..."))
                    _tv_data["class_ia"] = _tv_data["ia_risk"].map(class_risk)
                    _tv_data["class_euro"] = _tv_data["euroscore_calc"].map(class_risk)
                    _tv_data["class_sts"] = _tv_data["sts_score"].map(lambda x: class_risk(x) if pd.notna(x) else np.nan)

                    # Score columns and rename map
                    _tv_score_cols = ["ia_risk", "euroscore_calc"]
                    _tv_rename = {"ia_risk": "AI Risk", "euroscore_calc": "EuroSCORE II"}
                    if _tv_sts_ok:
                        _tv_score_cols.append("sts_score")
                        # Phase: nomenclature consistency — always use the
                        # full term "STS Score" in the temporal-validation UI.
                        _tv_rename["sts_score"] = "STS Score"

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
                            _cil = calibration_in_the_large(_sub["morte_30d"].values, _sub[_sc].values)
                            _ici = integrated_calibration_index(_sub["morte_30d"].values, _sub[_sc].values)
                            _tv_calib_rows.append({
                                "Score": _tv_rename.get(_sc, _sc),
                                "Calibration_Intercept": _cal["Calibration intercept"],
                                "Calibration_Slope": _cal["Calibration slope"],
                                "CIL": _cil["CIL"],
                                "ICI": _ici,
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
                        "date_range": (
                            f"{_tv_val_start} \u2014 {_tv_val_end} "
                            + (tr(
                                "(de-identified surrogate timeline — not real clinical dates)",
                                "(linha do tempo substituta desidentificada — não são datas reais)",
                            ) if _tv_is_surrogate else "")
                        ),
                        "n_complete": int(_comp_counts.get("complete", 0)),
                        "n_adequate": int(_comp_counts.get("adequate", 0)),
                        "n_partial": int(_comp_counts.get("partial", 0)),
                        "n_low": int(_comp_counts.get("low", 0)),
                    }

                    # ── Extra analytics for interactive graphs ─────────────────────────
                    for _sc in _tv_score_cols:
                        _sub = _tv_data[["morte_30d", _sc]].dropna()
                        if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                            _tv_youden[_sc] = youden_threshold(
                                _sub["morte_30d"].values, _sub[_sc].values
                            )
                            _tv_thresh_tables[_sc] = threshold_analysis_table(
                                _sub["morte_30d"].values,
                                _sub[_sc].values,
                                _tv_thresholds_fixed,
                            )

                    # ── Exploratory: all recalibration methods for each score ──────────
                    _tv_exploratory_recal = {}
                    for _sc in _tv_score_cols:
                        _sub = _tv_data[["morte_30d", _sc]].dropna()
                        if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                            _ery = _sub["morte_30d"].values
                            _erp = _sub[_sc].values
                            _tv_exploratory_recal[_sc] = {}
                            for _mkey, _mfn in (
                                ("intercept_only", recalibrate_intercept_only),
                                ("logistic",        recalibrate_logistic),
                                ("isotonic",        recalibrate_isotonic),
                            ):
                                try:
                                    _tv_exploratory_recal[_sc][_mkey] = _mfn(_ery, _erp)
                                except Exception:
                                    pass

                    # ── Exploratory: per-score threshold tables (locked + Youden + fixed) ──
                    _tv_exploratory_thresh_tables = {}
                    for _sc in _tv_score_cols:
                        _sub = _tv_data[["morte_30d", _sc]].dropna()
                        if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                            _ethr_set = set(_tv_thresholds_fixed) | {_tv_locked_threshold}
                            if _sc in _tv_youden:
                                _ethr_set.add(_tv_youden[_sc][0])
                            _tv_exploratory_thresh_tables[_sc] = threshold_analysis_table(
                                _sub["morte_30d"].values,
                                _sub[_sc].values,
                                sorted(_ethr_set),
                            )

                    # ── Common-cohort (STS-available subset) comparison ────────────────
                    _tv_common_perf = None
                    _tv_n_common = 0
                    if "sts_score" in _tv_data.columns and len(_tv_score_cols) >= 2:
                        _common_cols = ["morte_30d"] + _tv_score_cols
                        _common_sub = _tv_data[_common_cols].dropna()
                        _tv_n_common = len(_common_sub)
                        if _tv_n_common >= 10 and _common_sub["morte_30d"].nunique() >= 2:
                            _tv_common_perf = evaluate_scores_temporal(
                                _common_sub, "morte_30d", _tv_score_cols, _tv_locked_threshold
                            )
                            if not _tv_common_perf.empty:
                                _tv_common_perf["Score"] = _tv_common_perf["Score"].replace(_tv_rename)

                    _tv_prepare_info = dict(_tv_prepared.info) if _tv_prepared is not None else {}
                    _tv_progress.progress(1.0, text=tr("Done.", "Concluído."))
                    _tv_phase_slot.empty()
                    with st.expander(tr("View execution details", "Ver detalhes de execução"), expanded=False):
                        _sts_elig_n  = len([e for e in _tv_sts_eligibility_log if e["eligibility"] == "supported"])
                        _sts_ns_n    = len([e for e in _tv_sts_eligibility_log if e["eligibility"] == "not_supported"])
                        _sts_unc_n   = len([e for e in _tv_sts_eligibility_log if e["eligibility"] == "uncertain"])
                        _sts_skip_n  = _sts_ns_n + _sts_unc_n

                        # ── Pipeline provenance ───────────────────────────────────────────
                        st.markdown(tr("**Pipeline provenance**", "**Proveniência do pipeline**"))
                        st.caption(tr(
                            "Loading: `prepare_master_dataset` (risk_data.py)  \n"
                            "Surgery class: `classify_sts_eligibility` (sts_calculator.py)  \n"
                            "STS inputs: `build_sts_input_from_row` (sts_calculator.py)  \n"
                            "AI Risk: `apply_frozen_model_to_temporal_cohort` (ai_risk_inference.py)  \n"
                            "EuroSCORE II: `euroscore_from_row` (euroscore.py)",
                            "Carregamento: `prepare_master_dataset` (risk_data.py)  \n"
                            "Classe cirúrgica: `classify_sts_eligibility` (sts_calculator.py)  \n"
                            "Inputs STS: `build_sts_input_from_row` (sts_calculator.py)  \n"
                            "AI Risk: `apply_frozen_model_to_temporal_cohort` (ai_risk_inference.py)  \n"
                            "EuroSCORE II: `euroscore_from_row` (euroscore.py)",
                        ))

                        # ── Cohort assembly breakdown ─────────────────────────────────────
                        st.markdown(tr("**Cohort assembly**", "**Montagem da coorte**"))
                        _pi = _tv_prepare_info
                        if _pi.get("source_type") == "flat":
                            st.caption(tr(
                                f"Source: flat CSV/Parquet — no merge steps.  \n"
                                f"Rows loaded: **{_pi.get('n_rows', len(_tv_data))}**  \n"
                                f"_(Intermediate merge/filter counts are only available for XLSX uploads.)_",
                                f"Fonte: CSV/Parquet plano — sem etapas de mesclagem.  \n"
                                f"Linhas carregadas: **{_pi.get('n_rows', len(_tv_data))}**  \n"
                                f"_(Contagens intermediárias só estão disponíveis para uploads XLSX.)_",
                            ))
                        elif _pi.get("pre_rows_before_criteria", 0) > 0:
                            _pi_excl_crit = _pi.get("excluded_missing_surgery_or_date", 0)
                            _pi_excl_match = _pi.get("excluded_no_pre_post_match", 0)
                            st.caption(tr(
                                f"Preoperative rows (raw): **{_pi.get('pre_rows_before_criteria')}**  \n"
                                f"After surgery/date filter: **{_pi.get('pre_rows_after_criteria')}** "
                                f"(excluded: {_pi_excl_crit})  \n"
                                f"Unique pre-op patient–date pairs: **{_pi.get('pre_unique_patient_date_after_criteria')}**  \n"
                                f"Post-op unique patient–date pairs: **{_pi.get('post_unique_patient_date')}**  \n"
                                f"After pre–post inner join: **{_pi.get('matched_pre_post_rows')}** "
                                f"(unmatched: {_pi_excl_match})  \n"
                                f"Echocardiogram rows joined: **{_pi.get('echo_rows', '?')}**  \n"
                                f"Final cohort (after normalization): **{len(_tv_data)}** patients  \n"
                                f"Outcome encoding: `Death` → `morte_30d` via `map_death_30d`  \n"
                                f"⚠ Note: `Death = \"0\"` is interpreted as operative death on day 0 "
                                f"(not as boolean false). See canonical semantics note below.",
                                f"Linhas pré-operatórias (brutas): **{_pi.get('pre_rows_before_criteria')}**  \n"
                                f"Após filtro cirurgia/data: **{_pi.get('pre_rows_after_criteria')}** "
                                f"(excluídos: {_pi_excl_crit})  \n"
                                f"Pares paciente–data pré-op únicos: **{_pi.get('pre_unique_patient_date_after_criteria')}**  \n"
                                f"Pares paciente–data pós-op únicos: **{_pi.get('post_unique_patient_date')}**  \n"
                                f"Após junção interna pré–pós: **{_pi.get('matched_pre_post_rows')}** "
                                f"(sem correspondência: {_pi_excl_match})  \n"
                                f"Linhas de ecocardiograma: **{_pi.get('echo_rows', '?')}**  \n"
                                f"Coorte final (após normalização): **{len(_tv_data)}** pacientes  \n"
                                f"Codificação do desfecho: `Death` → `morte_30d` via `map_death_30d`  \n"
                                f"⚠ Nota: `Death = \"0\"` é interpretado como morte operatória no dia 0 "
                                f"(não como falso booleano). Ver nota de semântica canônica abaixo.",
                            ))
                        else:
                            st.caption(tr(
                                f"Final cohort: **{len(_tv_data)}** patients  \n"
                                f"_(Detailed intermediate counts not available for this source format.)_",
                                f"Coorte final: **{len(_tv_data)}** pacientes  \n"
                                f"_(Contagens intermediárias detalhadas não disponíveis para este formato.)_",
                            ))

                        # ── Per-step patient counts ───────────────────────────────────────
                        st.markdown(tr("**Per-step patient counts**", "**Contagem de pacientes por etapa**"))
                        st.caption(tr(
                            f"After `prepare_master_dataset`: **{len(_tv_data)}** patients "
                            f"({_tv_events} events, event rate {_tv_rate:.1%})  \n"
                            f"STS eligibility (`classify_sts_eligibility`):  \n"
                            f"  · supported: **{_sts_elig_n}** (eligible for query)  \n"
                            f"  · not_supported: **{_sts_ns_n}** (aorta/Bentall/dissection — skipped)  \n"
                            f"  · uncertain: **{_sts_unc_n}** (OBSERVATION ADMIT or unmapped — skipped)  \n"
                            f"  · total classified: **{len(_tv_sts_eligibility_log)}**  \n"
                            f"AI Risk inference incidents: {len(_tv_ai_incidents) if _tv_ai_incidents else 0}  \n"
                            f"{_sts_availability_details_caption(_tv_sts_availability, _tv_sts_n_score, _tv_sts_n_eligible, 'English')}  \n"
                            f"Temporal axis: {'de-identified surrogate' if _tv_is_surrogate else 'standard'}",
                            f"Após `prepare_master_dataset`: **{len(_tv_data)}** pacientes "
                            f"({_tv_events} eventos, taxa {_tv_rate:.1%})  \n"
                            f"Elegibilidade STS (`classify_sts_eligibility`):  \n"
                            f"  · suportados: **{_sts_elig_n}** (elegíveis para consulta)  \n"
                            f"  · não suportados: **{_sts_ns_n}** (aorta/Bentall/dissecção — ignorados)  \n"
                            f"  · incertos: **{_sts_unc_n}** (OBSERVATION ADMIT ou sem mapeamento — ignorados)  \n"
                            f"  · total classificados: **{len(_tv_sts_eligibility_log)}**  \n"
                            f"Incidentes de AI Risk: {len(_tv_ai_incidents) if _tv_ai_incidents else 0}  \n"
                            f"{_sts_availability_details_caption(_tv_sts_availability, _tv_sts_n_score, _tv_sts_n_eligible, 'Portuguese')}  \n"
                            f"Eixo temporal: {'substituto desidentificado' if _tv_is_surrogate else 'padrão'}",
                        ))

                        # ── Outcome semantics note ────────────────────────────────────────
                        st.markdown(tr("**Outcome encoding semantics**", "**Semântica da codificação do desfecho**"))
                        st.caption(tr(
                            "Note: `Death = \"0\"` is interpreted canonically as **operative death on day 0**, "
                            "not as boolean false (survivor). This is handled by `map_death_30d` / "
                            "`parse_postop_timing` in `risk_data.py`. Day 0 = day of surgery → event = 1.  \n"
                            "Survivor tokens: `\"No\"`, `\"Não\"`, `\"Nao\"`, `\"-\"`, `\"--\"` → 0.  \n"
                            "Event tokens: `\"Yes\"`, `\"Sim\"`, `\"Death\"`, `\"1\"`, `\"0\"` (day 0) → 1.",
                            "Nota: `Death = \"0\"` é interpretado canonicamente como **morte operatória no dia 0**, "
                            "não como falso booleano (sobrevivente). Isso é tratado por `map_death_30d` / "
                            "`parse_postop_timing` em `risk_data.py`. Dia 0 = dia da cirurgia → evento = 1.  \n"
                            "Tokens de sobrevivente: `\"No\"`, `\"Não\"`, `\"Nao\"`, `\"-\"`, `\"--\"` → 0.  \n"
                            "Tokens de evento: `\"Yes\"`, `\"Sim\"`, `\"Death\"`, `\"1\"`, `\"0\"` (dia 0) → 1.",
                        ))

                        # ── Cache / signature audit ───────────────────────────────────────
                        st.markdown(tr("**Cache / signature audit**", "**Auditoria de cache / assinatura**"))
                        st.caption(tr(
                            f"Upload content hash: `{_tv_file_content_hash}`  \n"
                            f"Context sig: `{_tv_context_sig}`  \n"
                            f"Model: `{forced_model}` · threshold: `{_tv_locked_threshold:.4f}` · "
                            f"STS mode: `{'on' if (HAS_STS and bool(_tv_include_sts)) else 'off'}`  \n"
                            f"Missing-value tokens: `MISSING_TOKENS` (risk_data.py)  \n"
                            f"Numeric coercion: `_maybe_numeric` / `clean_features` (modeling.py)",
                            f"Hash do conteúdo: `{_tv_file_content_hash}`  \n"
                            f"Assinatura: `{_tv_context_sig}`  \n"
                            f"Modelo: `{forced_model}` · limiar: `{_tv_locked_threshold:.4f}` · "
                            f"Modo STS: `{'ligado' if (HAS_STS and bool(_tv_include_sts)) else 'desligado'}`  \n"
                            f"Tokens de ausência: `MISSING_TOKENS` (risk_data.py)  \n"
                            f"Coerção numérica: `_maybe_numeric` / `clean_features` (modeling.py)",
                        ))

                        # ── STS execution summary ─────────────────────────────────────────
                        if _tv_exec_log:
                            _exec_counts: dict = {}
                            for _er in _tv_exec_log:
                                # Group by status for successful outcomes (fresh/refreshed/
                                # cached/stale_fallback) and by stage for failed outcomes
                                # (build_input/fetch/batch_abort).  Grouping by stage alone
                                # would collapse all successes under "done", making the
                                # "cached", "fresh", and "stale_fallback" display counts
                                # permanently zero.
                                _estatus = getattr(_er, "status", None) or "unknown"
                                _estage  = getattr(_er, "stage",  None) or ""
                                if _estatus == "failed" and _estage in ("build_input", "fetch", "batch_abort"):
                                    _key = _estage
                                else:
                                    _key = _estatus
                                _exec_counts[_key] = _exec_counts.get(_key, 0) + 1
                            _n_score_present = int(_tv_data["sts_score"].notna().sum()) if "sts_score" in _tv_data.columns else 0
                            _n_exec_abort = sum(1 for _fe in _tv_fail_log if _fe.get("stage") == "batch_abort")
                            st.markdown(tr("**STS execution summary**", "**Resumo de execução STS**"))
                            st.caption(
                                tr(
                                    f"· cached (session/disk): **{_exec_counts.get('cached', 0)}**  \n"
                                    f"· freshly queried: **{_exec_counts.get('fresh', 0) + _exec_counts.get('refreshed', 0)}**  \n"
                                    f"· stale fallback used: **{_exec_counts.get('stale_fallback', 0)}**  \n"
                                    f"· query failed (fetched but invalid): **{_exec_counts.get('fetch', 0)}**  \n"
                                    f"· input build failed: **{_exec_counts.get('build_input', 0)}**  \n"
                                    f"· unqueried due to batch abort: **{_abort_before_query_count}**  \n"
                                    f"· STS score present in final output: **{_n_score_present}**  \n"
                                    f"Â· {_sts_availability_details_caption(_tv_sts_availability, _tv_sts_n_score, _tv_sts_n_eligible, 'English')}  \n"
                                    "_Availability is classified against STS-eligible rows, not the full cohort._",
                                    f"· em cache (sessão/disco): **{_exec_counts.get('cached', 0)}**  \n"
                                    f"· consultados com sucesso: **{_exec_counts.get('fresh', 0) + _exec_counts.get('refreshed', 0)}**  \n"
                                    f"· fallback de cache antigo usado: **{_exec_counts.get('stale_fallback', 0)}**  \n"
                                    f"· falha na consulta: **{_exec_counts.get('fetch', 0)}**  \n"
                                    f"· falha na construção do input: **{_exec_counts.get('build_input', 0)}**  \n"
                                    f"· não consultados por aborto do lote: **{_abort_before_query_count}**  \n"
                                    f"· STS Score presente na saída final: **{_n_score_present}**  \n"
                                    f"Â· {_sts_availability_details_caption(_tv_sts_availability, _tv_sts_n_score, _tv_sts_n_eligible, 'Portuguese')}  \n"
                                    "_A disponibilidade é classificada em relação às linhas elegíveis para STS, não à coorte completa._",
                                )
                            )

                        # ── Batch-abort warning ───────────────────────────────────────────
                        if _batch_aborted_ctx and not (_tv_sts_state == "cancelling" if "_tv_sts_state" in dir() else False):
                            st.warning(
                                tr(
                                    f"⚠️ **STS batch incomplete:** execution stopped after repeated "
                                    f"chunk failures (likely endpoint unreachable). "
                                    f"{_abort_before_query_count} eligible patient(s) were not queried. "
                                    f"Final STS summaries reflect only the successfully queried subset. "
                                    f"Re-run when the STS endpoint is reachable to obtain a complete result.",
                                    f"⚠️ **Lote STS incompleto:** execução interrompida após falhas repetidas "
                                    f"(endpoint provavelmente inacessível). "
                                    f"{_abort_before_query_count} paciente(s) elegível(is) não foram consultados. "
                                    f"Os resumos STS refletem apenas o subconjunto consultado com sucesso. "
                                    f"Execute novamente quando o endpoint STS estiver acessível.",
                                )
                            )

                        # ── STS fail log (only shown when failures exist) ──────────────────
                        if _tv_fail_log:
                            _n_abort_rows = sum(1 for _fe in _tv_fail_log if _fe.get("stage") == "batch_abort")
                            _n_query_rows = len(_tv_fail_log) - _n_abort_rows
                            st.markdown(tr(
                                f"**STS per-patient failures ({len(_tv_fail_log)})**"
                                + (f" — {_n_abort_rows} not queried (batch abort), {_n_query_rows} query/parse failures"
                                   if _n_abort_rows else ""),
                                f"**Falhas STS por paciente ({len(_tv_fail_log)})**"
                                + (f" — {_n_abort_rows} não consultados (aborto do lote), {_n_query_rows} falhas de consulta/parse"
                                   if _n_abort_rows else ""),
                            ))
                            _fail_rows = []
                            for _fe in _tv_fail_log:
                                _fail_rows.append({
                                    tr("patient_id", "patient_id"):    _fe.get("patient_id") or _fe.get("name") or "?",
                                    tr("row_index",  "linha"):         _fe.get("idx", ""),
                                    tr("status",     "status"):        _fe.get("status", "failed"),
                                    tr("stage",      "etapa"):         _fe.get("stage") or "",
                                    tr("reason",     "motivo"):        _fe.get("reason", "?"),
                                    tr("retry",      "tentou_retry"):  _fe.get("retry_attempted", ""),
                                    tr("stale_cache","cache_antigo"):  _fe.get("used_previous_cache", ""),
                                })
                            st.dataframe(pd.DataFrame(_fail_rows), width="stretch", hide_index=True)

                        # ── Chunk-level execution log ─────────────────────────────────────
                        if _tv_chunk_log:
                            _n_abort_chunks = sum(1 for _cl in _tv_chunk_log if _cl.get("aborted_after_this_chunk"))
                            _n_endpoint_chunks = sum(1 for _cl in _tv_chunk_log if _cl.get("endpoint_failure_count", 0) > 0)
                            with st.expander(
                                tr(
                                    f"STS chunk execution log ({len(_tv_chunk_log)} chunk(s)"
                                    + (f", {_n_endpoint_chunks} endpoint failure(s)" if _n_endpoint_chunks else "")
                                    + (f", aborted after chunk {_n_abort_chunks}" if _n_abort_chunks else "") + ")",
                                    f"Log de execução por chunk STS ({len(_tv_chunk_log)} chunk(s)"
                                    + (f", {_n_endpoint_chunks} falha(s) de endpoint" if _n_endpoint_chunks else "")
                                    + (f", abortado após chunk {_n_abort_chunks}" if _n_abort_chunks else "") + ")",
                                ),
                                expanded=bool(_n_abort_chunks),
                            ):
                                _chunk_display_rows = []
                                for _cl in _tv_chunk_log:
                                    _chunk_display_rows.append({
                                        tr("chunk_index",  "chunk_index"):      _cl.get("chunk_index", ""),
                                        tr("row_count",    "n_pacientes"):      _cl.get("row_count", ""),
                                        tr("success",      "sucesso"):          _cl.get("success_count", 0),
                                        tr("failure",      "falha"):            _cl.get("failure_count", 0),
                                        tr("failure_type", "tipo_falha"):       _cl.get("failure_type") or "",
                                        tr("endpt_fails",  "falhas_endpoint"):  _cl.get("endpoint_failure_count", 0),
                                        tr("counted",      "conta_abort"):      _cl.get("counted_toward_abort", False),
                                        tr("exc_type",     "tipo_excecao"):     _cl.get("exception_type") or "",
                                        tr("exc_msg",      "msg_excecao"):      (_cl.get("exception_message") or "")[:80],
                                        tr("aborted",      "abortou_aqui"):     _cl.get("aborted_after_this_chunk", False),
                                    })
                                st.dataframe(pd.DataFrame(_chunk_display_rows), width="stretch", hide_index=True)

                        # ── Endpoint health summary ───────────────────────────────────────
                        _ehs = dict(getattr(calculate_sts_batch, "endpoint_health_summary", {}))
                        if not _ehs:
                            _ehs = dict(st.session_state.get("_tv_result", {}).get("endpoint_health_summary", {}))
                        if _ehs:
                            _ehs_c1, _ehs_c2, _ehs_c3, _ehs_c4 = st.columns(4)
                            _ehs_c1.metric(
                                tr("Sent to endpoint",  "Enviados ao endpoint"),
                                _ehs.get("n_queried", 0),
                                help=tr(
                                    f"Of {_ehs.get('n_eligible_for_fetch',0)} eligible after cache check",
                                    f"De {_ehs.get('n_eligible_for_fetch',0)} elegíveis após cache",
                                ),
                            )
                            _ehs_c2.metric(
                                tr("Received a score",  "Receberam escore"),
                                _ehs.get("n_queried_with_score", 0),
                            )
                            _ehs_c3.metric(
                                tr("Endpoint fail chunks", "Chunks c/ falha endpoint"),
                                _ehs.get("n_chunks_endpoint_failure", 0),
                                help=tr(
                                    f"Of {_ehs.get('n_chunks_attempted',0)} chunks attempted",
                                    f"De {_ehs.get('n_chunks_attempted',0)} chunks tentados",
                                ),
                            )
                            _ehs_c4.metric(
                                tr("Unqueried (abort)",  "Não consultados (abort)"),
                                _ehs.get("n_rows_unqueried", 0),
                            )
                            if _ehs.get("abort_reason"):
                                st.caption(tr(
                                    f"Abort reason: {_ehs['abort_reason']} "
                                    f"after {_ehs.get('abort_endpoint_failures',0)} consecutive endpoint failures.",
                                    f"Motivo do abort: {_ehs['abort_reason']} "
                                    f"após {_ehs.get('abort_endpoint_failures',0)} falhas consecutivas de endpoint.",
                                ))

                        # ── STS eligibility log ───────────────────────────────────────────
                        if _tv_sts_eligibility_log:
                            st.markdown(tr("**STS eligibility log:**", "**Log de elegibilidade STS:**"))
                            _elig_df_det = pd.DataFrame(_tv_sts_eligibility_log)
                            st.dataframe(_elig_df_det, width="stretch", hide_index=True)
                            _elig_det_c1, _elig_det_c2 = st.columns(2)
                            with _elig_det_c1:
                                st.download_button(
                                    label=tr(
                                        "Download STS eligibility (CSV)",
                                        "Baixar elegibilidade STS (CSV)",
                                    ),
                                    data=_elig_df_det.to_csv(index=False).encode("utf-8"),
                                    file_name="sts_eligibility.csv",
                                    mime="text/csv",
                                    key="tv_elig_dl_csv_det",
                                )
                            with _elig_det_c2:
                                _elig_det_xlsx = BytesIO()
                                with pd.ExcelWriter(_elig_det_xlsx, engine="openpyxl") as _ew2:
                                    _elig_df_det.to_excel(_ew2, sheet_name="eligibility", index=False)
                                st.download_button(
                                    label=tr(
                                        "Download STS eligibility (XLSX)",
                                        "Baixar elegibilidade STS (XLSX)",
                                    ),
                                    data=_elig_det_xlsx.getvalue(),
                                    file_name="sts_eligibility.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="tv_elig_dl_xlsx_det",
                                )

                    # ── Persist result state for tab-navigation session cache ──
                    # Saves only the computed data (no export bytes).  Export bytes
                    # are cheaply rebuilt from this data in the display section.
                    # Determine inline-path outcome type for message exclusivity.
                    if not (HAS_STS and _tv_include_sts):
                        _tv_inline_sts_outcome = "no_sts"
                    elif _tv_n_eligible == 0:
                        _tv_inline_sts_outcome = "no_eligible"
                    else:
                        _tv_inline_sts_outcome = "completed"
                    _tv_inline_fail_details = "\n".join(
                        f"- **patient={f.get('patient_id') or '?'}** | "
                        f"status={f.get('status','?')} | reason={f.get('reason','?')}"
                        for f in _tv_fail_log
                    ) if _tv_fail_log else ""
                    st.session_state["_tv_result"] = {
                        "data": _tv_data.copy(),
                        "perf": _tv_perf,
                        "pairwise": _tv_pairwise,
                        "risk_cat": _tv_risk_cat,
                        "calib_df": _tv_calib_df,
                        "cohort_summary": dict(_tv_cohort_summary),
                        "score_cols": list(_tv_score_cols),
                        "rename": dict(_tv_rename),
                        "sts_ok": bool(_tv_sts_ok),
                        "ai_incidents": list(_tv_ai_incidents) if _tv_ai_incidents else [],
                        "exec_log": _tv_exec_log,
                        "sts_status": str(_tv_sts_status),
                        "fail_log": list(_tv_fail_log) if _tv_fail_log else [],
                        "locked_threshold": float(_tv_locked_threshold),
                        "sts_eligibility_log": list(_tv_sts_eligibility_log),
                        "surrogate_timeline": bool(_tv_is_surrogate),
                        "prepare_info": dict(_tv_prepared.info) if _tv_prepared is not None else {},
                        # ── STS outcome fields (for exclusive message display) ──────────
                        "sts_outcome": _tv_inline_sts_outcome,
                        "sts_summary_line": "",
                        "sts_n_completed": 0,
                        "sts_n_failed": 0,
                        "sts_n_total_ctx": _tv_n,
                        "sts_n_cancelled_r": 0,
                        "sts_fail_details": _tv_inline_fail_details,
                        # ── STS audit fields ────────────────────────────────────────
                        "batch_aborted": False,
                        "abort_before_query_count": 0,
                        "chunk_log": list(getattr(calculate_sts_batch, "chunk_log", [])),
                        "sts_patient_audit": [],
                        "endpoint_health_summary": dict(
                            getattr(calculate_sts_batch, "endpoint_health_summary", {})
                        ),
                        "sts_availability": _tv_sts_availability,
                        "sts_n_score": _tv_sts_n_score,
                        "sts_n_eligible": _tv_sts_n_eligible,
                        "sts_n_not_supported": int(sum(
                            1 for e in _tv_sts_eligibility_log
                            if e.get("eligibility") == "not_supported"
                        )),
                        "sts_n_uncertain": int(sum(
                            1 for e in _tv_sts_eligibility_log
                            if e.get("eligibility") == "uncertain"
                        )),
                        # ── Common cohort ────────────────────────────────────────────
                        "common_perf": (
                            _tv_common_perf.to_dict(orient="list")
                            if _tv_common_perf is not None else None
                        ),
                        "n_common": int(_tv_n_common),
                        # ── Exploratory analytics ──────────────────────────────────────
                        "youden":                  dict(_tv_youden),
                        "exploratory_recal":       dict(_tv_exploratory_recal),
                        "exploratory_thresh_tables": dict(_tv_exploratory_thresh_tables),
                    }
                    st.session_state["_tv_result_sig"] = _tv_context_sig

                # Analytics variables computed in the _tv_run branch; initialize safe
                # defaults here so the shared display section never hits UnboundLocalError
                # on the cached-restore path.  The _tv_run branch overwrites these below.
                _tv_youden: dict = {}
                _tv_thresh_tables: dict = {}
                _tv_thresholds_fixed: list = [0.02, 0.05, 0.08, 0.10]
                _tv_common_perf = None
                _tv_n_common: int = 0
                _tv_exploratory_recal: dict = {}
                _tv_exploratory_thresh_tables: dict = {}
                _tv_n_not_supported: int = 0
                _tv_n_uncertain: int = 0
                _tv_endpoint_health: dict = dict(
                    getattr(calculate_sts_batch, "endpoint_health_summary", {})
                )

                # ── Display results (runs on fresh compute OR valid session cache) ──
                if _tv_run or _tv_has_cached:
                    if _tv_has_cached and not _tv_run:
                        # Restore computed state from session — no recomputation.
                        _saved = st.session_state["_tv_result"]
                        _tv_data = _saved["data"].copy()
                        _tv_perf = _saved["perf"]
                        _tv_pairwise = _saved["pairwise"]
                        _tv_risk_cat = _saved["risk_cat"]
                        _tv_calib_df = _saved["calib_df"]
                        _tv_cohort_summary = _saved["cohort_summary"]
                        _tv_score_cols = _saved["score_cols"]
                        _tv_rename = _saved["rename"]
                        _tv_sts_ok = _saved["sts_ok"]
                        _tv_ai_incidents = _saved.get("ai_incidents", [])
                        _tv_exec_log = _saved.get("exec_log", [])
                        _tv_sts_status = _saved.get("sts_status", "")
                        _tv_fail_log = _saved.get("fail_log", [])
                        _tv_locked_threshold = _saved.get("locked_threshold", _tv_locked_threshold)
                        _tv_sts_eligibility_log = _saved.get("sts_eligibility_log", [])
                        _tv_is_surrogate = _saved.get("surrogate_timeline", False)
                        _tv_prepare_info = _saved.get("prepare_info", {})
                        _batch_aborted_ctx = _saved.get("batch_aborted", False)
                        _abort_before_query_count = _saved.get("abort_before_query_count", 0)
                        _tv_chunk_log = _saved.get("chunk_log", [])
                        _tv_sts_audit_rows = _saved.get("sts_patient_audit", [])
                        _tv_endpoint_health = _saved.get("endpoint_health_summary", {})
                        _tv_sts_availability = _saved.get("sts_availability", "no_sts")
                        _tv_sts_n_score = int(_saved.get("sts_n_score", 0) or 0)
                        _tv_sts_n_eligible = int(_saved.get("sts_n_eligible", 0) or 0)
                        _tv_youden = _saved.get("youden", {})
                        _tv_exploratory_recal = _saved.get("exploratory_recal", {})
                        _tv_exploratory_thresh_tables = _saved.get("exploratory_thresh_tables", {})
                        _tv_n_not_supported = int(_saved.get("sts_n_not_supported", 0) or 0)
                        _tv_n_uncertain     = int(_saved.get("sts_n_uncertain",     0) or 0)
                        _tv_common_perf_dict = _saved.get("common_perf")
                        _tv_common_perf = (
                            pd.DataFrame(_tv_common_perf_dict)
                            if _tv_common_perf_dict else None
                        )
                        _tv_n_common = int(_saved.get("n_common", 0) or 0)

                        st.success(tr(
                            "Temporal validation results restored from session — "
                            "no recomputation performed. "
                            "Click **Run temporal validation** to recompute.",
                            "Resultados da validação temporal restaurados da sessão — "
                            "nenhuma recomputação realizada. "
                            "Clique em **Executar validação temporal** para recomputar.",
                        ))

                        # ── Exclusive STS outcome message + structured summary ──────────
                        # Only one of these branches fires; order mirrors severity.
                        _saved_outcome     = _saved.get("sts_outcome", "")
                        _saved_n_completed = _saved.get("sts_n_completed", 0)
                        _saved_n_failed    = _saved.get("sts_n_failed", 0)
                        _saved_n_total_ctx = _saved.get("sts_n_total_ctx", 0)
                        _saved_n_cancelled = _saved.get("sts_n_cancelled_r", 0)
                        _saved_summary     = _saved.get("sts_summary_line", "")
                        _saved_fail_details = _saved.get("sts_fail_details", "")
                        _saved_sts_status  = _saved.get("sts_status", "")

                        # Compute skipped breakdown from eligibility log (available here)
                        _sum_elig = len([e for e in _tv_sts_eligibility_log if e["eligibility"] == "supported"])
                        _sum_ns   = len([e for e in _tv_sts_eligibility_log if e["eligibility"] == "not_supported"])
                        _sum_unc  = len([e for e in _tv_sts_eligibility_log if e["eligibility"] == "uncertain"])
                        _sum_skip = _sum_ns + _sum_unc

                        if _saved_outcome == "cancelled":
                            st.warning(tr(
                                "STS execution cancelled by user — partial results preserved.",
                                "Execução STS cancelada pelo usuário — resultados parciais preservados.",
                            ))
                        elif _saved_outcome == "batch_aborted":
                            st.warning(tr(
                                "STS batch aborted after consecutive failures — "
                                "the web calculator may be unreachable. Partial results preserved.",
                                "Lote STS abortado por falhas consecutivas — "
                                "a calculadora web pode estar inacessível. Resultados parciais preservados.",
                            ))
                        elif _saved_outcome == "no_eligible":
                            pass  # shown inline during the run
                        elif _saved_outcome == "no_sts":
                            pass  # scoring-options label already explains

                        # Structured summary — shown for cancelled / batch_aborted / completed
                        # when the eligibility log is available.
                        if _saved_outcome in ("cancelled", "batch_aborted", "completed") and _tv_sts_eligibility_log:
                            _sm_c1, _sm_c2, _sm_c3, _sm_c4, _sm_c5 = st.columns(5)
                            _sm_c1.metric(tr("STS eligible",  "Elegíveis STS"),  _sum_elig)
                            _sm_c2.metric(tr("Completed",     "Concluídos"),     _saved_n_completed)
                            _sm_c3.metric(tr("Failed",        "Falhas"),         _saved_n_failed)
                            _sm_c4.metric(tr("Skipped",       "Ignorados"),      _sum_skip)
                            _sm_c5.metric(
                                tr("Cancelled remaining", "Cancelados restantes"),
                                _saved_n_cancelled,
                            )

                        if _saved_sts_status:
                            st.caption(tr(
                                f"STS Score cache status: {_saved_sts_status}",
                                f"Status do cache STS Score: {_saved_sts_status}",
                            ))
                        # ── STS consistency check ──────────────────────────────────────
                        if _tv_sts_audit_rows:
                            _aud_n_elig   = len(_tv_sts_audit_rows)
                            _aud_n_score  = sum(1 for _ar in _tv_sts_audit_rows if _ar["sts_score_present_final"])
                            if _aud_n_score < _aud_n_elig:
                                _aud_n_lost  = _aud_n_elig - _aud_n_score
                                _aud_n_bi    = sum(1 for _ar in _tv_sts_audit_rows if _ar["sts_failure_stage"] == "build_input")
                                _aud_n_fe    = sum(1 for _ar in _tv_sts_audit_rows if _ar["sts_failure_stage"] == "fetch" and not _ar["sts_score_present_final"])
                                _aud_n_ab    = sum(1 for _ar in _tv_sts_audit_rows if _ar["sts_batch_aborted_before_query"] and not _ar["sts_score_present_final"])
                                _aud_n_other = _aud_n_lost - _aud_n_bi - _aud_n_fe - _aud_n_ab
                                st.warning(tr(
                                    f"STS Score consistency: **{_aud_n_elig} eligible → {_aud_n_score} final scores** "
                                    f"({_aud_n_lost} patient(s) without a score: "
                                    f"{_aud_n_bi} input-build failure(s), "
                                    f"{_aud_n_fe} fetch/parse failure(s), "
                                    f"{_aud_n_ab} batch-aborted"
                                    + (f", {_aud_n_other} other" if _aud_n_other else "")
                                    + "). See **STS patient audit** below.",
                                    f"Consistência STS Score: **{_aud_n_elig} elegíveis → {_aud_n_score} escores finais** "
                                    f"({_aud_n_lost} paciente(s) sem escore: "
                                    f"{_aud_n_bi} falha(s) de construção de input, "
                                    f"{_aud_n_fe} falha(s) de consulta/parse, "
                                    f"{_aud_n_ab} abortados em lote"
                                    + (f", {_aud_n_other} outros" if _aud_n_other else "")
                                    + "). Veja **Auditoria de pacientes STS** abaixo.",
                                ))
                        if _tv_ai_incidents:
                            st.warning(tr(
                                f"AI Risk inference incidents (temporal validation): "
                                f"{len(_tv_ai_incidents)} patient(s) had issues on last run.",
                                f"Incidentes de inferência do AI Risk (validação temporal): "
                                f"{len(_tv_ai_incidents)} paciente(s) com problemas na última execução.",
                            ))
                        if _tv_fail_log:
                            st.warning(tr(
                                f"STS Score per-patient failures: {len(_tv_fail_log)} patient(s). "
                                "See **View execution details** below for the full failure log.",
                                f"Falhas STS Score por paciente: {len(_tv_fail_log)} paciente(s). "
                                "Veja **Ver detalhes de execução** abaixo para o log completo de falhas.",
                            ))

                        # ── Execution details (restored from session) ──────────────────
                        with st.expander(tr("View execution details", "Ver detalhes de execução"), expanded=False):
                            _rst_elig_n  = len([e for e in _tv_sts_eligibility_log if e["eligibility"] == "supported"])
                            _rst_ns_n    = len([e for e in _tv_sts_eligibility_log if e["eligibility"] == "not_supported"])
                            _rst_unc_n   = len([e for e in _tv_sts_eligibility_log if e["eligibility"] == "uncertain"])
                            _rst_skip_n  = _rst_ns_n + _rst_unc_n

                            # ── Pipeline provenance ───────────────────────────────────────
                            st.markdown(tr("**Pipeline provenance**", "**Proveniência do pipeline**"))
                            st.caption(tr(
                                "Loading: `prepare_master_dataset` (risk_data.py)  \n"
                                "Surgery class: `classify_sts_eligibility` (sts_calculator.py)  \n"
                                "STS inputs: `build_sts_input_from_row` (sts_calculator.py)  \n"
                                "AI Risk: `apply_frozen_model_to_temporal_cohort` (ai_risk_inference.py)  \n"
                                "EuroSCORE II: `euroscore_from_row` (euroscore.py)",
                                "Carregamento: `prepare_master_dataset` (risk_data.py)  \n"
                                "Classe cirúrgica: `classify_sts_eligibility` (sts_calculator.py)  \n"
                                "Inputs STS: `build_sts_input_from_row` (sts_calculator.py)  \n"
                                "AI Risk: `apply_frozen_model_to_temporal_cohort` (ai_risk_inference.py)  \n"
                                "EuroSCORE II: `euroscore_from_row` (euroscore.py)",
                            ))

                            # ── Cohort assembly breakdown ─────────────────────────────────
                            st.markdown(tr("**Cohort assembly**", "**Montagem da coorte**"))
                            _pi_r = _tv_prepare_info
                            if _pi_r.get("source_type") == "flat":
                                st.caption(tr(
                                    f"Source: flat CSV/Parquet — no merge steps.  \n"
                                    f"Rows loaded: **{_pi_r.get('n_rows', len(_tv_data))}**  \n"
                                    f"_(Intermediate merge/filter counts are only available for XLSX uploads.)_",
                                    f"Fonte: CSV/Parquet plano — sem etapas de mesclagem.  \n"
                                    f"Linhas carregadas: **{_pi_r.get('n_rows', len(_tv_data))}**  \n"
                                    f"_(Contagens intermediárias só estão disponíveis para uploads XLSX.)_",
                                ))
                            elif _pi_r.get("pre_rows_before_criteria", 0) > 0:
                                _pi_r_excl_crit  = _pi_r.get("excluded_missing_surgery_or_date", 0)
                                _pi_r_excl_match = _pi_r.get("excluded_no_pre_post_match", 0)
                                st.caption(tr(
                                    f"Preoperative rows (raw): **{_pi_r.get('pre_rows_before_criteria')}**  \n"
                                    f"After surgery/date filter: **{_pi_r.get('pre_rows_after_criteria')}** "
                                    f"(excluded: {_pi_r_excl_crit})  \n"
                                    f"Unique pre-op patient–date pairs: **{_pi_r.get('pre_unique_patient_date_after_criteria')}**  \n"
                                    f"Post-op unique patient–date pairs: **{_pi_r.get('post_unique_patient_date')}**  \n"
                                    f"After pre–post inner join: **{_pi_r.get('matched_pre_post_rows')}** "
                                    f"(unmatched: {_pi_r_excl_match})  \n"
                                    f"Echocardiogram rows joined: **{_pi_r.get('echo_rows', '?')}**  \n"
                                    f"Final cohort (after normalization): **{len(_tv_data)}** patients  \n"
                                    f"Outcome encoding: `Death` → `morte_30d` via `map_death_30d`  \n"
                                    f"⚠ Note: `Death = \"0\"` is interpreted as operative death on day 0 "
                                    f"(not as boolean false). See canonical semantics note below.",
                                    f"Linhas pré-operatórias (brutas): **{_pi_r.get('pre_rows_before_criteria')}**  \n"
                                    f"Após filtro cirurgia/data: **{_pi_r.get('pre_rows_after_criteria')}** "
                                    f"(excluídos: {_pi_r_excl_crit})  \n"
                                    f"Pares paciente–data pré-op únicos: **{_pi_r.get('pre_unique_patient_date_after_criteria')}**  \n"
                                    f"Pares paciente–data pós-op únicos: **{_pi_r.get('post_unique_patient_date')}**  \n"
                                    f"Após junção interna pré–pós: **{_pi_r.get('matched_pre_post_rows')}** "
                                    f"(sem correspondência: {_pi_r_excl_match})  \n"
                                    f"Linhas de ecocardiograma: **{_pi_r.get('echo_rows', '?')}**  \n"
                                    f"Coorte final (após normalização): **{len(_tv_data)}** pacientes  \n"
                                    f"Codificação do desfecho: `Death` → `morte_30d` via `map_death_30d`  \n"
                                    f"⚠ Nota: `Death = \"0\"` é interpretado como morte operatória no dia 0 "
                                    f"(não como falso booleano). Ver nota de semântica canônica abaixo.",
                                ))
                            else:
                                st.caption(tr(
                                    f"Final cohort: **{len(_tv_data)}** patients  \n"
                                    f"_(Detailed intermediate counts not available for this source format.)_",
                                    f"Coorte final: **{len(_tv_data)}** pacientes  \n"
                                    f"_(Contagens intermediárias detalhadas não disponíveis para este formato.)_",
                                ))

                            # ── Per-step patient counts ───────────────────────────────────
                            st.markdown(tr("**Per-step patient counts**", "**Contagem de pacientes por etapa**"))
                            st.caption(tr(
                                f"After `prepare_master_dataset`: **{len(_tv_data)}** patients "
                                f"({_tv_cohort_summary.get('n_events', '?')} events, "
                                f"event rate {_tv_cohort_summary.get('event_rate', 0):.1%})  \n"
                                f"STS eligibility (`classify_sts_eligibility`):  \n"
                                f"  · supported: **{_rst_elig_n}** (eligible for query)  \n"
                                f"  · not_supported: **{_rst_ns_n}** (aorta/Bentall/dissection — skipped)  \n"
                                f"  · uncertain: **{_rst_unc_n}** (OBSERVATION ADMIT or unmapped — skipped)  \n"
                                f"  · total classified: **{len(_tv_sts_eligibility_log)}**  \n"
                                f"AI Risk inference incidents: {len(_tv_ai_incidents) if _tv_ai_incidents else 0}  \n"
                                f"{_sts_availability_details_caption(_tv_sts_availability, _tv_sts_n_score, _tv_sts_n_eligible, 'English')}  \n"
                                f"Temporal axis: {'de-identified surrogate' if _tv_is_surrogate else 'standard'}",
                                f"Após `prepare_master_dataset`: **{len(_tv_data)}** pacientes "
                                f"({_tv_cohort_summary.get('n_events', '?')} eventos, "
                                f"taxa {_tv_cohort_summary.get('event_rate', 0):.1%})  \n"
                                f"Elegibilidade STS (`classify_sts_eligibility`):  \n"
                                f"  · suportados: **{_rst_elig_n}** (elegíveis para consulta)  \n"
                                f"  · não suportados: **{_rst_ns_n}** (aorta/Bentall/dissecção — ignorados)  \n"
                                f"  · incertos: **{_rst_unc_n}** (OBSERVATION ADMIT ou sem mapeamento — ignorados)  \n"
                                f"  · total classificados: **{len(_tv_sts_eligibility_log)}**  \n"
                                f"Incidentes de AI Risk: {len(_tv_ai_incidents) if _tv_ai_incidents else 0}  \n"
                                f"{_sts_availability_details_caption(_tv_sts_availability, _tv_sts_n_score, _tv_sts_n_eligible, 'Portuguese')}  \n"
                                f"Eixo temporal: {'substituto desidentificado' if _tv_is_surrogate else 'padrão'}",
                            ))

                            # ── Outcome semantics note ────────────────────────────────────
                            st.markdown(tr("**Outcome encoding semantics**", "**Semântica da codificação do desfecho**"))
                            st.caption(tr(
                                "Note: `Death = \"0\"` is interpreted canonically as **operative death on day 0**, "
                                "not as boolean false (survivor). This is handled by `map_death_30d` / "
                                "`parse_postop_timing` in `risk_data.py`. Day 0 = day of surgery → event = 1.  \n"
                                "Survivor tokens: `\"No\"`, `\"Não\"`, `\"Nao\"`, `\"-\"`, `\"--\"` → 0.  \n"
                                "Event tokens: `\"Yes\"`, `\"Sim\"`, `\"Death\"`, `\"1\"`, `\"0\"` (day 0) → 1.",
                                "Nota: `Death = \"0\"` é interpretado canonicamente como **morte operatória no dia 0**, "
                                "não como falso booleano (sobrevivente). Isso é tratado por `map_death_30d` / "
                                "`parse_postop_timing` em `risk_data.py`. Dia 0 = dia da cirurgia → evento = 1.  \n"
                                "Tokens de sobrevivente: `\"No\"`, `\"Não\"`, `\"Nao\"`, `\"-\"`, `\"--\"` → 0.  \n"
                                "Tokens de evento: `\"Yes\"`, `\"Sim\"`, `\"Death\"`, `\"1\"`, `\"0\"` (dia 0) → 1.",
                            ))

                            # ── Cache / signature audit ───────────────────────────────────
                            st.markdown(tr("**Cache / signature audit**", "**Auditoria de cache / assinatura**"))
                            st.caption(tr(
                                f"Upload content hash: `{_tv_file_content_hash}`  \n"
                                f"Context sig: `{_tv_context_sig}`  \n"
                                f"Model: `{forced_model}` · threshold: `{_tv_locked_threshold:.4f}` · "
                                f"STS mode: `{'on' if (HAS_STS and bool(_tv_include_sts)) else 'off'}`  \n"
                                f"Missing-value tokens: `MISSING_TOKENS` (risk_data.py)  \n"
                                f"Numeric coercion: `_maybe_numeric` / `clean_features` (modeling.py)",
                                f"Hash do conteúdo: `{_tv_file_content_hash}`  \n"
                                f"Assinatura: `{_tv_context_sig}`  \n"
                                f"Modelo: `{forced_model}` · limiar: `{_tv_locked_threshold:.4f}` · "
                                f"Modo STS: `{'ligado' if (HAS_STS and bool(_tv_include_sts)) else 'desligado'}`  \n"
                                f"Tokens de ausência: `MISSING_TOKENS` (risk_data.py)  \n"
                                f"Coerção numérica: `_maybe_numeric` / `clean_features` (modeling.py)",
                            ))

                            # ── STS fail log (only shown when failures exist) ──────────────
                            if _tv_fail_log:
                                st.markdown(tr(
                                    f"**STS per-patient failures ({len(_tv_fail_log)})**",
                                    f"**Falhas STS por paciente ({len(_tv_fail_log)})**",
                                ))
                                _fail_rows_r = []
                                for _fe_r in _tv_fail_log:
                                    _fail_rows_r.append({
                                        tr("patient_id", "patient_id"):    _fe_r.get("patient_id") or _fe_r.get("name") or "?",
                                        tr("row_index",  "linha"):         _fe_r.get("idx", ""),
                                        tr("status",     "status"):        _fe_r.get("status", "failed"),
                                        tr("stage",      "etapa"):         _fe_r.get("stage") or "",
                                        tr("reason",     "motivo"):        _fe_r.get("reason", "?"),
                                        tr("retry",      "tentou_retry"):  _fe_r.get("retry_attempted", ""),
                                        tr("stale_cache","cache_antigo"):  _fe_r.get("used_previous_cache", ""),
                                    })
                                st.dataframe(pd.DataFrame(_fail_rows_r), width="stretch", hide_index=True)

                            # ── STS eligibility log ───────────────────────────────────────
                            if _tv_sts_eligibility_log:
                                st.markdown(tr("**STS eligibility log:**", "**Log de elegibilidade STS:**"))
                                _elig_df_rst = pd.DataFrame(_tv_sts_eligibility_log)
                                st.dataframe(_elig_df_rst, width="stretch", hide_index=True)
                                _elig_rst_c1, _elig_rst_c2 = st.columns(2)
                                with _elig_rst_c1:
                                    st.download_button(
                                        label=tr(
                                            "Download STS eligibility (CSV)",
                                            "Baixar elegibilidade STS (CSV)",
                                        ),
                                        data=_elig_df_rst.to_csv(index=False).encode("utf-8"),
                                        file_name="sts_eligibility.csv",
                                        mime="text/csv",
                                        key="tv_elig_dl_csv_rst",
                                    )
                                with _elig_rst_c2:
                                    _elig_rst_xlsx = BytesIO()
                                    with pd.ExcelWriter(_elig_rst_xlsx, engine="openpyxl") as _ew3:
                                        _elig_df_rst.to_excel(_ew3, sheet_name="eligibility", index=False)
                                    st.download_button(
                                        label=tr(
                                            "Download STS eligibility (XLSX)",
                                            "Baixar elegibilidade STS (XLSX)",
                                        ),
                                        data=_elig_rst_xlsx.getvalue(),
                                        file_name="sts_eligibility.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key="tv_elig_dl_xlsx_rst",
                                    )

                        # ── STS patient audit expander (restore path) ────────────────
                        if _tv_sts_audit_rows:
                            _aud_has_loss = any(not _ar["sts_score_present_final"] for _ar in _tv_sts_audit_rows)
                            with st.expander(
                                tr(
                                    f"STS patient audit ({len(_tv_sts_audit_rows)} eligible patient(s))",
                                    f"Auditoria de pacientes STS ({len(_tv_sts_audit_rows)} paciente(s) elegível(is))",
                                ),
                                expanded=_aud_has_loss,
                            ):
                                _aud_df = pd.DataFrame(_tv_sts_audit_rows)
                                st.dataframe(_aud_df, width="stretch", hide_index=True)
                                # Reconciliation summary
                                _rec_score    = sum(1 for _ar in _tv_sts_audit_rows if _ar["sts_score_present_final"])
                                _rec_bi       = sum(1 for _ar in _tv_sts_audit_rows if _ar["sts_failure_stage"] == "build_input")
                                _rec_fetch_ns = sum(1 for _ar in _tv_sts_audit_rows if _ar["sts_failure_stage"] == "fetch" and not _ar["sts_score_present_final"])
                                _rec_abort_ns = sum(1 for _ar in _tv_sts_audit_rows if _ar["sts_batch_aborted_before_query"] and not _ar["sts_score_present_final"])
                                _rec_total    = _rec_score + _rec_bi + _rec_fetch_ns + _rec_abort_ns
                                _rec_ok       = _rec_total == len(_tv_sts_audit_rows)
                                st.caption(tr(
                                    f"Reconciliation: {len(_tv_sts_audit_rows)} eligible = "
                                    f"{_rec_score} with score + "
                                    f"{_rec_bi} input-build failure + "
                                    f"{_rec_fetch_ns} fetch/parse failure (no fallback) + "
                                    f"{_rec_abort_ns} batch-aborted (no fallback)"
                                    + (" ✓" if _rec_ok else f" ⚠ unaccounted: {len(_tv_sts_audit_rows) - _rec_total}"),
                                    f"Reconciliação: {len(_tv_sts_audit_rows)} elegíveis = "
                                    f"{_rec_score} com escore + "
                                    f"{_rec_bi} falha de input + "
                                    f"{_rec_fetch_ns} falha de consulta/parse (sem fallback) + "
                                    f"{_rec_abort_ns} abortados em lote (sem fallback)"
                                    + (" ✓" if _rec_ok else f" ⚠ não contabilizados: {len(_tv_sts_audit_rows) - _rec_total}"),
                                ))
                                _aud_c1, _aud_c2 = st.columns(2)
                                with _aud_c1:
                                    st.download_button(
                                        label=tr("Download STS audit (CSV)", "Baixar auditoria STS (CSV)"),
                                        data=_aud_df.to_csv(index=False).encode("utf-8"),
                                        file_name="sts_patient_audit.csv",
                                        mime="text/csv",
                                        key="tv_aud_dl_csv_rst",
                                    )
                                with _aud_c2:
                                    _aud_xlsx = BytesIO()
                                    with pd.ExcelWriter(_aud_xlsx, engine="openpyxl") as _ew_aud:
                                        _aud_df.to_excel(_ew_aud, sheet_name="sts_audit", index=False)
                                    st.download_button(
                                        label=tr("Download STS audit (XLSX)", "Baixar auditoria STS (XLSX)"),
                                        data=_aud_xlsx.getvalue(),
                                        file_name="sts_patient_audit.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key="tv_aud_dl_xlsx_rst",
                                    )

                    # ── Block 3: Main Validation Result ───────────────────
                    st.divider()
                    st.markdown(tr("### Main Validation Result", "### Resultado Principal da Validação"))
                    st.caption(tr(
                        "Locked model applied to the validation cohort. Fixed operational threshold. "
                        "This is the primary analysis — do not confuse with exploratory recalibration below.",
                        "Modelo congelado aplicado à coorte de validação. Limiar operacional fixo. "
                        "Esta é a análise principal — não confundir com recalibração exploratória abaixo.",
                    ))

                    if _tv_sts_availability in ("complete", "partial", "unavailable") and _tv_sts_n_eligible > 0:
                        _tv_sts_availability_note = build_sts_availability_summary(
                            _tv_sts_n_eligible,
                            _tv_sts_n_score,
                            language=language,
                        )
                        _tv_sts_score_label = _tv_sts_availability_note["score_label"]
                    else:
                        _tv_sts_availability_note = None
                        _tv_sts_score_label = "STS Score"

                    if _tv_sts_availability_note and _tv_sts_availability in (STS_AVAILABILITY_PARTIAL, STS_AVAILABILITY_UNAVAILABLE):
                        st.warning(_tv_sts_availability_note["banner_text"])
                        st.caption(_tv_sts_availability_note["coverage_text"])
                        if _tv_sts_availability == STS_AVAILABILITY_PARTIAL and _tv_sts_availability_note.get("subset_note"):
                            st.caption(_tv_sts_availability_note["subset_note"])
                        elif _tv_sts_availability == STS_AVAILABILITY_UNAVAILABLE and _tv_sts_availability_note.get("suppressed_note"):
                            st.caption(_tv_sts_availability_note["suppressed_note"])

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

                        # STS accounting table (only when STS was attempted)
                        if _tv_sts_availability != "no_sts" and _tv_sts_n_eligible > 0:
                            st.markdown(tr("**STS pipeline accounting**", "**Accounting do pipeline STS**"))
                            _acct_df = build_sts_accounting_table(
                                n_total=_cs["n_total"],
                                n_not_supported=_tv_n_not_supported,
                                n_uncertain=_tv_n_uncertain,
                                n_supported=_tv_sts_n_eligible,
                                n_final_usable=_tv_sts_n_score,
                                language=language,
                            )
                            st.dataframe(_acct_df, width="stretch", hide_index=True)

                    # 7.2 Performance table
                    if not _tv_perf.empty:
                        with st.expander(tr("Discrimination and calibration", "Discriminação e calibração"), expanded=True):
                            st.caption(tr(
                                f"95% CI by bootstrap ({AppConfig.N_BOOTSTRAP_SAMPLES} resamples). "
                                f"Classification metrics at locked threshold = {_tv_locked_threshold:.0%}.",
                                f"IC 95% por bootstrap ({AppConfig.N_BOOTSTRAP_SAMPLES} reamostras). "
                                f"Métricas de classificação no limiar bloqueado = {_tv_locked_threshold:.0%}.",
                            ))
                            if _tv_sts_availability_note and _tv_sts_availability in (STS_AVAILABILITY_PARTIAL, STS_AVAILABILITY_UNAVAILABLE):
                                if _tv_sts_availability == STS_AVAILABILITY_PARTIAL:
                                    st.caption(_tv_sts_availability_note["subset_note"])
                                else:
                                    st.caption(_tv_sts_availability_note["suppressed_note"])
                            _tv_perf_display = _tv_perf.copy()
                            # Format for display
                            for _fc in ["AUC", "AUPRC", "AUPRC_baseline", "Brier",
                                        "Calibration_Intercept", "Calibration_Slope",
                                        "Sensitivity", "Specificity", "PPV", "NPV",
                                        "CIL", "ICI"]:
                                if _fc in _tv_perf_display.columns:
                                    _tv_perf_display[_fc] = _tv_perf_display[_fc].map(
                                        lambda v: f"{v:.3f}" if pd.notna(v) else "—"
                                    )
                            for _fc in ["AUC_IC95_inf", "AUC_IC95_sup",
                                        "AUPRC_IC95_inf", "AUPRC_IC95_sup",
                                        "Calibration_Intercept_CI_low", "Calibration_Intercept_CI_high",
                                        "Calibration_Slope_CI_low", "Calibration_Slope_CI_high",
                                        "CIL_CI_low", "CIL_CI_high"]:
                                if _fc in _tv_perf_display.columns:
                                    _tv_perf_display[_fc] = _tv_perf_display[_fc].map(
                                        lambda v: f"{v:.3f}" if pd.notna(v) else "—"
                                    )
                            if "HL_p" in _tv_perf_display.columns:
                                _tv_perf_display["HL_p"] = _tv_perf_display["HL_p"].map(
                                    lambda v: f"{v:.4f}" if pd.notna(v) else "—"
                                )
                            st.dataframe(_tv_perf_display, width="stretch", hide_index=True)

                    # ── Calibration at a Glance ──────────────────────────
                    st.markdown(tr("**Calibration at a Glance**", "**Calibração em Resumo**"))
                    if not _tv_calib_df.empty:
                        st.caption(tr(
                            "Intercept near 0 and slope near 1 indicate good calibration. "
                            "HL p-value is complementary — do not interpret in isolation.",
                            "Intercepto próximo de 0 e slope próximo de 1 indicam boa calibração. "
                            "p-valor de HL é apenas complementar — não interpretar isoladamente.",
                        ))
                        _tv_cag_cols = [c for c in [
                            "Score", "Calibration_Intercept", "Calibration_Slope",
                            "CIL", "ICI", "HL_p",
                        ] if c in _tv_calib_df.columns]
                        _tv_calib_display = _tv_calib_df[_tv_cag_cols].copy()
                        for _fc in ["Calibration_Intercept", "Calibration_Slope", "CIL", "ICI"]:
                            if _fc in _tv_calib_display.columns:
                                _tv_calib_display[_fc] = _tv_calib_display[_fc].map(
                                    lambda v: f"{v:.3f}" if pd.notna(v) else "—"
                                )
                        if "HL_p" in _tv_calib_display.columns:
                            _tv_calib_display["HL_p"] = _tv_calib_display["HL_p"].map(
                                lambda v: f"{v:.4f}" if pd.notna(v) else "—"
                            )
                        st.dataframe(_tv_calib_display, width="stretch", hide_index=True)
                        st.caption(tr(
                            "ICI (Integrated Calibration Index): continuous calibration measure via isotonic "
                            "regression — lower is better (0 = perfect calibration). "
                            "CIL (Calibration-in-the-Large) = mean predicted − mean observed: "
                            "positive values indicate the model overestimates risk on average.",
                            "ICI (Índice Integrado de Calibração): medida contínua via regressão isotônica — "
                            "menor é melhor (0 = calibração perfeita). "
                            "CIL (Calibração Geral) = média predita − média observada: "
                            "valores positivos indicam que o modelo superestima o risco em média.",
                        ))
                    else:
                        st.caption(tr(
                            "Calibration summary not available for this cohort "
                            "(insufficient events or single outcome class).",
                            "Resumo de calibração não disponível para esta coorte "
                            "(eventos insuficientes ou classe de desfecho única).",
                        ))

                    # 7.3 Pairwise comparison
                    if not _tv_pairwise.empty:
                        with st.expander(tr("Pairwise comparison", "Comparação pareada"), expanded=True):
                            _tv_pw_display = _tv_pairwise.copy()
                            for _fc in ["Delta_AUC", "Delta_AUC_IC95_inf", "Delta_AUC_IC95_sup",
                                        "Bootstrap_CI_low", "Bootstrap_CI_high",
                                        "DeLong_SE",
                                        "NRI", "NRI_CI_low", "NRI_CI_high",
                                        "IDI", "IDI_CI_low", "IDI_CI_high"]:
                                if _fc in _tv_pw_display.columns:
                                    _tv_pw_display[_fc] = _tv_pw_display[_fc].map(
                                        lambda v: f"{v:.4f}" if pd.notna(v) else "—"
                                    )
                            for _fc in ["Bootstrap_p", "DeLong_p", "NRI_p", "IDI_p"]:
                                if _fc in _tv_pw_display.columns:
                                    _tv_pw_display[_fc] = _tv_pw_display[_fc].map(
                                        lambda v: f"{v:.4f}" if pd.notna(v) else "—"
                                    )
                            # DeLong is suppressed for sparse cohorts (<2 events
                            # or <2 non-events) — drop the internal reason column
                            # from the visible table and surface it as a caption
                            # below so the p-value cell shows an em dash with
                            # explanatory context instead of silently failing.
                            _tv_pw_skip_notes = []
                            if "DeLong_skip_reason" in _tv_pw_display.columns:
                                _tv_pw_skip_notes = [
                                    r for r in _tv_pw_display["DeLong_skip_reason"].tolist()
                                    if isinstance(r, str) and r
                                ]
                                _tv_pw_display = _tv_pw_display.drop(columns=["DeLong_skip_reason"])
                            st.dataframe(_tv_pw_display, width="stretch", hide_index=True)
                            if _tv_pw_skip_notes:
                                st.caption(
                                    tr(
                                        "DeLong p-value shown as '—' for one or more comparisons: "
                                        "the validation cohort has fewer than 2 events or fewer than "
                                        "2 non-events, below the minimum required for a stable "
                                        "variance estimate. Bootstrap ΔAUC with 95% CI remains valid.",
                                        "Valor-p de DeLong exibido como '—' em uma ou mais comparações: "
                                        "a coorte de validação tem menos de 2 eventos ou menos de 2 "
                                        "não-eventos, abaixo do mínimo necessário para uma estimativa "
                                        "estável de variância. O ΔAUC por bootstrap com IC95% "
                                        "permanece válido.",
                                    )
                                )

                    # 7.4 Risk categories
                    if not _tv_risk_cat.empty:
                        with st.expander(tr("Risk category distribution", "Distribuição por classe de risco"), expanded=True):
                            if _tv_sts_availability_note and _tv_sts_availability in (STS_AVAILABILITY_PARTIAL, STS_AVAILABILITY_UNAVAILABLE):
                                st.caption(_tv_sts_availability_note["risk_category_note"])
                            _tv_rc_display = _tv_risk_cat.copy()
                            _tv_rc_display["Observed_mortality"] = _tv_rc_display["Observed_mortality"].map(
                                lambda v: f"{v:.1%}" if pd.notna(v) else "—"
                            )
                            st.dataframe(_tv_rc_display, width="stretch", hide_index=True)

                    # ── 8. Interactive Graphs ──
                    st.divider()
                    st.markdown(tr("### Graphs", "### Gráficos"))

                    import plotly.graph_objects as go
                    from sklearn.metrics import (
                        precision_recall_curve as _pr_curve_fn,
                        average_precision_score as _auprc_fn,
                        roc_auc_score as _roc_auc_fn,
                    )

                    _tv_y = _tv_data["morte_30d"].values

                    def _model_color(name: str) -> str:
                        return _TV_MODEL_COLORS.get(name, "#636efa")

                    # 8.1 ROC curves
                    with st.expander(tr("ROC Curves", "Curvas ROC"), expanded=True):
                        _fig_roc = go.Figure()
                        _fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                           line=dict(dash="dash", color="gray", width=1))
                        for _sc in _tv_score_cols:
                            _sub = _tv_data[["morte_30d", _sc]].dropna()
                            if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                                _fpr, _tpr = roc_data(_sub["morte_30d"].values, _sub[_sc].values)
                                _auc_val = float(_roc_auc_fn(_sub["morte_30d"].values, _sub[_sc].values))
                                _label = _tv_rename.get(_sc, _sc)
                                _fig_roc.add_trace(go.Scatter(
                                    x=_fpr, y=_tpr, mode="lines",
                                    name=f"{_label} (AUC={_auc_val:.3f})",
                                    line=dict(color=_model_color(_label), width=2),
                                ))
                                # Mark locked threshold
                                _cls_thr = classification_metrics_at_threshold(
                                    _sub["morte_30d"].values, _sub[_sc].values, _tv_locked_threshold
                                )
                                _fig_roc.add_trace(go.Scatter(
                                    x=[1 - _cls_thr["Specificity"]], y=[_cls_thr["Sensitivity"]],
                                    mode="markers",
                                    marker=dict(symbol="x", size=10, color=_model_color(_label)),
                                    name=f"{_label} @ {_tv_locked_threshold:.0%}",
                                    showlegend=False,
                                    hovertemplate=(
                                        f"{_label}<br>"
                                        f"Threshold={_tv_locked_threshold:.0%}<br>"
                                        f"TPR={_cls_thr['Sensitivity']:.3f}<br>"
                                        f"FPR={1-_cls_thr['Specificity']:.3f}<extra></extra>"
                                    ),
                                ))
                                # Mark Youden threshold
                                if _sc in _tv_youden:
                                    _yt, _ = _tv_youden[_sc]
                                    _cls_y = classification_metrics_at_threshold(
                                        _sub["morte_30d"].values, _sub[_sc].values, _yt
                                    )
                                    _fig_roc.add_trace(go.Scatter(
                                        x=[1 - _cls_y["Specificity"]], y=[_cls_y["Sensitivity"]],
                                        mode="markers",
                                        marker=dict(symbol="diamond", size=10, color=_model_color(_label)),
                                        name=f"{_label} Youden",
                                        showlegend=False,
                                        hovertemplate=(
                                            f"{_label}<br>"
                                            f"Youden threshold={_yt:.3f}<br>"
                                            f"TPR={_cls_y['Sensitivity']:.3f}<br>"
                                            f"FPR={1-_cls_y['Specificity']:.3f}<extra></extra>"
                                        ),
                                    ))
                        _fig_roc.update_layout(
                            xaxis_title="1 − Specificity (FPR)",
                            yaxis_title=tr("Sensitivity (TPR)", "Sensibilidade (TPR)"),
                            title=tr("ROC Curves — Temporal Validation", "Curvas ROC — Validação Temporal"),
                            legend=dict(x=0.6, y=0.05),
                            height=420,
                        )
                        st.plotly_chart(_fig_roc, width="stretch")
                        st.caption(tr(
                            "× marker = locked threshold; ◆ marker = Youden's J optimal (exploratory).",
                            "Marcador × = limiar bloqueado; ◆ = ótimo de Youden (exploratório).",
                        ))

                    # 8.2 Precision-Recall curves
                    with st.expander(tr("Precision-Recall Curves", "Curvas Precisão-Recall"), expanded=False):
                        _tv_prevalence = float(_tv_data["morte_30d"].mean()) if _tv_n > 0 else 0.0
                        _fig_pr = go.Figure()
                        _fig_pr.add_shape(type="line", x0=0, y0=_tv_prevalence, x1=1, y1=_tv_prevalence,
                                          line=dict(dash="dash", color="gray", width=1))
                        for _sc in _tv_score_cols:
                            _sub = _tv_data[["morte_30d", _sc]].dropna()
                            if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                                _prec, _rec, _ = _pr_curve_fn(_sub["morte_30d"].values, _sub[_sc].values)
                                _auprc_val = float(_auprc_fn(_sub["morte_30d"].values, _sub[_sc].values))
                                _label = _tv_rename.get(_sc, _sc)
                                _fig_pr.add_trace(go.Scatter(
                                    x=_rec, y=_prec, mode="lines",
                                    name=f"{_label} (AUPRC={_auprc_val:.3f})",
                                    line=dict(color=_model_color(_label), width=2),
                                ))
                        _fig_pr.update_layout(
                            xaxis_title=tr("Recall (Sensitivity)", "Recall (Sensibilidade)"),
                            yaxis_title=tr("Precision (PPV)", "Precisão (PPV)"),
                            title=tr("Precision-Recall Curves — Temporal Validation", "Curvas Precisão-Recall — Validação Temporal"),
                            height=420,
                        )
                        st.plotly_chart(_fig_pr, width="stretch")
                        st.caption(tr(
                            f"Dashed baseline = prevalence ({_tv_prevalence:.1%}).",
                            f"Linha tracejada = prevalência ({_tv_prevalence:.1%}).",
                        ))

                    # 8.3 Calibration curves — interactive with bin selector and Wilson CI
                    with st.expander(tr("Calibration Curves", "Curvas de Calibração"), expanded=True):
                        _cal_n_bins = st.select_slider(
                            tr("Number of bins", "Número de bins"),
                            options=[5, 8, 10, 15, 20],
                            value=10,
                            key="_tv_cal_bins",
                        )
                        _fig_cal = go.Figure()
                        _fig_cal.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                           line=dict(dash="dash", color="gray", width=1))
                        for _sc in _tv_score_cols:
                            _sub = _tv_data[["morte_30d", _sc]].dropna()
                            if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                                _bins_df = calibration_bins_detail(
                                    _sub["morte_30d"].values, _sub[_sc].values,
                                    n_bins=_cal_n_bins, strategy="quantile",
                                )
                                _label = _tv_rename.get(_sc, _sc)
                                _fig_cal.add_trace(go.Scatter(
                                    x=_bins_df["Mean_Predicted"],
                                    y=_bins_df["Obs_Frequency"],
                                    error_y=dict(
                                        type="data",
                                        symmetric=False,
                                        array=(_bins_df["CI_upper"] - _bins_df["Obs_Frequency"]).tolist(),
                                        arrayminus=(_bins_df["Obs_Frequency"] - _bins_df["CI_lower"]).tolist(),
                                        color=_model_color(_label),
                                    ),
                                    mode="lines+markers",
                                    name=_label,
                                    line=dict(color=_model_color(_label), width=2),
                                    marker=dict(size=7),
                                    customdata=_bins_df[["N", "CI_lower", "CI_upper"]].values,
                                    hovertemplate=(
                                        f"{_label}<br>"
                                        "Predicted: %{x:.3f}<br>"
                                        "Observed: %{y:.3f}<br>"
                                        "N: %{customdata[0]}<br>"
                                        "CI 95%%: [%{customdata[1]:.3f}, %{customdata[2]:.3f}]<extra></extra>"
                                    ),
                                ))
                        _fig_cal.update_layout(
                            xaxis_title=tr("Mean predicted probability", "Probabilidade predita média"),
                            yaxis_title=tr("Observed frequency", "Frequência observada"),
                            title=tr("Calibration Curves — Temporal Validation", "Curvas de Calibração — Validação Temporal"),
                            height=420,
                        )
                        st.plotly_chart(_fig_cal, width="stretch")
                        st.caption(tr(
                            "Error bars = Wilson 95% CI per bin. Quantile binning (equal-size bins).",
                            "Barras de erro = IC 95% de Wilson por bin. Binagem por quantil (bins de tamanho igual).",
                        ))

                    # 8.4 Distribution of predicted probabilities
                    with st.expander(tr("Risk Score Distribution", "Distribuição dos Escores de Risco"), expanded=False):
                        _col_log, _col_outcome = st.columns(2)
                        _dist_log = _col_log.checkbox(tr("Log scale", "Escala log"), value=False, key="_tv_dist_log")
                        _dist_by_outcome = _col_outcome.checkbox(
                            tr("Split by outcome", "Separar por desfecho"), value=False, key="_tv_dist_outcome"
                        )
                        _fig_dist = go.Figure()
                        for _sc in _tv_score_cols:
                            _label = _tv_rename.get(_sc, _sc)
                            _color = _model_color(_label)
                            if _dist_by_outcome:
                                for _out_val, _out_label in [(0, tr("Alive", "Vivo")), (1, tr("Died", "Óbito"))]:
                                    _mask_out = _tv_data["morte_30d"] == _out_val
                                    _vals = _tv_data.loc[_mask_out, _sc].dropna().values
                                    if len(_vals) > 0:
                                        _fig_dist.add_trace(go.Histogram(
                                            x=_vals, name=f"{_label} — {_out_label}",
                                            opacity=0.6, nbinsx=30,
                                            histnorm="probability density",
                                        ))
                            else:
                                _vals = _tv_data[_sc].dropna().values
                                if len(_vals) > 0:
                                    _fig_dist.add_trace(go.Histogram(
                                        x=_vals, name=_label, opacity=0.6,
                                        nbinsx=30, histnorm="probability density",
                                        marker_color=_color,
                                    ))
                        _fig_dist.update_layout(
                            barmode="overlay",
                            xaxis_title=tr("Predicted probability", "Probabilidade predita"),
                            yaxis_title=tr("Density", "Densidade"),
                            yaxis_type="log" if _dist_log else "linear",
                            title=tr("Distribution of Predicted Probabilities", "Distribuição das Probabilidades Preditas"),
                            height=380,
                        )
                        st.plotly_chart(_fig_dist, width="stretch")

                    # 8.5 Decision Curve Analysis
                    with st.expander(tr("Decision Curve Analysis (DCA)", "Análise de Curva de Decisão (DCA)"), expanded=False):
                        _tv_dca_scores = {}
                        _dca_mask = _tv_data["morte_30d"].notna()
                        for _sc in _tv_score_cols:
                            _dca_mask = _dca_mask & _tv_data[_sc].notna()
                        _tv_dca_sub = _tv_data.loc[_dca_mask]
                        for _sc in _tv_score_cols:
                            if len(_tv_dca_sub) >= 10 and _tv_dca_sub["morte_30d"].nunique() >= 2:
                                _tv_dca_scores[_tv_rename.get(_sc, _sc)] = _tv_dca_sub[_sc].values
                        if _tv_dca_scores:
                            _tv_dca_y = _tv_dca_sub["morte_30d"].values
                            _tv_dca_thresholds = np.linspace(0.01, 0.30, 60)
                            _tv_dca_df = decision_curve(_tv_dca_y, _tv_dca_scores, _tv_dca_thresholds)
                            _fig_dca = go.Figure()
                            for _strat in _tv_dca_df["Strategy"].unique():
                                _dca_s = _tv_dca_df[_tv_dca_df["Strategy"] == _strat]
                                _is_ref = _strat in ("Treat all", "Treat none")
                                _fig_dca.add_trace(go.Scatter(
                                    x=_dca_s["Threshold"], y=_dca_s["Net Benefit"],
                                    mode="lines",
                                    name=_strat,
                                    line=dict(
                                        dash="dot" if _is_ref else "solid",
                                        color=_model_color(_strat) if not _is_ref else ("gray" if _strat == "Treat none" else "black"),
                                        width=1.5 if _is_ref else 2.5,
                                    ),
                                    opacity=0.6 if _is_ref else 1.0,
                                ))
                            # Threshold markers for 2%, 5%, 8%, 10%
                            for _t_mark in [0.02, 0.05, 0.08, 0.10]:
                                _fig_dca.add_vline(
                                    x=_t_mark, line_dash="dash", line_color="rgba(100,100,100,0.4)",
                                    annotation_text=f"{_t_mark:.0%}",
                                    annotation_position="top",
                                )
                            _fig_dca.update_layout(
                                xaxis_title=tr("Threshold probability", "Probabilidade limiar"),
                                yaxis_title=tr("Net benefit", "Benefício líquido"),
                                title=tr("Decision Curve Analysis — Temporal Validation", "Análise de Curva de Decisão — Validação Temporal"),
                                yaxis=dict(range=[-0.05, None]),
                                height=420,
                            )
                            st.plotly_chart(_fig_dca, width="stretch")
                        else:
                            st.info(tr("Insufficient data for DCA.", "Dados insuficientes para DCA."))

                    # 8.6 Observed vs Predicted by risk decile
                    with st.expander(tr("Observed vs Predicted by Decile", "Observado vs Predito por Decil"), expanded=False):
                        _fig_decile = go.Figure()
                        _fig_decile.add_shape(type="line", x0=0, y0=0, x1=0.5, y1=0.5,
                                              line=dict(dash="dash", color="gray", width=1))
                        _has_decile = False
                        for _sc in _tv_score_cols:
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
                                    n=(_sc, "count"),
                                ).reset_index()
                                _label = _tv_rename.get(_sc, _sc)
                                _fig_decile.add_trace(go.Scatter(
                                    x=_grp["predicted"], y=_grp["observed"],
                                    mode="markers",
                                    name=_label,
                                    marker=dict(color=_model_color(_label), size=10),
                                    customdata=_grp["n"].values,
                                    hovertemplate=(
                                        f"{_label}<br>"
                                        "Mean pred: %{x:.3f}<br>"
                                        "Observed: %{y:.3f}<br>"
                                        "N: %{customdata}<extra></extra>"
                                    ),
                                ))
                                _has_decile = True
                        if _has_decile:
                            _fig_decile.update_layout(
                                xaxis_title=tr("Mean predicted probability", "Probabilidade predita média"),
                                yaxis_title=tr("Observed mortality", "Mortalidade observada"),
                                title=tr("Observed vs Predicted by Decile", "Observado vs Predito por Decil"),
                                height=400,
                            )
                            st.plotly_chart(_fig_decile, width="stretch")
                        else:
                            st.info(tr("At least 20 rows required per score.", "Mínimo de 20 linhas por escore necessárias."))

                    # 8.7 Confusion matrices at locked threshold (Plotly heatmaps)
                    with st.expander(
                        tr(f"Confusion Matrices (threshold={_tv_locked_threshold:.0%})",
                           f"Matrizes de Confusão (limiar={_tv_locked_threshold:.0%})"),
                        expanded=False,
                    ):
                        _cm_cols = st.columns(max(1, len(_tv_score_cols)))
                        for _ax_i, _sc in enumerate(_tv_score_cols):
                            _sub = _tv_data[["morte_30d", _sc]].dropna()
                            if len(_sub) >= 10 and _sub["morte_30d"].nunique() >= 2:
                                _cls = classification_metrics_at_threshold(
                                    _sub["morte_30d"].values, _sub[_sc].values, _tv_locked_threshold
                                )
                                _z = [[_cls["TN"], _cls["FP"]], [_cls["FN"], _cls["TP"]]]
                                _x_labels = [tr("Predicted −", "Predito −"), tr("Predicted +", "Predito +")]
                                _y_labels = [tr("Actual −", "Real −"), tr("Actual +", "Real +")]
                                _label = _tv_rename.get(_sc, _sc)
                                _fig_cm_i = go.Figure(go.Heatmap(
                                    z=_z, x=_x_labels, y=_y_labels,
                                    colorscale="Blues", showscale=False,
                                    text=[[str(v) for v in row] for row in _z],
                                    texttemplate="%{text}",
                                    textfont=dict(size=18),
                                ))
                                _fig_cm_i.update_layout(
                                    title=f"{_label} @ {_tv_locked_threshold:.0%}",
                                    height=280, margin=dict(l=50, r=20, t=50, b=40),
                                )
                                with _cm_cols[_ax_i]:
                                    st.plotly_chart(_fig_cm_i, width="stretch")

                    # ── 9. Threshold Analysis ──
                    st.divider()
                    st.markdown(tr("### Threshold Analysis", "### Análise de Limiar"))
                    st.caption(tr(
                        "Fixed thresholds: 2%, 5%, 8%, 10%. "
                        "Youden's J optimal threshold is data-driven and **exploratory** — "
                        "do not use it as the operative locked threshold.",
                        "Limiares fixos: 2%, 5%, 8%, 10%. "
                        "O limiar ótimo de Youden é orientado pelos dados e **exploratório** — "
                        "não deve ser usado como limiar operacional bloqueado.",
                    ))
                    for _sc in _tv_score_cols:
                        if _sc in _tv_thresh_tables:
                            _label = _tv_rename.get(_sc, _sc)
                            with st.expander(f"{_label} — {tr('threshold sweep', 'varredura de limiar')}", expanded=False):
                                _tbl = _tv_thresh_tables[_sc].copy()
                                _tbl["Threshold"] = _tbl["Threshold"].map(lambda v: f"{v:.0%}")
                                _tbl_disp = _tbl.rename(columns={
                                    "Sensitivity": tr("Sensitivity", "Sensibilidade"),
                                    "Specificity": tr("Specificity", "Especificidade"),
                                    "PPV": "PPV",
                                    "NPV": "NPV",
                                    "TP": "TP",
                                    "FP": "FP",
                                    "TN": "TN",
                                    "FN": "FN",
                                    "N_Flagged": tr("N Flagged", "N Sinalizados"),
                                    "Positives_per_1000": tr("Per 1000", "Por 1000"),
                                    "Flag_Rate_pct": tr("Flag Rate %", "Taxa sinalizados %"),
                                })
                                # Put TP/FP/TN/FN right after threshold for readability
                                _ordered_cols = [
                                    "Threshold",
                                    "TP", "FP", "TN", "FN",
                                    tr("Sensitivity", "Sensibilidade"),
                                    tr("Specificity", "Especificidade"),
                                    "PPV", "NPV",
                                    tr("N Flagged", "N Sinalizados"),
                                    tr("Per 1000", "Por 1000"),
                                    tr("Flag Rate %", "Taxa sinalizados %"),
                                ]
                                _tbl_disp = _tbl_disp[[c for c in _ordered_cols if c in _tbl_disp.columns]]
                                st.dataframe(_tbl_disp, hide_index=True)
                                if _sc in _tv_youden:
                                    _yt, _yj = _tv_youden[_sc]
                                    st.info(tr(
                                        f"Youden's J optimum (exploratory): threshold = {_yt:.3f} → J = {_yj:.3f}",
                                        f"Ótimo de Youden (exploratório): limiar = {_yt:.3f} → J = {_yj:.3f}",
                                    ))
                                # Sens/Spec sweep chart
                                _full_thr_vals = np.linspace(0.01, 0.50, 100)
                                _sub2 = _tv_data[["morte_30d", _sc]].dropna()
                                if len(_sub2) >= 10 and _sub2["morte_30d"].nunique() >= 2:
                                    _sweep_tbl = threshold_analysis_table(
                                        _sub2["morte_30d"].values, _sub2[_sc].values,
                                        list(_full_thr_vals),
                                    )
                                    _fig_sweep = go.Figure()
                                    _fig_sweep.add_trace(go.Scatter(
                                        x=_sweep_tbl["Threshold"], y=_sweep_tbl["Sensitivity"],
                                        mode="lines", name=tr("Sensitivity", "Sensibilidade"),
                                        line=dict(color="#1f77b4", width=2),
                                    ))
                                    _fig_sweep.add_trace(go.Scatter(
                                        x=_sweep_tbl["Threshold"], y=_sweep_tbl["Specificity"],
                                        mode="lines", name=tr("Specificity", "Especificidade"),
                                        line=dict(color="#ff7f0e", width=2),
                                    ))
                                    _fig_sweep.add_vline(
                                        x=_tv_locked_threshold, line_dash="dash", line_color="gray",
                                        annotation_text=tr(f"Locked {_tv_locked_threshold:.0%}", f"Bloqueado {_tv_locked_threshold:.0%}"),
                                        annotation_position="top right",
                                    )
                                    _fig_sweep.update_layout(
                                        xaxis_title=tr("Threshold", "Limiar"),
                                        yaxis_title=tr("Metric value", "Valor da métrica"),
                                        title=f"{_label} — {tr('Sensitivity / Specificity Trade-off', 'Trade-off Sensibilidade / Especificidade')}",
                                        height=320,
                                    )
                                    st.plotly_chart(_fig_sweep, width="stretch")

                    # ── Block 4: Supplementary / Exploratory Analyses ─────
                    _tv_has_common = (
                        _tv_common_perf is not None and not _tv_common_perf.empty
                    )
                    st.divider()
                    st.markdown(tr(
                        "### Supplementary Analyses, Predictions & Export",
                        "### Análises Suplementares, Predições e Exportação",
                    ))
                    if _tv_has_common:
                        st.caption(tr(
                            "Common cohort and post-hoc recalibration are exploratory — they do not "
                            "replace the primary result above. Case-level predictions and export "
                            "options follow below.",
                            "Coorte comum e recalibração pós-hoc são exploratórias — não substituem "
                            "o resultado principal acima. Predições por paciente e opções de "
                            "exportação seguem abaixo.",
                        ))
                    else:
                        st.caption(tr(
                            "Post-hoc recalibration is exploratory and does not replace the primary "
                            "result above. Case-level predictions and export options follow below.",
                            "A recalibração pós-hoc é exploratória e não substitui o resultado "
                            "principal acima. Predições por paciente e opções de exportação seguem abaixo.",
                        ))

                    # ── 10. Common Cohort Comparison ──
                    if _tv_common_perf is not None and not _tv_common_perf.empty:
                        with st.expander(
                            tr(
                                f"Common Cohort — All models on STS-available subset (n={_tv_n_common})",
                                f"Coorte Comum — Todos os modelos no subconjunto com STS disponível (n={_tv_n_common})",
                            ),
                            expanded=False,
                        ):
                            st.caption(tr(
                                "Metrics below use only rows where **all three** models "
                                "(AI Risk, EuroSCORE II, and STS Score) have non-missing predictions. "
                                "This 'common cohort' is strictly the STS-available subset; "
                                "it will typically be smaller than the full validation cohort. "
                                "Use this to compare all three scores on identical patients.",
                                "As métricas abaixo usam apenas as linhas em que os **três** modelos "
                                "(AI Risk, EuroSCORE II e STS Score) têm predições disponíveis. "
                                "Essa 'coorte comum' corresponde ao subconjunto com STS disponível; "
                                "tende a ser menor que a coorte completa de validação. "
                                "Use para comparar os três escores nos mesmos pacientes.",
                            ))
                            st.dataframe(_tv_common_perf, hide_index=True)

                    # ── 11. Post-hoc Recalibration (exploratory) ──
                    with st.expander(
                        tr(
                            "Post-hoc Recalibration — Exploratory analysis only",
                            "Recalibração Pós-hoc — Apenas análise exploratória",
                        ),
                        expanded=False,
                    ):
                        st.warning(tr(
                            "**Exploratory only.** Recalibration is applied after-the-fact to the "
                            "validation cohort. Results are not clinically validated and must **not** "
                            "be used to report model performance or guide clinical decisions. "
                            "The locked model and threshold remain unchanged.",
                            "**Apenas exploratório.** A recalibração é aplicada post-hoc à "
                            "coorte de validação. Os resultados não foram validados clinicamente "
                            "e **não devem** ser usados para reportar desempenho do modelo ou "
                            "orientar decisões clínicas. O modelo bloqueado e o limiar permanecem inalterados.",
                        ))
                        _recal_score_opts = [_tv_rename.get(_sc, _sc) for _sc in _tv_score_cols]
                        _recal_score_sel = st.selectbox(
                            tr("Score to recalibrate", "Escore para recalibrar"),
                            _recal_score_opts,
                            key="_tv_recal_score",
                        )
                        _recal_method = st.radio(
                            tr("Method", "Método"),
                            [
                                tr("Intercept-only (slope=1)", "Apenas intercepto (slope=1)"),
                                tr("Intercept + slope (logistic)", "Intercepto + slope (logístico)"),
                                tr("Isotonic (non-parametric)", "Isotônica (não-paramétrica)"),
                            ],
                            horizontal=True,
                            key="_tv_recal_method",
                        )
                        _recal_sc_raw = next(
                            (_sc for _sc in _tv_score_cols if _tv_rename.get(_sc, _sc) == _recal_score_sel),
                            _tv_score_cols[0] if _tv_score_cols else None,
                        )
                        if _recal_sc_raw is not None:
                            _rsub = _tv_data[["morte_30d", _recal_sc_raw]].dropna()
                            if len(_rsub) >= 10 and _rsub["morte_30d"].nunique() >= 2:
                                _ry = _rsub["morte_30d"].values
                                _rp = _rsub[_recal_sc_raw].values
                                if "slope=1" in _recal_method or "slope=1)" in _recal_method:
                                    _rcal = recalibrate_intercept_only(_ry, _rp)
                                elif "logistic" in _recal_method or "logístico" in _recal_method:
                                    _rcal = recalibrate_logistic(_ry, _rp)
                                else:
                                    _rcal = recalibrate_isotonic(_ry, _rp)
                                _rp_new = _rcal["recalibrated_probs"]
                                # Before/after summary table
                                _rcal_summary = {
                                    tr("Metric", "Métrica"): [
                                        tr("Brier score", "Escore de Brier"),
                                        tr("Calibration intercept", "Intercepto de calibração"),
                                        tr("Calibration slope", "Inclinação de calibração"),
                                    ],
                                    tr("Before", "Antes"): [
                                        f"{_rcal['brier_before']:.4f}",
                                        f"{_rcal.get('cal_intercept_before', float('nan')):.3f}",
                                        f"{_rcal.get('cal_slope_before', float('nan')):.3f}",
                                    ],
                                    tr("After", "Depois"): [
                                        f"{_rcal['brier_after']:.4f}",
                                        f"{_rcal.get('cal_intercept_after', float('nan')):.3f}",
                                        f"{_rcal.get('cal_slope_after', float('nan')):.3f}",
                                    ],
                                }
                                st.dataframe(pd.DataFrame(_rcal_summary), hide_index=True)
                                # Before/after calibration plot
                                _fig_rcal = go.Figure()
                                _fig_rcal.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                                    line=dict(dash="dash", color="gray", width=1))
                                for _rp_vals, _rp_name, _rp_color in [
                                    (_rp, tr("Original", "Original"), "#aaaaaa"),
                                    (_rp_new, tr("Recalibrated", "Recalibrado"), "#1f77b4"),
                                ]:
                                    _rbins = calibration_bins_detail(_ry, _rp_vals, n_bins=10, strategy="quantile")
                                    _fig_rcal.add_trace(go.Scatter(
                                        x=_rbins["Mean_Predicted"], y=_rbins["Obs_Frequency"],
                                        mode="lines+markers", name=_rp_name,
                                        line=dict(color=_rp_color, width=2),
                                    ))
                                _fig_rcal.update_layout(
                                    xaxis_title=tr("Mean predicted probability", "Probabilidade predita média"),
                                    yaxis_title=tr("Observed frequency", "Frequência observada"),
                                    title=tr("Calibration: Before vs After Recalibration", "Calibração: Antes vs Depois da Recalibração"),
                                    height=350,
                                )
                                st.plotly_chart(_fig_rcal, width="stretch")
                            else:
                                st.info(tr("Insufficient data for recalibration.", "Dados insuficientes para recalibração."))

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
                    if _tv_sts_availability != "no_sts" and "sts_score" in _tv_data.columns:
                        _tv_case_cols.append("sts_score")
                    _tv_case_cols.extend(["class_ia", "class_euro"])
                    if _tv_sts_availability != "no_sts" and "class_sts" in _tv_data.columns:
                        _tv_case_cols.append("class_sts")
                    _tv_case_cols.append("_completeness")
                    _tv_case_cols = [c for c in _tv_case_cols if c in _tv_data.columns]
                    _tv_export_case_cols = list(_tv_case_cols)
                    if "sts_score" in _tv_data.columns and "sts_score" not in _tv_export_case_cols:
                        _tv_export_case_cols.append("sts_score")
                    if "class_sts" in _tv_data.columns and "class_sts" not in _tv_export_case_cols:
                        _tv_export_case_cols.append("class_sts")
                    # case_uid is a metadata identifier (not a feature) — include in exports for traceability
                    if "case_uid" in _tv_data.columns and "case_uid" not in _tv_export_case_cols:
                        _tv_export_case_cols.insert(0, "case_uid")

                    _tv_case_df = _tv_data[_tv_case_cols].copy()
                    # Rename for display
                    # Phase: nomenclature consistency — "STS Score" everywhere.
                    _tv_case_rename = {
                        "morte_30d": tr("Outcome", "Desfecho"),
                        "ia_risk": "AI Risk",
                        "euroscore_calc": "EuroSCORE II",
                        "sts_score": _tv_sts_score_label,
                        "class_ia": tr("AI Risk class", "Classe AI Risk"),
                        "class_euro": tr("EuroSCORE II class", "Classe EuroSCORE II"),
                        "class_sts": tr("STS Score class", "Classe STS Score"),
                        "_completeness": tr("Completeness", "Completude"),
                    }
                    _tv_case_df = _tv_case_df.rename(columns=_tv_case_rename)

                    # Format probabilities as percentages
                    for _pc in ["AI Risk", "EuroSCORE II", _tv_sts_score_label]:
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

                    # Build markdown report (primary)
                    def _build_tv_exploratory_summaries():
                        _expl_recal = build_exploratory_recalibration_summary(
                            _tv_exploratory_recal, _tv_rename, language,
                        )
                        _expl_thresh = build_exploratory_threshold_summary(
                            _tv_exploratory_thresh_tables, _tv_locked_threshold,
                            _tv_youden, _tv_rename, language,
                        )
                        return _expl_recal, _expl_thresh

                    def _build_tv_markdown_bytes() -> bytes:
                        _sts_acct_for_report = {
                            "n_total":         _tv_cohort_summary.get("n_total", 0),
                            "n_not_supported": _tv_n_not_supported,
                            "n_uncertain":     _tv_n_uncertain,
                            "n_supported":     _tv_sts_n_eligible,
                            "n_final_usable":  _tv_sts_n_score,
                        } if _tv_sts_n_eligible > 0 else None
                        _tv_md = build_temporal_validation_summary(
                            _tv_cohort_summary, _tv_perf, _tv_pairwise, _tv_calib_df,
                            _tv_risk_cat, _tv_meta, _tv_locked_threshold, language,
                            sts_availability=_tv_sts_availability_note,
                            normalization_report=_tv_norm_report,
                            sts_accounting=_sts_acct_for_report,
                            common_cohort_perf=_tv_common_perf,
                            n_common=_tv_n_common,
                        )
                        _expl_recal_sum, _expl_thresh_sum = _build_tv_exploratory_summaries()
                        _expl_md = build_exploratory_temporal_validation_section(
                            _expl_recal_sum, _expl_thresh_sum, language,
                        )
                        if _expl_md:
                            _tv_md = _tv_md + "\n" + _expl_md
                        return _tv_md.encode("utf-8")

                    def _build_tv_xlsx_bytes() -> bytes:
                        _expl_recal_sum, _expl_thresh_sum = _build_tv_exploratory_summaries()
                        _tv_xlsx_buf = BytesIO()
                        with pd.ExcelWriter(_tv_xlsx_buf, engine="openpyxl") as _tv_writer:
                            # Manifest sheet — written first so the auditor sees
                            # the bundle/version/threshold provenance as the
                            # opening tab of the workbook.
                            _tv_manifest = build_export_manifest(
                                export_kind="temporal_validation",
                                model_version=MODEL_VERSION,
                                active_model_name=bundle_info.get("active_model_name") or forced_model,
                                threshold_mode=("youden" if _tv_use_youden else "clinical_fixed"),
                                threshold_value=float(_tv_locked_threshold),
                                dataset_fingerprint=bundle_info.get("dataset_fingerprint"),
                                bundle_fingerprint=bundle_info.get("bundle_fingerprint"),
                                bundle_saved_at=bundle_info.get("saved_at"),
                                training_source=bundle_info.get("training_source"),
                                current_analysis_file=getattr(_tv_file, "name", None),
                                extra={
                                    "n_total": _tv_cohort_summary.get("n_total"),
                                    "n_events": _tv_events,
                                    "language": language,
                                },
                            )
                            pd.DataFrame(
                                [{"Property": k, "Value": v} for k, v in _tv_manifest.items() if k != "extra"]
                                + [{"Property": f"extra.{k}", "Value": v} for k, v in (_tv_manifest.get("extra") or {}).items()]
                            ).to_excel(_tv_writer, sheet_name="manifest", index=False)

                            _cs_export = pd.DataFrame([
                                {"Property": k, "Value": v} for k, v in _tv_cohort_summary.items()
                            ])
                            _cs_export.to_excel(_tv_writer, sheet_name="cohort_summary", index=False)
                            if not _tv_perf.empty:
                                _tv_perf.to_excel(_tv_writer, sheet_name="performance", index=False)
                            if not _tv_pairwise.empty:
                                _tv_pairwise.to_excel(_tv_writer, sheet_name="pairwise_comparison", index=False)
                            if not _tv_risk_cat.empty:
                                _tv_risk_cat.to_excel(_tv_writer, sheet_name="risk_categories", index=False)
                            if not _tv_calib_df.empty:
                                _tv_calib_df.to_excel(_tv_writer, sheet_name="calibration", index=False)
                            _tv_case_export = _tv_data[_tv_export_case_cols].copy()
                            _tv_case_export.to_excel(_tv_writer, sheet_name="case_level_predictions", index=False)
                            if _tv_common_perf is not None and not _tv_common_perf.empty:
                                _tv_common_perf.to_excel(_tv_writer, sheet_name="common_cohort", index=False)
                            if _tv_norm_report is not None:
                                pd.DataFrame(_tv_norm_report.to_export_rows()).to_excel(
                                    _tv_writer, sheet_name="Normalization_Summary", index=False
                                )
                            if _expl_recal_sum.get("available") and not _expl_recal_sum["table"].empty:
                                _expl_recal_sum["table"].to_excel(
                                    _tv_writer, sheet_name="Exploratory_Recalibration", index=False
                                )
                            if _expl_thresh_sum.get("available") and not _expl_thresh_sum["table"].empty:
                                _expl_thresh_sum["table"].to_excel(
                                    _tv_writer, sheet_name="Exploratory_Thresholds", index=False
                                )
                        return _tv_xlsx_buf.getvalue()

                    def _build_tv_csv_bytes() -> bytes:
                        return _tv_data[_tv_export_case_cols].copy().to_csv(index=False).encode("utf-8")

                    def _build_tv_pdf_bytes() -> bytes:
                        return statistical_summary_to_pdf(_build_tv_markdown_bytes().decode("utf-8"))

                    def _lazy_tv_export_button(label: str, filename: str, mime: str, kind: str, build_fn):
                        _exports = st.session_state.get("_tv_exports", {})
                        if _exports.get("sig") != _tv_context_sig:
                            _exports = {"sig": _tv_context_sig}
                            st.session_state["_tv_exports"] = _exports

                        _slot = st.empty()
                        if kind in _exports and _exports[kind]:
                            _slot.download_button(
                                label,
                                data=_exports[kind],
                                file_name=filename,
                                mime=mime,
                                key=f"dl_tv_{kind}",
                                on_click="ignore",
                            )
                            return

                        if _slot.button(label, key=f"prepare_tv_{kind}"):
                            with st.spinner(tr("Preparing export...", "Preparando export...")):
                                _exports[kind] = build_fn()
                                st.session_state["_tv_exports"] = _exports
                            if _exports[kind]:
                                _slot.download_button(
                                    label,
                                    data=_exports[kind],
                                    file_name=filename,
                                    mime=mime,
                                    key=f"dl_tv_{kind}",
                                    on_click="ignore",
                                )
                                st.caption(tr(
                                    "File ready. Click the same button to download.",
                                    "Arquivo pronto. Clique no mesmo botão para baixar.",
                                ))
                            else:
                                st.caption(tr("Export unavailable for this environment.", "Export indisponível neste ambiente."))

                    # Download buttons — rendered here for both fresh runs and
                    # session-cache restores so the user never has to recompute
                    # just to download.
                    import datetime as _tv_dt
                    _tv_date_tag = _tv_dt.datetime.now().strftime("%Y%m%d")
                    _tv_export_base = f"ai_risk_temporal_{MODEL_VERSION}_{_tv_date_tag}"

                    _ex_c1, _ex_c2, _ex_c3, _ex_c4 = st.columns(4)
                    with _ex_c1:
                        _lazy_tv_export_button(
                            tr("XLSX summary", "XLSX resumo"),
                            f"{_tv_export_base}_summary.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            "xlsx",
                            _build_tv_xlsx_bytes,
                        )
                    with _ex_c2:
                        _lazy_tv_export_button(
                            tr("CSV predictions", "CSV predições"),
                            f"{_tv_export_base}_predictions.csv",
                            "text/csv",
                            "csv",
                            _build_tv_csv_bytes,
                        )
                    with _ex_c3:
                        _lazy_tv_export_button(
                            tr("PDF report", "Relatório PDF"),
                            f"{_tv_export_base}_report.pdf",
                            "application/pdf",
                            "pdf",
                            _build_tv_pdf_bytes,
                        )
                    with _ex_c4:
                        _lazy_tv_export_button(
                            tr("Markdown", "Markdown"),
                            f"{_tv_export_base}_report.md",
                            "text/markdown",
                            "md",
                            _build_tv_markdown_bytes,
                        )
                    if _tv_run:
                        # Log audit — only on fresh computation, not cache restores.
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
                                "sts_availability": _tv_sts_availability,
                                "sts_n_eligible": _tv_sts_n_eligible,
                                "sts_n_score": _tv_sts_n_score,
                            },
                        )
