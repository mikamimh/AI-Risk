"""Batch & Export tab — extracted from app.py (tab index 4).

Pure extraction: all logic, text, i18n, and UI elements are identical to the
original inline code.  The only structural change is that shared state is
accessed through ``ctx`` (:class:`tabs.TabContext`) instead of bare local
variables in ``app.py``.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st

from ai_risk_inference import (
    _get_numeric_columns_from_pipeline,
    _patient_identifier_from_row,
    _run_ai_risk_inference_row,
)
from euroscore import euroscore_from_inputs
from export_helpers import statistical_summary_to_pdf
from model_metadata import log_analysis
from risk_data import FLAT_ALIAS_TO_APP_COLUMNS
from stats_compare import class_risk
from sts_calculator import calculate_sts_batch

if TYPE_CHECKING:
    from tabs import TabContext


def render(ctx: TabContext) -> None:  # noqa: C901 – extracted verbatim, complexity preserved
    """Render the Batch & Export tab (tab index 4)."""
    tr = ctx.tr
    artifacts = ctx.artifacts
    df = ctx.df
    prepared = ctx.prepared
    forced_model = ctx.forced_model
    language = ctx.language
    _default_threshold = ctx.default_threshold
    MODEL_VERSION = ctx.model_version
    HAS_STS = ctx.has_sts

    # Alias helpers
    _csv_download_btn = ctx.csv_download_btn
    _bytes_download_btn = ctx.bytes_download_btn
    _txt_download_btn = ctx.txt_download_btn
    _update_phase = ctx.update_phase
    _sts_score_patient_ids = ctx.sts_score_patient_ids
    general_table_column_config = ctx.general_table_column_config

    # ── Begin original tab body (verbatim) ───────────────────────────────

    st.subheader(tr("Batch & Export", "Lote e Exportação"))
    st.caption(tr(
        "Research exports for the current cohort and batch prediction for new patient files.",
        "Exports de pesquisa da coorte atual e predição em lote para arquivos de novos pacientes.",
    ))

    export_df = df.copy()

    # Add OOF predictions from ALL models (for research)
    for _model_name, _oof_probs in artifacts.oof_predictions.items():
        export_df[f"oof_{_model_name}"] = _oof_probs

    export_df["classe_ia"] = export_df["ia_risk_oof"].map(class_risk)
    export_df["classe_euro"] = export_df["euroscore_calc"].map(class_risk)
    export_df["classe_sts"] = export_df["sts_score"].map(lambda x: class_risk(x) if pd.notna(x) else np.nan)

    _n_current = len(export_df)
    _n_ai_current = int(export_df["ia_risk_oof"].notna().sum()) if "ia_risk_oof" in export_df.columns else 0
    _n_sts_current = int(export_df["sts_score"].notna().sum()) if "sts_score" in export_df.columns else 0
    _n_euro_current = int(export_df["euroscore_calc"].notna().sum()) if "euroscore_calc" in export_df.columns else 0
    _top1, _top2, _top3, _top4 = st.columns(4)
    _top1.metric(tr("Current cohort", "Coorte atual"), f"{_n_current}", border=True)
    _top2.metric(tr("AI Risk available", "AI Risk disponível"), f"{_n_ai_current}", border=True)
    _top3.metric(tr("EuroSCORE II available", "EuroSCORE II disponível"), f"{_n_euro_current}", border=True)
    _top4.metric(tr("STS available", "STS disponível"), f"{_n_sts_current}", border=True)

    st.divider()
    st.markdown(tr("### Research Export", "### Exportação de Pesquisa"))
    st.caption(tr(
        "Export the active research cohort with app-calculated AI Risk, EuroSCORE II, STS Score, risk classes, and optional OOF predictions from all AI models.",
        "Exporte a coorte de pesquisa ativa com AI Risk, EuroSCORE II, STS Score, classes de risco e predições OOF opcionais de todos os modelos de IA.",
    ))

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
    st.markdown(tr("**Preview**", "**Prévia**"))
    st.dataframe(
        export_df[cols_show].head(25),
        width="stretch",
        column_config=general_table_column_config("export"),
    )
    with st.expander(tr("Show full research export table", "Mostrar tabela completa do export de pesquisa"), expanded=False):
        st.dataframe(export_df[cols_show], width="stretch", column_config=general_table_column_config("export"))

    # Download always includes all models
    st.markdown(tr("**Downloads**", "**Downloads**"))
    _rx1, _rx2 = st.columns(2)
    with _rx1:
        _csv_download_btn(export_df, "ia_risk_resultados.csv", tr("Download results (CSV)", "Baixar resultados (CSV)"))
    _xlsx_export_buf = BytesIO()
    export_df.to_excel(_xlsx_export_buf, index=False, engine="openpyxl")
    with _rx2:
        _bytes_download_btn(
            _xlsx_export_buf.getvalue(),
            "ia_risk_resultados.xlsx",
            tr("Download results (XLSX)", "Baixar resultados (XLSX)"),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_export_xlsx",
        )

    st.caption(
        tr(
            "Note: All scores are calculated by the app — not read from the input file. EuroSCORE II is computed from the published logistic equation (Nashef et al., 2012). STS Score is obtained via automated query to the STS Score web calculator.",
            "Nota: Todos os escores são calculados pelo app — não lidos do arquivo de entrada. EuroSCORE II é calculado pela equação logística publicada (Nashef et al., 2012). STS Score é obtido via consulta automatizada à calculadora web do STS Score.",
        )
    )

    # --- Batch prediction for NEW patients ---
    st.divider()
    st.subheader(tr("Batch Prediction for New Patients", "Predição em Lote de Novos Pacientes"))
    st.caption(
        tr(
            "Upload a CSV or Excel file with the same clinical variables used in training. "
            "Each row will receive AI Risk, EuroSCORE II, and STS Score predictions. "
            "Outcome column (morte_30d) is NOT required.",
            "Faça upload de um arquivo CSV ou Excel com as mesmas variáveis clínicas usadas no treinamento. "
            "Cada linha receberá predições de AI Risk, EuroSCORE II e STS Score. "
            "A coluna de desfecho (morte_30d) NÃO é necessária.",
        )
    )
    with st.expander(tr("Methodological note", "Nota metodológica"), expanded=False):
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
            _compat_cols = st.columns(4)
            _compat_cols[0].metric(tr("Rows loaded", "Linhas carregadas"), f"{len(new_df)}", border=True)
            _compat_cols[1].metric(tr("Input columns", "Colunas no arquivo"), f"{len(new_df.columns)}", border=True)
            _compat_cols[2].metric(
                tr("Matched features", "Variáveis encontradas"),
                f"{len(matched_cols)}/{len(artifacts.feature_columns)}",
                border=True,
            )
            _compat_cols[3].metric(tr("Missing features", "Variáveis ausentes"), f"{len(missing_cols)}", border=True)
            st.caption(
                tr(
                    "Missing model features will be imputed by the model pipeline.",
                    "Variáveis do modelo ausentes serão imputadas pelo pipeline do modelo.",
                )
            )
            if missing_cols:
                with st.expander(tr("Show missing features", "Ver variáveis ausentes")):
                    st.write(", ".join(missing_cols))

            st.markdown(tr("**Prediction options**", "**Opções de predição**"))
            _show_all_models = st.checkbox(
                tr("Show predictions from all AI models", "Mostrar predições de todos os modelos de IA"),
                value=False,
                key="batch_show_all_models",
            )
            _include_sts = st.checkbox(
                tr("Include STS Score (requires internet, ~1 min per 50 patients)", "Incluir STS Score (requer internet, ~1 min a cada 50 pacientes)"),
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
                _batch_local_phase_slot = st.empty()
                _update_phase(_batch_local_phase_slot, 1, 2, tr(
                    "applying AI Risk + EuroSCORE II",
                    "aplicando AI Risk + EuroSCORE II",
                ))
                _progress_bar = st.progress(0, text=tr(
                    f"Computing AI Risk + EuroSCORE: 0/{_n_total}",
                    f"Calculando AI Risk + EuroSCORE: 0/{_n_total}",
                ))
                _n_errors = 0
                _batch_ai_incidents: list = []
                _num_cols_batch = _get_numeric_columns_from_pipeline(artifacts.fitted_models[forced_model])
                for idx, row_data in new_df.iterrows():
                    _i = len(results)
                    form_map = row_data.to_dict()
                    _patient_id_batch = _patient_identifier_from_row(form_map, idx)
                    _infer_batch = _run_ai_risk_inference_row(
                        model_pipeline=artifacts.fitted_models[forced_model],
                        feature_columns=artifacts.feature_columns,
                        reference_df=ref_df,
                        row_dict=form_map,
                        patient_id=_patient_id_batch,
                        numeric_cols=_num_cols_batch,
                        language=language,
                    )
                    if _infer_batch["incident"] is not None:
                        _n_errors += 1
                        _batch_ai_incidents.append(_infer_batch["incident"])
                        results.append({
                            tr("Row", "Linha"): idx + 1,
                            tr("Name", "Nome"): form_map.get("Name", form_map.get("Nome", f"Patient {idx+1}")),
                            tr("Error", "Erro"): _infer_batch["incident"]["reason"],
                        })
                        batch_rows_for_sts.append({})
                    else:
                        ia_prob = _infer_batch["probability"]
                        model_input = _infer_batch["model_input"]
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

                    _pct = (_i + 1) / _n_total
                    _progress_bar.progress(_pct, text=tr(
                        f"Computing AI Risk + EuroSCORE: {_i + 1}/{_n_total}",
                        f"Calculando AI Risk + EuroSCORE: {_i + 1}/{_n_total}",
                    ))

                _progress_bar.progress(1.0, text=tr(
                    f"AI Risk + EuroSCORE complete: {_n_total - _n_errors} OK, {_n_errors} errors",
                    f"AI Risk + EuroSCORE completo: {_n_total - _n_errors} OK, {_n_errors} erros",
                ))
                with st.expander(tr("View AI Risk + EuroSCORE execution details", "Ver detalhes de execução do AI Risk + EuroSCORE"), expanded=False):
                    st.caption(tr(
                        f"Processed: {_n_total} | OK: {_n_total - _n_errors} | Errors: {_n_errors}",
                        f"Processados: {_n_total} | OK: {_n_total - _n_errors} | Erros: {_n_errors}",
                    ))
                    if _batch_ai_incidents:
                        st.caption(tr(
                            f"AI Risk incidents: {len(_batch_ai_incidents)} — see warning below for details.",
                            f"Incidentes de AI Risk: {len(_batch_ai_incidents)} — veja o aviso abaixo para detalhes.",
                        ))

                # Surface per-patient AI Risk incidents (mirrors temporal validation UI).
                if _batch_ai_incidents:
                    st.warning(tr(
                        f"AI Risk inference incidents (batch): {len(_batch_ai_incidents)} patient(s) failed.",
                        f"Incidentes de inferência do AI Risk (lote): {len(_batch_ai_incidents)} paciente(s) falharam.",
                    ))
                    with st.expander(
                        tr(
                            f"AI Risk per-patient incidents (batch) ({len(_batch_ai_incidents)})",
                            f"Incidentes por paciente do AI Risk (lote) ({len(_batch_ai_incidents)})",
                        ),
                        expanded=False,
                    ):
                        st.dataframe(
                            pd.DataFrame(_batch_ai_incidents),
                            width="stretch",
                            hide_index=True,
                        )

                # --- Phase 2: STS Score (routed through Phase 2 cache) ---
                _update_phase(_batch_local_phase_slot, 2, 2, tr(
                    "consolidating results",
                    "consolidando resultados",
                ))
                sts_probs = [np.nan] * len(results)
                if HAS_STS and _include_sts:
                    _batch_local_phase_slot.empty()
                    _sts_phase_slot = st.empty()
                    _sts_progress = st.progress(0, text=tr("STS Score…", "STS Score…"))
                    _sts_last_phase: list = ["", ""]  # [label, detail]
                    try:
                        def _sts_phase_cb(phase_num, phase_total, label, detail=""):
                            _sts_last_phase[0] = label
                            _sts_last_phase[1] = detail
                            try:
                                # Secondary / sub-status line — lighter than the macro phase slot above
                                _sts_phase_slot.caption(tr(
                                    f"↳ STS Score subphase {phase_num}/{phase_total}: {label}",
                                    f"↳ STS Score subfase {phase_num}/{phase_total}: {label}",
                                ))
                            except Exception:
                                pass

                        def _sts_progress_cb(done, total):
                            try:
                                _sts_progress.progress(
                                    done / max(total, 1),
                                    text=tr(
                                        f"STS Score: {done}/{total}",
                                        f"STS Score: {done}/{total}",
                                    ),
                                )
                            except Exception:
                                pass
                        _batch_pids = _sts_score_patient_ids(batch_rows_for_sts)
                        sts_results = calculate_sts_batch(
                            batch_rows_for_sts,
                            progress_callback=_sts_progress_cb,
                            phase_callback=_sts_phase_cb,
                            patient_ids=_batch_pids,
                        )
                        _sts_phase_slot.empty()  # clear phase label on completion
                        if sts_results:
                            for _ri, _sr in enumerate(sts_results):
                                if isinstance(_sr, dict) and "predmort" in _sr:
                                    sts_probs[_ri] = _sr["predmort"]
                        _n_sts_ok = sum(1 for p in sts_probs if pd.notna(p))
                        _sts_progress.progress(1.0, text=tr(
                            f"STS Score complete: {_n_sts_ok}/{_n_total} resolved",
                            f"STS Score completo: {_n_sts_ok}/{_n_total} resolvidos",
                        ))
                        # Phase 3: compact cache status summary.
                        _sts_exec_log = getattr(calculate_sts_batch, 'last_execution_log', [])
                        _sts_fail_log = getattr(calculate_sts_batch, 'failure_log', [])
                        _sts_hits = sum(1 for r in _sts_exec_log if getattr(r, 'status', '') == 'cached')
                        _sts_fresh = sum(1 for r in _sts_exec_log if getattr(r, 'status', '') == 'fresh')
                        _sts_refreshed = sum(1 for r in _sts_exec_log if getattr(r, 'status', '') == 'refreshed')
                        _sts_stale = sum(1 for r in _sts_exec_log if getattr(r, 'status', '') == 'stale_fallback')
                        _sts_failed_n = sum(1 for r in _sts_exec_log if getattr(r, 'status', '') == 'failed')
                        st.caption(tr(
                            f"STS Score — Cache hits: {_sts_hits} | Misses: {_sts_fresh} | "
                            f"Refreshed: {_sts_refreshed} | Stale fallback: {_sts_stale} | Failed: {_sts_failed_n}",
                            f"STS Score — Cache acertos: {_sts_hits} | Novas buscas: {_sts_fresh} | "
                            f"Atualizados: {_sts_refreshed} | Cache expirado reutilizado: {_sts_stale} | Falhas: {_sts_failed_n}",
                        ))
                        # Phase 3: execution details expander.
                        with st.expander(tr("View STS Score execution details", "Ver detalhes de execução do STS Score"), expanded=False):
                            st.caption(tr(
                                f"Last phase: {_sts_last_phase[0]} | Detail: {_sts_last_phase[1]}",
                                f"Última fase: {_sts_last_phase[0]} | Detalhe: {_sts_last_phase[1]}",
                            ))
                            if _sts_fail_log:
                                st.markdown(tr("**Incidents:**", "**Incidentes:**"))
                                for _sf in _sts_fail_log[:10]:
                                    st.text(
                                        f"patient={_sf.get('patient_id') or '?'} | "
                                        f"status={_sf.get('status','?')} | stage={_sf.get('stage','?')} | "
                                        f"reason={_sf.get('reason','?')}"
                                    )
                                if len(_sts_fail_log) > 10:
                                    st.caption(tr(
                                        f"... and {len(_sts_fail_log) - 10} more incidents",
                                        f"... e mais {len(_sts_fail_log) - 10} incidentes",
                                    ))
                            else:
                                st.caption(tr("No incidents.", "Nenhum incidente."))
                        if _n_sts_ok < _n_total:
                            if _sts_fail_log:
                                _fail_details = "\n".join(
                                    f"- **patient={f.get('patient_id') or '?'}** | "
                                    f"status={f.get('status','?')} | stage={f.get('stage','?')} | "
                                    f"reason={f.get('reason','?')} | "
                                    f"retry={f.get('retry_attempted', False)} | "
                                    f"used_previous_cache={f.get('used_previous_cache', False)}"
                                    for f in _sts_fail_log
                                )
                                st.warning(tr(
                                    f"STS Score resolved for {_n_sts_ok}/{_n_total} patients. Incidents:",
                                    f"STS Score resolvido para {_n_sts_ok}/{_n_total} pacientes. Incidentes:",
                                ))
                                with st.expander(tr("STS incident details", "Detalhes dos incidentes STS"), expanded=False):
                                    st.markdown(_fail_details)
                            else:
                                st.warning(tr(
                                    f"STS Score resolved for {_n_sts_ok}/{_n_total} patients.",
                                    f"STS Score resolvido para {_n_sts_ok}/{_n_total} pacientes.",
                                ))
                    except Exception as _sts_err:
                        _sts_progress.progress(1.0, text=tr(
                            f"STS Score failed: {_sts_err}", f"STS Score falhou: {_sts_err}",
                        ))
                        st.warning(tr(
                            f"STS Score calculation failed: {_sts_err}. Results shown without STS Score.",
                            f"Cálculo do STS Score falhou: {_sts_err}. Resultados exibidos sem STS Score.",
                        ))
                else:
                    _batch_local_phase_slot.empty()

                for i, sp in enumerate(sts_probs):
                    results[i]["STS (%)"] = round(sp * 100, 2) if pd.notna(sp) else np.nan

                result_df = pd.DataFrame(results)
                _n_ai_ok = max(_n_total - _n_errors, 0)
                _n_sts_ok_final = sum(1 for sp in sts_probs if pd.notna(sp))
                _n_sts_incidents = len(_sts_fail_log) if "_sts_fail_log" in locals() else 0
                _n_total_incidents = int(_n_errors + _n_sts_incidents)
                _success_rate = (_n_ai_ok / _n_total) if _n_total else np.nan

                st.markdown(tr("**Processing summary**", "**Resumo do processamento**"))
                _sum1, _sum2, _sum3, _sum4, _sum5 = st.columns(5)
                _sum1.metric(tr("Rows processed", "Linhas processadas"), f"{_n_total}", border=True)
                _sum2.metric(tr("AI Risk OK", "AI Risk OK"), f"{_n_ai_ok}", border=True)
                _sum3.metric(tr("STS resolved", "STS resolvido"), f"{_n_sts_ok_final}", border=True)
                _sum4.metric(tr("Incidents", "Incidentes"), f"{_n_total_incidents}", border=True)
                _sum5.metric(
                    tr("AI success rate", "Sucesso AI"),
                    "N/A" if not np.isfinite(_success_rate) else f"{_success_rate*100:.1f}%",
                    border=True,
                )

                # Column visibility: hide individual model columns unless checkbox is checked
                _ia_detail_cols = [f"IA-{_mn} (%)" for _mn in _all_model_names]
                if not _show_all_models:
                    display_df = result_df.drop(columns=_ia_detail_cols, errors="ignore")
                else:
                    display_df = result_df

                st.markdown(tr("**Result preview**", "**Prévia dos resultados**"))
                st.dataframe(display_df.head(25), width="stretch")
                with st.expander(tr("Show full batch prediction table", "Mostrar tabela completa da predição em lote"), expanded=False):
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
                st.markdown(tr("**Downloads**", "**Downloads**"))
                st.caption(tr(
                    "Downloads include the full result table; CSV/XLSX always include all AI model prediction columns.",
                    "Os downloads incluem a tabela completa; CSV/XLSX sempre incluem as colunas de predição de todos os modelos de IA.",
                ))
                _dl1, _dl2, _dl3, _dl4 = st.columns(4)
                with _dl1:
                    _csv_download_btn(result_df, "ia_risk_batch_predictions.csv", tr("CSV", "CSV"))
                with _dl2:
                    _xlsx_buf = BytesIO()
                    result_df.to_excel(_xlsx_buf, index=False, engine="openpyxl")
                    _bytes_download_btn(
                        _xlsx_buf.getvalue(),
                        "ia_risk_batch_predictions.xlsx",
                        "XLSX",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_batch_xlsx",
                    )
                with _dl3:
                    _txt_download_btn(_batch_md, "ia_risk_batch_predictions.md", "MD")
                with _dl4:
                    _batch_pdf = statistical_summary_to_pdf(_batch_md)
                    if _batch_pdf:
                        _bytes_download_btn(
                            _batch_pdf,
                            "ia_risk_batch_predictions.pdf",
                            "PDF",
                            "application/pdf",
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
