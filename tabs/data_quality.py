"""Data Quality tab — extracted from app.py (tab index 7).

Pure extraction: all logic, text, i18n, and UI elements are identical to the
original inline code.  The only structural change is that shared state is
accessed through ``ctx`` (:class:`tabs.TabContext`) instead of bare local
variables in ``app.py``.

``_build_audit_package_xlsx`` is moved here from app.py as a module-level
private function.
"""

from __future__ import annotations

import datetime
from io import BytesIO
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from model_metadata import (
    build_model_metadata,
    check_validation_readiness,
    compute_data_quality_summary,
    read_audit_log,
)

if TYPE_CHECKING:
    from tabs import TabContext


def _build_audit_package_xlsx(
    dq: dict,
    prepared,
    artifacts,
    bundle_info: dict,
    val_checks: list,
    miss_df: pd.DataFrame,
) -> bytes:
    """Build a multi-sheet XLSX audit package for the Data Quality tab."""
    from variable_contract import VARIABLE_CONTRACT

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _manifest = getattr(artifacts, "training_manifest", None)

        # README
        readme_rows = [
            {"Field": "Generated at", "Value": datetime.datetime.now().isoformat()},
            {"Field": "Model version", "Value": bundle_info.get("model_version", "?")},
            {"Field": "Best model", "Value": getattr(artifacts, "best_model_name", "?")},
            {"Field": "Calibration method", "Value": getattr(artifacts, "calibration_method", "?")},
            {"Field": "N rows", "Value": dq["n_total"]},
            {"Field": "N events", "Value": dq["n_events"]},
            {"Field": "Event rate", "Value": f"{dq['event_rate']:.4f}"},
            {"Field": "N features", "Value": len(prepared.feature_columns)},
            {"Field": "Sheets in this workbook", "Value": "See tabs below for complete audit data"},
        ]
        if _manifest and _manifest.get("dataset_hash"):
            readme_rows.append({"Field": "Dataset fingerprint", "Value": _manifest["dataset_hash"]})
        pd.DataFrame(readme_rows).to_excel(writer, sheet_name="README", index=False)

        # 01_COHORT_SUMMARY
        cohort_rows = [
            {"Metric": "Total eligible surgeries", "Value": dq["n_total"]},
            {"Metric": "Deaths (primary outcome)", "Value": dq["n_events"]},
            {"Metric": "Event rate", "Value": f"{dq['event_rate']:.4f}"},
            {"Metric": "Triple cohort (AI + Euro + STS)", "Value": dq["n_triple"]},
            {"Metric": "N features", "Value": len(prepared.feature_columns)},
        ]
        pd.DataFrame(cohort_rows).to_excel(writer, sheet_name="01_COHORT_SUMMARY", index=False)

        # 02_TRAINING_MANIFEST
        if _manifest:
            manifest_rows = [
                {"Key": k, "Value": str(v)}
                for k, v in _manifest.items()
                if k != "feature_columns"
            ]
            for i, fc in enumerate((_manifest.get("feature_columns") or [])):
                manifest_rows.append({"Key": f"feature_{i + 1:03d}", "Value": fc})
            pd.DataFrame(manifest_rows).to_excel(writer, sheet_name="02_TRAINING_MANIFEST", index=False)

        # 03_MISSING_RATES
        if not miss_df.empty:
            miss_df.to_excel(writer, sheet_name="03_MISSING_RATES", index=False)

        # 04_SCORE_AVAILABILITY
        score_rows = [
            {"Score": "AI Risk (OOF)", "Patients": dq.get("n_total", 0)},
            {"Score": "EuroSCORE II (app-calculated)", "Patients": dq["n_euro_calc"]},
            {"Score": "STS (app-calculated)", "Patients": dq["n_sts"]},
            {"Score": "EuroSCORE II (sheet)", "Patients": dq["n_euro_sheet"]},
            {"Score": "EuroSCORE II Auto (sheet)", "Patients": dq["n_euro_auto"]},
            {"Score": "STS (sheet)", "Patients": dq["n_sts_sheet"]},
            {"Score": "Triple cohort", "Patients": dq["n_triple"]},
        ]
        pd.DataFrame(score_rows).to_excel(writer, sheet_name="04_SCORE_AVAILABILITY", index=False)

        # 05_VALIDATION_READINESS
        if val_checks:
            pd.DataFrame(val_checks).to_excel(writer, sheet_name="05_VALIDATION_READINESS", index=False)

        # 06_FEATURE_EXCLUSION_POLICY
        nf = dq.get("never_feature_audit", {})
        nf_rows = []
        for category, cols in nf.items():
            for col in (cols if isinstance(cols, list) else []):
                nf_rows.append({"Category": category, "Column": col})
        if nf_rows:
            pd.DataFrame(nf_rows).to_excel(writer, sheet_name="06_FEATURE_EXCLUSION", index=False)

        # 07_SURGERY_DISTRIBUTION
        if dq.get("surgery_dist"):
            surg_rows = [{"Procedure": k, "Count": v} for k, v in dq["surgery_dist"].items()]
            pd.DataFrame(surg_rows).to_excel(writer, sheet_name="07_SURGERY_DIST", index=False)

        # 08_PROCEDURE_GROUPS
        if dq.get("procedure_group_dist"):
            pg_total = sum(dq["procedure_group_dist"].values())
            pg_rows = [
                {
                    "Group": k,
                    "Count": v,
                    "%": f"{v / pg_total:.1%}" if pg_total > 0 else "—",
                }
                for k, v in sorted(
                    dq["procedure_group_dist"].items(), key=lambda x: x[1], reverse=True
                )
            ]
            pd.DataFrame(pg_rows).to_excel(writer, sheet_name="08_PROCEDURE_GROUPS", index=False)

        # 09_PREVIOUS_SURGERY_AUDIT
        ps = dq.get("previous_surgery_audit", {})
        if ps:
            ps_rows = [{"Field": k, "Value": str(v)} for k, v in ps.items()]
            pd.DataFrame(ps_rows).to_excel(writer, sheet_name="09_PREV_SURGERY_AUDIT", index=False)

        # 10_INGESTION_ACTIONS
        ir = getattr(prepared, "ingestion_report", None)
        if ir:
            lines = ir.summary_lines()
            if lines:
                pd.DataFrame({"Action": lines}).to_excel(
                    writer, sheet_name="10_INGESTION_ACTIONS", index=False
                )

        # 11_CORRECTION_RECORDS
        if ir:
            corr_df = ir.audit_dataframe()
            if not corr_df.empty:
                corr_df.to_excel(writer, sheet_name="11_CORRECTION_RECORDS", index=False)

        # 12_VARIABLE_CONTRACT
        contract_rows = []
        for var_name, spec in VARIABLE_CONTRACT.items():
            row = {"variable": var_name}
            row.update({k: str(v) for k, v in spec.items()})
            contract_rows.append(row)
        if contract_rows:
            pd.DataFrame(contract_rows).to_excel(
                writer, sheet_name="12_VARIABLE_CONTRACT", index=False
            )

        # 13_LEADERBOARD
        lb = getattr(artifacts, "leaderboard", None)
        if lb is not None and not lb.empty:
            lb.to_excel(writer, sheet_name="13_LEADERBOARD", index=False)

    return buf.getvalue()


def render(ctx: "TabContext") -> None:
    tr = ctx.tr
    artifacts = ctx.artifacts
    prepared = ctx.prepared
    df = ctx.df
    bundle_info = ctx.bundle_info
    best_model_name = ctx.best_model_name
    xlsx_path = ctx.xlsx_path
    language = ctx.language
    _bytes_download_btn = ctx.bytes_download_btn
    HAS_STS = ctx.has_sts

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
        model_version=bundle_info.get("model_version"),
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

    _macro_dist = _dq.get("procedure_group_dist", {})
    _surg_cov = _dq.get("surgery_coverage", {})
    if _macro_dist or _surg_cov:
        with st.expander(tr("Procedure macro-group coverage", "Cobertura por macrogrupo cirúrgico"), expanded=False):
            st.caption(tr(
                "Taxonomy mapping audit: how many surgeries resolved to a known macro group vs. unrecognised text (Other) or absent/blank (Unknown).",
                "Auditoria de mapeamento taxonômico: quantas cirurgias foram resolvidas para um macrogrupo conhecido vs. texto não reconhecido (Other) ou ausente/branco (Unknown).",
            ))
            if _surg_cov:
                _cov_n = _surg_cov.get("total", 0)
                _cov_mapped = _surg_cov.get("n_mapped", 0)
                _cov_unk = _surg_cov.get("n_unknown", 0)
                _cov_other = _surg_cov.get("n_other", 0)
                _cov_rate = _surg_cov.get("coverage_rate", 0.0)
                _cov1, _cov2, _cov3, _cov4 = st.columns(4)
                _cov1.metric(tr("Total surgeries", "Total de cirurgias"), _cov_n)
                _cov2.metric(tr("Mapped", "Mapeados"), f"{_cov_mapped} ({_cov_rate:.0%})")
                _cov3.metric(tr("Other (unrecognised text)", "Other (texto não reconhecido)"), _cov_other)
                _cov4.metric(tr("Unknown (absent/blank)", "Unknown (ausente/branco)"), _cov_unk)
                if _cov_other > 0:
                    st.warning(tr(
                        f"{_cov_other} surgery value(s) resolved to 'Other' — text present but not in procedure taxonomy. See unrecognised values below.",
                        f"{_cov_other} valor(es) de cirurgia resolvido(s) como 'Other' — texto presente mas fora da taxonomia. Veja os valores não reconhecidos abaixo.",
                    ))
                    _unrec = _surg_cov.get("top_unrecognized", [])
                    if _unrec:
                        st.dataframe(
                            pd.DataFrame(_unrec, columns=[tr("Surgery text", "Texto da cirurgia"), tr("Count", "Contagem")]),
                            width="stretch",
                            hide_index=True,
                        )
            if _macro_dist:
                st.divider()
                _n_total_macro = sum(_macro_dist.values())
                _macro_rows = sorted(_macro_dist.items(), key=lambda x: x[1], reverse=True)
                _macro_df = pd.DataFrame([
                    {
                        tr("Macro group", "Macrogrupo"): grp,
                        tr("Count", "Contagem"): cnt,
                        tr("%", "%"): f"{cnt / _n_total_macro:.1%}" if _n_total_macro > 0 else "—",
                    }
                    for grp, cnt in _macro_rows
                ])
                st.dataframe(_macro_df, width="stretch", hide_index=True)

    _nf_audit = _dq.get("never_feature_audit", {})
    if _nf_audit:
        _nf_leaked = _nf_audit.get("leaked_into_features", [])
        with st.expander(
            tr("Column exclusion policy audit", "Auditoria de política de exclusão de colunas"),
            expanded=bool(_nf_leaked),
        ):
            st.caption(tr(
                "Columns present in the loaded dataset that are excluded from AI Risk features by policy category. "
                "The 'Leaked into features' row should always be empty — if it is not, this is a data integrity issue.",
                "Colunas presentes no dataset carregado que são excluídas das variáveis do AI Risk por categoria de política. "
                "A linha 'Vazaram para features' deve sempre ser vazia — se não estiver, há um problema de integridade nos dados.",
            ))
            if _nf_leaked:
                st.error(tr(
                    f"DATA INTEGRITY ISSUE: {len(_nf_leaked)} never-feature column(s) leaked into feature set: {_nf_leaked}",
                    f"PROBLEMA DE INTEGRIDADE: {len(_nf_leaked)} coluna(s) proibida(s) vazaram para o conjunto de features: {_nf_leaked}",
                ))
            _nf_rows = [
                {
                    tr("Category", "Categoria"): tr("Outcome (target variable)", "Desfecho (variável alvo)"),
                    tr("Columns in data", "Colunas no dataset"): ", ".join(_nf_audit.get("outcome", [])) or "—",
                    tr("Count", "Qtd"): len(_nf_audit.get("outcome", [])),
                },
                {
                    tr("Category", "Categoria"): tr("Postoperative / future info", "Pós-operatório / informação futura"),
                    tr("Columns in data", "Colunas no dataset"): ", ".join(_nf_audit.get("postoperative", [])) or "—",
                    tr("Count", "Qtd"): len(_nf_audit.get("postoperative", [])),
                },
                {
                    tr("Category", "Categoria"): tr("Comparator scores (STS / EuroSCORE)", "Scores comparadores (STS / EuroSCORE)"),
                    tr("Columns in data", "Colunas no dataset"): ", ".join(_nf_audit.get("comparator_score", [])) or "—",
                    tr("Count", "Qtd"): len(_nf_audit.get("comparator_score", [])),
                },
                {
                    tr("Category", "Categoria"): tr("Reference / metadata / IDs", "Referência / metadados / IDs"),
                    tr("Columns in data", "Colunas no dataset"): ", ".join(_nf_audit.get("metadata", [])) or "—",
                    tr("Count", "Qtd"): len(_nf_audit.get("metadata", [])),
                },
                {
                    tr("Category", "Categoria"): tr("Leaked into features (must be 0)", "Vazaram para features (deve ser 0)"),
                    tr("Columns in data", "Colunas no dataset"): ", ".join(_nf_leaked) or "—",
                    tr("Count", "Qtd"): len(_nf_leaked),
                },
            ]
            st.dataframe(pd.DataFrame(_nf_rows), width="stretch", hide_index=True)

    _ps_audit = _dq.get("previous_surgery_audit", {})
    if _ps_audit:
        with st.expander(tr("Previous surgery structured audit", "Auditoria estruturada de cirurgia prévia"), expanded=False):
            st.caption(tr(
                "Grammar-aware decomposition of the Previous surgery free-text field. "
                "These columns are for auditing only — they are not model features.",
                "Decomposição gramatical do campo de texto livre Cirurgia Prévia. "
                "Estas colunas são apenas para auditoria — não são features do modelo.",
            ))
            _n_any = _ps_audit.get("n_with_prior_surgery", 0)
            _pct_any = _ps_audit.get("pct_with_prior_surgery", 0.0)
            _ps_rows = [
                {
                    tr("Field", "Campo"): tr("Patients with any prior surgery", "Pacientes com cirurgia prévia"),
                    tr("Value", "Valor"): f"{_n_any} ({_pct_any:.1%})",
                },
                {
                    tr("Field", "Campo"): tr("Episodes with combined procedures (+)", "Episódios com procedimentos combinados (+)"),
                    tr("Value", "Valor"): str(_ps_audit.get("n_combined_episode", 0)),
                },
                {
                    tr("Field", "Campo"): tr("Episodes with repeat marker (xN)", "Episódios com marcador de repetição (xN)"),
                    tr("Value", "Valor"): str(_ps_audit.get("n_repeat_marker", 0)),
                },
                {
                    tr("Field", "Campo"): tr("Episodes with year annotation (YYYY)", "Episódios com marcação de ano (YYYY)"),
                    tr("Value", "Valor"): str(_ps_audit.get("n_year_marker", 0)),
                },
                {
                    tr("Field", "Campo"): tr("Mean estimated episode count (redo only)", "Contagem média estimada de episódios (apenas redo)"),
                    tr("Value", "Valor"): f"{_ps_audit.get('mean_count_est_among_redo', 0.0):.2f}",
                },
            ]
            st.dataframe(pd.DataFrame(_ps_rows), width="stretch", hide_index=True)

    st.divider()
    st.markdown(tr("### Audit Package", "### Pacote de Auditoria"))
    st.caption(tr(
        "Download a consolidated XLSX workbook with all data quality, ingestion, training, "
        "and governance information for external audit or dissertation appendix.",
        "Baixe um workbook XLSX consolidado com todas as informações de qualidade, ingestão, "
        "treino e governança para auditoria externa ou apêndice da dissertação.",
    ))
    _miss_df_for_export = _miss_df.copy() if not _miss_df.empty else pd.DataFrame(
        [{"Variable": var, "Missing rate": rate}
         for var, rate in sorted(_dq["missing_rates"].items(), key=lambda x: x[1], reverse=True)]
    )
    _audit_xlsx_bytes = _build_audit_package_xlsx(
        dq=_dq,
        prepared=prepared,
        artifacts=artifacts,
        bundle_info=bundle_info,
        val_checks=_val_checks,
        miss_df=_miss_df_for_export,
    )
    _bytes_download_btn(
        _audit_xlsx_bytes,
        "ai_risk_audit_package.xlsx",
        tr("Download Audit Package (XLSX)", "Baixar Pacote de Auditoria (XLSX)"),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_audit_package_xlsx",
    )

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
