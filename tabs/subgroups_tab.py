"""Subgroups tab — extracted from app.py (tab index 6).

Pure extraction: all logic, text, i18n, and UI elements are identical to the
original inline code.  The only structural change is that shared state is
accessed through ``ctx`` (:class:`tabs.TabContext`) instead of bare local
variables in ``app.py``.
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st

from bundle_io import bundle_signature as _bundle_signature
from export_helpers import statistical_summary_to_pdf
from subgroups import (
    evaluate_subgroup,
    lvef_group as _lvef_group_impl,
    renal_group as _renal_group_impl,
    surgery_type_group as _surgery_type_group_impl,
)

if TYPE_CHECKING:
    from tabs import TabContext


# ---------------------------------------------------------------------------
# Subgroup helper functions (moved from app.py — only used in subgroups tab)
# ---------------------------------------------------------------------------

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


def _markdown_table(df: pd.DataFrame, cols: list, max_rows: int = 12) -> str:
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


def render(ctx: "TabContext") -> None:
    tr = ctx.tr
    df = ctx.df
    _default_threshold = ctx.default_threshold
    language = ctx.language
    xlsx_path = ctx.xlsx_path
    _bytes_download_btn = ctx.bytes_download_btn
    _csv_download_btn = ctx.csv_download_btn
    HAS_STS = ctx.has_sts
    prepared = ctx.prepared
    _format_ppv_npv = ctx.format_ppv_npv
    stats_table_column_config = ctx.stats_table_column_config

    # Create local tr-bound wrappers (same pattern as app.py)
    def _surgery_type_group(text: object) -> str:
        return _surgery_type_group_impl(text, tr)

    def _lvef_group(value: object, fallback: object = None) -> str:
        return _lvef_group_impl(value, fallback, tr)

    def _renal_group(clearance: object, dialysis: object,
                     creatinine: object = None, age: object = None,
                     weight: object = None, sex: object = None) -> str:
        return _renal_group_impl(clearance, dialysis, creatinine, age, weight, sex, tr)

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
