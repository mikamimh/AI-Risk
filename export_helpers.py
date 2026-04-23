"""Statistical summary export helpers for AI Risk.

Extracted from model_metadata.py to isolate self-contained report/export
logic from core model metadata functions.  All functions here are pure
transformers: they accept DataFrames or Markdown strings as input and return
formatted output (Markdown, DataFrames, bytes).  They carry no project-level
state and import nothing from the AI Risk project modules.

Provides:
- build_statistical_summary       — Markdown statistical report (with Calibration at a Glance)
- build_comparison_xlsx           — structured XLSX with numbered/named sheets + README
- build_comparison_summary_pdf    — curated editorial PDF (main metrics, calibration, pairwise)
- build_comparison_full_pdf       — comprehensive PDF (all sections: exec summary, DCA, NRI/IDI, appendix)
- build_comparison_full_package   — ZIP bytes (summary PDF + full PDF + full MD + XLSX + CSV)
- build_export_manifest           — single source-of-truth manifest dict for any export
- manifest_to_md_lines            — render a manifest as Markdown header lines
- statistical_summary_to_dataframes — Markdown → dict of DataFrames
- statistical_summary_to_xlsx       — Markdown → XLSX bytes (legacy, flat layout)
- statistical_summary_to_csv        — Markdown → CSV string
- statistical_summary_to_pdf        — Markdown → PDF bytes (requires fpdf2)
"""

import io
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Export manifest — the canonical metadata block embedded in every export
# ---------------------------------------------------------------------------
# Every artifact written to disk (CSV, XLSX, PDF, MD, ZIP) MUST be able to
# answer four questions without ambiguity:
#   1. which model_version produced these numbers
#   2. which active_model_name was scored
#   3. which threshold (mode + value) was applied
#   4. which dataset and bundle produced them
#
# The manifest below answers all four with a single dict, so callers cannot
# accidentally mix sources — every export reads from the same builder.

def build_export_manifest(
    *,
    export_kind: str,
    model_version: str,
    active_model_name: Optional[str],
    threshold_mode: str,
    threshold_value: float,
    dataset_fingerprint: Optional[str] = None,
    bundle_fingerprint: Optional[str] = None,
    bundle_saved_at: Optional[str] = None,
    training_source: Optional[str] = None,
    current_analysis_file: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the canonical manifest dict for an export.

    The fields are deliberately flat and JSON-serialisable so the same dict
    can be embedded in a ZIP entry, written into a Markdown header, or
    rendered into a PDF/XLSX cover sheet.

    Args:
        export_kind: short identifier (e.g. ``"comparison"``, ``"temporal_validation"``,
            ``"batch_prediction"``, ``"individual_report"``).
        threshold_mode: e.g. ``"clinical_fixed"``, ``"youden"``, ``"locked"``.
        threshold_value: the numeric probability threshold actually used.
        dataset_fingerprint / bundle_fingerprint: opaque identifiers from
            ``bundle_metadata_from_payload`` — pass through unchanged.
    """
    manifest = {
        "export_kind": export_kind,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "active_model_name": active_model_name,
        "threshold_mode": threshold_mode,
        "threshold_value": float(threshold_value),
        "dataset_fingerprint": dataset_fingerprint,
        "bundle_fingerprint": bundle_fingerprint,
        "bundle_saved_at": bundle_saved_at,
        "training_source": training_source,
        "current_analysis_file": current_analysis_file,
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def manifest_to_json_bytes(manifest: Dict[str, Any]) -> bytes:
    """Serialise a manifest to UTF-8 JSON bytes for ZIP embedding."""
    return json.dumps(manifest, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def manifest_to_md_lines(manifest: Dict[str, Any], language: str = "English") -> List[str]:
    """Render the manifest as Markdown header lines for inline embedding.

    Returned list has no trailing blank line — callers append as needed.
    """
    def _tr(en: str, pt: str) -> str:
        return en if language == "English" else pt

    lines = [
        f"**{_tr('Model version', 'Versão do modelo')}:** {manifest.get('model_version', 'N/A')}",
        f"**{_tr('Active model', 'Modelo ativo')}:** {manifest.get('active_model_name') or _tr('N/A', 'N/A')}",
        f"**{_tr('Threshold', 'Limiar')}:** {manifest.get('threshold_value', 0):.0%} ({manifest.get('threshold_mode', '?')})",
        f"**{_tr('Generated', 'Gerado em')}:** {str(manifest.get('generated_at', '') or '')[:19].replace('T', ' ')}",
    ]
    if manifest.get("training_source"):
        lines.append(f"**{_tr('Training source', 'Fonte do treino')}:** {manifest['training_source']}")
    if manifest.get("current_analysis_file"):
        lines.append(f"**{_tr('Current analysis file', 'Arquivo de análise atual')}:** {manifest['current_analysis_file']}")
    if manifest.get("bundle_fingerprint"):
        lines.append(f"**{_tr('Bundle fingerprint', 'Fingerprint do bundle')}:** `{manifest['bundle_fingerprint']}`")
    return lines


# ---------------------------------------------------------------------------
# Markdown statistical summary generation
# ---------------------------------------------------------------------------

def build_statistical_summary(
    triple_ci: pd.DataFrame,
    calib_df: pd.DataFrame,
    formal_df: pd.DataFrame,
    delong_df: pd.DataFrame,
    reclass_df: pd.DataFrame,
    threshold: float,
    threshold_metrics: pd.DataFrame,
    n_triple: int,
    model_version: str,
    language: str = "English",
) -> str:
    """Build an exportable statistical summary as Markdown."""
    def _tr(en, pt):
        return en if language == "English" else pt

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# {_tr('Statistical Analysis Summary', 'Resumo da Análise Estatística')}",
        "",
        f"**{_tr('Generated', 'Gerado em')}:** {now}",
        f"**{_tr('Model version', 'Versão do modelo')}:** {model_version}",
        f"**{_tr('Triple comparison sample', 'Amostra da comparação tripla')}:** n = {n_triple}",
        f"**{_tr('Decision threshold', 'Limiar de decisão')}:** {threshold:.0%}",
        "",
    ]

    # Discrimination with CI
    if not triple_ci.empty:
        lines.append(f"## {_tr('Main Performance (95% CI, triple cohort)', 'Desempenho Principal (IC 95%, coorte tripla)')}")
        lines.append("")
        lines.append(f"| {_tr('Score', 'Escore')} | n | AUC (95% CI) | AUPRC (95% CI) | Brier (95% CI) |")
        lines.append("|:--|:--|:--|:--|:--|")
        for _, row in triple_ci.iterrows():
            auc_ci = f"{row['AUC']:.3f} ({row.get('AUC_IC95_inf', np.nan):.3f}-{row.get('AUC_IC95_sup', np.nan):.3f})"
            auprc_ci = f"{row['AUPRC']:.3f} ({row.get('AUPRC_IC95_inf', np.nan):.3f}-{row.get('AUPRC_IC95_sup', np.nan):.3f})"
            brier_ci = f"{row['Brier']:.4f} ({row.get('Brier_IC95_inf', np.nan):.4f}-{row.get('Brier_IC95_sup', np.nan):.4f})"
            lines.append(f"| {row['Score']} | {row.get('n', '')} | {auc_ci} | {auprc_ci} | {brier_ci} |")
        lines.append("")

        # Narrative: brief primary result synthesis
        try:
            best_auc_row = triple_ci.sort_values("AUC", ascending=False).iloc[0]
            best_brier_row = triple_ci.sort_values("Brier", ascending=True).iloc[0]
            lines.append(
                f"> {_tr('Primary result', 'Resultado principal')}: "
                f"{_tr('best discrimination', 'melhor discriminação')} — {best_auc_row['Score']} "
                f"(AUC {best_auc_row['AUC']:.3f}); "
                f"{_tr('best calibration', 'melhor calibração')} — {best_brier_row['Score']} "
                f"(Brier {best_brier_row['Brier']:.4f})."
            )
            lines.append("")
        except Exception:
            pass

    # Threshold metrics
    if not threshold_metrics.empty:
        lines.append(f"## {_tr('Classification at threshold', 'Classificação no limiar')} {threshold:.0%}")
        lines.append("")
        lines.append(f"| {_tr('Score', 'Escore')} | {_tr('Sensitivity', 'Sensibilidade')} | {_tr('Specificity', 'Especificidade')} | PPV | NPV |")
        lines.append("|:--|:--|:--|:--|:--|")
        for _, row in threshold_metrics.iterrows():
            ppv = f"{row['PPV']:.3f}" if pd.notna(row.get('PPV')) else "-"
            npv = f"{row['NPV']:.3f}" if pd.notna(row.get('NPV')) else "-"
            lines.append(f"| {row['Score']} | {row.get('Sensitivity', np.nan):.3f} | {row.get('Specificity', np.nan):.3f} | {ppv} | {npv} |")
        lines.append("")

    # Calibration at a Glance — compact summary before full calibration table
    if not calib_df.empty:
        lines.append(f"## {_tr('Calibration at a Glance', 'Calibração em Resumo')}")
        lines.append("")
        lines.append(
            _tr(
                "Intercept near 0 and slope near 1 indicate good calibration. "
                "Brier score measures probabilistic accuracy (lower is better). "
                "Hosmer-Lemeshow p-value is complementary only.",
                "Intercepto próximo de 0 e slope próximo de 1 indicam boa calibração. "
                "Brier score mede acurácia probabilística (menor é melhor). "
                "p-valor de Hosmer-Lemeshow é apenas complementar.",
            )
        )
        lines.append("")
        lines.append(f"| {_tr('Score', 'Escore')} | {_tr('Intercept', 'Intercepto')} | Slope | CIL | ICI | Brier | HL p |")
        lines.append("|:--|:--|:--|:--|:--|:--|:--|")
        for _, row in calib_df.iterrows():
            brier_val = f"{row['Brier']:.4f}" if pd.notna(row.get('Brier')) else "-"
            cil_val = f"{row['CIL']:.4f}" if pd.notna(row.get('CIL')) else "-"
            ici_val = f"{row['ICI']:.4f}" if pd.notna(row.get('ICI')) else "-"
            lines.append(
                f"| {row['Score']} | {row.get('Calibration intercept', np.nan):.4f} "
                f"| {row.get('Calibration slope', np.nan):.4f} "
                f"| {cil_val} "
                f"| {ici_val} "
                f"| {brier_val} "
                f"| {row.get('HL p-value', np.nan):.4f} |"
            )
        lines.append("")

    # Full calibration table
    if not calib_df.empty:
        lines.append(f"## {_tr('Calibration (full)', 'Calibração (completa)')}")
        lines.append("")
        lines.append(f"| {_tr('Score', 'Escore')} | {_tr('Intercept', 'Intercepto')} | Slope | CIL | ICI | HL chi² | HL p | Brier |")
        lines.append("|:--|:--|:--|:--|:--|:--|:--|:--|")
        for _, row in calib_df.iterrows():
            brier_val = f"{row['Brier']:.4f}" if pd.notna(row.get('Brier')) else "-"
            cil_val = f"{row['CIL']:.4f}" if pd.notna(row.get('CIL')) else "-"
            ici_val = f"{row['ICI']:.4f}" if pd.notna(row.get('ICI')) else "-"
            lines.append(
                f"| {row['Score']} "
                f"| {row.get('Calibration intercept', np.nan):.4f} "
                f"| {row.get('Calibration slope', np.nan):.4f} "
                f"| {cil_val} "
                f"| {ici_val} "
                f"| {row.get('HL chi-square', np.nan):.2f} "
                f"| {row.get('HL p-value', np.nan):.4f} "
                f"| {brier_val} |"
            )
        lines.append("")

    # DeLong
    if not delong_df.empty:
        lines.append(f"## {_tr('DeLong Test', 'Teste de DeLong')}")
        lines.append("")
        comp_col = [c for c in delong_df.columns if "Comparison" in c or "Comparação" in c]
        comp_key = comp_col[0] if comp_col else "Comparison"
        lines.append(f"| {_tr('Comparison', 'Comparação')} | ΔAUC | z | p |")
        lines.append("|:--|:--|:--|:--|")
        for _, row in delong_df.iterrows():
            lines.append(f"| {row.get(comp_key, '')} | {row.get('Delta AUC', np.nan):.3f} | {row.get('z', np.nan):.2f} | {row.get('p (DeLong)', np.nan):.4f} |")
        lines.append("")

    # Bootstrap comparison
    if not formal_df.empty:
        lines.append(f"## {_tr('Bootstrap AUC Comparison', 'Comparação de AUC por Bootstrap')}")
        lines.append("")
        comp_col = [c for c in formal_df.columns if "Comparison" in c or "Comparação" in c]
        comp_key = comp_col[0] if comp_col else "Comparison"
        ci_lo_col = [c for c in formal_df.columns if "CI low" in c or "IC95% inf" in c]
        ci_hi_col = [c for c in formal_df.columns if "CI high" in c or "IC95% sup" in c]
        lo_key = ci_lo_col[0] if ci_lo_col else "95% CI low"
        hi_key = ci_hi_col[0] if ci_hi_col else "95% CI high"
        lines.append(f"| {_tr('Comparison', 'Comparação')} | ΔAUC | 95% CI | p |")
        lines.append("|:--|:--|:--|:--|")
        for _, row in formal_df.iterrows():
            lines.append(f"| {row.get(comp_key, '')} | {row.get('Delta AUC (A-B)', np.nan):.3f} | {row.get(lo_key, np.nan):.3f}-{row.get(hi_key, np.nan):.3f} | {row.get('p (bootstrap)', np.nan):.4f} |")
        lines.append("")

    # NRI/IDI
    if not reclass_df.empty:
        lines.append(f"## {_tr('Reclassification (NRI/IDI)', 'Reclassificação (NRI/IDI)')}")
        lines.append("")
        comp_col = [c for c in reclass_df.columns if "Comparison" in c or "Comparação" in c]
        comp_key = comp_col[0] if comp_col else "Comparison"
        _has_nri_ci = "NRI 95% CI" in reclass_df.columns
        _has_idi_ci = "IDI 95% CI" in reclass_df.columns
        _has_nri_p = "NRI p" in reclass_df.columns
        _has_idi_p = "IDI p" in reclass_df.columns
        _hdr = f"| {_tr('Comparison', 'Comparação')} | NRI events | NRI non-events | NRI total"
        if _has_nri_ci:
            _hdr += " | NRI 95% CI"
        if _has_nri_p:
            _hdr += " | NRI p"
        _hdr += " | IDI"
        if _has_idi_ci:
            _hdr += " | IDI 95% CI"
        if _has_idi_p:
            _hdr += " | IDI p"
        _hdr += " |"
        lines.append(_hdr)
        _sep = "|:--|:--|:--|:--"
        if _has_nri_ci: _sep += "|:--"
        if _has_nri_p: _sep += "|:--"
        _sep += "|:--"
        if _has_idi_ci: _sep += "|:--"
        if _has_idi_p: _sep += "|:--"
        _sep += "|"
        lines.append(_sep)
        for _, row in reclass_df.iterrows():
            _r = (f"| {row.get(comp_key, '')} "
                  f"| {row.get('NRI events', np.nan):.3f} "
                  f"| {row.get('NRI non-events', np.nan):.3f} "
                  f"| {row.get('NRI total', np.nan):.3f}")
            if _has_nri_ci:
                _r += f" | {row.get('NRI 95% CI', '—')}"
            if _has_nri_p:
                _nri_p = row.get("NRI p", np.nan)
                _r += f" | {_nri_p:.4f}" if pd.notna(_nri_p) else " | —"
            _r += f" | {row.get('IDI', np.nan):.4f}"
            if _has_idi_ci:
                _r += f" | {row.get('IDI 95% CI', '—')}"
            if _has_idi_p:
                _idi_p = row.get("IDI p", np.nan)
                _r += f" | {_idi_p:.4f}" if pd.notna(_idi_p) else " | —"
            _r += " |"
            lines.append(_r)
        lines.append("")

    lines.append("---")
    lines.append(f"*{_tr('Generated by AI Risk', 'Gerado pelo AI Risk')}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown parsing and conversion utilities
# ---------------------------------------------------------------------------

def _parse_md_tables(md_text: str) -> List[dict]:
    """Extract Markdown tables from summary text as list of {title, headers, rows}."""
    tables = []
    lines = md_text.split("\n")
    i = 0
    current_title = ""
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("## "):
            current_title = line.lstrip("# ").strip()
        elif line.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            headers = [c.strip() for c in line.strip("|").split("|")]
            i += 1  # skip separator
            rows = []
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                rows.append(cells)
                i += 1
            tables.append({"title": current_title, "headers": headers, "rows": rows})
            continue
        i += 1
    return tables


def _df_to_md_table(df: pd.DataFrame, float_fmt: str = ".4f") -> List[str]:
    """Render a DataFrame as Markdown table lines (header + separator + data rows)."""
    if df.empty:
        return []
    headers = [str(c) for c in df.columns]
    lines: List[str] = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([":--"] * len(headers)) + "|",
    ]
    for _, row in df.iterrows():
        cells = []
        for val in row:
            try:
                na = pd.isna(val)
            except (TypeError, ValueError):
                na = False
            if na:
                cells.append("-")
            elif isinstance(val, float):
                cells.append(f"{val:{float_fmt}}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def statistical_summary_to_dataframes(md_text: str) -> Dict[str, pd.DataFrame]:
    """Convert the Markdown statistical summary into a dict of DataFrames (one per table)."""
    tables = _parse_md_tables(md_text)
    result = {}
    for t in tables:
        key = t["title"] or f"Table_{len(result) + 1}"
        df = pd.DataFrame(t["rows"], columns=t["headers"])
        result[key] = df
    return result


def statistical_summary_to_xlsx(md_text: str) -> bytes:
    """Convert statistical summary to XLSX with one sheet per table."""
    dfs = statistical_summary_to_dataframes(md_text)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in dfs.items():
            sheet_name = re.sub(r'[\\/*?\[\]:/]', '_', name)[:31]  # sanitize for Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buf.getvalue()


def statistical_summary_to_csv(md_text: str) -> str:
    """Convert statistical summary to CSV (all tables concatenated with section headers)."""
    dfs = statistical_summary_to_dataframes(md_text)
    parts = []
    for name, df in dfs.items():
        parts.append(f"# {name}")
        parts.append(df.to_csv(index=False))
    return "\n".join(parts)


def _line_plot_png(
    plot_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    diagonal: bool = False,
) -> bytes:
    """Render a deterministic line plot PNG for full-package figure exports."""
    if plot_df is None or plot_df.empty:
        return b""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return b""

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, grp in plot_df.groupby(group_col, observed=True):
        grp = grp.sort_values(x_col)
        ax.plot(grp[x_col], grp[y_col], label=str(name))
    if diagonal:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _comparison_figure_pngs(
    roc_plot_df: pd.DataFrame = None,
    calibration_plot_df: pd.DataFrame = None,
    dca_plot_df: pd.DataFrame = None,
) -> Dict[str, bytes]:
    """Build PNG bytes for Comparison figure exports from source-data tables."""
    pngs: Dict[str, bytes] = {}
    if roc_plot_df is not None and not roc_plot_df.empty:
        png = _line_plot_png(
            roc_plot_df,
            x_col="fpr",
            y_col="tpr",
            group_col="score",
            title="ROC Curves",
            xlabel="1 - Specificity (FPR)",
            ylabel="Sensitivity (TPR)",
            diagonal=True,
        )
        if png:
            pngs["figures/roc.png"] = png
    if calibration_plot_df is not None and not calibration_plot_df.empty:
        png = _line_plot_png(
            calibration_plot_df,
            x_col="mean_predicted_risk",
            y_col="observed_event_rate",
            group_col="score",
            title="Calibration Curves",
            xlabel="Predicted probability",
            ylabel="Observed frequency",
            diagonal=True,
        )
        if png:
            pngs["figures/calibration.png"] = png
    if dca_plot_df is not None and not dca_plot_df.empty:
        png = _line_plot_png(
            dca_plot_df,
            x_col="threshold",
            y_col="net_benefit",
            group_col="strategy",
            title="Decision Curve Analysis",
            xlabel="Decision threshold",
            ylabel="Net benefit",
            diagonal=False,
        )
        if png:
            pngs["figures/dca.png"] = png
    return pngs


def build_comparison_xlsx(
    triple_ci: pd.DataFrame,
    calib_df: pd.DataFrame,
    formal_df: pd.DataFrame,
    delong_df: pd.DataFrame,
    reclass_df: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    threshold: float,
    n_triple: int,
    model_version: str,
    language: str = "English",
    source_file: str = "",
    endpoint: str = "30-day / in-hospital mortality",
    cohort_note: str = "",
    extra_sheets: Dict[str, pd.DataFrame] = None,
    dca_df: pd.DataFrame = None,
    metrics_all: pd.DataFrame = None,
    pair_df: pd.DataFrame = None,
    threshold_comparison_df: pd.DataFrame = None,
    roc_plot_df: pd.DataFrame = None,
    calibration_plot_df: pd.DataFrame = None,
    dca_plot_df: pd.DataFrame = None,
) -> bytes:
    """Build a structured XLSX export with numbered, named sheets.

    Sheet layout:
      00_README            — artifact description, generation metadata
      01_EXECUTIVE_SUMMARY — headline metrics derived from triple cohort
      02_MAIN_METRICS      — discrimination with 95% CI
      03_THRESHOLD_PERF    — classification metrics at fixed threshold
      04_CALIBRATION       — intercept, slope, HL, Brier
      05_PAIRWISE          — DeLong + bootstrap comparisons (triple cohort)
      06_RECLASSIFICATION  — NRI / IDI (exploratory)
      07_CLINICAL_UTILITY  — DCA net benefit table (all thresholds)
      08_OVERALL_COMPARE   — each score with all available patients (non-matched)
      09_ALLPAIRS_FULL     — bootstrap pairwise (full cohort, before triple matching)
      10_THRESHOLD_COMPARISON — AI Risk threshold comparison (5/8/10/15/Youden)
      11_FIG_ROC_DATA      — source data for ROC figure
      12_FIG_CALIBRATION_DATA — source data for calibration figure
      13_FIG_DCA_DATA      — source data for DCA figure
      14_APPENDIX          — any extra_sheets passed by the caller

    Extra sheets from ``extra_sheets`` dict are appended as-is with their
    key used as the sheet name (truncated to 31 chars).
    """
    def _tr(en, pt):
        return en if language == "English" else pt

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M")
    date_tag = now.strftime("%Y%m%d")
    buf = io.BytesIO()
    _threshold_comp = threshold_comparison_df if threshold_comparison_df is not None else pd.DataFrame()
    _roc_plot = roc_plot_df if roc_plot_df is not None else pd.DataFrame()
    _cal_plot = calibration_plot_df if calibration_plot_df is not None else pd.DataFrame()
    _dca_plot = dca_plot_df if dca_plot_df is not None else pd.DataFrame()

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:

        # ── 00_README ──────────────────────────────────────────────────────
        readme_rows = [
            (_tr("Artifact", "Artefato"), "AI Risk — Statistical Comparison Export"),
            (_tr("Generated", "Gerado em"), now_str),
            (_tr("Model version", "Versão do modelo"), model_version),
            (_tr("Source file", "Arquivo fonte"), source_file or _tr("(not specified)", "(não informado)")),
            (_tr("Endpoint", "Desfecho"), endpoint),
            (_tr("Decision threshold", "Limiar de decisão"), f"{threshold:.0%}"),
            (_tr("Triple cohort n", "n coorte tripla"), str(n_triple)),
            (_tr("Cohort note", "Nota de coorte"), cohort_note or _tr("Triple cohort: patients with simultaneous AI Risk, EuroSCORE II, and STS Score.", "Coorte tripla: pacientes com AI Risk, EuroSCORE II e STS Score simultâneos.")),
            ("", ""),
            (_tr("Sheet guide", "Guia das abas"), ""),
            ("00_README", _tr("This sheet — artifact metadata and legend", "Esta aba — metadados do artefato e legenda")),
            ("01_EXECUTIVE_SUMMARY", _tr("Headline metrics (best AUC, best Brier, n)", "Métricas-síntese (melhor AUC, melhor Brier, n)")),
            ("02_MAIN_METRICS", _tr("Discrimination with 95% CI — AUC, AUPRC, Brier", "Discriminação com IC 95% — AUC, AUPRC, Brier")),
            ("03_THRESHOLD_PERF", _tr("Classification at fixed threshold — Sens/Spec/PPV/NPV", "Classificação no limiar fixo — Sens/Espec/VPP/VPN")),
            ("04_CALIBRATION", _tr("Calibration — intercept, slope, HL chi², Brier", "Calibração — intercepto, slope, HL chi², Brier")),
            ("05_PAIRWISE", _tr("Pairwise comparisons — DeLong test + bootstrap delta AUC", "Comparações pareadas — teste de DeLong + delta AUC por bootstrap")),
            ("06_RECLASSIFICATION", _tr("Reclassification (exploratory) — NRI / IDI", "Reclassificação (exploratória) — NRI / IDI")),
            ("07_CLINICAL_UTILITY", _tr("Decision curve analysis — net benefit at all thresholds", "Decision curve analysis — benefício líquido em todos os limiares")),
            ("08_OVERALL_COMPARE", _tr("Each score with all available patients (non-matched cohort)", "Cada escore com todos os pacientes disponíveis (coorte não pareada)")),
            ("09_ALLPAIRS_FULL", _tr("Bootstrap pairwise — full cohort (before triple matching)", "Bootstrap por pares — coorte completa (antes do pareamento triplo)")),
            ("10_THRESHOLD_COMPARISON", _tr("AI Risk threshold comparison — 5%, 8%, 10%, 15%, and Youden", "Comparação de limiares do AI Risk — 5%, 8%, 10%, 15% e Youden")),
            ("11_FIG_ROC_DATA", _tr("ROC figure source data — FPR, TPR, thresholds", "Dados-base da figura ROC — FPR, TPR, limiares")),
            ("12_FIG_CALIBRATION_DATA", _tr("Calibration figure source data — bins, observed event rate", "Dados-base da figura de calibração — bins, taxa observada")),
            ("13_FIG_DCA_DATA", _tr("DCA figure source data — net benefit by threshold", "Dados-base da figura DCA — benefício líquido por limiar")),
            ("14_APPENDIX", _tr("Supplementary sheets appended by the caller", "Abas suplementares adicionadas pelo chamador")),
            ("", ""),
            (_tr("Analysis note", "Nota metodológica"), _tr(
                "Primary analysis: triple cohort (AI Risk + EuroSCORE II + STS Score in the same patients). "
                "The 8% threshold remains the primary operational threshold; additional threshold rows are supplementary. "
                "Reclassification (NRI/IDI) is complementary and exploratory.",
                "Análise principal: coorte tripla (AI Risk + EuroSCORE II + STS Score nos mesmos pacientes). "
                "O limiar de 8% permanece como limiar operacional principal; as demais linhas de limiar são suplementares. "
                "Reclassificação (NRI/IDI) é complementar e exploratória.",
            )),
        ]
        readme_df = pd.DataFrame(readme_rows, columns=[_tr("Field", "Campo"), _tr("Value", "Valor")])
        readme_df.to_excel(writer, sheet_name="00_README", index=False)

        # ── 01_EXECUTIVE_SUMMARY ───────────────────────────────────────────
        exec_rows = []
        if not triple_ci.empty:
            try:
                best_auc = triple_ci.sort_values("AUC", ascending=False).iloc[0]
                best_brier = triple_ci.sort_values("Brier", ascending=True).iloc[0]
                exec_rows += [
                    (_tr("Triple cohort n", "n coorte tripla"), n_triple),
                    (_tr("Decision threshold", "Limiar de decisão"), f"{threshold:.0%}"),
                    (_tr("Best AUC (score)", "Melhor AUC (escore)"), best_auc["Score"]),
                    ("AUC", f"{best_auc['AUC']:.3f}"),
                    ("AUC 95% CI", f"{best_auc.get('AUC_IC95_inf', ''):.3f}–{best_auc.get('AUC_IC95_sup', ''):.3f}"),
                    (_tr("Best Brier (score)", "Melhor Brier (escore)"), best_brier["Score"]),
                    ("Brier", f"{best_brier['Brier']:.4f}"),
                ]
            except Exception:
                pass
        if not calib_df.empty:
            for _, row in calib_df.iterrows():
                exec_rows.append((
                    f"{row['Score']} — {_tr('Intercept / Slope', 'Intercepto / Slope')}",
                    f"{row.get('Calibration intercept', ''):.4f} / {row.get('Calibration slope', ''):.4f}",
                ))
        if not threshold_metrics.empty:
            for _, row in threshold_metrics.iterrows():
                exec_rows.append((
                    f"{row['Score']} — {_tr('Sensitivity / Specificity', 'Sensibilidade / Especificidade')} @ {threshold:.0%}",
                    f"{row.get('Sensitivity', ''):.3f} / {row.get('Specificity', ''):.3f}",
                ))
        exec_df = pd.DataFrame(exec_rows, columns=[_tr("Metric", "Métrica"), _tr("Value", "Valor")])
        exec_df.to_excel(writer, sheet_name="01_EXECUTIVE_SUMMARY", index=False)

        # ── 02_MAIN_METRICS ────────────────────────────────────────────────
        if not triple_ci.empty:
            triple_ci.to_excel(writer, sheet_name="02_MAIN_METRICS", index=False)

        # ── 03_THRESHOLD_PERF ─────────────────────────────────────────────
        if not threshold_metrics.empty:
            threshold_metrics.to_excel(writer, sheet_name="03_THRESHOLD_PERF", index=False)

        # ── 04_CALIBRATION ────────────────────────────────────────────────
        if not calib_df.empty:
            calib_df.to_excel(writer, sheet_name="04_CALIBRATION", index=False)

        # ── 05_PAIRWISE ───────────────────────────────────────────────────
        pairwise_frames = []
        if not delong_df.empty:
            dl = delong_df.copy()
            dl.insert(0, _tr("Method", "Método"), "DeLong")
            pairwise_frames.append(dl)
        if not formal_df.empty:
            bt = formal_df.copy()
            bt.insert(0, _tr("Method", "Método"), "Bootstrap")
            pairwise_frames.append(bt)
        if pairwise_frames:
            pd.concat(pairwise_frames, ignore_index=True).to_excel(
                writer, sheet_name="05_PAIRWISE", index=False
            )

        # ── 06_RECLASSIFICATION ───────────────────────────────────────────
        if not reclass_df.empty:
            reclass_df.to_excel(writer, sheet_name="06_RECLASSIFICATION", index=False)

        # ── 07_CLINICAL_UTILITY (DCA) ─────────────────────────────────────
        if dca_df is not None and not dca_df.empty:
            dca_df.to_excel(writer, sheet_name="07_CLINICAL_UTILITY", index=False)

        # ── 08_OVERALL_COMPARE ────────────────────────────────────────────
        if metrics_all is not None and not metrics_all.empty:
            metrics_all.to_excel(writer, sheet_name="08_OVERALL_COMPARE", index=False)

        # ── 09_ALLPAIRS_FULL ──────────────────────────────────────────────
        if pair_df is not None and not pair_df.empty:
            pair_df.to_excel(writer, sheet_name="09_ALLPAIRS_FULL", index=False)

        # ── 10_THRESHOLD_COMPARISON ───────────────────────────────────────
        if threshold_comparison_df is not None:
            _threshold_comp.to_excel(writer, sheet_name="10_THRESHOLD_COMPARISON", index=False)

        # ── 11-13 figure source-data sheets ───────────────────────────────
        if roc_plot_df is not None:
            _roc_plot.to_excel(writer, sheet_name="11_FIG_ROC_DATA", index=False)
        if calibration_plot_df is not None:
            _cal_plot.to_excel(writer, sheet_name="12_FIG_CALIBRATION_DATA", index=False)
        if dca_plot_df is not None:
            _dca_plot.to_excel(writer, sheet_name="13_FIG_DCA_DATA", index=False)

        # ── 14_APPENDIX (caller-supplied extra sheets) ────────────────────
        if extra_sheets:
            for sheet_key, sheet_df in extra_sheets.items():
                safe_name = re.sub(r'[\\/*?\[\]:/]', '_', str(sheet_key))[:31]
                sheet_df.to_excel(writer, sheet_name=safe_name, index=False)

    return buf.getvalue()


def statistical_summary_to_pdf(md_text: str) -> bytes:
    """Convert a Markdown report to PDF.

    Renders all Markdown elements in a single pass:
      H1 / H2 / H3 headers, metadata lines (**Key:** Value),
      pipe tables, bullet lists, paragraphs, horizontal rules.

    Backward-compatible: statistical summaries (headers + tables only) render
    identically to the previous implementation.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return b""

    def _latin_safe(text: str) -> str:
        """Replace Unicode characters unsupported by Helvetica with ASCII equivalents."""
        replacements = {
            "\u0394": "Delta ",  # Δ
            "\u2264": "<=",      # ≤
            "\u2265": ">=",      # ≥
            "\u00b2": "^2",      # ²  → ^2 (e.g. m² → m^2)
            "\u2013": "-",       # –
            "\u2014": "--",      # —
            "\u00b3": "^3",      # ³  → ^3 (e.g. 10³/μL → 10^3/uL)
            "\u03c7": "chi",     # χ
            "\u2022": "-",       # •
            "\u2192": "->",      # →
            "\u2190": "<-",      # ←
            "\u00b0": " deg",    # °
            "\u00b5": "u",       # µ  (micro sign, U+00B5)
            "\u03bc": "u",       # μ  (Greek mu, U+03BC — used in μL, μg)
            "\u03b1": "alpha",   # α
            "\u03b2": "beta",    # β
            "\u03c3": "sigma",   # σ
            "\u00d7": "x",       # ×  (multiplication sign)
            "\u00b1": "+/-",     # ±
            "\u2215": "/",       # ∕  (division slash)
            "\u2248": "~",       # ≈
            "\u221e": "inf",     # ∞
            "\u2020": "+",       # †  (dagger used in footnotes)
            "\u2021": "++",      # ‡  (double dagger)
        }
        # NOTE: Latin-1 accented characters (é ç ã õ à â ê ó ú í and their
        # uppercase equivalents) are intentionally NOT listed here.  They are
        # all valid ISO-8859-1 / CP1252 code points and are rendered correctly
        # by fpdf2's Helvetica core font without any substitution.  Converting
        # them to ASCII equivalents would degrade Portuguese text.
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def _strip_inline_md(text: str) -> str:
        """Strip inline Markdown markers (bold, italic) for plain-text rendering."""
        return re.sub(r'\*+', '', text)

    def _is_separator_row(cells: list) -> bool:
        """True if every non-empty cell contains only dashes, colons, and spaces."""
        return bool(cells) and all(re.fullmatch(r'[-:\s]+', c) for c in cells if c)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    available_w = pdf.w - pdf.l_margin - pdf.r_margin

    # --- Table render buffer ---
    tbl_headers: list = []
    tbl_rows: list = []

    def _flush_table() -> None:
        """Render the buffered table with proportional column widths.

        Column-width strategy
        ─────────────────────
        Each column is allocated width proportional to the length of its
        header label, with the *first* column (Score / Comparison) given
        extra weight because it holds the longest label text.  The widths
        are then scaled so the total equals ``available_w``.

        Font-size strategy
        ──────────────────
        * n_cols ≤ 6  → 8 pt  (normal table)
        * n_cols ≤ 9  → 7 pt  (split sub-table after our column reduction)
        * n_cols > 9  → 6 pt  (fallback for any remaining wide tables)

        The approximate character capacity per mm is derived from the
        Helvetica core font metrics (empirically ≈ 0.24 mm/pt × font_pt).
        Cells are truncated to fit, preventing overlap.
        """
        nonlocal tbl_headers, tbl_rows
        if not tbl_headers:
            return
        n_cols = len(tbl_headers)

        # ── Choose font size based on column count ────────────────────────
        if n_cols <= 6:
            font_sz = 8
        elif n_cols <= 9:
            font_sz = 7
        else:
            font_sz = 6
        # Empirical Helvetica character width (mm) at given pt size
        char_w_mm = font_sz * 0.24
        row_h_hdr = 5.5 if font_sz >= 8 else 5.0
        row_h_dat = 5.0 if font_sz >= 8 else 4.5

        # ── Proportional column widths ─────────────────────────────────────
        # Weight = max(header_length, 4).
        # First column gets 1.6× its natural weight (Score/Comparison labels).
        weights = []
        for i, h in enumerate(tbl_headers):
            w = max(4, len(_strip_inline_md(h)))
            if i == 0:
                w = int(w * 1.6)
            weights.append(w)
        total_weight = sum(weights)
        col_widths = [available_w * wt / total_weight for wt in weights]

        # ── Header row ────────────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", font_sz)
        for h, cw in zip(tbl_headers, col_widths):
            max_chars = max(3, int(cw / char_w_mm))
            txt = _latin_safe(_strip_inline_md(h)[:max_chars])
            pdf.cell(cw, row_h_hdr, txt, border=1, align="C")
        pdf.ln()

        # ── Data rows ─────────────────────────────────────────────────────
        pdf.set_font("Helvetica", "", font_sz)
        for row in tbl_rows:
            # Pad short rows so cell count always equals n_cols
            padded = (row + [""] * n_cols)[:n_cols]
            for j, (cell, cw) in enumerate(zip(padded, col_widths)):
                max_chars = max(3, int(cw / char_w_mm))
                txt = _latin_safe(_strip_inline_md(cell)[:max_chars])
                pdf.cell(cw, row_h_dat, txt, border=1,
                         align="L" if j == 0 else "C")
            pdf.ln()

        pdf.ln(4)
        tbl_headers = []
        tbl_rows = []

    # --- Single-pass line renderer ---
    for raw_line in md_text.split("\n"):
        stripped = raw_line.strip()

        # --- Table lines ---
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if _is_separator_row(cells):
                pass  # skip alignment row
            elif not tbl_headers:
                tbl_headers = cells
            else:
                tbl_rows.append(cells)
            continue

        # Non-table line: flush any pending table first
        _flush_table()

        if not stripped:
            pdf.ln(3)

        elif stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(available_w, 7, _latin_safe(_strip_inline_md(stripped[4:])),
                     new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(available_w, 8, _latin_safe(_strip_inline_md(stripped[3:])),
                     new_x="LMARGIN", new_y="NEXT")
            # Thin grey underline to visually anchor the section heading.
            _uy = pdf.get_y()
            pdf.set_draw_color(160, 160, 160)
            pdf.set_line_width(0.2)
            pdf.line(pdf.l_margin, _uy, pdf.w - pdf.r_margin, _uy)
            pdf.set_draw_color(0, 0, 0)
            pdf.set_line_width(0.2)
            pdf.ln(3)

        elif stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(available_w, 10, _latin_safe(_strip_inline_md(stripped[2:])),
                     new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

        elif stripped == "---":
            pdf.ln(2)  # breathing room before the rule
            y = pdf.get_y()
            pdf.set_draw_color(180, 180, 180)
            pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.set_draw_color(0, 0, 0)
            pdf.set_line_width(0.2)
            pdf.ln(4)

        elif stripped.startswith("- "):
            pdf.set_font("Helvetica", "", 9)
            content = _latin_safe(_strip_inline_md(stripped[2:]))
            # Indent bullet text by 5 mm; use middle-dot (U+00B7, Latin-1 0xB7)
            # as the bullet marker — renders reliably in Helvetica without Unicode.
            indent = 5
            pdf.set_x(pdf.l_margin + indent)
            pdf.multi_cell(available_w - indent, 5.5, "\xb7 " + content)

        elif stripped.startswith("**") and ":**" in stripped:
            # Bold metadata line: **Key:** value  →  Key: value
            pdf.set_font("Helvetica", "", 9)
            meta = _latin_safe(_strip_inline_md(stripped))
            pdf.cell(available_w, 5, meta, new_x="LMARGIN", new_y="NEXT")

        elif stripped.startswith("*") and stripped.endswith("*"):
            # Italic line (e.g. *Generated by AI Risk*)
            pdf.set_font("Helvetica", "I", 8)
            pdf.cell(available_w, 5, _latin_safe(_strip_inline_md(stripped)),
                     new_x="LMARGIN", new_y="NEXT")

        else:
            # Regular paragraph — slightly looser line spacing and a small gap
            # after so the text doesn't visually merge with what follows.
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(available_w, 5.5, _latin_safe(_strip_inline_md(stripped)))
            pdf.ln(2)

    _flush_table()  # flush any table at end of input

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Comparison Summary Report and Full Package
# ---------------------------------------------------------------------------

def _build_comparison_summary_md(
    triple_ci: pd.DataFrame,
    calib_df: pd.DataFrame,
    formal_df: pd.DataFrame,
    delong_df: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    threshold: float,
    n_triple: int,
    model_version: str,
    language: str = "English",
) -> str:
    """Internal: curated Markdown for the Summary Report PDF.

    Includes: main performance + narrative, Calibration at a Glance, threshold
    performance, pairwise comparisons (DeLong + bootstrap), interpretation note.
    Excludes: full calibration table and NRI/IDI (those are in the Full Package).
    """
    def _tr(en, pt):
        return en if language == "English" else pt

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# {_tr('Statistical Comparison — Summary Report', 'Comparação Estatística — Relatório Sumário')}",
        "",
        f"**{_tr('Generated', 'Gerado em')}:** {now}",
        f"**{_tr('Model version', 'Versão do modelo')}:** {model_version}",
        f"**{_tr('Triple comparison sample', 'Amostra da comparação tripla')}:** n = {n_triple}",
        f"**{_tr('Decision threshold', 'Limiar de decisão')}:** {threshold:.0%}",
        "",
    ]

    # Main performance with 95% CI
    if not triple_ci.empty:
        lines.append(f"## {_tr('Main Performance (95% CI)', 'Desempenho Principal (IC 95%)')}")
        lines.append("")
        lines.append(f"| {_tr('Score', 'Escore')} | n | AUC (95% CI) | AUPRC (95% CI) | Brier (95% CI) |")
        lines.append("|:--|:--|:--|:--|:--|")
        for _, row in triple_ci.iterrows():
            auc_ci = f"{row['AUC']:.3f} ({row.get('AUC_IC95_inf', np.nan):.3f}-{row.get('AUC_IC95_sup', np.nan):.3f})"
            auprc_ci = f"{row['AUPRC']:.3f} ({row.get('AUPRC_IC95_inf', np.nan):.3f}-{row.get('AUPRC_IC95_sup', np.nan):.3f})"
            brier_ci = f"{row['Brier']:.4f} ({row.get('Brier_IC95_inf', np.nan):.4f}-{row.get('Brier_IC95_sup', np.nan):.4f})"
            lines.append(f"| {row['Score']} | {row.get('n', '')} | {auc_ci} | {auprc_ci} | {brier_ci} |")
        lines.append("")
        try:
            best_auc_row = triple_ci.sort_values("AUC", ascending=False).iloc[0]
            best_brier_row = triple_ci.sort_values("Brier", ascending=True).iloc[0]
            lines.append(
                f"> {_tr('Primary result', 'Resultado principal')}: "
                f"{_tr('best discrimination', 'melhor discriminação')} — {best_auc_row['Score']} "
                f"(AUC {best_auc_row['AUC']:.3f}); "
                f"{_tr('best calibration', 'melhor calibração')} — {best_brier_row['Score']} "
                f"(Brier {best_brier_row['Brier']:.4f})."
            )
            lines.append("")
        except Exception:
            pass

    # Calibration at a Glance
    if not calib_df.empty:
        lines.append(f"## {_tr('Calibration at a Glance', 'Calibração em Resumo')}")
        lines.append("")
        lines.append(
            _tr(
                "Intercept near 0 and slope near 1 indicate good calibration. "
                "Brier score measures probabilistic accuracy (lower is better).",
                "Intercepto próximo de 0 e slope próximo de 1 indicam boa calibração. "
                "Brier score mede acurácia probabilística (menor é melhor).",
            )
        )
        lines.append("")
        _has_cil = "CIL" in calib_df.columns
        _has_ici = "ICI" in calib_df.columns
        _hdr = f"| {_tr('Score', 'Escore')} | {_tr('Intercept', 'Intercepto')} | Slope"
        if _has_cil:
            _hdr += " | CIL"
        if _has_ici:
            _hdr += " | ICI"
        _hdr += " | Brier | HL p |"
        _sep = "|:--|:--|:--" + ("|:--" if _has_cil else "") + ("|:--" if _has_ici else "") + "|:--|:--|"
        lines.append(_hdr)
        lines.append(_sep)
        for _, row in calib_df.iterrows():
            brier_val = f"{row['Brier']:.4f}" if pd.notna(row.get('Brier')) else "-"
            _r = (
                f"| {row['Score']} | {row.get('Calibration intercept', np.nan):.4f} "
                f"| {row.get('Calibration slope', np.nan):.4f}"
            )
            if _has_cil:
                _cil = f"{row['CIL']:.4f}" if pd.notna(row.get('CIL')) else "-"
                _r += f" | {_cil}"
            if _has_ici:
                _ici = f"{row['ICI']:.4f}" if pd.notna(row.get('ICI')) else "-"
                _r += f" | {_ici}"
            _r += f" | {brier_val} | {row.get('HL p-value', np.nan):.4f} |"
            lines.append(_r)
        lines.append("")

    # Threshold classification
    if not threshold_metrics.empty:
        lines.append(f"## {_tr('Classification at threshold', 'Classificação no limiar')} {threshold:.0%}")
        lines.append("")
        lines.append(f"| {_tr('Score', 'Escore')} | {_tr('Sensitivity', 'Sensibilidade')} | {_tr('Specificity', 'Especificidade')} | PPV | NPV |")
        lines.append("|:--|:--|:--|:--|:--|")
        for _, row in threshold_metrics.iterrows():
            ppv = f"{row['PPV']:.3f}" if pd.notna(row.get('PPV')) else "-"
            npv = f"{row['NPV']:.3f}" if pd.notna(row.get('NPV')) else "-"
            lines.append(f"| {row['Score']} | {row.get('Sensitivity', np.nan):.3f} | {row.get('Specificity', np.nan):.3f} | {ppv} | {npv} |")
        lines.append("")

    # Pairwise — DeLong
    if not delong_df.empty:
        lines.append(f"## {_tr('Pairwise Comparisons', 'Comparações Pareadas')}")
        lines.append("")
        comp_col = [c for c in delong_df.columns if "Comparison" in c or "Comparação" in c]
        comp_key = comp_col[0] if comp_col else "Comparison"
        lines.append(f"### {_tr('DeLong Test', 'Teste de DeLong')}")
        lines.append("")
        lines.append(f"| {_tr('Comparison', 'Comparação')} | ΔAUC | z | p |")
        lines.append("|:--|:--|:--|:--|")
        for _, row in delong_df.iterrows():
            lines.append(f"| {row.get(comp_key, '')} | {row.get('Delta AUC', np.nan):.3f} | {row.get('z', np.nan):.2f} | {row.get('p (DeLong)', np.nan):.4f} |")
        lines.append("")

    # Pairwise — Bootstrap
    if not formal_df.empty:
        comp_col = [c for c in formal_df.columns if "Comparison" in c or "Comparação" in c]
        comp_key = comp_col[0] if comp_col else "Comparison"
        ci_lo_col = [c for c in formal_df.columns if "CI low" in c or "IC95% inf" in c]
        ci_hi_col = [c for c in formal_df.columns if "CI high" in c or "IC95% sup" in c]
        lo_key = ci_lo_col[0] if ci_lo_col else "95% CI low"
        hi_key = ci_hi_col[0] if ci_hi_col else "95% CI high"
        lines.append(f"### {_tr('Bootstrap AUC Comparison', 'Comparação de AUC por Bootstrap')}")
        lines.append("")
        lines.append(f"| {_tr('Comparison', 'Comparação')} | ΔAUC | 95% CI | p |")
        lines.append("|:--|:--|:--|:--|")
        for _, row in formal_df.iterrows():
            lines.append(f"| {row.get(comp_key, '')} | {row.get('Delta AUC (A-B)', np.nan):.3f} | {row.get(lo_key, np.nan):.3f}-{row.get(hi_key, np.nan):.3f} | {row.get('p (bootstrap)', np.nan):.4f} |")
        lines.append("")

    # Interpretation note
    lines.append(f"## {_tr('Interpretation note', 'Nota interpretativa')}")
    lines.append("")
    lines.append(
        _tr(
            "This summary report covers primary discrimination, calibration, and pairwise comparison metrics. "
            "Reclassification statistics (NRI/IDI), the full calibration table, and threshold comparison across candidate cutoffs are included in the Full Package export.",
            "Este relatório sumário cobre métricas primárias de discriminação, calibração e comparação pareada. "
            "Estatísticas de reclassificação (NRI/IDI), a tabela completa de calibração e a comparação de limiares entre cortes candidatos estão no pacote completo.",
        )
    )
    lines.append("")
    lines.append("---")
    lines.append(f"*{_tr('Generated by AI Risk', 'Gerado pelo AI Risk')}*")

    return "\n".join(lines)


def _build_comparison_full_md(
    triple_ci: pd.DataFrame,
    calib_df: pd.DataFrame,
    formal_df: pd.DataFrame,
    delong_df: pd.DataFrame,
    reclass_df: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    threshold: float,
    n_triple: int,
    model_version: str,
    language: str = "English",
    metrics_all: pd.DataFrame = None,
    dca_df: pd.DataFrame = None,
    pair_df: pd.DataFrame = None,
    threshold_comparison_df: pd.DataFrame = None,
) -> str:
    """Internal: comprehensive Markdown for the Full Report PDF.

    Covers every section shown in the Comparison UI:
      Executive Summary, Overall Comparison, Main Performance (95% CI),
      Calibration at a Glance, Full Calibration, Threshold Classification,
      All-Pairs Pairwise (full cohort), DeLong, Bootstrap, DCA,
      Reclassification (NRI/IDI), Interpretation, Methodological Appendix.
    """
    def _tr(en, pt):
        return en if language == "English" else pt

    _metrics_all = metrics_all if metrics_all is not None else pd.DataFrame()
    _dca_df = dca_df if dca_df is not None else pd.DataFrame()
    _pair_df = pair_df if pair_df is not None else pd.DataFrame()
    _threshold_comp = threshold_comparison_df if threshold_comparison_df is not None else pd.DataFrame()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# {_tr('Statistical Comparison — Full Report', 'Comparação Estatística — Relatório Completo')}",
        "",
        f"**{_tr('Generated', 'Gerado em')}:** {now}",
        f"**{_tr('Model version', 'Versão do modelo')}:** {model_version}",
        f"**{_tr('Triple comparison sample', 'Amostra da comparação tripla')}:** n = {n_triple}",
        f"**{_tr('Decision threshold', 'Limiar de decisão')}:** {threshold:.0%}",
        "",
    ]

    # ── Section 1: Executive Summary ──────────────────────────────────────
    lines.append(f"## {_tr('Executive Summary', 'Resumo Executivo')}")
    lines.append("")
    if not triple_ci.empty:
        try:
            best_auc = triple_ci.sort_values("AUC", ascending=False).iloc[0]
            best_brier = triple_ci.sort_values("Brier", ascending=True).iloc[0]
            best_auprc = triple_ci.sort_values("AUPRC", ascending=False).iloc[0]
            lines += [
                f"| {_tr('Metric', 'Métrica')} | {_tr('Best score', 'Melhor escore')} | {_tr('Value', 'Valor')} |",
                "|:--|:--|:--|",
                f"| {_tr('Discrimination (AUC)', 'Discriminação (AUC)')} | {best_auc['Score']} | "
                f"{best_auc['AUC']:.3f} (95% CI {best_auc.get('AUC_IC95_inf', np.nan):.3f}-{best_auc.get('AUC_IC95_sup', np.nan):.3f}) |",
                f"| AUPRC | {best_auprc['Score']} | "
                f"{best_auprc['AUPRC']:.3f} (95% CI {best_auprc.get('AUPRC_IC95_inf', np.nan):.3f}-{best_auprc.get('AUPRC_IC95_sup', np.nan):.3f}) |",
                f"| {_tr('Calibration (Brier)', 'Calibração (Brier)')} | {best_brier['Score']} | "
                f"{best_brier['Brier']:.4f} (95% CI {best_brier.get('Brier_IC95_inf', np.nan):.4f}-{best_brier.get('Brier_IC95_sup', np.nan):.4f}) |",
                f"| n (triple cohort) | — | {n_triple} |",
                f"| {_tr('Decision threshold', 'Limiar de decisão')} | — | {threshold:.0%} |",
                "",
            ]
        except Exception:
            pass

    # ── Section 2: Overall comparison (all available patients) ────────────
    if not _metrics_all.empty:
        lines.append(f"## {_tr('Overall Comparison (All Available Patients)', 'Comparação Geral (Todos os Pacientes Disponíveis)')}")
        lines.append("")
        lines.append(_tr(
            "Each score evaluated with all patients for whom it is available. Sample sizes differ — "
            "use the triple cohort analysis below for direct head-to-head comparison.",
            "Cada escore avaliado com todos os pacientes disponíveis para ele. Tamanhos de amostra diferem — "
            "use a análise da coorte tripla abaixo para comparação direta.",
        ))
        lines.append("")
        lines.extend(_df_to_md_table(_metrics_all))
        lines.append("")

    # ── Section 3: Main performance — triple cohort (95% CI) ─────────────
    if not triple_ci.empty:
        lines.append(f"## {_tr('Main Performance — Triple Cohort (95% CI)', 'Desempenho Principal — Coorte Tripla (IC 95%)')}")
        lines.append("")
        lines.append(_tr(
            "Primary analysis: all three scores evaluated in the same patients (matched cohort). "
            "95% CI via 2000 bootstrap resamples (seed=42).",
            "Análise principal: os três escores avaliados nos mesmos pacientes (coorte pareada). "
            "IC 95% via 2000 reamostras bootstrap (seed=42).",
        ))
        lines.append("")
        lines.append(f"| {_tr('Score', 'Escore')} | n | AUC (95% CI) | AUPRC (95% CI) | Brier (95% CI) |")
        lines.append("|:--|:--|:--|:--|:--|")
        for _, row in triple_ci.iterrows():
            auc_ci = f"{row['AUC']:.3f} ({row.get('AUC_IC95_inf', np.nan):.3f}-{row.get('AUC_IC95_sup', np.nan):.3f})"
            auprc_ci = f"{row['AUPRC']:.3f} ({row.get('AUPRC_IC95_inf', np.nan):.3f}-{row.get('AUPRC_IC95_sup', np.nan):.3f})"
            brier_ci = f"{row['Brier']:.4f} ({row.get('Brier_IC95_inf', np.nan):.4f}-{row.get('Brier_IC95_sup', np.nan):.4f})"
            lines.append(f"| {row['Score']} | {row.get('n', '')} | {auc_ci} | {auprc_ci} | {brier_ci} |")
        lines.append("")
        try:
            best_auc = triple_ci.sort_values("AUC", ascending=False).iloc[0]
            best_brier = triple_ci.sort_values("Brier", ascending=True).iloc[0]
            lines.append(
                f"> {_tr('Primary result', 'Resultado principal')}: "
                f"{_tr('best AUC', 'melhor AUC')} — {best_auc['Score']} ({best_auc['AUC']:.3f}); "
                f"{_tr('best Brier', 'melhor Brier')} — {best_brier['Score']} ({best_brier['Brier']:.4f})."
            )
            lines.append("")
        except Exception:
            pass

    # ── Section 4: Calibration at a Glance ───────────────────────────────
    if not calib_df.empty:
        lines.append(f"## {_tr('Calibration at a Glance', 'Calibração em Resumo')}")
        lines.append("")
        lines.append(_tr(
            "Intercept near 0 and slope near 1 indicate good calibration. "
            "Brier measures probabilistic accuracy (lower is better). HL p-value is complementary only.",
            "Intercepto próximo de 0 e slope próximo de 1 indicam boa calibração. "
            "Brier mede acurácia probabilística (menor é melhor). p-valor de HL é apenas complementar.",
        ))
        lines.append("")
        _has_cil = "CIL" in calib_df.columns
        _has_ici = "ICI" in calib_df.columns
        _hdr = f"| {_tr('Score', 'Escore')} | {_tr('Intercept', 'Intercepto')} | Slope"
        if _has_cil:
            _hdr += " | CIL"
        if _has_ici:
            _hdr += " | ICI"
        _hdr += " | Brier | HL p |"
        _sep = "|:--|:--|:--" + ("|:--" if _has_cil else "") + ("|:--" if _has_ici else "") + "|:--|:--|"
        lines.append(_hdr)
        lines.append(_sep)
        for _, row in calib_df.iterrows():
            brier_val = f"{row['Brier']:.4f}" if pd.notna(row.get('Brier')) else "-"
            _r = (
                f"| {row['Score']} | {row.get('Calibration intercept', np.nan):.4f} "
                f"| {row.get('Calibration slope', np.nan):.4f}"
            )
            if _has_cil:
                _cil = f"{row['CIL']:.4f}" if pd.notna(row.get('CIL')) else "-"
                _r += f" | {_cil}"
            if _has_ici:
                _ici = f"{row['ICI']:.4f}" if pd.notna(row.get('ICI')) else "-"
                _r += f" | {_ici}"
            _r += f" | {brier_val} | {row.get('HL p-value', np.nan):.4f} |"
            lines.append(_r)
        lines.append("")

    # ── Section 5: Full calibration table ────────────────────────────────
    if not calib_df.empty:
        lines.append(f"## {_tr('Calibration (Full)', 'Calibração (Completa)')}")
        lines.append("")
        _has_cil = "CIL" in calib_df.columns
        _has_ici = "ICI" in calib_df.columns
        _hdr = f"| {_tr('Score', 'Escore')} | {_tr('Intercept', 'Intercepto')} | Slope"
        if _has_cil:
            _hdr += " | CIL"
        if _has_ici:
            _hdr += " | ICI"
        _hdr += " | HL chi2 | HL p | Brier |"
        _sep = "|:--|:--|:--" + ("|:--" if _has_cil else "") + ("|:--" if _has_ici else "") + "|:--|:--|:--|"
        lines.append(_hdr)
        lines.append(_sep)
        for _, row in calib_df.iterrows():
            brier_val = f"{row['Brier']:.4f}" if pd.notna(row.get('Brier')) else "-"
            _r = (
                f"| {row['Score']} | {row.get('Calibration intercept', np.nan):.4f} "
                f"| {row.get('Calibration slope', np.nan):.4f}"
            )
            if _has_cil:
                _cil = f"{row['CIL']:.4f}" if pd.notna(row.get('CIL')) else "-"
                _r += f" | {_cil}"
            if _has_ici:
                _ici = f"{row['ICI']:.4f}" if pd.notna(row.get('ICI')) else "-"
                _r += f" | {_ici}"
            _r += (
                f" | {row.get('HL chi-square', np.nan):.2f} "
                f"| {row.get('HL p-value', np.nan):.4f} | {brier_val} |"
            )
            lines.append(_r)
        lines.append("")

    # ── Section 6: Threshold classification ──────────────────────────────
    if not threshold_metrics.empty:
        lines.append(f"## {_tr('Classification at Threshold', 'Classificação no Limiar')} {threshold:.0%}")
        lines.append("")
        lines.append(_tr(
            "PPV and NPV depend strongly on event prevalence and selected threshold.",
            "PPV e NPV dependem fortemente da prevalência do evento e do limiar selecionado.",
        ))
        lines.append("")
        lines.append(f"| {_tr('Score', 'Escore')} | {_tr('Sensitivity', 'Sensibilidade')} | {_tr('Specificity', 'Especificidade')} | PPV | NPV |")
        lines.append("|:--|:--|:--|:--|:--|")
        for _, row in threshold_metrics.iterrows():
            ppv = f"{row['PPV']:.3f}" if pd.notna(row.get('PPV')) else "-"
            npv = f"{row['NPV']:.3f}" if pd.notna(row.get('NPV')) else "-"
            lines.append(f"| {row['Score']} | {row.get('Sensitivity', np.nan):.3f} | {row.get('Specificity', np.nan):.3f} | {ppv} | {npv} |")
        lines.append("")

    # ── Section 7: Threshold comparison across candidate cutoffs ─────────
    lines.append(
        f"## {_tr('Threshold Performance Across Candidate Cutoffs', 'Desempenho por Limiar nos Cortes Candidatos')}"
    )
    lines.append("")
    lines.append(_tr(
        "Operational note: 8% remains the primary clinical threshold. The other thresholds below are supplementary for comparison.",
        "Nota operacional: 8% permanece como limiar clínico principal. Os demais limiares abaixo são suplementares para comparação.",
    ))
    lines.append("")
    if not _threshold_comp.empty:
        lines.extend(_df_to_md_table(_threshold_comp, float_fmt=".4f"))
    else:
        lines.append(_tr(
            "Threshold comparison table unavailable for this export.",
            "Tabela de comparação de limiares indisponível nesta exportação.",
        ))
    lines.append("")

    # ── Section 8: Pairwise comparisons ──────────────────────────────────
    if not _pair_df.empty or not delong_df.empty or not formal_df.empty:
        lines.append(f"## {_tr('Pairwise Comparisons', 'Comparações Pareadas')}")
        lines.append("")

    if not _pair_df.empty:
        lines.append(f"### {_tr('Bootstrap — All Pairs, Full Cohort', 'Bootstrap — Todos os Pares, Coorte Completa')}")
        lines.append("")
        lines.append(_tr(
            "Uses all available patients per pair (larger n than triple cohort). Complementary to the triple analysis.",
            "Usa todos os pacientes disponíveis por par (n maior que coorte tripla). Complementar à análise tripla.",
        ))
        lines.append("")
        lines.extend(_df_to_md_table(_pair_df))
        lines.append("")

    if not delong_df.empty:
        comp_col = [c for c in delong_df.columns if "Comparison" in c or "Comparação" in c]
        comp_key = comp_col[0] if comp_col else "Comparison"
        lines.append(f"### {_tr('DeLong Test — Triple Cohort', 'Teste de DeLong — Coorte Tripla')}")
        lines.append("")
        lines.append(f"| {_tr('Comparison', 'Comparação')} | ΔAUC | z | p |")
        lines.append("|:--|:--|:--|:--|")
        for _, row in delong_df.iterrows():
            lines.append(f"| {row.get(comp_key, '')} | {row.get('Delta AUC', np.nan):.3f} | {row.get('z', np.nan):.2f} | {row.get('p (DeLong)', np.nan):.4f} |")
        lines.append("")

    if not formal_df.empty:
        comp_col = [c for c in formal_df.columns if "Comparison" in c or "Comparação" in c]
        comp_key = comp_col[0] if comp_col else "Comparison"
        ci_lo_col = [c for c in formal_df.columns if "CI low" in c or "IC95% inf" in c]
        ci_hi_col = [c for c in formal_df.columns if "CI high" in c or "IC95% sup" in c]
        lo_key = ci_lo_col[0] if ci_lo_col else "95% CI low"
        hi_key = ci_hi_col[0] if ci_hi_col else "95% CI high"
        lines.append(f"### {_tr('Bootstrap AUC — Triple Cohort', 'Bootstrap AUC — Coorte Tripla')}")
        lines.append("")
        lines.append(f"| {_tr('Comparison', 'Comparação')} | ΔAUC | 95% CI | p |")
        lines.append("|:--|:--|:--|:--|")
        for _, row in formal_df.iterrows():
            lines.append(f"| {row.get(comp_key, '')} | {row.get('Delta AUC (A-B)', np.nan):.3f} | {row.get(lo_key, np.nan):.3f}-{row.get(hi_key, np.nan):.3f} | {row.get('p (bootstrap)', np.nan):.4f} |")
        lines.append("")

    # ── Section 9: Clinical Utility — DCA ────────────────────────────────
    if not _dca_df.empty:
        lines.append(f"## {_tr('Clinical Utility — Decision Curve Analysis', 'Utilidade Clínica — Decision Curve Analysis')}")
        lines.append("")
        lines.append(_tr(
            "Net benefit across risk thresholds 5%-20%. Higher net benefit indicates greater clinical utility. "
            "Treat-all and treat-none are reference strategies.",
            "Benefício líquido nos limiares de risco 5%-20%. Benefício líquido maior indica maior utilidade clínica. "
            "Tratar todos e tratar nenhum são estratégias de referência.",
        ))
        lines.append("")
        _dca_compact = (
            _dca_df[_dca_df["Threshold"].isin([0.05, 0.10, 0.15, 0.20])].copy()
            if "Threshold" in _dca_df.columns
            else _dca_df
        )
        lines.extend(_df_to_md_table(_dca_compact, float_fmt=".4f"))
        lines.append("")

    # ── Section 10: Reclassification (NRI/IDI) ───────────────────────────
    if not reclass_df.empty:
        lines.append(f"## {_tr('Reclassification (NRI/IDI)', 'Reclassificação (NRI/IDI)')}")
        lines.append("")
        lines.append(_tr(
            "NRI: movement to more appropriate risk categories (<5% low, 5-15% intermediate, >15% high). "
            "IDI: average improvement in separation between events and non-events. "
            "Both are complementary — do not use as sole evidence of superiority.",
            "NRI: movimento para categorias de risco mais apropriadas (<5% baixo, 5-15% intermediário, >15% alto). "
            "IDI: melhora média na separação entre eventos e não-eventos. "
            "Ambas são complementares — não usar como única evidência de superioridade.",
        ))
        lines.append("")
        comp_col = [c for c in reclass_df.columns if "Comparison" in c or "Comparação" in c]
        comp_key = comp_col[0] if comp_col else "Comparison"
        lines.append(f"| {_tr('Comparison', 'Comparação')} | NRI events | NRI non-events | NRI total | IDI |")
        lines.append("|:--|:--|:--|:--|:--|")
        for _, row in reclass_df.iterrows():
            lines.append(
                f"| {row.get(comp_key, '')} | {row.get('NRI events', np.nan):.3f} "
                f"| {row.get('NRI non-events', np.nan):.3f} | {row.get('NRI total', np.nan):.3f} "
                f"| {row.get('IDI', np.nan):.4f} |"
            )
        lines.append("")

    # ── Section 11: Interpretation ────────────────────────────────────────
    lines.append(f"## {_tr('Interpretation', 'Interpretação')}")
    lines.append("")
    if not triple_ci.empty:
        try:
            best_auc = triple_ci.sort_values("AUC", ascending=False).iloc[0]
            best_brier = triple_ci.sort_values("Brier", ascending=True).iloc[0]
            best_auprc = triple_ci.sort_values("AUPRC", ascending=False).iloc[0]
            lines.append(
                _tr(
                    f"In the triple cohort (n={n_triple}), {best_auc['Score']} achieved the highest discrimination "
                    f"(AUC {best_auc['AUC']:.3f}, 95% CI {best_auc.get('AUC_IC95_inf', np.nan):.3f}-{best_auc.get('AUC_IC95_sup', np.nan):.3f}) "
                    f"and {best_auprc['Score']} the highest AUPRC ({best_auprc['AUPRC']:.3f}). "
                    f"Best probabilistic calibration (Brier) was {best_brier['Score']} "
                    f"({best_brier['Brier']:.4f}, 95% CI {best_brier.get('Brier_IC95_inf', np.nan):.4f}-{best_brier.get('Brier_IC95_sup', np.nan):.4f}). "
                    f"DCA and reclassification results are complementary to discrimination and calibration findings.",
                    f"Na coorte tripla (n={n_triple}), {best_auc['Score']} alcançou a maior discriminação "
                    f"(AUC {best_auc['AUC']:.3f}, IC 95% {best_auc.get('AUC_IC95_inf', np.nan):.3f}-{best_auc.get('AUC_IC95_sup', np.nan):.3f}) "
                    f"e {best_auprc['Score']} o maior AUPRC ({best_auprc['AUPRC']:.3f}). "
                    f"Melhor calibração probabilística (Brier) foi {best_brier['Score']} "
                    f"({best_brier['Brier']:.4f}, IC 95% {best_brier.get('Brier_IC95_inf', np.nan):.4f}-{best_brier.get('Brier_IC95_sup', np.nan):.4f}). "
                    f"Resultados de DCA e reclassificação são complementares aos achados de discriminação e calibração.",
                )
            )
        except Exception:
            pass
    lines.append("")

    # ── Section 12: Methodological Appendix ──────────────────────────────
    lines.append(f"## {_tr('Methodological Appendix', 'Apêndice Metodológico')}")
    lines.append("")
    bullet = lambda en, pt: f"- {_tr(en, pt)}"  # noqa: E731
    lines += [
        bullet(
            "Triple cohort: patients with simultaneous AI Risk, EuroSCORE II, and STS Score — ensures identical observations for all comparisons.",
            "Coorte tripla: pacientes com AI Risk, EuroSCORE II e STS Score simultâneos — garante observações idênticas em todas as comparações.",
        ),
        bullet(
            "Confidence intervals: 2000 bootstrap resamples, percentile method, seed=42.",
            "Intervalos de confiança: 2000 reamostras bootstrap, método percentil, seed=42.",
        ),
        bullet(
            "DeLong test: formal AUC comparison for correlated samples (same patients). Complements bootstrap delta AUC.",
            "Teste de DeLong: comparação formal de AUC em amostras correlacionadas (mesmos pacientes). Complementa delta AUC por bootstrap.",
        ),
        bullet(
            "Calibration: intercept-slope method; Hosmer-Lemeshow (10 bins); Brier score. HL p-value should not be interpreted in isolation.",
            "Calibração: método intercepto-slope; Hosmer-Lemeshow (10 bins); Brier score. p-valor de HL não deve ser interpretado isoladamente.",
        ),
        bullet(
            "NRI/IDI: complementary reclassification metrics. Risk cutoffs: <5% low, 5-15% intermediate, >15% high.",
            "NRI/IDI: métricas complementares de reclassificação. Cortes: <5% baixo, 5-15% intermediário, >15% alto.",
        ),
        bullet(
            "Decision curve analysis: net benefit evaluated across threshold range 5%-20%. Treat-all and treat-none as reference strategies.",
            "Decision curve analysis: benefício líquido avaliado no intervalo 5%-20%. Tratar todos e tratar nenhum como referências.",
        ),
        bullet(
            "Figures (ROC curves, calibration plots, boxplots, DCA plot) are rendered interactively in the application and are not included in this export.",
            "Figuras (curvas ROC, gráficos de calibração, boxplots, gráfico de DCA) são renderizadas interativamente no aplicativo e não estão incluídas nesta exportação.",
        ),
    ]
    lines += ["", "---", f"*{_tr('Generated by AI Risk', 'Gerado pelo AI Risk')}*"]

    return "\n".join(lines)


def build_comparison_full_pdf(
    triple_ci: pd.DataFrame,
    calib_df: pd.DataFrame,
    formal_df: pd.DataFrame,
    delong_df: pd.DataFrame,
    reclass_df: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    threshold: float,
    n_triple: int,
    model_version: str,
    language: str = "English",
    metrics_all: pd.DataFrame = None,
    dca_df: pd.DataFrame = None,
    pair_df: pd.DataFrame = None,
    threshold_comparison_df: pd.DataFrame = None,
) -> bytes:
    """Build a comprehensive Full Report PDF for the Comparison tab.

    Covers all sections shown in the Comparison UI: executive summary,
    overall comparison, main performance (95% CI), calibration (at-a-glance
    and full), threshold classification, threshold comparison across candidate
    cutoffs, pairwise comparisons (all-pairs + DeLong + bootstrap), DCA,
    NRI/IDI, interpretation, and methodological appendix. Returns empty bytes
    if fpdf2 is not installed.
    """
    md = _build_comparison_full_md(
        triple_ci=triple_ci,
        calib_df=calib_df,
        formal_df=formal_df,
        delong_df=delong_df,
        reclass_df=reclass_df,
        threshold_metrics=threshold_metrics,
        threshold=threshold,
        n_triple=n_triple,
        model_version=model_version,
        language=language,
        metrics_all=metrics_all,
        dca_df=dca_df,
        pair_df=pair_df,
        threshold_comparison_df=threshold_comparison_df,
    )
    return statistical_summary_to_pdf(md)


def build_comparison_summary_pdf(
    triple_ci: pd.DataFrame,
    calib_df: pd.DataFrame,
    formal_df: pd.DataFrame,
    delong_df: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    threshold: float,
    n_triple: int,
    model_version: str,
    language: str = "English",
) -> bytes:
    """Build a curated editorial PDF for the Comparison tab Summary Report.

    Covers: main performance (95% CI), Calibration at a Glance, threshold
    classification, pairwise comparisons (DeLong + bootstrap).
    NRI/IDI and the full calibration table are intentionally excluded
    (available in the Full Package).
    Returns empty bytes if fpdf2 is not installed.
    """
    md = _build_comparison_summary_md(
        triple_ci=triple_ci,
        calib_df=calib_df,
        formal_df=formal_df,
        delong_df=delong_df,
        threshold_metrics=threshold_metrics,
        threshold=threshold,
        n_triple=n_triple,
        model_version=model_version,
        language=language,
    )
    return statistical_summary_to_pdf(md)


def build_comparison_full_package(
    triple_ci: pd.DataFrame,
    calib_df: pd.DataFrame,
    formal_df: pd.DataFrame,
    delong_df: pd.DataFrame,
    reclass_df: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    threshold: float,
    n_triple: int,
    model_version: str,
    language: str = "English",
    source_file: str = "",
    endpoint: str = "30-day / in-hospital mortality",
    cohort_note: str = "",
    dca_df: pd.DataFrame = None,
    metrics_all: pd.DataFrame = None,
    pair_df: pd.DataFrame = None,
    threshold_comparison_df: pd.DataFrame = None,
    roc_plot_df: pd.DataFrame = None,
    calibration_plot_df: pd.DataFrame = None,
    dca_plot_df: pd.DataFrame = None,
    manifest: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Build a Full Package ZIP for the Comparison tab.

    ZIP contents:
      comparison_summary.pdf      — curated Summary Report PDF (no NRI/IDI, no DCA)
      comparison_full_report.pdf  — comprehensive PDF (all sections incl. DCA, NRI/IDI, appendix)
      comparison_full_report.md   — full narrative Markdown report
      comparison_tables.xlsx      — structured XLSX workbook (numbered sheets + DCA/overall/allpairs/threshold comparison)
      comparison_metrics.csv      — flat CSV (all tables concatenated)
      figures/*.png               — exportable ROC, calibration, and DCA figures when available
      comparison_*_data.csv       — source data for exported figures
    """
    import zipfile

    summary_pdf = build_comparison_summary_pdf(
        triple_ci=triple_ci,
        calib_df=calib_df,
        formal_df=formal_df,
        delong_df=delong_df,
        threshold_metrics=threshold_metrics,
        threshold=threshold,
        n_triple=n_triple,
        model_version=model_version,
        language=language,
    )

    full_pdf = build_comparison_full_pdf(
        triple_ci=triple_ci,
        calib_df=calib_df,
        formal_df=formal_df,
        delong_df=delong_df,
        reclass_df=reclass_df,
        threshold_metrics=threshold_metrics,
        threshold=threshold,
        n_triple=n_triple,
        model_version=model_version,
        language=language,
        metrics_all=metrics_all,
        dca_df=dca_df,
        pair_df=pair_df,
        threshold_comparison_df=threshold_comparison_df,
    )

    full_md = _build_comparison_full_md(
        triple_ci=triple_ci,
        calib_df=calib_df,
        formal_df=formal_df,
        delong_df=delong_df,
        reclass_df=reclass_df,
        threshold_metrics=threshold_metrics,
        threshold=threshold,
        n_triple=n_triple,
        model_version=model_version,
        language=language,
        metrics_all=metrics_all,
        dca_df=dca_df,
        pair_df=pair_df,
        threshold_comparison_df=threshold_comparison_df,
    )

    xlsx_bytes = build_comparison_xlsx(
        triple_ci=triple_ci,
        calib_df=calib_df,
        formal_df=formal_df,
        delong_df=delong_df,
        reclass_df=reclass_df,
        threshold_metrics=threshold_metrics,
        threshold=threshold,
        n_triple=n_triple,
        model_version=model_version,
        language=language,
        source_file=source_file,
        endpoint=endpoint,
        cohort_note=cohort_note,
        dca_df=dca_df,
        metrics_all=metrics_all,
        pair_df=pair_df,
        threshold_comparison_df=threshold_comparison_df,
        roc_plot_df=roc_plot_df,
        calibration_plot_df=calibration_plot_df,
        dca_plot_df=dca_plot_df,
    )

    csv_bytes = statistical_summary_to_csv(full_md).encode("utf-8")
    figure_pngs = _comparison_figure_pngs(
        roc_plot_df=roc_plot_df,
        calibration_plot_df=calibration_plot_df,
        dca_plot_df=dca_plot_df,
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Manifest is the canonical metadata block — written first so any
        # consumer extracting the ZIP sees the version/threshold/model
        # answer up-front and never has to infer it from filenames.
        if manifest is not None:
            zf.writestr("manifest.json", manifest_to_json_bytes(manifest))
        if summary_pdf:
            zf.writestr("comparison_summary.pdf", summary_pdf)
        if full_pdf:
            zf.writestr("comparison_full_report.pdf", full_pdf)
        zf.writestr("comparison_full_report.md", full_md.encode("utf-8"))
        zf.writestr("comparison_tables.xlsx", xlsx_bytes)
        zf.writestr("comparison_metrics.csv", csv_bytes)
        if roc_plot_df is not None and not roc_plot_df.empty:
            zf.writestr("comparison_roc_data.csv", roc_plot_df.to_csv(index=False).encode("utf-8"))
        if calibration_plot_df is not None and not calibration_plot_df.empty:
            zf.writestr("comparison_calibration_data.csv", calibration_plot_df.to_csv(index=False).encode("utf-8"))
        if dca_plot_df is not None and not dca_plot_df.empty:
            zf.writestr("comparison_dca_data.csv", dca_plot_df.to_csv(index=False).encode("utf-8"))
        for fig_name, fig_bytes in figure_pngs.items():
            if fig_bytes:
                zf.writestr(fig_name, fig_bytes)
    return buf.getvalue()
