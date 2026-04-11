"""Statistical summary export helpers for AI Risk.

Extracted from model_metadata.py to isolate self-contained report/export
logic from core model metadata functions.  All functions here are pure
transformers: they accept DataFrames or Markdown strings as input and return
formatted output (Markdown, DataFrames, bytes).  They carry no project-level
state and import nothing from the AI Risk project modules.

Provides:
- build_statistical_summary  — Markdown statistical report
- statistical_summary_to_dataframes — Markdown → dict of DataFrames
- statistical_summary_to_xlsx       — Markdown → XLSX bytes
- statistical_summary_to_csv        — Markdown → CSV string
- statistical_summary_to_pdf        — Markdown → PDF bytes (requires fpdf2)
"""

import io
import re
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd


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
        lines.append(f"## {_tr('Discrimination (95% CI)', 'Discriminação (IC 95%)')}")
        lines.append("")
        lines.append(f"| {_tr('Score', 'Escore')} | n | AUC (95% CI) | AUPRC (95% CI) | Brier (95% CI) |")
        lines.append("|:--|:--|:--|:--|:--|")
        for _, row in triple_ci.iterrows():
            auc_ci = f"{row['AUC']:.3f} ({row.get('AUC_IC95_inf', np.nan):.3f}-{row.get('AUC_IC95_sup', np.nan):.3f})"
            auprc_ci = f"{row['AUPRC']:.3f} ({row.get('AUPRC_IC95_inf', np.nan):.3f}-{row.get('AUPRC_IC95_sup', np.nan):.3f})"
            brier_ci = f"{row['Brier']:.4f} ({row.get('Brier_IC95_inf', np.nan):.4f}-{row.get('Brier_IC95_sup', np.nan):.4f})"
            lines.append(f"| {row['Score']} | {row.get('n', '')} | {auc_ci} | {auprc_ci} | {brier_ci} |")
        lines.append("")

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

    # Calibration
    if not calib_df.empty:
        lines.append(f"## {_tr('Calibration', 'Calibração')}")
        lines.append("")
        lines.append(f"| {_tr('Score', 'Escore')} | {_tr('Intercept', 'Intercepto')} | Slope | HL chi² | HL p |")
        lines.append("|:--|:--|:--|:--|:--|")
        for _, row in calib_df.iterrows():
            lines.append(f"| {row['Score']} | {row.get('Calibration intercept', np.nan):.4f} | {row.get('Calibration slope', np.nan):.4f} | {row.get('HL chi-square', np.nan):.2f} | {row.get('HL p-value', np.nan):.4f} |")
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
        lines.append(f"| {_tr('Comparison', 'Comparação')} | NRI events | NRI non-events | NRI total | IDI |")
        lines.append("|:--|:--|:--|:--|:--|")
        for _, row in reclass_df.iterrows():
            lines.append(f"| {row.get(comp_key, '')} | {row.get('NRI events', np.nan):.3f} | {row.get('NRI non-events', np.nan):.3f} | {row.get('NRI total', np.nan):.3f} | {row.get('IDI', np.nan):.4f} |")
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


def statistical_summary_to_pdf(md_text: str) -> bytes:
    """Convert statistical summary to PDF."""
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
            "\u00b2": "2",       # ²
            "\u2013": "-",       # –
            "\u2014": "--",      # —
            "\u00b3": "3",       # ³
            "\u03c7": "chi",     # χ
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    tables = _parse_md_tables(md_text)

    # Extract header metadata from markdown
    title = "Statistical Summary"
    header_lines = []
    for line in md_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("**") and ":**" in stripped:
            header_lines.append(stripped.replace("**", ""))
        elif stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped.lstrip("# ").strip()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _latin_safe(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Header metadata
    pdf.set_font("Helvetica", "", 9)
    for hl in header_lines:
        pdf.cell(0, 5, _latin_safe(hl), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    for t in tables:
        # Section title
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, _latin_safe(t["title"]), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        if not t["headers"]:
            continue

        n_cols = len(t["headers"])
        available_width = pdf.w - pdf.l_margin - pdf.r_margin
        col_w = available_width / n_cols

        # Header row
        pdf.set_font("Helvetica", "B", 8)
        for h in t["headers"]:
            pdf.cell(col_w, 6, _latin_safe(h[:20]), border=1, align="C")
        pdf.ln()

        # Data rows
        pdf.set_font("Helvetica", "", 8)
        for row in t["rows"]:
            for j, cell in enumerate(row):
                txt = _latin_safe(cell[:22] if j > 0 else cell[:25])
                pdf.cell(col_w, 5, txt, border=1, align="C" if j > 0 else "L")
            pdf.ln()

        pdf.ln(4)

    return bytes(pdf.output())
