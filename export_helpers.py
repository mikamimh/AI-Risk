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
