"""Plot helper functions for AI Risk charts.

Functions that call st.* or tr() accept a ``tr`` callable so they can
be used independently of app.py's module-level closure.
"""

from __future__ import annotations

from io import BytesIO
from typing import Callable, Dict

import numpy as np
import pandas as pd
import streamlit as st


def _fig_to_png_bytes(fig) -> bytes:
    """Convert a Matplotlib figure to PNG bytes (300 DPI)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


@st.fragment
def _chart_download_buttons(data_df: pd.DataFrame, png_bytes: "bytes | None", chart_name: str):
    """Add XLSX + PNG download buttons below a chart.

    Isolated as a fragment so that clicking either button does not trigger a
    full-page rerun — preventing the MediaFileStorageError that occurs when a
    rerun replaces the registered file before the browser can fetch it.
    """
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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
            patch.set_facecolor("#4C78A8")
            patch.set_alpha(0.6)
        ax.set_title(grp, fontsize=9)
        ax.set_ylabel(y_col if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    png = _fig_to_png_bytes(fig)
    plt.close(fig)
    return png


def _plot_roc(
    scores: Dict[str, np.ndarray],
    y: np.ndarray,
    tr: Callable[[str, str], str],
    roc_data: Callable,
) -> "bytes | None":
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


def _plot_calibration(
    scores: Dict[str, np.ndarray],
    y: np.ndarray,
    tr: Callable[[str, str], str],
    calibration_data: Callable,
) -> "bytes | None":
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


def _plot_boxplots(
    df_plot: pd.DataFrame,
    tr: Callable[[str, str], str],
) -> "bytes | None":
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


def _plot_ia_model_boxplots(
    y_true: np.ndarray,
    oof_predictions: Dict[str, np.ndarray],
    tr: Callable[[str, str], str],
) -> "bytes | None":
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


def _plot_dca(
    curve_df: pd.DataFrame,
    tr: Callable[[str, str], str],
) -> "bytes | None":
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
