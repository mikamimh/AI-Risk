"""Statistical Comparison tab — extracted from app.py (tab index 2).

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
from sklearn.metrics import roc_curve

from stats_compare import (
    bootstrap_auc_diff,
    calibration_bins_detail,
    calibration_in_the_large,
    calibration_intercept_slope,
    classification_metrics_at_threshold,
    compute_idi_with_ci,
    compute_nri_with_ci,
    decision_curve,
    delong_roc_test,
    evaluate_scores,
    evaluate_scores_with_ci,
    evaluate_scores_with_threshold,
    hosmer_lemeshow_test,
    integrated_calibration_index,
    threshold_analysis_table,
    THRESHOLD_ROLE_PRIMARY,
    THRESHOLD_ROLE_FIXED_COMPARATOR,
    THRESHOLD_ROLE_HISTORICAL_COMPARATOR,
    THRESHOLD_ROLE_EXPLORATORY,
)
from export_helpers import (
    build_comparison_full_package,
    build_comparison_summary_pdf,
    build_comparison_xlsx,
    build_export_manifest,
    build_statistical_summary,
    statistical_summary_to_csv,
    statistical_summary_to_pdf,
)

if TYPE_CHECKING:
    from tabs import TabContext


# Canonical display labels for the three scores.
# Any alias that ever appears in triple_ci["Score"] is mapped here so the
# Brier merge never silently returns NaN due to a naming variation.
_CANONICAL_SCORE_LABEL: dict[str, str] = {
    "AI Risk": "AI Risk",
    "EuroSCORE II": "EuroSCORE II",
    "STS": "STS",
    "STS Score": "STS",    # alias used in some display contexts
    "STS_Score": "STS",    # underscore variant
}


def _canon(name: str) -> str:
    """Return the canonical display label for a score name."""
    return _CANONICAL_SCORE_LABEL.get(str(name).strip(), str(name).strip())


def _build_threshold_comparison_export_df(
    df: pd.DataFrame,
    artifacts,
    forced_model: str,
) -> pd.DataFrame:
    """Build a supplementary AI Risk threshold comparison table for exports.

    Includes a Threshold Role column distinguishing Primary (sens-constrained
    90%), Fixed comparator, Historical comparator (8%), and Exploratory (Youden).
    The Primary row is derived from the training bundle's threshold_policy when
    available; it is omitted (as NaN) for legacy bundles.
    """
    cols = [
        "Threshold label",
        "Threshold Role",
        "Threshold",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "Accuracy",
        "TP",
        "FP",
        "TN",
        "FN",
        "High risk (%)",
        "High risk (n)",
        "n",
    ]
    if "ia_risk_oof" not in df.columns or "morte_30d" not in df.columns:
        return pd.DataFrame(columns=cols)

    _mask = df["ia_risk_oof"].notna() & df["morte_30d"].notna()
    _p = df.loc[_mask, "ia_risk_oof"].astype(float).values
    _y = df.loc[_mask, "morte_30d"].astype(int).values
    if len(_p) < 2 or len(np.unique(_y)) < 2:
        return pd.DataFrame(columns=cols)

    _youden_map = getattr(artifacts, "youden_thresholds", None) or {}
    _youden_thr = _youden_map.get(forced_model, getattr(artifacts, "best_youden_threshold", np.nan))
    _youden_thr = float(_youden_thr) if _youden_thr is not None else float("nan")

    # Primary threshold from bundle threshold_policy (NaN for legacy bundles)
    _tp_dict = getattr(artifacts, "threshold_policy", None)
    _primary_thr = float("nan")
    if isinstance(_tp_dict, dict) and _tp_dict.get("status") == "ok":
        _v = _tp_dict.get("selected_threshold")
        if _v is not None and 0.0 < float(_v) < 1.0:
            _primary_thr = float(_v)

    # (label, role, threshold_value)
    candidates: list[tuple[str, str, float]] = [
        ("Sens-90% (Primary)", THRESHOLD_ROLE_PRIMARY,              _primary_thr),
        ("2%",                  THRESHOLD_ROLE_FIXED_COMPARATOR,     0.02),
        ("5%",                  THRESHOLD_ROLE_FIXED_COMPARATOR,     0.05),
        ("8%",                  THRESHOLD_ROLE_HISTORICAL_COMPARATOR, 0.08),
        ("10%",                 THRESHOLD_ROLE_FIXED_COMPARATOR,     0.10),
        ("15%",                 THRESHOLD_ROLE_FIXED_COMPARATOR,     0.15),
        ("Youden",              THRESHOLD_ROLE_EXPLORATORY,          _youden_thr),
    ]
    valid_thresholds = [float(t) for _, _, t in candidates if np.isfinite(t)]
    analysis = threshold_analysis_table(_y, _p, valid_thresholds) if valid_thresholds else pd.DataFrame()

    rows: list[dict] = []
    valid_idx = 0
    n_total = int(len(_y))
    for label, role, thr in candidates:
        if not np.isfinite(thr):
            rows.append({
                "Threshold label": label,
                "Threshold Role": role,
                "Threshold": np.nan,
                "Sensitivity": np.nan,
                "Specificity": np.nan,
                "PPV": np.nan,
                "NPV": np.nan,
                "Accuracy": np.nan,
                "TP": np.nan,
                "FP": np.nan,
                "TN": np.nan,
                "FN": np.nan,
                "High risk (%)": np.nan,
                "High risk (n)": np.nan,
                "n": n_total,
            })
            continue

        _row = analysis.iloc[valid_idx]
        valid_idx += 1
        tp = int(_row.get("TP", 0))
        tn = int(_row.get("TN", 0))
        fp = int(_row.get("FP", 0))
        fn = int(_row.get("FN", 0))
        acc = ((tp + tn) / n_total) if n_total else np.nan
        rows.append({
            "Threshold label": label,
            "Threshold Role": role,
            "Threshold": float(thr),
            "Sensitivity": _row.get("Sensitivity", np.nan),
            "Specificity": _row.get("Specificity", np.nan),
            "PPV": _row.get("PPV", np.nan),
            "NPV": _row.get("NPV", np.nan),
            "Accuracy": acc,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "High risk (%)": _row.get("Flag_Rate_pct", np.nan),
            "High risk (n)": int(_row.get("N_Flagged", 0)),
            "n": n_total,
        })
    return pd.DataFrame(rows, columns=cols)


def _build_figure_export_data(triple: pd.DataFrame, dca_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build figure source-data tables for the Comparison full package."""
    score_cols = {
        "AI Risk": "ia_risk_oof",
        "EuroSCORE II": "euroscore_calc",
        "STS": "sts_score",
    }
    roc_rows = []
    cal_frames = []
    if len(triple) >= 2 and triple["morte_30d"].nunique() > 1:
        y = triple["morte_30d"].astype(int).values
        for score, col in score_cols.items():
            fpr, tpr, thresholds = roc_curve(y, triple[col].values)
            roc_rows.extend(
                {
                    "score": score,
                    "fpr": float(_fpr),
                    "tpr": float(_tpr),
                    "threshold": float(_thr),
                    "cohort": "triple",
                }
                for _fpr, _tpr, _thr in zip(fpr, tpr, thresholds)
            )
            cal = calibration_bins_detail(y, triple[col].values).rename(
                columns={
                    "Bin": "bin",
                    "Mean_Predicted": "mean_predicted_risk",
                    "Obs_Frequency": "observed_event_rate",
                    "N": "n_in_bin",
                }
            )
            if not cal.empty:
                cal.insert(0, "score", score)
                cal["cohort"] = "triple"
                cal_frames.append(
                    cal[["score", "bin", "mean_predicted_risk", "observed_event_rate", "n_in_bin", "cohort"]]
                )

    roc_df = pd.DataFrame(roc_rows, columns=["score", "fpr", "tpr", "threshold", "cohort"])
    cal_df = (
        pd.concat(cal_frames, ignore_index=True)
        if cal_frames
        else pd.DataFrame(columns=["score", "bin", "mean_predicted_risk", "observed_event_rate", "n_in_bin", "cohort"])
    )
    dca_plot_df = pd.DataFrame(columns=["score", "threshold", "net_benefit", "strategy", "cohort"])
    if dca_df is not None and not dca_df.empty:
        dca_plot_df = dca_df.rename(
            columns={
                "Strategy": "strategy",
                "Threshold": "threshold",
                "Net Benefit": "net_benefit",
            }
        ).copy()
        dca_plot_df["score"] = dca_plot_df["strategy"]
        dca_plot_df["cohort"] = "triple"
        dca_plot_df = dca_plot_df[["score", "threshold", "net_benefit", "strategy", "cohort"]]
    return roc_df, cal_df, dca_plot_df


@st.cache_data(show_spinner=False)
def _cached_evaluate_scores_with_ci(
    df: pd.DataFrame,
    y_col: str,
    score_cols: tuple[str, ...],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    return evaluate_scores_with_ci(
        df,
        y_col=y_col,
        score_cols=list(score_cols),
        n_boot=n_boot,
        seed=seed,
    )


@st.cache_data(show_spinner=False)
def _cached_bootstrap_auc_diff(
    y: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    return bootstrap_auc_diff(y, p1, p2, n_boot=n_boot, seed=seed)


@st.cache_data(show_spinner=False)
def _cached_delong_roc_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    return delong_roc_test(y, p1, p2)


@st.cache_data(show_spinner=False)
def _cached_decision_curve(
    y: np.ndarray,
    ai_risk: np.ndarray,
    euroscore: np.ndarray,
    sts_score: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    return decision_curve(
        y,
        {
            "AI Risk": ai_risk,
            "EuroSCORE II": euroscore,
            "STS": sts_score,
        },
        thresholds,
    )


@st.cache_data(show_spinner=False)
def _cached_statistical_summary(
    triple_ci: pd.DataFrame,
    calib_df: pd.DataFrame,
    formal_df: pd.DataFrame,
    delong_df: pd.DataFrame,
    reclass_df: pd.DataFrame,
    threshold: float,
    threshold_metrics: pd.DataFrame,
    n_triple: int,
    model_version: str,
    language: str,
) -> str:
    return build_statistical_summary(
        triple_ci=triple_ci,
        calib_df=calib_df,
        formal_df=formal_df,
        delong_df=delong_df,
        reclass_df=reclass_df,
        threshold=threshold,
        threshold_metrics=threshold_metrics,
        n_triple=n_triple,
        model_version=model_version,
        language=language,
    )


@st.cache_data(show_spinner=False)
def _cached_summary_pdf(
    triple_ci: pd.DataFrame,
    calib_df: pd.DataFrame,
    formal_df: pd.DataFrame,
    delong_df: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    threshold: float,
    n_triple: int,
    model_version: str,
    language: str,
) -> bytes:
    return build_comparison_summary_pdf(
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


@st.cache_data(show_spinner=False)
def _cached_full_package(
    triple_ci: pd.DataFrame,
    calib_df: pd.DataFrame,
    formal_df: pd.DataFrame,
    delong_df: pd.DataFrame,
    reclass_df: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    threshold: float,
    n_triple: int,
    model_version: str,
    language: str,
    dca_df: pd.DataFrame,
    metrics_all: pd.DataFrame,
    pair_df: pd.DataFrame,
    threshold_comparison_df: pd.DataFrame,
    roc_plot_df: pd.DataFrame,
    calibration_plot_df: pd.DataFrame,
    dca_plot_df: pd.DataFrame,
    manifest: dict | None = None,
    # Bump to invalidate stale cache when export_helpers formatting changes.
    _export_version: str = "2026-04-24-p-str",
) -> bytes:
    return build_comparison_full_package(
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
        dca_df=dca_df,
        metrics_all=metrics_all,
        pair_df=pair_df,
        threshold_comparison_df=threshold_comparison_df,
        roc_plot_df=roc_plot_df,
        calibration_plot_df=calibration_plot_df,
        dca_plot_df=dca_plot_df,
        manifest=manifest,
    )


@st.cache_data(show_spinner=False)
def _cached_structured_xlsx(
    triple_ci: pd.DataFrame,
    calib_df: pd.DataFrame,
    formal_df: pd.DataFrame,
    delong_df: pd.DataFrame,
    reclass_df: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    threshold: float,
    n_triple: int,
    model_version: str,
    language: str,
    dca_df: pd.DataFrame,
    metrics_all: pd.DataFrame,
    pair_df: pd.DataFrame,
    threshold_comparison_df: pd.DataFrame,
    roc_plot_df: pd.DataFrame,
    calibration_plot_df: pd.DataFrame,
    dca_plot_df: pd.DataFrame,
) -> bytes:
    return build_comparison_xlsx(
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
        dca_df=dca_df,
        metrics_all=metrics_all,
        pair_df=pair_df,
        threshold_comparison_df=threshold_comparison_df,
        roc_plot_df=roc_plot_df,
        calibration_plot_df=calibration_plot_df,
        dca_plot_df=dca_plot_df,
    )


@st.cache_data(show_spinner=False)
def _cached_pdf_from_markdown(md_text: str) -> bytes:
    return statistical_summary_to_pdf(md_text)


@st.cache_data(show_spinner=False)
def _cached_csv_from_markdown(md_text: str) -> bytes:
    return statistical_summary_to_csv(md_text).encode("utf-8")


def render(ctx: TabContext) -> None:  # noqa: C901 – extracted verbatim, complexity preserved
    """Render the Statistical Comparison tab (tab index 2)."""
    tr = ctx.tr
    artifacts = ctx.artifacts
    df = ctx.df
    forced_model = ctx.forced_model
    language = ctx.language
    _default_threshold = ctx.default_threshold
    MODEL_VERSION = ctx.model_version

    # Alias helpers
    _format_ppv_npv = ctx.format_ppv_npv
    _bytes_download_btn = ctx.bytes_download_btn
    _txt_download_btn = ctx.txt_download_btn
    stats_table_column_config = ctx.stats_table_column_config
    _plot_roc = ctx.plot_roc
    _plot_calibration = ctx.plot_calibration
    _plot_boxplots = ctx.plot_boxplots
    _plot_ia_model_boxplots = ctx.plot_ia_model_boxplots
    _plot_dca = ctx.plot_dca
    build_methods_text = ctx.build_methods_text
    build_results_text = ctx.build_results_text

    # ── Begin tab body ───────────────────────────────────────────────────

    st.subheader(tr("Comparison", "Comparação"))
    st.caption(tr(
        "Primary analysis: head-to-head comparison between risk scores in the matched triple cohort. "
        "Supplementary sections cover candidate thresholds, DCA, pairwise analyses, and diagnostics.",
        "Análise principal: comparação direta entre escores de risco na coorte tripla pareada. "
        "Seções suplementares cobrem limiares candidatos, DCA, análises pareadas e diagnósticos.",
    ))

    # ── Threshold mode selector ──
    _best_youden = getattr(artifacts, "best_youden_threshold", None)
    _youden_available = _best_youden is not None

    # Determine primary threshold label: sens-constrained 90% when available,
    # else legacy fixed 8%.
    _tp_dict = getattr(artifacts, "threshold_policy", None)
    _primary_available = (
        isinstance(_tp_dict, dict)
        and _tp_dict.get("status") == "ok"
        and _tp_dict.get("selected_threshold") is not None
        and 0.0 < float(_tp_dict["selected_threshold"]) < 1.0
    )
    _mode_primary_label = (
        tr(
            f"Primary: Sens-90% = {_default_threshold*100:.1f}%",
            f"Primario: Sens-90% = {_default_threshold*100:.1f}%",
        )
        if _primary_available
        else tr(f"Primary: legacy fixed {_default_threshold:.0%}", f"Primario: legado fixo {_default_threshold:.0%}")
    )
    _mode_youden_label = (
        tr(
            f"Exploratory: Youden = {_best_youden*100:.1f}%",
            f"Exploratorio: Youden = {_best_youden*100:.1f}%",
        )
        if _youden_available
        else tr("Youden (not available — retrain required)", "Youden (indisponivel — retreine para ativar)")
    )
    st.markdown(tr("#### Operational threshold", "#### Limiar operacional"))
    _threshold_mode = st.radio(
        tr("Threshold mode", "Modo de limiar"),
        options=[_mode_primary_label, _mode_youden_label],
        index=0,
        horizontal=True,
        disabled=not _youden_available and True,
        help=tr(
            "Primary threshold: sensitivity-constrained 90% policy (largest OOF threshold keeping sensitivity >= 90%). "
            "Youden is data-driven and exploratory — not for prospective use.",
            "Limiar primario: politica de sensibilidade minima de 90% (maior limiar OOF com sensibilidade >= 90%). "
            "Youden e orientado por dados e exploratorio — nao para uso prospectivo.",
        ),
    )
    _use_youden = _youden_available and _threshold_mode == _mode_youden_label
    _slider_default = _best_youden if _use_youden else _default_threshold

    if _use_youden:
        st.caption(tr(
            f"**Active threshold: Best model Youden = {_slider_default*100:.1f}%** (probability {_slider_default:.4f})",
            f"**Limiar ativo: Youden do melhor modelo = {_slider_default*100:.1f}%** (probabilidade {_slider_default:.4f})",
        ))
    else:
        st.caption(tr(
            f"**Active threshold: Fixed clinical = {_default_threshold*100:.1f}%** (probability {_default_threshold:.4f})",
            f"**Limiar ativo: Clínico fixo = {_default_threshold*100:.1f}%** (probabilidade {_default_threshold:.4f})",
        ))

    decision_threshold = st.slider(
        tr("Decision threshold", "Limiar de decisão"),
        min_value=0.01,
        max_value=0.99,
        value=_slider_default,
        step=0.01,
    )
    with st.expander(tr("How to read this section", "Como ler esta seção"), expanded=False):
        st.markdown(
            tr(
                f"""
**Decision threshold**

The decision threshold converts a predicted probability into a binary decision: **positive** (high risk) if the predicted risk is ≥ {decision_threshold:.0%}, **negative** (low risk) otherwise. A lower threshold increases sensitivity at the cost of specificity; a higher threshold does the opposite. AUC, AUPRC, and Brier score are **not affected** by the threshold — they evaluate the full probability distribution.

The default value ({_default_threshold:.0%}) is conservative for cardiac surgery: it sits just above the average mortality rate (3–8% globally) and is aligned with EuroSCORE II stratification (low <3%, intermediate 3–8%, high >8%). Missing a high-risk patient (false negative) is much more costly than an unnecessary alert (false positive), and the conservative choice reflects that asymmetry.

**Analysis layout**

The main analysis is the three-way head-to-head, where AI Risk, EuroSCORE II, and STS are evaluated in the same patients (matched cohort). All-pairs comparisons are complementary and use larger samples when one of the three scores is unavailable. Threshold-dependent metrics (sensitivity, specificity, PPV, NPV) change with the decision threshold; calibration metrics evaluate agreement between predicted and observed risk; DCA evaluates clinical usefulness across thresholds.
""",
                f"""
**Limiar de decisão**

O limiar de decisão converte uma probabilidade predita em uma decisão binária: **positivo** (alto risco) se o risco predito for ≥ {decision_threshold:.0%}, **negativo** (baixo risco) caso contrário. Um limiar mais baixo aumenta a sensibilidade ao custo da especificidade; um limiar mais alto faz o oposto. AUC, AUPRC e Brier score **não são afetados** pelo limiar — eles avaliam a distribuição completa de probabilidades.

O valor padrão ({_default_threshold:.0%}) é conservador para cirurgia cardíaca: está logo acima da mortalidade média (3–8% mundialmente) e alinhado com a estratificação do EuroSCORE II (baixo <3%, intermediário 3–8%, alto >8%). Não identificar um paciente de alto risco (falso negativo) é muito mais custoso do que um alerta desnecessário (falso positivo), e a escolha conservadora reflete essa assimetria.

**Layout da análise**

A análise principal é a comparação tripla (head-to-head), em que AI Risk, EuroSCORE II e STS são avaliados exatamente nos mesmos pacientes (coorte pareada). Comparações com todas as amostras são complementares e usam amostras maiores quando um dos três escores está ausente. Métricas dependentes do limiar (sensibilidade, especificidade, PPV, NPV) mudam quando o limiar muda; métricas de calibração avaliam a concordância entre risco previsto e observado; a DCA avalia utilidade clínica ao longo dos limiares.
""",
            )
        )

    st.divider()
    st.markdown(tr("### Main Result — Matched Triple Cohort", "### Resultado Principal — Coorte Tripla Pareada"))
    st.caption(tr(
        "AI Risk, EuroSCORE II, and STS Score evaluated in exactly the same patients (matched cohort). "
        "This is the primary comparative analysis.",
        "AI Risk, EuroSCORE II e STS Score avaliados exatamente nos mesmos pacientes (coorte pareada). "
        "Esta é a análise comparativa principal.",
    ))

    metrics_all = evaluate_scores(
        df,
        y_col="morte_30d",
        score_cols=["ia_risk_oof", "euroscore_calc", "sts_score"],
    )
    _score_rename = {"ia_risk_oof": "AI Risk", "euroscore_calc": "EuroSCORE II", "sts_score": "STS"}
    if not metrics_all.empty:
        metrics_all["Score"] = metrics_all["Score"].replace(_score_rename)

    triple = df[["morte_30d", "ia_risk_oof", "euroscore_calc", "sts_score"]].dropna()
    st.markdown(tr("**Primary head-to-head table**", "**Tabela principal head-to-head**"))
    st.caption(tr(
        f"Matched triple cohort: n = {len(triple)}. Each score is evaluated in exactly the same patients. "
        f"STS Score is only computed for surgeries within the STS ACSD scope "
        f"(CABG, AVR, MVR, MV Repair, and their CABG combinations). "
        f"Out-of-scope surgeries (transplant, thoracic aorta, Bentall, Ross, homograft) "
        f"are excluded from this cohort by design — they do not have a valid STS Score.",
        f"Coorte tripla pareada: n = {len(triple)}. Cada escore é avaliado exatamente nos mesmos pacientes. "
        f"O STS Score é calculado apenas para cirurgias dentro do escopo STS ACSD "
        f"(CABG, AVR, MVR, MV Repair e suas combinações com CABG). "
        f"Cirurgias fora do escopo (transplante, aorta torácica, Bentall, Ross, homograft) "
        f"são excluídas desta coorte por definição — não possuem STS Score válido.",
    ))
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
        triple_ci = _cached_evaluate_scores_with_ci(
            triple,
            y_col="morte_30d",
            score_cols=("ia_risk_oof", "euroscore_calc", "sts_score"),
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

        # Normalise triple_ci["Score"] to canonical labels once before any merge.
        # This guards against minor naming variations (e.g. "STS Score" vs "STS")
        # that would otherwise produce a silent NaN in the Brier column.
        _triple_ci_canon = triple_ci.copy()
        if not _triple_ci_canon.empty:
            _triple_ci_canon["Score"] = _triple_ci_canon["Score"].map(_canon)

        calib_rows = []
        for label, col in [("AI Risk", "ia_risk_oof"), ("EuroSCORE II", "euroscore_calc"), ("STS", "sts_score")]:
            _y = triple["morte_30d"].values
            _p = triple[col].values
            ci_vals = calibration_intercept_slope(_y, _p)
            hl_vals = hosmer_lemeshow_test(_y, _p)
            try:
                _cil_val = calibration_in_the_large(_y, _p)["CIL"]
            except Exception:
                _cil_val = np.nan
            try:
                _ici_val = integrated_calibration_index(_y, _p)
            except Exception:
                _ici_val = np.nan
            brier_val = np.nan
            if not _triple_ci_canon.empty:
                _br = _triple_ci_canon[_triple_ci_canon["Score"] == _canon(label)]
                if not _br.empty:
                    brier_val = float(_br.iloc[0].get("Brier", np.nan))
            calib_rows.append({
                "Score": label,
                **ci_vals,
                **hl_vals,
                "CIL": _cil_val,
                "ICI": _ici_val,
                "Brier": brier_val,
            })
        calib_df = pd.DataFrame(calib_rows)

        _best_auc_label = None
        _best_auc_value = np.nan
        _best_brier_label = None
        _best_brier_value = np.nan
        if not triple_ci.empty and "AUC" in triple_ci.columns:
            _best_auc_row = triple_ci.sort_values("AUC", ascending=False).iloc[0]
            _best_auc_label = _best_auc_row["Score"]
            _best_auc_value = float(_best_auc_row["AUC"])
        if not triple_ci.empty and "Brier" in triple_ci.columns:
            _best_brier_row = triple_ci.sort_values("Brier", ascending=True).iloc[0]
            _best_brier_label = _best_brier_row["Score"]
            _best_brier_value = float(_best_brier_row["Brier"])
        _best_cal_label = None
        _best_cal_value = np.nan
        if not calib_df.empty and "Calibration slope" in calib_df.columns:
            _cal_rank = calib_df.assign(_slope_distance=(calib_df["Calibration slope"] - 1.0).abs())
            _best_cal_row = _cal_rank.sort_values("_slope_distance", ascending=True).iloc[0]
            _best_cal_label = _best_cal_row["Score"]
            _best_cal_value = float(_best_cal_row["Calibration slope"])

        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric(tr("Triple cohort", "Coorte tripla"), f"n = {len(triple)}", border=True)
        _m2.metric(
            tr("Best AUC", "Melhor AUC"),
            "N/A" if not np.isfinite(_best_auc_value) else f"{_best_auc_value:.3f}",
            delta=_best_auc_label or "",
            border=True,
        )
        _m3.metric(
            tr("Lowest Brier", "Menor Brier"),
            "N/A" if not np.isfinite(_best_brier_value) else f"{_best_brier_value:.3f}",
            delta=_best_brier_label or "",
            delta_color="off",
            border=True,
        )
        _m4.metric(
            tr("Calibration slope closest to 1", "Slope de calibração mais próximo de 1"),
            "N/A" if not np.isfinite(_best_cal_value) else f"{_best_cal_value:.2f}",
            delta=_best_cal_label or "",
            delta_color="off",
            border=True,
        )

        # ── Calibration at a Glance ──────────────────────────────────────
        st.divider()
        st.markdown(tr("### Calibration at a Glance", "### Calibração em Resumo"))
        st.caption(tr(
            "Intercept near 0 and slope near 1 indicate good calibration. "
            "CIL (Calibration-in-the-Large) = mean predicted − mean observed — close to 0 is ideal. "
            "ICI (Integrated Calibration Index) measures average absolute calibration error — lower is better. "
            "HL p-value is complementary only — do not interpret in isolation.",
            "Intercepto próximo de 0 e slope próximo de 1 indicam boa calibração. "
            "CIL (Calibração Geral) = média predita − média observada — próximo de 0 é ideal. "
            "ICI (Índice Integrado de Calibração) mede o erro médio absoluto de calibração — menor é melhor. "
            "p-valor de HL é apenas complementar — não interpretar isoladamente.",
        ))
        _calib_glance_cols = ["Score", "Calibration intercept", "Calibration slope", "CIL", "ICI", "Brier", "HL p-value"]
        _calib_glance_df = calib_df[[c for c in _calib_glance_cols if c in calib_df.columns]].copy()
        st.dataframe(_calib_glance_df, width="stretch", column_config=stats_table_column_config("calibration"), hide_index=True)

        st.markdown(tr("**Full calibration table**", "**Tabela de calibração completa**"))
        st.caption(tr(
            "Brier measures probabilistic accuracy. Intercept close to 0 and slope close to 1 are desirable. "
            "CIL (Calibration-in-the-Large) = mean predicted − mean observed — close to 0 is ideal. "
            "ICI (Integrated Calibration Index) measures average absolute calibration error via isotonic regression — lower is better. "
            "Hosmer-Lemeshow should be interpreted as complementary, not in isolation.",
            "O Brier mede a acurácia probabilística. Intercepto próximo de 0 e slope próximo de 1 são desejáveis. "
            "CIL (Calibração Geral) = média predita − média observada — próximo de 0 é ideal. "
            "ICI (Índice Integrado de Calibração) mede o erro médio absoluto de calibração via regressão isotônica — menor é melhor. "
            "O teste de Hosmer-Lemeshow deve ser interpretado como complementar, e não isoladamente.",
        ))
        st.dataframe(calib_df, width="stretch", column_config=stats_table_column_config("calibration"))

        st.markdown(tr("### Threshold Comparison", "### Comparação entre Limiares"))
        st.caption(tr(
            "Primary: sensitivity-constrained 90% (largest OOF threshold keeping sensitivity >= 90%). "
            "Fixed 8% is a historical comparator. Youden is exploratory. "
            "All other fixed thresholds are supplementary benchmarks.",
            "Primario: sensibilidade minima 90% (maior limiar OOF com sensibilidade >= 90%). "
            "Fixo 8% e comparador historico. Youden e exploratorio. "
            "Os demais limiares fixos sao referencias suplementares.",
        ))
        _threshold_comparison_for_display = _build_threshold_comparison_export_df(
            df=df,
            artifacts=artifacts,
            forced_model=forced_model,
        )
        if not _threshold_comparison_for_display.empty:
            st.dataframe(
                _format_ppv_npv(_threshold_comparison_for_display),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(tr(
                "Threshold comparison is unavailable because the AI Risk outcome sample is insufficient.",
                "A comparação entre limiares não está disponível porque a amostra de desfecho do AI Risk é insuficiente.",
            ))

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

        st.markdown(tr("#### Probability distributions", "#### Distribuições de probabilidade"))
        _bx1, _bx2 = st.columns(2)
        with _bx1:
            st.markdown(tr("**By outcome (triple cohort)**", "**Por desfecho (coorte tripla)**"))
            _plot_boxplots(box_df)
        with _bx2:
            st.markdown(tr("**Per AI model (OOF)**", "**Por modelo de IA (OOF)**"))
            _plot_ia_model_boxplots(df["morte_30d"].values, artifacts.oof_predictions)
    else:
        st.warning(tr("Insufficient sample for complete triple comparison.", "Amostra insuficiente para comparação tripla completa."))

    st.divider()
    st.markdown(tr("### Overall Comparison — Available Data", "### Comparação Geral — Dados Disponíveis"))
    st.caption(
        tr(
            "Supplementary overview using all available observations for each score separately. "
            "Because sample sizes can differ by score, the matched triple cohort above remains the primary head-to-head result.",
            "Visão suplementar usando todas as observações disponíveis para cada escore separadamente. "
            "Como os tamanhos amostrais podem diferir por escore, a coorte tripla pareada acima continua sendo o resultado head-to-head principal.",
        )
    )
    st.dataframe(metrics_all, width="stretch", column_config=stats_table_column_config("overall"))

    # ── Read-only probability distribution diagnostics ────────────────────
    # Diagnostic-only panel: does NOT change thresholds, model logic,
    # methodology, or any metric shown above. Inspects the exact
    # probability source already used by this tab (calibrated OOF).
    with st.expander(
        tr(
            "Advanced diagnostics — probability distribution (read-only)",
            "Diagnóstico avançado — distribuição de probabilidades (somente leitura)",
        ),
        expanded=False,
    ):
        st.caption(tr(
            "Read-only inspection of AI Risk probability distributions and threshold behavior. "
            "This panel is for auditing only — it does not change the fixed 8% clinical threshold, "
            "the model outputs, the ranking logic, or any methodology.",
            "Inspeção somente leitura das distribuições de probabilidade do AI Risk e do comportamento do limiar. "
            "Este painel serve apenas para auditoria — não altera o limiar clínico fixo de 8%, "
            "as saídas do modelo, a lógica de ranqueamento ou qualquer aspecto metodológico.",
        ))

        # Collected as each panel renders; used at the end of the
        # expander to build a single multi-sheet XLSX download bundling
        # the whole diagnostics area.
        _diag_sheets: dict[str, pd.DataFrame] = {}

        def _describe_prob_array(arr) -> dict:
            a = np.asarray(arr, dtype=float)
            a = a[~np.isnan(a)]
            if len(a) == 0:
                return {}
            return {
                "n": int(len(a)),
                "min": float(np.min(a)),
                "p01": float(np.quantile(a, 0.01)),
                "p05": float(np.quantile(a, 0.05)),
                "p25": float(np.quantile(a, 0.25)),
                "median": float(np.median(a)),
                "p75": float(np.quantile(a, 0.75)),
                "p95": float(np.quantile(a, 0.95)),
                "p99": float(np.quantile(a, 0.99)),
                "max": float(np.max(a)),
                "frac<0.08": float((a < 0.08).mean()),
            }

        _youden_dict = getattr(artifacts, "youden_thresholds", None) or {}

        # Panel 1 — Calibrated OOF, per model
        st.markdown(tr(
            "**1. Calibrated OOF probabilities — per model**",
            "**1. Probabilidades OOF calibradas — por modelo**",
        ))
        st.caption(tr(
            "Source: `artifacts.oof_predictions` (nested-CV calibrated OOF — the honest evaluation probabilities used by this tab).",
            "Origem: `artifacts.oof_predictions` (OOF calibrado por CV aninhada — as probabilidades honestas usadas nesta aba).",
        ))
        _cal_rows = []
        for _mn, _probs in artifacts.oof_predictions.items():
            _d = _describe_prob_array(_probs)
            if not _d:
                continue
            _d = {"model": _mn, **_d, "youden": float(_youden_dict.get(_mn, np.nan))}
            _cal_rows.append(_d)
        if _cal_rows:
            _cal_diag_df = pd.DataFrame(_cal_rows)[
                ["model", "n", "min", "p01", "p05", "p25", "median",
                 "p75", "p95", "p99", "max", "frac<0.08", "youden"]
            ]
            st.dataframe(_cal_diag_df, width="stretch", hide_index=True)
            _diag_sheets["1_OOF_calibrated"] = _cal_diag_df
        else:
            st.info(tr("No calibrated OOF predictions available.",
                       "Sem predições OOF calibradas disponíveis."))

        # Panel 2 — Raw OOF, per model (if stored in bundle)
        st.markdown(tr(
            "**2. Raw OOF probabilities — per model (uncalibrated, audit only)**",
            "**2. Probabilidades OOF brutas — por modelo (não calibradas, apenas auditoria)**",
        ))
        st.caption(tr(
            "Source: `artifacts.oof_raw`. Comparing panel 1 vs panel 2 exposes the effect of post-hoc calibration (per-model strategy) on the distribution floor.",
            "Origem: `artifacts.oof_raw`. Comparar os painéis 1 e 2 mostra o efeito da calibração pós-hoc (estratégia por modelo) no piso da distribuição.",
        ))
        _oof_raw = getattr(artifacts, "oof_raw", None)
        if _oof_raw:
            _raw_rows = []
            for _mn, _probs in _oof_raw.items():
                _d = _describe_prob_array(_probs)
                if not _d:
                    continue
                _raw_rows.append({"model": _mn, **_d})
            if _raw_rows:
                _raw_diag_df = pd.DataFrame(_raw_rows)[
                    ["model", "n", "min", "p01", "p05", "p25", "median",
                     "p75", "p95", "p99", "max", "frac<0.08"]
                ]
                st.dataframe(_raw_diag_df, width="stretch", hide_index=True)
                _diag_sheets["2_OOF_raw"] = _raw_diag_df
            else:
                st.info(tr("Raw OOF dict is empty.", "Dicionário OOF bruto vazio."))
        else:
            st.info(tr(
                "No raw OOF predictions stored in this bundle (retrain to populate).",
                "Sem predições OOF brutas armazenadas neste bundle (retreinar para popular).",
            ))

        # Panel 3 — Focus on the active AI Risk OOF series (calibrated)
        st.markdown(tr(
            f"**3. Active AI Risk OOF series — calibrated (model: `{forced_model}`)**",
            f"**3. Série OOF do AI Risk ativa — calibrada (modelo: `{forced_model}`)**",
        ))
        st.caption(tr(
            "This is the exact probability series used by the Comparison tab above. "
            "Source: `df['ia_risk_oof']` = `artifacts.oof_predictions[forced_model]`.",
            "Esta é a série exata de probabilidades usada na aba de Comparação acima. "
            "Origem: `df['ia_risk_oof']` = `artifacts.oof_predictions[forced_model]`.",
        ))
        _active_mask = df["ia_risk_oof"].notna() & df["morte_30d"].notna()
        _active_np = df.loc[_active_mask, "ia_risk_oof"].values.astype(float)
        _y_active = df.loc[_active_mask, "morte_30d"].astype(int).values
        _forced_youden = float(_youden_dict.get(forced_model, np.nan)) if _youden_dict else float("nan")

        if len(_active_np) >= 2 and len(np.unique(_y_active)) >= 2:
            _desc = _describe_prob_array(_active_np)
            _n_below_8 = int((_active_np < 0.08).sum())
            _frac_below_8 = float(_n_below_8 / len(_active_np))
            _min_above_8 = bool(_desc["min"] >= 0.08)
            if not np.isnan(_forced_youden):
                _n_below_yd = int((_active_np < _forced_youden).sum())
                _frac_below_yd = float(_n_below_yd / len(_active_np))
                _yd_label = f"{_forced_youden:.4f}"
            else:
                _n_below_yd = np.nan
                _frac_below_yd = np.nan
                _yd_label = "N/A"

            _active_summary = pd.DataFrame([
                {"metric": "n (non-null, y known)", "value": _desc["n"]},
                {"metric": "min", "value": _desc["min"]},
                {"metric": "p01", "value": _desc["p01"]},
                {"metric": "p05", "value": _desc["p05"]},
                {"metric": "p25", "value": _desc["p25"]},
                {"metric": "median", "value": _desc["median"]},
                {"metric": "p75", "value": _desc["p75"]},
                {"metric": "p95", "value": _desc["p95"]},
                {"metric": "p99", "value": _desc["p99"]},
                {"metric": "max", "value": _desc["max"]},
                {"metric": "count < 0.08", "value": _n_below_8},
                {"metric": "fraction < 0.08", "value": _frac_below_8},
                {"metric": f"count < Youden ({_yd_label})", "value": _n_below_yd},
                {"metric": f"fraction < Youden ({_yd_label})", "value": _frac_below_yd},
                {"metric": "min >= 0.08 ?", "value": _min_above_8},
            ])
            st.dataframe(_active_summary, width="stretch", hide_index=True)
            _diag_sheets[f"3_active_{forced_model}"[:31]] = _active_summary.assign(
                model=forced_model
            )

            # Panel 4 — calibration intercept/slope on the same series
            st.markdown(tr(
                "**4. Calibration intercept and slope (active AI Risk series)**",
                "**4. Intercepto e slope de calibração (série ativa do AI Risk)**",
            ))
            st.caption(tr(
                "Computed on the exact series shown in panel 3. Intercept near 0 and slope near 1 indicate good calibration-in-the-large and spread.",
                "Calculado na série exata mostrada no painel 3. Intercepto próximo de 0 e slope próximo de 1 indicam boa calibração global e de dispersão.",
            ))
            _cis = calibration_intercept_slope(_y_active, _active_np)
            _cis_df = pd.DataFrame([
                {"metric": "Calibration intercept", "value": _cis["Calibration intercept"]},
                {"metric": "Calibration slope", "value": _cis["Calibration slope"]},
            ])
            st.dataframe(_cis_df, width="stretch", hide_index=True)
            _diag_sheets["4_calibration_int_slope"] = _cis_df.assign(model=forced_model)

            # Panel 5 — side-by-side threshold comparison
            st.markdown(tr(
                "**5. Threshold comparison: fixed 8% vs stored Youden (this model)**",
                "**5. Comparação de limiares: 8% fixo vs Youden armazenado (este modelo)**",
            ))
            st.caption(tr(
                "Classification of the active AI Risk OOF series at both thresholds. Diagnostic only — does not change the metrics above.",
                "Classificação da série OOF ativa do AI Risk em ambos os limiares. Somente diagnóstico — não altera as métricas acima.",
            ))
            _thr_rows = []
            _thr_pairs = [("Fixed 8%", 0.08),
                          (f"Youden ({forced_model})", _forced_youden)]
            for _label, _thr in _thr_pairs:
                if np.isnan(_thr):
                    _thr_rows.append({
                        "threshold_label": _label,
                        "threshold_value": np.nan,
                        "n_positive": np.nan,
                        "Sensitivity": np.nan,
                        "Specificity": np.nan,
                        "PPV": np.nan,
                        "NPV": np.nan,
                    })
                    continue
                _cls = classification_metrics_at_threshold(_y_active, _active_np, float(_thr))
                _n_pos = int((_active_np >= float(_thr)).sum())
                _thr_rows.append({
                    "threshold_label": _label,
                    "threshold_value": float(_thr),
                    "n_positive": _n_pos,
                    "Sensitivity": _cls["Sensitivity"],
                    "Specificity": _cls["Specificity"],
                    "PPV": _cls["PPV"],
                    "NPV": _cls["NPV"],
                })
            _thr_df = pd.DataFrame(_thr_rows)
            st.dataframe(_thr_df, width="stretch", hide_index=True)
            _diag_sheets["5_threshold_comparison"] = _thr_df.assign(model=forced_model)
        else:
            st.info(tr(
                "Active AI Risk OOF series is empty or has only one outcome class — cannot diagnose.",
                "Série OOF do AI Risk ativa está vazia ou tem apenas uma classe — não é possível diagnosticar.",
            ))

        # ── Single-click download of the entire diagnostics area ──────
        if _diag_sheets:
            _diag_xlsx_buf = BytesIO()
            try:
                with pd.ExcelWriter(_diag_xlsx_buf, engine="openpyxl") as _writer:
                    for _sheet_name, _sheet_df in _diag_sheets.items():
                        _sheet_df.to_excel(
                            _writer, sheet_name=_sheet_name[:31], index=False
                        )
                _bytes_download_btn(
                    _diag_xlsx_buf.getvalue(),
                    "probability_distribution_diagnostics.xlsx",
                    tr(
                        "Download all diagnostics (XLSX)",
                        "Baixar todos os diagnósticos (XLSX)",
                    ),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_prob_diag_all",
                )
            except Exception as _diag_dl_err:
                st.caption(tr(
                    f"Diagnostics bundle unavailable: {_diag_dl_err}",
                    f"Pacote de diagnósticos indisponível: {_diag_dl_err}",
                ))

    st.divider()
    st.markdown(tr("### Supplementary Pairwise Comparisons", "### Comparações Pareadas Suplementares"))
    st.caption(tr(
        "Formal ROC comparison between pairs of scores, positioned as supplementary evidence below the matched triple cohort. "
        "Bootstrap and DeLong results are complementary — interpret together.",
        "Comparação formal entre pares de escores, posicionada como evidência suplementar abaixo da coorte tripla pareada. "
        "Bootstrap e DeLong são complementares — interpretar em conjunto.",
    ))

    st.markdown(tr("**All-pairs comparison (full cohort)**", "**Comparação por pares (coorte completa)**"))
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
        boot = _cached_bootstrap_auc_diff(sub["morte_30d"].values, sub[a].values, sub[b].values)
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

    st.markdown(tr("**Pairwise bootstrap (matched cohort)**", "**Bootstrap por pares (coorte pareada)**"))
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
            boot = _cached_bootstrap_auc_diff(
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

    st.markdown(tr("**DeLong test (matched cohort)**", "**Teste de DeLong (coorte pareada)**"))
    st.caption(
        tr(
            "DeLong formally compares correlated AUCs in the same patients. It complements the bootstrap-based delta AUC analysis.",
            "O teste de DeLong compara formalmente AUCs correlacionadas nos mesmos pacientes. Ele complementa a análise de delta AUC por bootstrap.",
        )
    )
    delong_rows = []
    if len(triple) >= 30 and triple["morte_30d"].nunique() > 1:
        for a, b in [("ia_risk_oof", "euroscore_calc"), ("ia_risk_oof", "sts_score"), ("euroscore_calc", "sts_score")]:
            dtest = _cached_delong_roc_test(triple["morte_30d"].values, triple[a].values, triple[b].values)
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

    st.divider()
    st.markdown(tr("### Supplementary Clinical Utility", "### Utilidade Clínica Suplementar"))

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
        dca_df = _cached_decision_curve(
            triple["morte_30d"].values,
            triple["ia_risk_oof"].values,
            triple["euroscore_calc"].values,
            triple["sts_score"].values,
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
            nri = compute_nri_with_ci(triple["morte_30d"].values, triple[old_col].values, triple[new_col].values, cutoffs=(0.05, 0.15), n_boot=2000, seed=42)
            idi = compute_idi_with_ci(triple["morte_30d"].values, triple[old_col].values, triple[new_col].values, n_boot=2000, seed=42)
            _comp_col = tr("Comparison", "Comparação")
            _nri_ci_low = nri.get("NRI_CI_low", float("nan"))
            _nri_ci_high = nri.get("NRI_CI_high", float("nan"))
            _idi_ci_low = idi.get("IDI_CI_low", float("nan"))
            _idi_ci_high = idi.get("IDI_CI_high", float("nan"))
            reclass_rows.append(
                {
                    _comp_col: f"{new_name} vs {old_name}",
                    "NRI events": round(nri["NRI events"], 4),
                    "NRI non-events": round(nri["NRI non-events"], 4),
                    "NRI total": round(nri["NRI total"], 4),
                    "NRI 95% CI": f"({_nri_ci_low:.3f}, {_nri_ci_high:.3f})" if pd.notna(_nri_ci_low) else "—",
                    # Always string: keeps Arrow dtype consistent (object),
                    # avoids ArrowInvalid when mixing float and "< 0.0005".
                    "NRI p": (
                        f"< {nri['NRI_p_lower_bound']:.4f}"
                        if nri.get("NRI_p_lower_bound") is not None
                        else (f"{nri.get('NRI_p', float('nan')):.4f}"
                              if pd.notna(nri.get("NRI_p")) else "—")
                    ),
                    "IDI": round(idi["IDI"], 4),
                    "IDI 95% CI": f"({_idi_ci_low:.3f}, {_idi_ci_high:.3f})" if pd.notna(_idi_ci_low) else "—",
                    "IDI p": (
                        f"< {idi['IDI_p_lower_bound']:.4f}"
                        if idi.get("IDI_p_lower_bound") is not None
                        else (f"{idi.get('IDI_p', float('nan')):.4f}"
                              if pd.notna(idi.get("IDI_p")) else "—")
                    ),
                }
            )
        _comp_col = tr("Comparison", "Comparação")
        reclass_df = pd.DataFrame(reclass_rows)
        st.dataframe(reclass_df, width="stretch")
        if not reclass_df.empty:
            best_nri = reclass_df.sort_values("NRI total", ascending=False).iloc[0]
            best_idi = reclass_df.sort_values("IDI", ascending=False).iloc[0]
            # "NRI p"/"IDI p" are always pre-formatted strings in reclass_df.
            _nri_p_str = f", p = {best_nri['NRI p']}" if best_nri["NRI p"] not in ("—", "") else ""
            _idi_p_str = f", p = {best_idi['IDI p']}" if best_idi["IDI p"] not in ("—", "") else ""
            st.info(
                tr(
                    f"The highest NRI was observed for {best_nri[_comp_col]} "
                    f"(NRI total = {best_nri['NRI total']:.3f}{_nri_p_str}). "
                    f"The highest IDI was observed for {best_idi[_comp_col]} "
                    f"(IDI = {best_idi['IDI']:.3f}{_idi_p_str}). "
                    f"These are complementary reclassification metrics and should be interpreted alongside discrimination and calibration results.",
                    f"O maior NRI foi observado em {best_nri[_comp_col]} "
                    f"(NRI total = {best_nri['NRI total']:.3f}{_nri_p_str}). "
                    f"O maior IDI foi observado em {best_idi[_comp_col]} "
                    f"(IDI = {best_idi['IDI']:.3f}{_idi_p_str}). "
                    f"Essas são métricas complementares de reclassificação e devem ser interpretadas em conjunto com os resultados de discriminação e calibração.",
                )
            )
    else:
        st.info(tr("NRI/IDI are unavailable because the triple comparison sample is insufficient.", "NRI/IDI não estão disponíveis porque a amostra da comparação tripla é insuficiente."))

    st.divider()
    st.markdown(tr("### Interpretation & Export", "### Interpretação e Exportação"))
    st.caption(tr(
        "Narrative interpretation and on-demand exports. Files are generated only when requested.",
        "Interpretação narrativa e exports sob demanda. Os arquivos são gerados apenas quando solicitados.",
    ))

    with st.expander(tr("Clinical interpretation", "Interpretação clínica"), expanded=False):
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
            # Defaults so the narrative block can still run if the triple cohort is present
            best_brier = None
            best_sens = None
            best_spec = None
            best_ppv = None
            best_npv = None

    with st.expander(tr("Manuscript text (Methods & Results)", "Texto para manuscrito (Métodos e Resultados)"), expanded=False):
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

    # ── Statistical summary export ──
    import datetime as _dt
    _export_date_tag = _dt.datetime.now().strftime("%Y%m%d")
    _export_base = f"ai_risk_comparison_{MODEL_VERSION}_{_export_date_tag}"

    # Canonical manifest — single source of truth for every export emitted
    # by this tab.  Built once here so the ZIP, the cached helpers, and any
    # future export entry point all read from the same dict.
    _bi = ctx.bundle_info if isinstance(ctx.bundle_info, dict) else {}
    _comparison_manifest = build_export_manifest(
        export_kind="comparison",
        model_version=MODEL_VERSION,
        active_model_name=_bi.get("active_model_name") or forced_model,
        threshold_mode=("youden" if _use_youden else "clinical_fixed"),
        threshold_value=float(decision_threshold),
        dataset_fingerprint=_bi.get("dataset_fingerprint"),
        bundle_fingerprint=_bi.get("bundle_fingerprint"),
        bundle_saved_at=_bi.get("saved_at"),
        training_source=_bi.get("training_source"),
        current_analysis_file=Path(ctx.xlsx_path).name if ctx.xlsx_path else None,
        extra={
            "n_triple": int(len(triple)) if 'triple' in locals() else 0,
            "default_threshold": float(_default_threshold),
            "language": language,
        },
    )

    _calib_df_for_export = calib_df if 'calib_df' in locals() else pd.DataFrame()
    _formal_df_for_export = formal_df if 'formal_df' in locals() else pd.DataFrame()
    _delong_df_for_export = delong_df if 'delong_df' in locals() else pd.DataFrame()
    _reclass_df_for_export = reclass_df if 'reclass_df' in locals() else pd.DataFrame()
    _n_triple_export = len(triple) if 'triple' in locals() else 0
    _dca_df_for_export = dca_df if 'dca_df' in locals() else pd.DataFrame()
    _metrics_all_for_export = metrics_all if 'metrics_all' in locals() else pd.DataFrame()
    _pair_df_for_export = pd.DataFrame(pair_rows) if 'pair_rows' in locals() else pd.DataFrame()
    _threshold_comparison_for_export = (
        _threshold_comparison_for_display
        if '_threshold_comparison_for_display' in locals()
        else _build_threshold_comparison_export_df(
            df=df,
            artifacts=artifacts,
            forced_model=forced_model,
        )
    )
    _roc_plot_for_export, _calibration_plot_for_export, _dca_plot_for_export = _build_figure_export_data(
        triple=triple if 'triple' in locals() else pd.DataFrame(),
        dca_df=_dca_df_for_export,
    )

    def _build_stat_summary_export() -> str:
        return _cached_statistical_summary(
            triple_ci=triple_ci,
            calib_df=_calib_df_for_export,
            formal_df=_formal_df_for_export,
            delong_df=_delong_df_for_export,
            reclass_df=_reclass_df_for_export,
            threshold=decision_threshold,
            threshold_metrics=threshold_metrics,
            n_triple=_n_triple_export,
            model_version=MODEL_VERSION,
            language=language,
        )

    _comparison_export_sig = (
        MODEL_VERSION,
        language,
        forced_model,
        _default_threshold,
        decision_threshold,
        ctx.xlsx_path,
        ctx.bundle_info.get("saved_at") if isinstance(ctx.bundle_info, dict) else None,
        id(df),
        len(df),
        len(triple_ci),
        len(_calib_df_for_export),
        len(_formal_df_for_export),
        len(_delong_df_for_export),
        len(_reclass_df_for_export),
        len(_dca_df_for_export),
        len(_threshold_comparison_for_export),
    )

    def _lazy_export_button(label: str, filename: str, mime: str, key: str, build_fn):
        _exports = st.session_state.get("_comparison_exports", {})
        if _exports.get("sig") != _comparison_export_sig:
            _exports = {"sig": _comparison_export_sig}
            st.session_state["_comparison_exports"] = _exports

        _slot = st.empty()
        _data = _exports.get(key)
        if _data:
            _slot.download_button(
                label,
                data=_data,
                file_name=filename,
                mime=mime,
                key=f"{key}_download",
                on_click="ignore",
            )
            return

        if _slot.button(label, key=f"{key}_prepare"):
            with st.spinner(tr("Preparing export...", "Preparando export...")):
                _data = build_fn()
            if _data:
                _exports[key] = _data
                st.session_state["_comparison_exports"] = _exports
                _slot.download_button(
                    label,
                    data=_data,
                    file_name=filename,
                    mime=mime,
                    key=f"{key}_download",
                    on_click="ignore",
                )
                st.caption(tr(
                    "File ready. Click the same button to download.",
                    "Arquivo pronto. Clique no mesmo botão para baixar.",
                ))
            else:
                st.caption(tr("Export unavailable for this environment.", "Export indisponível neste ambiente."))

    # Primary downloads: two consolidated buttons
    _dl_col1, _dl_col2 = st.columns(2)
    with _dl_col1:
        _lazy_export_button(
            tr("Download Summary Report (PDF)", "Baixar Relatório Sumário (PDF)"),
            f"{_export_base}_summary.pdf",
            "application/pdf",
            "dl_summary_pdf",
            lambda: _cached_summary_pdf(
                triple_ci=triple_ci,
                calib_df=_calib_df_for_export,
                formal_df=_formal_df_for_export,
                delong_df=_delong_df_for_export,
                threshold_metrics=threshold_metrics,
                threshold=decision_threshold,
                n_triple=_n_triple_export,
                model_version=MODEL_VERSION,
                language=language,
            ),
        )
        st.caption(tr(
            "Curated editorial PDF — main performance, calibration, pairwise comparisons.",
            "PDF editorial curado — desempenho principal, calibração, comparações pareadas.",
        ))
    with _dl_col2:
        _lazy_export_button(
            tr("Download Full Package (ZIP)", "Baixar Pacote Completo (ZIP)"),
            f"{_export_base}_full.zip",
            "application/zip",
            "dl_full_zip",
            lambda: _cached_full_package(
                triple_ci=triple_ci,
                calib_df=_calib_df_for_export,
                formal_df=_formal_df_for_export,
                delong_df=_delong_df_for_export,
                reclass_df=_reclass_df_for_export,
                threshold_metrics=threshold_metrics,
                threshold=decision_threshold,
                n_triple=_n_triple_export,
                model_version=MODEL_VERSION,
                language=language,
                dca_df=_dca_df_for_export,
                metrics_all=_metrics_all_for_export,
                pair_df=_pair_df_for_export,
                threshold_comparison_df=_threshold_comparison_for_export,
                roc_plot_df=_roc_plot_for_export,
                calibration_plot_df=_calibration_plot_for_export,
                dca_plot_df=_dca_plot_for_export,
                manifest=_comparison_manifest,
                _export_version="2026-04-24-p-str",
            ),
        )
        st.caption(tr(
            "ZIP: Summary PDF + Full Report PDF (all sections) + Markdown + structured XLSX + CSV.",
            "ZIP: PDF sumário + PDF completo (todas as seções) + Markdown + XLSX estruturado + CSV.",
        ))

    # Advanced / Raw exports — individual format downloads
    with st.expander(tr("Advanced / Raw exports", "Exportações avançadas / brutas"), expanded=False):
        st.caption(tr(
            "Individual format downloads for custom analysis or archiving.",
            "Downloads individuais por formato para análise personalizada ou arquivamento.",
        ))
        st.text_area(
            tr("Markdown preview", "Pré-visualização Markdown"),
            value=tr("Generated on demand with the Markdown export.", "Gerado sob demanda com a exportação Markdown."),
            height=180,
            disabled=True,
            key="md_preview_area",
        )
        _exp_col1, _exp_col2, _exp_col3, _exp_col4 = st.columns(4)
        with _exp_col1:
            _lazy_export_button(
                tr("PDF report", "Relatório PDF"),
                f"{_export_base}.pdf",
                "application/pdf",
                "dl_stat_pdf",
                lambda: _cached_pdf_from_markdown(_build_stat_summary_export()),
            )
        with _exp_col2:
            _lazy_export_button(
                tr("XLSX (structured)", "XLSX (estruturado)"),
                f"{_export_base}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "dl_stat_xlsx",
                lambda: _cached_structured_xlsx(
                    triple_ci=triple_ci,
                    calib_df=_calib_df_for_export,
                    formal_df=_formal_df_for_export,
                    delong_df=_delong_df_for_export,
                    reclass_df=_reclass_df_for_export,
                    threshold_metrics=threshold_metrics,
                    threshold=decision_threshold,
                    n_triple=_n_triple_export,
                    model_version=MODEL_VERSION,
                    language=language,
                    dca_df=_dca_df_for_export,
                    metrics_all=_metrics_all_for_export,
                    pair_df=_pair_df_for_export,
                    threshold_comparison_df=_threshold_comparison_for_export,
                    roc_plot_df=_roc_plot_for_export,
                    calibration_plot_df=_calibration_plot_for_export,
                    dca_plot_df=_dca_plot_for_export,
                ),
            )
        with _exp_col3:
            _lazy_export_button(
                tr("CSV (flat)", "CSV (plano)"),
                f"{_export_base}.csv",
                "text/csv",
                "dl_stat_csv",
                lambda: _cached_csv_from_markdown(_build_stat_summary_export()),
            )
        with _exp_col4:
            _lazy_export_button(
                tr("Markdown", "Markdown"),
                f"{_export_base}.md",
                "text/markdown",
                "dl_stat_md",
                lambda: _build_stat_summary_export().encode("utf-8"),
            )
