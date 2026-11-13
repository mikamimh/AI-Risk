"""Dissertation exports: SHAP interpretability (Block A) and operational
threshold-policy analyses (Block B).

Pure logic (no Streamlit).  Everything here is *derived* analysis written to
files — the model, the bundle, the stored threshold policy and the existing
metrics are never modified.

Inputs
------
* Development cohort (Block B): the AI Risk out-of-fold predictions already
  used by the Comparison tab — ``df["ia_risk_oof"]`` (=
  ``artifacts.oof_predictions[forced_model]``) — with outcome ``morte_30d``.
* Triple cohort: patients with AI Risk, EuroSCORE II and STS available,
  built by :func:`build_triple_cohort` (shared with ``tabs/comparison.py``).

Files produced by :func:`build_dissertation_export_zip`
------------------------------------------------------
A1_shap_global_top15.csv, A2_fig_shap_bar.{png,svg},
A3_fig_shap_beeswarm.{png,svg}, B1_sensitivity_targets.csv,
B2_threshold_bootstrap_summary.csv, B2_threshold_bootstrap_draws.csv,
B2_fig_threshold_bootstrap.{png,svg}, B3_risk_bands_triple.csv,
B4_ruleout_primary_threshold.csv, metadata.json.
"""

from __future__ import annotations

import json
import math
import zipfile
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from stats_compare import (
    categorize_risk,
    classification_metrics_at_threshold,
    sensitivity_constrained_threshold,
    threshold_analysis_table,
)

OUTCOME_COL = "morte_30d"
AI_RISK_COL = "ia_risk_oof"
EURO_COL = "euroscore_calc"
STS_COL = "sts_score"
TRIPLE_COLUMNS = [OUTCOME_COL, AI_RISK_COL, EURO_COL, STS_COL]
METHOD_COLUMNS = {"AI Risk": AI_RISK_COL, "EuroSCORE II": EURO_COL, "STS": STS_COL}

COHORT_DEV = "desenvolvimento_OOF"
COHORT_TRIPLE = "tripla"

SENSITIVITY_TARGETS = (0.80, 0.85, 0.90)
NRI_CUTOFFS = (0.05, 0.15)
# categorize_risk: low = p < 5%; intermediate = 5% <= p <= 15%; high = p > 15%
_BAND_ORDER = [("low", "<5%"), ("intermediate", "5–15%"), ("high", ">15%")]

FIG_DPI = 300
_FONT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "none",  # keep text editable in SVG
    "axes.spines.top": False,
    "axes.spines.right": False,
}
_BAR_COLOR = "#3b6e9e"


# ---------------------------------------------------------------------------
# Cohorts
# ---------------------------------------------------------------------------

def build_triple_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Matched triple cohort: rows with outcome, AI Risk OOF, EuroSCORE II and STS."""
    return df[TRIPLE_COLUMNS].dropna()


def development_oof_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """(y, p) for the full development cohort with AI Risk OOF available."""
    mask = df[AI_RISK_COL].notna() & df[OUTCOME_COL].notna()
    y = df.loc[mask, OUTCOME_COL].astype(int).to_numpy()
    p = df.loc[mask, AI_RISK_COL].astype(float).to_numpy()
    return y, p


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


# ---------------------------------------------------------------------------
# B1 — cost of the sensitivity target
# ---------------------------------------------------------------------------

def _metrics_row_from_policy(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "limiar": res["threshold"],
        "sensibilidade": res["sensitivity"],
        "especificidade": res["specificity"],
        "VPP": res["PPV"],
        "VPN": res["NPV"],
        "TP": res["TP"], "FP": res["FP"], "TN": res["TN"], "FN": res["FN"],
        "n_flagged": res["n_flagged"],
        "flag_rate": res["flag_rate"],
        "event_rate_above": res["event_rate_above"],
        "event_rate_below": res["event_rate_below"],
    }


def _metrics_row_from_table(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, Any]:
    r = threshold_analysis_table(y, p, [threshold]).iloc[0]
    tp, fp, tn, fn = int(r["TP"]), int(r["FP"]), int(r["TN"]), int(r["FN"])
    n_flag = int(r["N_Flagged"])
    return {
        "limiar": float(threshold),
        "sensibilidade": r["Sensitivity"],
        "especificidade": r["Specificity"],
        "VPP": r["PPV"],
        "VPN": r["NPV"],
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "n_flagged": n_flag,
        "flag_rate": _safe_div(n_flag, len(y)),
        "event_rate_above": _safe_div(tp, n_flag),
        "event_rate_below": _safe_div(fn, fn + tn),
    }


def sensitivity_targets_table(
    y_dev: np.ndarray,
    p_dev: np.ndarray,
    y_triple: np.ndarray,
    p_triple: np.ndarray,
    targets: Sequence[float] = SENSITIVITY_TARGETS,
) -> pd.DataFrame:
    """Long-format table: one row per (target, cohort).

    The threshold is derived on the development OOF predictions and then
    applied, locked, to AI Risk in the triple cohort.
    """
    rows = []
    n_dev, n_tri = len(y_dev), len(y_triple)
    for tgt in targets:
        res = sensitivity_constrained_threshold(y_dev, p_dev, min_sensitivity=float(tgt))
        base = {"alvo_sensibilidade": float(tgt), "status": res["status"]}
        rows.append({"coorte": COHORT_DEV, **base, "n": n_dev, "eventos": int(np.sum(y_dev)),
                     **_metrics_row_from_policy(res)})
        thr = res["threshold"]
        if np.isfinite(thr) and n_tri:
            tri = _metrics_row_from_table(y_triple, p_triple, thr)
        else:
            tri = {k: float("nan") for k in _metrics_row_from_policy(res)}
        rows.append({"coorte": COHORT_TRIPLE, **base, "n": n_tri, "eventos": int(np.sum(y_triple)), **tri})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B2 — threshold stability + optimism-corrected sensitivity (bootstrap)
# ---------------------------------------------------------------------------

def defining_event_info(y: np.ndarray, p: np.ndarray, threshold: float, target: float = 0.90) -> Dict[str, Any]:
    """Describe the death whose predicted probability defines the threshold.

    Under the sensitivity-constrained policy the selected threshold equals the
    predicted probability of the death at ascending rank ``FN + 1`` among
    deaths (deaths ranked below it are the false negatives).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n_events = int(y.sum())
    p_death = np.sort(p[y == 1])
    fn = int(np.sum(p_death < threshold)) if np.isfinite(threshold) else 0
    rank = fn + 1
    p_def = float(p_death[rank - 1]) if np.isfinite(threshold) and rank <= len(p_death) else float("nan")
    return {
        "eventos": n_events,
        "fn_max_permitido": int(math.floor(round((1.0 - target) * n_events, 9))),
        "fn_observado": fn,
        "obito_definidor_rank_ascendente": rank,
        "obito_definidor_prob_prevista": p_def,
        "obito_definidor_coincide_com_limiar": bool(np.isfinite(p_def) and np.isclose(p_def, threshold)),
    }


def threshold_bootstrap(
    y: np.ndarray,
    p: np.ndarray,
    target: float = 0.90,
    n_boot: int = 2000,
    seed: int = 42,
    stored_threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Outcome-stratified bootstrap of the sensitivity-constrained threshold.

    For each replicate b: derive ``t_b`` on the bootstrap sample, record its
    in-sample sensitivity/specificity/flag rate, then apply ``t_b`` to the
    **original** sample (test performance).  Two corrected estimates are
    reported:

    * ``*_corrigida_media_teste`` — mean over b of the performance of ``t_b``
      on the original sample (the estimate requested for the dissertation);
    * ``*_corrigida_harrell`` — Harrell's optimism correction:
      apparent − mean_b(bootstrap apparent − test).

    Returns (summary_df, draws_df, info_dict).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)

    orig = sensitivity_constrained_threshold(y, p, min_sensitivity=target)
    t0 = float(orig["threshold"])

    draws = []
    for b in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        yb, pb = y[idx], p[idx]
        rb = sensitivity_constrained_threshold(yb, pb, min_sensitivity=target)
        tb = float(rb["threshold"])
        if np.isfinite(tb):
            m = classification_metrics_at_threshold(y, p, tb)
            fr_orig = float(np.mean(p >= tb))
        else:
            m = {"Sensitivity": float("nan"), "Specificity": float("nan")}
            fr_orig = float("nan")
        draws.append({
            "b": b,
            "t_b": tb,
            "flag_rate_boot": rb["flag_rate"],
            "sens_boot_aparente": rb["sensitivity"],
            "espec_boot_aparente": rb["specificity"],
            "sens_original": m["Sensitivity"],
            "espec_original": m["Specificity"],
            "flag_rate_original": fr_orig,
        })
    draws_df = pd.DataFrame(draws)

    def _q(col: str) -> Tuple[float, float, float]:
        v = draws_df[col].dropna().to_numpy()
        if v.size == 0:
            return (float("nan"),) * 3
        return float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    t_med, t_lo, t_hi = _q("t_b")
    f_med, f_lo, f_hi = _q("flag_rate_boot")
    sens_app, spec_app = orig["sensitivity"], orig["specificity"]
    sens_test = float(draws_df["sens_original"].mean())
    spec_test = float(draws_df["espec_original"].mean())
    sens_opt = float((draws_df["sens_boot_aparente"] - draws_df["sens_original"]).mean())
    spec_opt = float((draws_df["espec_boot_aparente"] - draws_df["espec_original"]).mean())
    defining = defining_event_info(y, p, t0, target=target)

    info: Dict[str, Any] = {
        "alvo_sensibilidade": float(target),
        "B": int(n_boot),
        "seed": int(seed),
        "estratificado_por_desfecho": True,
        "n": int(n),
        "limiar_original": t0,
        "limiar_armazenado": float(stored_threshold) if stored_threshold is not None else float("nan"),
        "t_b_mediana": t_med, "t_b_p2_5": t_lo, "t_b_p97_5": t_hi,
        "flag_rate_original": orig["flag_rate"],
        "flag_rate_boot_mediana": f_med, "flag_rate_boot_p2_5": f_lo, "flag_rate_boot_p97_5": f_hi,
        "flag_rate_corrigida_media_teste": float(draws_df["flag_rate_original"].mean()),
        "sensibilidade_aparente": sens_app,
        "sensibilidade_corrigida_media_teste": sens_test,
        "sensibilidade_otimismo": sens_opt,
        "sensibilidade_corrigida_harrell": float(sens_app - sens_opt),
        "especificidade_aparente": spec_app,
        "especificidade_corrigida_media_teste": spec_test,
        "especificidade_otimismo": spec_opt,
        "especificidade_corrigida_harrell": float(spec_app - spec_opt),
        "prop_reamostragens_sens_original_abaixo_alvo": float((draws_df["sens_original"] < target - 1e-12).mean()),
        "n_reamostragens_sem_limiar": int(draws_df["t_b"].isna().sum()),
        **defining,
    }
    summary_df = pd.DataFrame([{"metrica": k, "valor": v} for k, v in info.items()])
    return summary_df, draws_df, info


# ---------------------------------------------------------------------------
# B3 — risk bands (triple cohort)
# ---------------------------------------------------------------------------

def risk_bands_table(triple: pd.DataFrame, cutoffs: Tuple[float, float] = NRI_CUTOFFS) -> pd.DataFrame:
    """Per method × band: n, % of cohort, deaths, observed mortality, mean predicted risk.

    Bands use ``stats_compare.categorize_risk`` with the NRI cut-offs, so a
    value of exactly 15% falls in the intermediate band (as in the NRI).
    """
    y = triple[OUTCOME_COL].astype(int).to_numpy()
    n = len(y)
    rows = []
    for method, col in METHOD_COLUMNS.items():
        p = triple[col].astype(float).to_numpy()
        cats = np.array([categorize_risk(float(v), cutoffs) for v in p])
        for key, label in _BAND_ORDER:
            m = cats == key
            k = int(m.sum())
            d = int(y[m].sum())
            rows.append({
                "metodo": method,
                "faixa": label,
                "n": k,
                "pct_coorte": _safe_div(k, n) * 100 if n else float("nan"),
                "obitos": d,
                "mortalidade_observada": _safe_div(d, k),
                "risco_previsto_medio": float(p[m].mean()) if k else float("nan"),
                "n_coorte": n,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B4 — rule-out profile at the stored primary threshold
# ---------------------------------------------------------------------------

def ruleout_row(y: np.ndarray, p: np.ndarray, threshold: float, cohort: str) -> Dict[str, Any]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    m = classification_metrics_at_threshold(y, p, threshold)
    tp, fp, tn, fn = m["TP"], m["FP"], m["TN"], m["FN"]
    n_below, n_above = tn + fn, tp + fp
    lo, hi = wilson_ci(fn, n_below)
    return {
        "coorte": cohort,
        "limiar": float(threshold),
        "n": int(len(y)),
        "n_abaixo": int(n_below),
        "obitos_abaixo": int(fn),
        "mortalidade_abaixo": _safe_div(fn, n_below),
        "mortalidade_abaixo_ic95_inf": lo,
        "mortalidade_abaixo_ic95_sup": hi,
        "n_acima": int(n_above),
        "obitos_acima": int(tp),
        "mortalidade_acima": _safe_div(tp, n_above),
        "flag_rate": _safe_div(n_above, len(y)),
    }


def ruleout_table(
    y_dev: np.ndarray, p_dev: np.ndarray, y_triple: np.ndarray, p_triple: np.ndarray, threshold: float,
) -> pd.DataFrame:
    return pd.DataFrame([
        ruleout_row(y_dev, p_dev, threshold, COHORT_DEV),
        ruleout_row(y_triple, p_triple, threshold, COHORT_TRIPLE),
    ])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_bytes(fig) -> Tuple[bytes, bytes]:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out = []
    for fmt in ("png", "svg"):
        buf = BytesIO()
        fig.savefig(buf, format=fmt, dpi=FIG_DPI, bbox_inches="tight")
        out.append(buf.getvalue())
    plt.close(fig)
    return out[0], out[1]


def threshold_bootstrap_figure(draws: pd.DataFrame, stored_threshold: float) -> Tuple[bytes, bytes]:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    with matplotlib.rc_context(_FONT_RC):
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        v = draws["t_b"].dropna().to_numpy() * 100
        ax.hist(v, bins=40, color=_BAR_COLOR, edgecolor="white", linewidth=0.4)
        if stored_threshold is not None and np.isfinite(stored_threshold):
            ax.axvline(stored_threshold * 100, color="#b2182b", linestyle="--", linewidth=1.4,
                       label=f"Limiar armazenado ({stored_threshold * 100:.1f}%)".replace(".", ","))
            ax.legend(frameon=False, fontsize=9)
        ax.set_xlabel("Limiar bootstrap t_b (%)")
        ax.set_ylabel("Reamostragens (n)")
        fig.tight_layout()
        return _fig_bytes(fig)


def shap_bar_figure(table: pd.DataFrame) -> Tuple[bytes, bytes]:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    with matplotlib.rc_context(_FONT_RC):
        k = len(table)
        fig, ax = plt.subplots(figsize=(6.5, 0.32 * k + 1.0))
        ax.barh(table["feature"].iloc[::-1], table["mean_abs_shap"].iloc[::-1], color=_BAR_COLOR)
        ax.set_xlabel("Média |SHAP| (probabilidade prevista)")
        ax.tick_params(axis="y", labelsize=9)
        fig.tight_layout()
        return _fig_bytes(fig)


def shap_beeswarm_figure(summary: Dict[str, Any], max_display: int = 15) -> Tuple[bytes, bytes]:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import shap  # noqa: PLC0415

    with matplotlib.rc_context(_FONT_RC):
        import inspect  # noqa: PLC0415

        seed = int(summary.get("random_state", 42))
        kwargs: Dict[str, Any] = {}
        if "rng" in inspect.signature(shap.summary_plot).parameters:
            kwargs["rng"] = np.random.default_rng(seed)  # deterministic jitter/ordering
        else:  # older shap: jitter uses the global NumPy RNG
            np.random.seed(seed)
        plt.figure()
        shap.summary_plot(
            summary["shap_values"],
            summary["X_proc"],
            feature_names=summary["feature_names"],
            plot_type="dot",
            max_display=max_display,
            show=False,
            **kwargs,
        )
        fig = plt.gcf()
        for ax in fig.axes:
            ax.set_title("")
        main_ax, *cbar_axes = fig.axes
        main_ax.set_xlabel("Valor SHAP (impacto na probabilidade prevista)")
        for cax in cbar_axes:
            cax.set_ylabel("Valor da variável")
            labels = [t.get_text() for t in cax.get_yticklabels()]
            if labels:
                cax.set_yticklabels([{"High": "Alto", "Low": "Baixo"}.get(t, t) for t in labels])
        fig.suptitle("")
        return _fig_bytes(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def build_dissertation_exports(
    df: pd.DataFrame,
    stored_threshold: float,
    pipe=None,
    X: Optional[pd.DataFrame] = None,
    *,
    model_name: str = "",
    model_version: str = "",
    bundle_info: Optional[Dict[str, Any]] = None,
    n_boot: int = 2000,
    seed: int = 42,
    shap_top_n: int = 15,
    shap_random_state: int = 42,
    targets: Iterable[float] = SENSITIVITY_TARGETS,
) -> Dict[str, bytes]:
    """Compute every export and return ``{filename: bytes}`` (incl. metadata.json).

    Block A is skipped (and recorded in metadata) when ``pipe``/``X`` is not
    given or the model is not tree-based.
    """
    files: Dict[str, bytes] = {}
    y_dev, p_dev = development_oof_arrays(df)
    triple = build_triple_cohort(df)
    y_tri = triple[OUTCOME_COL].astype(int).to_numpy()
    p_tri = triple[AI_RISK_COL].astype(float).to_numpy()

    meta: Dict[str, Any] = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "model_version": model_version,
        "modelo": model_name,
        "bundle_saved_at": (bundle_info or {}).get("saved_at"),
        "bundle_schema_version": (bundle_info or {}).get("bundle_schema_version"),
        "bundle_training_source": (bundle_info or {}).get("training_source"),
        "limiar_armazenado": float(stored_threshold),
        "coorte_desenvolvimento": {"n": int(len(y_dev)), "eventos": int(y_dev.sum())},
        "coorte_tripla": {"n": int(len(y_tri)), "eventos": int(y_tri.sum())},
        "seeds": {"bootstrap": int(seed), "shap": int(shap_random_state)},
        "bootstrap_B": int(n_boot),
        "alvos_sensibilidade": [float(t) for t in targets],
        "faixas_risco_cortes": list(NRI_CUTOFFS),
        "fonte_predicoes": "df['ia_risk_oof'] = artifacts.oof_predictions[modelo] (OOF calibrado)",
    }

    # ── Block A ───────────────────────────────────────────────────────────
    try:
        import shap  # noqa: PLC0415
        meta["shap_version"] = shap.__version__
    except ImportError:
        shap = None
        meta["shap_version"] = None
    if shap is not None and pipe is not None and X is not None and hasattr(pipe.named_steps["model"], "feature_importances_"):
        from explainability import shap_global_summary  # noqa: PLC0415

        summ = shap_global_summary(pipe, X, top_n=shap_top_n, random_state=shap_random_state)
        tbl = summ["table"][["rank", "feature", "feature_raw", "mean_abs_shap", "spearman_value_vs_shap"]]
        files["A1_shap_global_top15.csv"] = _csv_bytes(tbl)
        files["A2_fig_shap_bar.png"], files["A2_fig_shap_bar.svg"] = shap_bar_figure(summ["table"])
        files["A3_fig_shap_beeswarm.png"], files["A3_fig_shap_beeswarm.svg"] = shap_beeswarm_figure(summ, shap_top_n)
        meta["shap"] = {
            "explained_output": summ["explained_output"],
            "explained_estimator": summ["explained_estimator"],
            "final_output_calibrated": summ["final_output_calibrated"],
            "calibration_method": summ["calibration_method"],
            "expected_value": summ["expected_value"],
            "n": int(summ["X_proc"].shape[0]),
            "n_features_processadas": int(summ["X_proc"].shape[1]),
        }
    else:
        meta["shap"] = {"status": "skipped"}

    # ── Block B ───────────────────────────────────────────────────────────
    files["B1_sensitivity_targets.csv"] = _csv_bytes(
        sensitivity_targets_table(y_dev, p_dev, y_tri, p_tri, targets=tuple(targets))
    )
    b2_summary, b2_draws, b2_info = threshold_bootstrap(
        y_dev, p_dev, target=0.90, n_boot=n_boot, seed=seed, stored_threshold=stored_threshold,
    )
    files["B2_threshold_bootstrap_summary.csv"] = _csv_bytes(b2_summary)
    files["B2_threshold_bootstrap_draws.csv"] = _csv_bytes(b2_draws)
    files["B2_fig_threshold_bootstrap.png"], files["B2_fig_threshold_bootstrap.svg"] = (
        threshold_bootstrap_figure(b2_draws, stored_threshold)
    )
    meta["B2_limiar_original_rederivado"] = b2_info["limiar_original"]
    files["B3_risk_bands_triple.csv"] = _csv_bytes(risk_bands_table(triple))
    files["B4_ruleout_primary_threshold.csv"] = _csv_bytes(
        ruleout_table(y_dev, p_dev, y_tri, p_tri, stored_threshold)
    )

    files["metadata.json"] = json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default).encode("utf-8")
    return files


def build_dissertation_export_zip(*args, **kwargs) -> bytes:
    """ZIP of :func:`build_dissertation_exports` output."""
    files = build_dissertation_exports(*args, **kwargs)
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()
