"""Subgroup assignment and per-subgroup metric evaluation.

Extracted from ``app.py`` during the Phase 4 conservative refactor.
These are pure data-transform helpers with **no** Streamlit dependency:

* :func:`surgery_family` - three-way classification (coronary / valve /
  mixed / other) from the Surgery procedure text.
* :func:`surgery_type_group` - finer-grained surgery-type label used by
  the subgroup-analysis tab.
* :func:`lvef_group` - ejection-fraction bucket (preserved / mildly
  reduced / reduced / unknown) with an optional fallback value.
* :func:`renal_group` - renal-function bucket based on measured
  creatinine clearance, with a Cockcroft-Gault fallback when clearance
  is missing but creatinine, age, weight, and sex are available.
* :func:`evaluate_subgroup` - run ``evaluate_scores_with_threshold``
  across groupby segments and attach bootstrap 95% CI columns.

The bilingual ``tr(en, pt) -> str`` helper is passed in as an explicit
argument so this module has no hidden globals and stays unit-testable
without Streamlit.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd

from risk_data import parse_number, split_surgery_procedures
from stats_compare import bootstrap_metrics_ci, evaluate_scores_with_threshold

TrFn = Callable[[str, str], str]


def surgery_family(text: object, tr: TrFn) -> str:
    """Coarse surgery family: coronary / valve / mixed / other."""
    parts = set(split_surgery_procedures(text))
    coronary = bool(parts & {"cabg", "opcab"})
    valve = bool(parts & {"avr", "av repair", "mv repair", "mvr", "tv repair", "tvr", "ross"})
    if coronary and not valve:
        return tr("Coronary", "Coronária")
    if valve and not coronary:
        return tr("Valve", "Valvar")
    if coronary and valve:
        return tr("Mixed", "Mista")
    return tr("Other", "Outra")


def surgery_type_group(text: object, tr: TrFn) -> str:
    """Surgery-type grouping used by the subgroup-analysis tab."""
    parts = set(split_surgery_procedures(text))
    coronary = bool(parts & {"cabg", "opcab"})
    valve = bool(parts & {"avr", "av repair", "mv repair", "mvr", "tv repair", "tvr", "ross"})
    aorta = bool(parts & {"aortic aneurism repair", "aortic dissection repair", "bentall-de bono procedure", "valve sparing aortic root replacement (david procedure)"})
    transplant = bool(parts & {"heart transplant"})
    classes = [coronary, valve, aorta, transplant]
    n_active = sum(classes)
    if n_active > 1:
        return tr("Mixed", "Mista")
    if coronary:
        return tr("Myocardial revascularization", "Revascularização do miocárdio")
    if valve:
        return tr("Valve surgery", "Cirurgia valvar")
    if aorta:
        return tr("Aortic surgery", "Cirurgia da aorta")
    if transplant:
        return tr("Transplant / assist device", "Transplante / assist device")
    return tr("Other", "Outra")


def lvef_group(value: object, fallback: object, tr: TrFn) -> str:
    """Ejection-fraction bucket with optional fallback."""
    v = parse_number(value)
    if pd.isna(v) and fallback is not None:
        v = parse_number(fallback)
    if pd.isna(v):
        return tr("Unknown", "Desconhecida")
    if v >= 50:
        return tr("Preserved", "Preservada")
    if v >= 41:
        return tr("Mildly reduced", "Levemente reduzida")
    return tr("Reduced", "Reduzida")


def renal_group(
    clearance: object,
    dialysis: object,
    creatinine: object,
    age: object,
    weight: object,
    sex: object,
    tr: TrFn,
) -> str:
    """Renal function bucket from creatinine clearance (or Cockcroft-Gault fallback)."""
    if str(dialysis).strip().lower() in {"yes", "sim", "1", "true"}:
        return tr("Dialysis", "Diálise")

    v = parse_number(clearance)

    # Fallback: compute Cockcroft-Gault if clearance is missing but
    # creatinine, age, weight, and sex are available
    if pd.isna(v) and creatinine is not None:
        scr = parse_number(creatinine)
        a = parse_number(age)
        w = parse_number(weight)
        sx = str(sex).strip().upper()[:1] if pd.notna(sex) else ""
        if not pd.isna(scr) and scr > 0 and not pd.isna(a) and not pd.isna(w) and sx in {"M", "F"}:
            factor = 0.85 if sx == "F" else 1.0
            v = ((140.0 - a) * w * factor) / (72.0 * scr)

    if pd.isna(v):
        return tr("Unknown", "Desconhecida")
    if v > 85:
        return ">85"
    if v > 50:
        return "51-85"
    return "<=50"


def evaluate_subgroup(
    df_in: pd.DataFrame,
    subgroup_col: str,
    score_cols: List[str],
    threshold: float,
) -> pd.DataFrame:
    """Evaluate each score on every subgroup with a fixed decision threshold.

    Small groups (<20 rows) or groups with only one outcome class are
    silently skipped.  Adds bootstrap 95% CI columns for AUC / AUPRC /
    Brier (500 resamples, fixed seed=42 for reproducibility).
    """
    rows = []
    for subgroup_name, sub in df_in.groupby(subgroup_col):
        for score in score_cols:
            s = sub[["morte_30d", score]].dropna()
            if len(s) < 20 or s["morte_30d"].nunique() < 2:
                continue
            metrics = evaluate_scores_with_threshold(s, "morte_30d", [score], threshold)
            if metrics.empty:
                continue
            rec = metrics.iloc[0].to_dict()
            rec["Subgroup"] = subgroup_col
            rec["Group"] = subgroup_name
            rec["Deaths"] = int(s["morte_30d"].sum())
            # Bootstrap CI for AUC, AUPRC, Brier (reduced resamples for speed in subgroups)
            ci = bootstrap_metrics_ci(s["morte_30d"].values, s[score].values, n_boot=500, seed=42)
            rec["AUC_IC95_inf"] = ci.get("AUC_IC95_inf", np.nan)
            rec["AUC_IC95_sup"] = ci.get("AUC_IC95_sup", np.nan)
            rec["AUPRC_IC95_inf"] = ci.get("AUPRC_IC95_inf", np.nan)
            rec["AUPRC_IC95_sup"] = ci.get("AUPRC_IC95_sup", np.nan)
            rec["Brier_IC95_inf"] = ci.get("Brier_IC95_inf", np.nan)
            rec["Brier_IC95_sup"] = ci.get("Brier_IC95_sup", np.nan)
            rows.append(rec)
    return pd.DataFrame(rows)
