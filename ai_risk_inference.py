"""Frozen-model inference helpers for AI Risk.

Extracted from app.py to isolate the core inference logic from Streamlit
UI code.  All seven functions here are pure computation: they carry no
Streamlit state, no ``tr()`` closure, and no UI side-effects.

Provides
--------
_get_numeric_columns_from_pipeline  – extract numeric columns from a Pipeline
_safe_select_features               – select/pad feature columns from a DataFrame
_build_input_row                    – assemble a single-row DataFrame from form data
_align_input_to_training_schema     – coerce dtypes to match training-time schema
_patient_identifier_from_row        – best-effort patient ID for incident reporting
_run_ai_risk_inference_row          – unified per-row frozen-inference core
apply_frozen_model_to_temporal_cohort – batch temporal cohort inference
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from modeling import clean_features
from model_metadata import assess_input_completeness
from risk_data import (
    BLANK_MEANS_NO_COLUMNS,
    MISSING_TOKENS,
    is_combined_surgery,
    procedure_weight,
    thoracic_aorta_surgery,
)


def _is_missing_token(value: object) -> bool:
    """Return True for scalar values treated as missing by dataset ingestion."""
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in MISSING_TOKENS


def _get_numeric_columns_from_pipeline(model_pipeline) -> set:
    """Extract the set of numeric column names from a trained sklearn Pipeline."""
    try:
        prep = model_pipeline.named_steps.get("prep")
        if prep and hasattr(prep, "transformers"):
            for name, _trans, cols in prep.transformers:
                if name == "num":
                    return set(cols)
    except Exception:
        pass
    return set()


def _safe_select_features(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Select feature columns from df, adding missing ones as NaN."""
    out = df.copy()
    for c in feature_columns:
        if c not in out.columns:
            out[c] = np.nan
    return out[feature_columns]


def _build_input_row(feature_columns, form: Dict[str, object]) -> pd.DataFrame:
    row = {c: np.nan for c in feature_columns}
    # Build normalized lookups to handle whitespace/truncation in Excel column names
    _norm = {c.strip(): c for c in feature_columns}
    # Also map by prefix to handle truncated names (e.g. "Surgical Priorit" matches "Surgical Priority")
    for k, v in form.items():
        if k in row:
            row[k] = v
        elif k.strip() in _norm:
            row[_norm[k.strip()]] = v
        else:
            # Check if any feature column is a prefix of form key (truncation)
            k_stripped = k.strip()
            for fc in feature_columns:
                fc_stripped = fc.strip()
                if fc_stripped != k_stripped and (k_stripped.startswith(fc_stripped) or fc_stripped.startswith(k_stripped)):
                    row[fc] = v
                    break
    surg = form.get("Surgery", "")
    row["cirurgia_combinada"] = is_combined_surgery(surg)
    row["peso_procedimento"] = procedure_weight(surg)
    row["thoracic_aorta_flag"] = thoracic_aorta_surgery(surg)

    # Clean numeric fields that may contain string values from CSV
    _susp = row.get("Suspension of Anticoagulation (day)")
    if isinstance(_susp, str):
        _susp_clean = _susp.strip().replace(">", "").strip()
        if _susp_clean.lower() in MISSING_TOKENS:
            row["Suspension of Anticoagulation (day)"] = np.nan
        else:
            try:
                row["Suspension of Anticoagulation (day)"] = float(_susp_clean)
            except (ValueError, TypeError):
                row["Suspension of Anticoagulation (day)"] = np.nan

    defaults = {col: "No" for col in BLANK_MEANS_NO_COLUMNS}
    defaults.update({
        "Aortic Stenosis": "None",
        "Aortic Regurgitation": "None",
        "Mitral Stenosis": "None",
        "Mitral Regurgitation": "None",
        "Tricuspid Regurgitation": "None",
    })
    # Apply defaults to the dict before constructing the DataFrame so that
    # columns receiving string defaults are not first typed as float64 (NaN),
    # which would trigger a pandas FutureWarning on mixed-type assignment.
    for c, v in defaults.items():
        if c in row:
            _cur = row[c]
            if c in BLANK_MEANS_NO_COLUMNS and _is_missing_token(_cur):
                row[c] = v
            elif (
                c not in BLANK_MEANS_NO_COLUMNS
                and (_cur is None or (isinstance(_cur, float) and pd.isna(_cur)) or str(_cur).strip() == "")
            ):
                row[c] = v

    out = pd.DataFrame([row])
    return out


def _align_input_to_training_schema(input_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    aligned = input_df.copy()
    for col in aligned.columns:
        if col in reference_df.columns:
            if pd.api.types.is_numeric_dtype(reference_df[col]):
                # Force numeric: strip symbols, fix comma decimals, coerce strings to NaN
                if aligned[col].dtype == object:
                    aligned[col] = (
                        aligned[col].astype(str)
                        .str.replace(r'[><~]', '', regex=True)
                        .str.strip()
                        .str.replace(',', '.', regex=False)
                    )
                aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
            elif reference_df[col].dtype == object and aligned[col].dtype != object:
                # Force categorical
                aligned[col] = aligned[col].astype(str)
    return aligned


def _patient_identifier_from_row(row_dict: Dict[str, object], fallback_index: int) -> str:
    """Best-effort patient identifier for incident reporting across all inference flows.

    Looks for common identifier columns (``Name``, ``Nome``, ``_patient_key``)
    and falls back to a 1-based row index when none are present.
    Used by individual, batch, and temporal-validation inference paths.
    """
    for key in ("Name", "Nome", "_patient_key"):
        val = row_dict.get(key)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            text = str(val).strip()
            if text:
                return text
    return f"row_{fallback_index + 1}"


def _run_ai_risk_inference_row(
    *,
    model_pipeline,
    feature_columns: list,
    reference_df: pd.DataFrame,
    row_dict: dict,
    patient_id: str,
    numeric_cols: set,
    language: str = "English",
) -> dict:
    """Unified per-row AI Risk frozen-inference core.

    This is the single inference path shared by individual, batch, and
    temporal-validation flows.  It encapsulates every step between a raw input
    dict and a calibrated probability so that parsing, schema alignment, feature
    cleaning, completeness assessment, and incident capture are always identical
    across all three callers.

    Parameters
    ----------
    model_pipeline : sklearn-compatible pipeline
        Frozen estimator (preprocessing + fitted estimator + calibration).
    feature_columns : list[str]
        Locked training feature schema (``artifacts.feature_columns``).
    reference_df : pd.DataFrame
        Training-time feature DataFrame used to drive numeric/categorical
        coercion inside ``_align_input_to_training_schema``.
    row_dict : dict
        Raw input values (form data or a row from an uploaded file).
    patient_id : str
        Best-effort patient identifier used in incident reporting.
    numeric_cols : set[str]
        Set of column names that should be numeric (extracted from the
        preprocessing pipeline).  Passed to ``clean_features`` to prevent
        single-row categorical loss.
    language : str
        Forwarded to ``assess_input_completeness`` for label localisation.

    Returns
    -------
    dict
        ``probability``  – float or None (None means inference failed),
        ``completeness`` – dict from ``assess_input_completeness``, or None,
        ``incident``     – ``{"patient_id", "stage", "reason"}`` or None,
        ``model_input``  – aligned/cleaned DataFrame ready for ``predict_proba``,
                           or None on failure; available so callers can run
                           additional models without rebuilding the input.
    """
    try:
        input_row = _build_input_row(feature_columns, row_dict)
        input_row = _align_input_to_training_schema(input_row, reference_df)
        model_input = clean_features(input_row[feature_columns], numeric_columns=numeric_cols)
        # Final safety: force any numeric columns that are still object-dtype
        # after the main cleaning pass (can happen with mixed-type CSV imports).
        for _c in model_input.columns:
            if (
                model_input[_c].dtype == object
                and _c in reference_df.columns
                and pd.api.types.is_numeric_dtype(reference_df[_c])
            ):
                model_input[_c] = pd.to_numeric(
                    model_input[_c].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce",
                )
        prob = float(model_pipeline.predict_proba(model_input)[:, 1][0])
        comp = assess_input_completeness(feature_columns, input_row, language)
        return {
            "probability": prob,
            "completeness": comp,
            "incident": None,
            "model_input": model_input,
        }
    except Exception as exc:  # noqa: BLE001 – surfaced to caller as incident
        return {
            "probability": None,
            "completeness": None,
            "incident": {
                "patient_id": patient_id,
                "stage": "ai_risk_inference",
                "reason": f"{type(exc).__name__}: {exc}",
            },
            "model_input": None,
        }


def apply_frozen_model_to_temporal_cohort(
    *,
    model_pipeline,
    feature_columns,
    reference_df: pd.DataFrame,
    temporal_data: pd.DataFrame,
    language: str = "English",
    progress_callback=None,
) -> dict:
    """Apply the frozen AI Risk model to a prepared temporal cohort.

    This is the single reusable entry point for temporal AI Risk inference.
    Behaviour is intentionally identical to the previous in-line loop:

    * inputs are routed through ``_build_input_row`` then aligned to the
      training schema via ``_align_input_to_training_schema`` (the shared
      numeric/categorical normalizer);
    * the frozen pipeline (``model_pipeline.predict_proba``) is applied
      exactly as saved — no retraining, no recalibration;
    * per-row completeness is assessed with ``assess_input_completeness``.

    The only behavioural addition is structured incident capture: when a
    row fails inference, instead of silently producing ``NaN`` we still
    record ``NaN`` in ``probabilities`` (so downstream metric code keeps
    working) **and** append a per-patient incident dict that the UI layer
    can surface alongside STS Score incidents.

    Parameters
    ----------
    model_pipeline : sklearn-compatible pipeline
        The frozen estimator (preprocessing + fitted estimator + calibration).
    feature_columns : list[str]
        The locked training feature schema.
    reference_df : pd.DataFrame
        Training-time feature dataframe used to drive numeric/categorical
        coercion in ``_align_input_to_training_schema``.
    temporal_data : pd.DataFrame
        The prepared temporal cohort (already passed through
        ``prepare_master_dataset``).
    language : str
        Forwarded to ``assess_input_completeness`` for label localisation.
    progress_callback : callable, optional
        ``progress_callback(i, n)`` invoked roughly every 5% of rows so the
        caller can drive a Streamlit progress bar without this helper
        importing Streamlit.

    Returns
    -------
    dict
        ``probabilities`` (list[float], NaN on failure),
        ``completeness`` (list[str]),
        ``incidents`` (list[dict] with ``patient_id`` / ``stage`` / ``reason``),
        ``n_total`` (int), ``n_failed`` (int).
    """
    n_total = len(temporal_data)
    probabilities: List[float] = []
    completeness: List[str] = []
    incidents: List[Dict[str, object]] = []

    numeric_cols = _get_numeric_columns_from_pipeline(model_pipeline)
    rows = temporal_data.to_dict(orient="records")

    for idx, row_dict in enumerate(rows):
        patient_id = _patient_identifier_from_row(row_dict, idx)
        _infer = _run_ai_risk_inference_row(
            model_pipeline=model_pipeline,
            feature_columns=feature_columns,
            reference_df=reference_df,
            row_dict=row_dict,
            patient_id=patient_id,
            numeric_cols=numeric_cols,
            language=language,
        )
        if _infer["incident"] is not None:
            probabilities.append(np.nan)
            completeness.append("error")
            incidents.append(_infer["incident"])
        else:
            probabilities.append(_infer["probability"])
            completeness.append(_infer["completeness"]["level"])

        if progress_callback is not None:
            try:
                progress_callback(idx + 1, n_total)
            except Exception:
                pass

    return {
        "probabilities": probabilities,
        "completeness": completeness,
        "incidents": incidents,
        "n_total": n_total,
        "n_failed": len(incidents),
    }
