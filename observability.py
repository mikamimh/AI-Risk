"""Phase 3: Observability and transparency layer for AI Risk.

Minimal, localized module that turns already-populated data structures
(IngestionReport, prepared.info, training leaderboard, STS Score execution
log) into a uniform shape for end-of-run reporting.

Design goals
------------
* Build on existing structures - do NOT duplicate or mutate them.
* Pure data transformations (no Streamlit imports at module top level) so the
  module is trivially unit-testable and safe to persist in the joblib bundle.
* A single Streamlit renderer (``render_run_report``) draws short
  user-facing messages plus expandable technical details.
* Does not change methodology. No config or model changes.

Used by
-------
* ``app.py._compute_bundle`` - builds a ``RunReport`` after each major phase
  and attaches it to the bundle under key ``"run_report"``.
* The training tab renders the report via ``render_run_report``.
* The temporal validation path reuses ``build_step_sts_score`` and
  ``render_sts_score_incidents`` to mirror the batch flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Data shapes
# --------------------------------------------------------------------------- #

STATUS_OK = "ok"
STATUS_WARNING = "warning"
STATUS_ERROR = "error"

_STATUS_RANK = {STATUS_OK: 0, STATUS_WARNING: 1, STATUS_ERROR: 2}


@dataclass
class RunStep:
    """One row in the run observability report.

    Attributes
    ----------
    name : str
        Short phase label, e.g. ``"Ingestion"``, ``"STS Score"``.
    status : str
        One of ``ok``, ``warning``, ``error``.
    summary : str
        Single-line human-readable headline shown without expanding.
    counters : dict
        Flat mapping of counter label -> number or short string, rendered as a
        small table inside the expander.
    details : list of str
        Short bullet strings with extra context.
    incidents : list of dict
        Row-shaped incidents (e.g. per-patient STS Score failures) rendered
        as a dataframe inside the expander.  Keys should be consistent across
        rows.
    """

    name: str
    status: str
    summary: str
    counters: Dict[str, Any] = field(default_factory=dict)
    details: List[str] = field(default_factory=list)
    incidents: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunReport:
    """End-of-run observability report aggregating every major phase."""

    steps: List[RunStep] = field(default_factory=list)

    def add(self, step: Optional[RunStep]) -> None:
        if step is not None:
            self.steps.append(step)

    def overall_status(self) -> str:
        if not self.steps:
            return STATUS_OK
        return max((s.status for s in self.steps), key=lambda s: _STATUS_RANK.get(s, 0))

    def has_errors(self) -> bool:
        return any(s.status == STATUS_ERROR for s in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "RunReport":
        if not payload:
            return cls()
        steps_raw = payload.get("steps", []) or []
        steps = [
            RunStep(
                name=s.get("name", ""),
                status=s.get("status", STATUS_OK),
                summary=s.get("summary", ""),
                counters=dict(s.get("counters", {})),
                details=list(s.get("details", [])),
                incidents=list(s.get("incidents", [])),
            )
            for s in steps_raw
        ]
        return cls(steps=steps)


# --------------------------------------------------------------------------- #
# Builders - convert existing structures into RunStep
# --------------------------------------------------------------------------- #


def build_step_ingestion(ingestion_report: Any) -> Optional[RunStep]:
    """Build a ``RunStep`` from a ``risk_data.IngestionReport``.

    Returns ``None`` if ``ingestion_report`` is falsy (e.g. legacy path that
    did not populate one).  The caller is expected to check ``has_errors()``
    independently and hard-stop the run before training if required columns
    are missing - this builder still returns a populated step in that case so
    the error is visible in the report.
    """
    if ingestion_report is None:
        return None

    required_failures = list(getattr(ingestion_report, "required_failures", []) or [])
    columns_converted = list(getattr(ingestion_report, "columns_converted", []) or [])
    missing_normalized = list(getattr(ingestion_report, "missing_normalized", []) or [])
    columns_dropped = list(getattr(ingestion_report, "columns_dropped", []) or [])
    warnings_list = list(getattr(ingestion_report, "warnings", []) or [])

    total_missing_tokens = sum(int(getattr(a, "count", 0) or 0) for a in missing_normalized)
    total_values_parsed = sum(int(getattr(a, "count", 0) or 0) for a in columns_converted)
    sparse_count = sum(
        1 for a in columns_dropped if getattr(a, "action", "") == "dropped_sparse"
    )
    constant_count = sum(
        1 for a in columns_dropped if getattr(a, "action", "") == "dropped_constant"
    )

    counters: Dict[str, Any] = {
        "Rows in": int(getattr(ingestion_report, "n_rows_input", 0) or 0),
        "Rows out": int(getattr(ingestion_report, "n_rows_output", 0) or 0),
        "Columns in": int(getattr(ingestion_report, "n_columns_input", 0) or 0),
        "Columns out": int(getattr(ingestion_report, "n_columns_output", 0) or 0),
        "Columns converted to numeric": len(columns_converted),
        "Values parsed as numeric": total_values_parsed,
        "Columns with missing tokens normalized": len(missing_normalized),
        "Missing tokens replaced": total_missing_tokens,
        "Sparse columns flagged (>95% missing)": sparse_count,
        "Constant columns flagged": constant_count,
        "Warnings": len(warnings_list),
    }

    details: List[str] = []
    for a in columns_converted[:10]:
        details.append(f"Converted to numeric: {a.column} ({a.detail})")
    if len(columns_converted) > 10:
        details.append(f"... and {len(columns_converted) - 10} more converted column(s)")
    for a in missing_normalized[:10]:
        details.append(f"Missing tokens normalized: {a.column} ({a.detail})")
    if len(missing_normalized) > 10:
        details.append(f"... and {len(missing_normalized) - 10} more column(s) with missing tokens")
    for a in columns_dropped[:10]:
        details.append(f"Flagged: {a.column} ({a.detail})")
    if len(columns_dropped) > 10:
        details.append(f"... and {len(columns_dropped) - 10} more flagged column(s)")
    for a in warnings_list[:10]:
        details.append(f"Warning: {a.column} ({a.detail})")
    if len(warnings_list) > 10:
        details.append(f"... and {len(warnings_list) - 10} more warning(s)")

    incidents: List[Dict[str, Any]] = []
    for a in required_failures:
        incidents.append({
            "column": getattr(a, "column", ""),
            "action": getattr(a, "action", "required_missing"),
            "detail": getattr(a, "detail", ""),
        })

    if required_failures:
        status = STATUS_ERROR
        summary = (
            f"Ingestion halted: {len(required_failures)} required column(s) missing. "
            "See incidents for details."
        )
    elif warnings_list or sparse_count or constant_count:
        status = STATUS_WARNING
        summary = (
            f"Ingestion completed with {len(warnings_list)} warning(s); "
            f"{sparse_count} sparse and {constant_count} constant column(s) flagged."
        )
    else:
        status = STATUS_OK
        summary = (
            f"Ingestion OK: {counters['Rows out']} rows x {counters['Columns out']} columns, "
            f"{len(columns_converted)} column(s) auto-converted to numeric."
        )

    return RunStep(
        name="Ingestion & normalization",
        status=status,
        summary=summary,
        counters=counters,
        details=details,
        incidents=incidents,
    )


def build_step_eligibility(info: Optional[Dict[str, Any]]) -> Optional[RunStep]:
    """Build the cohort assembly / eligibility RunStep from ``prepared.info``.

    Surfaces exclusions (missing surgery or date, unmatched pre/post rows)
    and the final positive-class prevalence.
    """
    if not info:
        return None

    n_rows = int(info.get("n_rows", 0) or 0)
    n_features = int(info.get("n_features", 0) or 0)
    positive_rate = float(info.get("positive_rate", 0.0) or 0.0)
    pre_before = int(info.get("pre_rows_before_criteria", 0) or 0)
    pre_after = int(info.get("pre_rows_after_criteria", 0) or 0)
    excluded_missing = int(info.get("excluded_missing_surgery_or_date", 0) or 0)
    matched = int(info.get("matched_pre_post_rows", 0) or 0)
    excluded_match = int(info.get("excluded_no_pre_post_match", 0) or 0)
    echo_rows = int(info.get("echo_rows", 0) or 0)
    optional_tables = list(info.get("available_optional_tables", []) or [])

    counters: Dict[str, Any] = {
        "Preoperative rows (input)": pre_before,
        "Preoperative rows after exclusions": pre_after,
        "Rows excluded (missing surgery or date)": excluded_missing,
        "Rows matched pre <-> post": matched,
        "Rows excluded (no pre/post match)": excluded_match,
        "Echocardiogram rows joined": echo_rows,
        "Final cohort rows": n_rows,
        "Predictor variables": n_features,
        "Positive-class prevalence": f"{positive_rate * 100:.2f}%",
        "Optional tables available": ", ".join(optional_tables) if optional_tables else "-",
    }

    details: List[str] = []
    if excluded_missing > 0:
        details.append(
            f"{excluded_missing} preoperative row(s) excluded due to missing surgery or surgery date."
        )
    if excluded_match > 0:
        details.append(
            f"{excluded_match} preoperative row(s) excluded because no matching postoperative row was found."
        )

    total_excluded = excluded_missing + excluded_match
    if total_excluded == 0:
        status = STATUS_OK
        summary = (
            f"Cohort assembled: {n_rows} patients, {n_features} predictors, "
            f"prevalence {positive_rate * 100:.2f}%."
        )
    else:
        status = STATUS_WARNING
        summary = (
            f"Cohort assembled with exclusions: {n_rows} patients kept, "
            f"{total_excluded} row(s) excluded."
        )

    return RunStep(
        name="Cohort eligibility",
        status=status,
        summary=summary,
        counters=counters,
        details=details,
    )


def build_step_training(
    leaderboard: Any,
    best_model_name: str,
    n_features: int,
    prevalence: float,
) -> Optional[RunStep]:
    """Build the training RunStep from the modeling leaderboard."""
    if leaderboard is None:
        return None

    try:
        n_models = int(len(leaderboard))
    except Exception:
        n_models = 0

    counters: Dict[str, Any] = {
        "Candidate models evaluated": n_models,
        "Best model (auto-selected)": best_model_name or "-",
        "Predictors used": int(n_features),
        "Positive-class prevalence": f"{prevalence * 100:.2f}%",
    }

    details: List[str] = []
    try:
        if hasattr(leaderboard, "columns") and "Modelo" in leaderboard.columns:
            model_names = list(leaderboard["Modelo"].tolist())
            details.append("Candidates: " + ", ".join(model_names))
            metric_cols = [c for c in ("AUC", "AUPRC", "Brier") if c in leaderboard.columns]
            if metric_cols and best_model_name in model_names:
                row = leaderboard[leaderboard["Modelo"] == best_model_name].iloc[0]
                metrics_fragment = ", ".join(
                    f"{c}={float(row[c]):.4f}" for c in metric_cols
                )
                details.append(f"Best model metrics: {metrics_fragment}")
    except Exception:
        pass

    return RunStep(
        name="Model training",
        status=STATUS_OK,
        summary=(
            f"Training OK: {n_models} candidate(s) evaluated, "
            f"auto-selected '{best_model_name or '-'}'."
        ),
        counters=counters,
        details=details,
    )


def build_step_sts_score(execution_log: Optional[Sequence[Any]]) -> Optional[RunStep]:
    """Build the STS Score RunStep from a ``calculate_sts_batch`` execution log.

    Uses ``sts_cache.summarise_execution_log`` for counts and exposes the
    per-patient incidents for ``stale_fallback`` and ``failed`` statuses.
    """
    if execution_log is None:
        return None

    try:
        import sts_cache as _sc  # lazy so observability stays importable standalone
        summary_counts = _sc.summarise_execution_log(execution_log)
    except Exception:
        summary_counts = {}

    total = sum(int(v) for v in summary_counts.values()) if summary_counts else 0
    if total == 0 and not execution_log:
        return None

    counters: Dict[str, Any] = {
        "Total patients": total,
        "Fresh fetches": int(summary_counts.get("fresh", 0) or 0),
        "Cache hits": int(summary_counts.get("cached", 0) or 0),
        "Refreshed (expired -> refetched)": int(summary_counts.get("refreshed", 0) or 0),
        "Stale fallback": int(summary_counts.get("stale_fallback", 0) or 0),
        "Failed": int(summary_counts.get("failed", 0) or 0),
    }

    incidents: List[Dict[str, Any]] = []
    for rec in execution_log:
        status = _record_field(rec, "status", "")
        if status not in ("stale_fallback", "failed"):
            continue
        incidents.append({
            "patient_id": _record_field(rec, "patient_id", "?"),
            "status": status,
            "stage": _record_field(rec, "stage", "-"),
            "reason": _record_field(rec, "reason", "-"),
            "retry_attempted": _record_field(rec, "retry_attempted", False),
            "used_previous_cache": _record_field(rec, "used_previous_cache", False),
            "cache_age_days": _record_field(rec, "cache_age_days", None),
        })

    n_failed = counters["Failed"]
    n_stale = counters["Stale fallback"]
    if n_failed > 0:
        status = STATUS_ERROR
        summary = (
            f"STS Score: {n_failed} failure(s), {n_stale} stale fallback(s) "
            f"(total {total})."
        )
    elif n_stale > 0:
        status = STATUS_WARNING
        summary = (
            f"STS Score: {n_stale} stale fallback(s) out of {total} patients."
        )
    else:
        status = STATUS_OK
        summary = (
            f"STS Score OK: {counters['Fresh fetches']} fresh, "
            f"{counters['Cache hits']} cached, "
            f"{counters['Refreshed (expired -> refetched)']} refreshed (total {total})."
        )

    return RunStep(
        name="STS Score execution",
        status=status,
        summary=summary,
        counters=counters,
        details=[],
        incidents=incidents,
    )


def _record_field(rec: Any, key: str, default: Any) -> Any:
    if isinstance(rec, dict):
        return rec.get(key, default)
    return getattr(rec, key, default)


# --------------------------------------------------------------------------- #
# Streamlit renderers
# --------------------------------------------------------------------------- #


_STATUS_ICON = {STATUS_OK: "[OK]", STATUS_WARNING: "[!]", STATUS_ERROR: "[ERR]"}


def render_run_report(report: RunReport, *, tr=None, title: Optional[str] = None) -> None:
    """Render a ``RunReport`` inside the Streamlit page.

    Parameters
    ----------
    report : RunReport
    tr : callable, optional
        Bilingual helper ``tr(en, pt)``.  When absent, English strings are
        used.
    title : str, optional
        Section title override.
    """
    import streamlit as st  # local import keeps module importable without streamlit
    import pandas as pd

    def _t(en: str, pt: Optional[str] = None) -> str:
        if tr is None:
            return en
        try:
            return tr(en, pt if pt is not None else en)
        except Exception:
            return en

    if not report.steps:
        return

    heading = title or _t("Run observability report", "Relatório de observabilidade da execução")
    overall = report.overall_status()
    overall_label = {
        STATUS_OK: _t("all steps OK", "todas as etapas OK"),
        STATUS_WARNING: _t("with warnings", "com avisos"),
        STATUS_ERROR: _t("with errors", "com erros"),
    }[overall]
    st.markdown(f"**{heading}** — {overall_label}")

    for step in report.steps:
        icon = _STATUS_ICON.get(step.status, "[?]")
        header = f"{icon} {step.name} — {step.summary}"
        expanded = step.status != STATUS_OK
        with st.expander(header, expanded=expanded):
            if step.counters:
                rows = [
                    {_t("Metric", "Métrica"): k, _t("Value", "Valor"): v}
                    for k, v in step.counters.items()
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            if step.details:
                st.markdown("\n".join(f"- {d}" for d in step.details))
            if step.incidents:
                st.markdown(f"**{_t('Incidents', 'Incidentes')}**")
                st.dataframe(pd.DataFrame(step.incidents), hide_index=True, width="stretch")


def render_sts_score_incidents(
    execution_log: Optional[Sequence[Any]],
    *,
    tr=None,
    header: Optional[str] = None,
) -> None:
    """Render per-patient STS Score incidents (stale_fallback / failed).

    Used by the temporal validation path to mirror the batch flow.  Emits
    nothing when there are no incidents.
    """
    import streamlit as st
    import pandas as pd

    def _t(en: str, pt: Optional[str] = None) -> str:
        if tr is None:
            return en
        try:
            return tr(en, pt if pt is not None else en)
        except Exception:
            return en

    step = build_step_sts_score(execution_log)
    if step is None or not step.incidents:
        return

    label = header or _t(
        "STS Score per-patient incidents",
        "Incidentes por paciente do STS Score",
    )
    with st.expander(f"{label} ({len(step.incidents)})", expanded=False):
        st.dataframe(pd.DataFrame(step.incidents), hide_index=True, width="stretch")
