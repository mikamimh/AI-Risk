"""Temporal validation helpers for AI Risk.

Extracted from model_metadata.py to isolate temporal-cohort logic from core
model metadata functions.

Provides:
- extract_year_quarter_range  — parse year-quarter range from a DataFrame
- check_temporal_overlap      — compare training vs. validation date ranges
- format_locked_model_for_display — tabular locked-model summary (Temporal Validation tab)
- build_temporal_validation_summary — Markdown report for temporal validation results
- classify_sts_availability   — explicit STS coverage rule for temporal cohorts
- build_sts_availability_summary — shared UI/report text for partial STS coverage

Internal helpers (year-quarter timestamp conversion) are kept private to this
module; they are not part of the public API.

Backward compatibility
----------------------
``_extract_year_quarter_range`` is provided as a module-level alias for
``extract_year_quarter_range`` so that any existing ``from temporal_validation
import _extract_year_quarter_range`` call continues to work.  The canonical
public name is ``extract_year_quarter_range`` (no leading underscore).
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Internal year-quarter constants
# ---------------------------------------------------------------------------

_QUARTER_TO_MONTH = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
_QUARTER_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

# ---------------------------------------------------------------------------
# STS availability status
# ---------------------------------------------------------------------------

STS_AVAILABILITY_COMPLETE: str = "complete"
STS_AVAILABILITY_PARTIAL: str = "partial"
STS_AVAILABILITY_UNAVAILABLE: str = "unavailable"

STS_AVAILABILITY_STATES: tuple = (
    STS_AVAILABILITY_COMPLETE,
    STS_AVAILABILITY_PARTIAL,
    STS_AVAILABILITY_UNAVAILABLE,
)


def classify_sts_availability(n_eligible: int, n_score: int) -> str:
    """Classify STS final-score availability for a temporal cohort.

    Rule:
      - complete    = final STS score present for all eligible rows
      - partial     = final STS score present for some but not all eligible rows
      - unavailable = final STS score present for none of the eligible rows
    """
    if n_eligible <= 0 or n_score <= 0:
        return STS_AVAILABILITY_UNAVAILABLE
    if n_score >= n_eligible:
        return STS_AVAILABILITY_COMPLETE
    return STS_AVAILABILITY_PARTIAL


def build_sts_availability_summary(
    n_eligible: int,
    n_score: int,
    language: str = "English",
) -> dict:
    """Return shared STS availability text for UI and report layers.

    Intended for cohorts where STS was enabled and at least one row was eligible.
    Callers that want separate handling for "STS disabled" or "no eligible rows"
    should gate those cases before calling this helper.
    """
    def _tr(en: str, pt: str) -> str:
        return en if language == "English" else pt

    status = classify_sts_availability(n_eligible, n_score)
    coverage_pct = (n_score / n_eligible * 100.0) if n_eligible > 0 else 0.0
    coverage_text = _tr(
        f"Coverage: {coverage_pct:.1f}% of eligible rows ({n_score}/{n_eligible}).",
        f"Cobertura: {coverage_pct:.1f}% das linhas elegíveis ({n_score}/{n_eligible}).",
    )

    comparator_scope_note = ""  # populated for PARTIAL status only

    if status == STS_AVAILABILITY_COMPLETE:
        status_label = _tr("COMPLETE", "COMPLETA")
        banner_text = ""
        execution_details_text = _tr(
            f"STS availability: complete ({n_score}/{n_eligible} eligible).",
            f"Disponibilidade do STS: completa ({n_score}/{n_eligible} elegíveis).",
        )
        report_note = _tr(
            f"STS availability: COMPLETE. Usable final STS scores were present for all {n_eligible} eligible rows.",
            f"Disponibilidade do STS: COMPLETA. Escores finais utilizáveis estavam presentes para todas as {n_eligible} linhas elegíveis.",
        )
        subset_note = ""
        suppressed_note = ""
        risk_category_note = ""
        score_label = "STS Score"
    elif status == STS_AVAILABILITY_PARTIAL:
        status_label = _tr("PARTIAL", "PARCIAL")
        banner_text = _tr(
            f"STS availability: PARTIAL. {n_score} of {n_eligible} eligible rows produced a usable final STS score. "
            "STS summaries below reflect only this subset and should not be interpreted as a complete cohort-level comparator.",
            f"Disponibilidade do STS: PARCIAL. {n_score} de {n_eligible} linhas elegíveis produziram um escore STS final utilizável. "
            "Os resumos STS abaixo refletem apenas esse subconjunto e não devem ser interpretados como um comparador em nível de coorte completa.",
        )
        execution_details_text = _tr(
            f"STS availability: partial ({n_score}/{n_eligible} eligible).",
            f"Disponibilidade do STS: parcial ({n_score}/{n_eligible} elegíveis).",
        )
        report_note = _tr(
            f"STS availability: PARTIAL. STS-based summaries and comparisons reflect only the subset with usable final STS scores "
            f"({n_score} of {n_eligible} eligible rows; {coverage_pct:.1f}% coverage) and should not be interpreted as full-cohort STS results.",
            f"Disponibilidade do STS: PARCIAL. Os resumos e comparações baseados em STS refletem apenas o subconjunto com escore STS final utilizável "
            f"({n_score} de {n_eligible} linhas elegíveis; cobertura de {coverage_pct:.1f}%) e não devem ser interpretados como resultados STS da coorte completa.",
        )
        subset_note = _tr(
            f"STS entries shown here are subset-only and reflect the {n_score}/{n_eligible} eligible rows with usable final STS scores.",
            f"As entradas de STS mostradas aqui são apenas do subconjunto e refletem as {n_score}/{n_eligible} linhas elegíveis com escore STS final utilizável.",
        )
        suppressed_note = ""
        risk_category_note = _tr(
            f"STS rows below reflect only the subset with usable final STS output ({n_score}/{n_eligible} eligible).",
            f"As linhas de STS abaixo refletem apenas o subconjunto com saída final STS utilizável ({n_score}/{n_eligible} elegíveis).",
        )
        comparator_scope_note = _tr(
            f"Comparability note: STS results are subset-only ({n_score}/{n_eligible} eligible rows"
            f" with usable final STS scores; {coverage_pct:.1f}% coverage). "
            "AI Risk model and EuroSCORE II metrics are computed at full cohort level and"
            " are not affected by STS subset coverage.",
            f"Nota de comparabilidade: Os resultados do STS são apenas do subconjunto"
            f" ({n_score}/{n_eligible} linhas elegíveis com escore STS final utilizável;"
            f" cobertura de {coverage_pct:.1f}%). "
            "As métricas do modelo de AI Risk e do EuroSCORE II são calculadas no nível da coorte"
            " completa e não são afetadas pela cobertura do subconjunto STS.",
        )
        score_label = _tr(
            f"STS Score (subset: {n_score}/{n_eligible} eligible)",
            f"STS Score (subconjunto: {n_score}/{n_eligible} elegíveis)",
        )
    else:
        status_label = _tr("UNAVAILABLE", "INDISPONÍVEL")
        banner_text = _tr(
            "STS availability: UNAVAILABLE. No eligible rows produced a usable final STS score.",
            "Disponibilidade do STS: INDISPONÍVEL. Nenhuma linha elegível produziu um escore STS final utilizável.",
        )
        execution_details_text = _tr(
            f"STS availability: unavailable ({n_score}/{n_eligible} eligible).",
            f"Disponibilidade do STS: indisponível ({n_score}/{n_eligible} elegíveis).",
        )
        report_note = _tr(
            f"STS availability: UNAVAILABLE. No eligible rows produced a usable final STS score "
            f"({n_score} of {n_eligible} eligible rows; {coverage_pct:.1f}% coverage). "
            "STS-specific tables and plots are therefore omitted from cohort-level interpretation.",
            f"Disponibilidade do STS: INDISPONÍVEL. Nenhuma linha elegível produziu um escore STS final utilizável "
            f"({n_score} de {n_eligible} linhas elegíveis; cobertura de {coverage_pct:.1f}%). "
            "As tabelas e gráficos específicos de STS são, portanto, omitidos da interpretação em nível de coorte.",
        )
        subset_note = ""
        suppressed_note = _tr(
            "STS-specific distributions and plots are not shown because no eligible rows produced a usable final STS score.",
            "As distribuições e gráficos específicos de STS não são mostrados porque nenhuma linha elegível produziu um escore STS final utilizável.",
        )
        risk_category_note = _tr(
            f"STS was unavailable for eligible rows ({n_score}/{n_eligible}) and is omitted from this section.",
            f"O STS ficou indisponível para as linhas elegíveis ({n_score}/{n_eligible}) e foi omitido desta seção.",
        )
        score_label = _tr(
            f"STS Score (available for {n_score}/{n_eligible} eligible)",
            f"STS Score (disponível para {n_score}/{n_eligible} elegíveis)",
        )

    return {
        "status": status,
        "status_label": status_label,
        "n_eligible": int(n_eligible),
        "n_score": int(n_score),
        "coverage_pct": coverage_pct,
        "coverage_text": coverage_text,
        "banner_text": banner_text,
        "execution_details_text": execution_details_text,
        "report_note": report_note,
        "subset_note": subset_note,
        "suppressed_note": suppressed_note,
        "risk_category_note": risk_category_note,
        "comparator_scope_note": comparator_scope_note,
        "score_label": score_label,
    }

# ---------------------------------------------------------------------------
# STS accounting table
# ---------------------------------------------------------------------------

def build_sts_accounting_table(
    n_total: int,
    n_not_supported: int,
    n_uncertain: int,
    n_supported: int,
    n_final_usable: int,
    language: str = "English",
) -> pd.DataFrame:
    """Build an explicit STS pipeline accounting table.

    Parameters
    ----------
    n_total:          Total cohort size.
    n_not_supported:  Patients excluded (not in STS ACSD scope).
    n_uncertain:      Patients with uncertain eligibility (missing/ambiguous surgery).
    n_supported:      Patients classified as STS-eligible (supported).
    n_final_usable:   Patients with a final usable STS score in the output.
    language:         ``"English"`` or any other value for Portuguese.

    Returns
    -------
    pd.DataFrame with columns ``Category`` and ``Count``
    (plus ``Pct_of_Supported`` for the final-usable coverage row).
    """
    _pt = language not in (None, "", "English")

    def _tr(en: str, pt: str) -> str:
        return pt if _pt else en

    n_supported_no_score = max(0, n_supported - n_final_usable)
    coverage_pct = (n_final_usable / n_supported * 100.0) if n_supported > 0 else 0.0

    rows = [
        (_tr("Total cohort",                         "Total da coorte"),              n_total,             ""),
        (_tr("Not supported (out of STS scope)",      "Não suportado (fora do escopo STS)"), n_not_supported, ""),
        (_tr("Uncertain (ambiguous surgery type)",    "Incerto (tipo de cirurgia ambíguo)"), n_uncertain,    ""),
        (_tr("Supported (STS-eligible)",              "Suportado (STS-elegível)"),     n_supported,         ""),
        (_tr("Final usable STS score",                "Escore STS final utilizável"),  n_final_usable,      f"{coverage_pct:.1f}%"),
        (_tr("Supported but no final STS score",      "Suportado sem escore STS final"), n_supported_no_score, ""),
    ]

    col_cat   = _tr("Category", "Categoria")
    col_count = _tr("Count", "Contagem")
    col_cov   = _tr("Coverage (% of supported)", "Cobertura (% dos suportados)")

    df = pd.DataFrame(rows, columns=[col_cat, col_count, col_cov])
    return df


# ---------------------------------------------------------------------------
# Surrogate / de-identified timeline detection
# ---------------------------------------------------------------------------

# Years above this threshold are treated as artificially shifted for patient
# de-identification and should NOT be interpreted as real clinical dates.
SURROGATE_YEAR_THRESHOLD: int = 2050


def is_surrogate_timeline(data: pd.DataFrame) -> bool:
    """Return True when the dataset appears to use a de-identified surrogate year.

    Datasets produced under temporal de-identification shift calendar years to
    an artificial range (e.g. 2111–2195) that is recognisably not real clinical
    time.  This helper detects that pattern so the UI can add an appropriate
    disclaimer without blocking execution.
    """
    if "surgery_year" not in data.columns:
        return False
    years = pd.to_numeric(data["surgery_year"], errors="coerce").dropna()
    return bool((not years.empty) and (years.max() > SURROGATE_YEAR_THRESHOLD))


def build_surrogate_timeline_note(language: str = "English") -> str:
    """Return a formatted UI note explaining a de-identified surrogate timeline.

    Should be shown whenever ``is_surrogate_timeline`` returns True so that
    readers do not misinterpret shifted years as real procedure dates.
    """
    if language == "English":
        return (
            "**De-identified surrogate timeline detected.** "
            "Calendar years shown (e.g. 2111\u20132195) were artificially shifted for patient "
            "privacy and do **not** represent real clinical procedure dates. "
            "Temporal ordering and overlap checks use only the relative sequence encoded in "
            "`surgery_year` + `surgery_quarter` — the absolute year values carry no clinical meaning."
        )
    return (
        "**Linha do tempo substituta desidentificada detectada.** "
        "Os anos exibidos (ex: 2111\u20132195) foram deslocados artificialmente para proteger a "
        "privacidade dos pacientes e **não** representam datas clínicas reais. "
        "A ordenação temporal e a verificação de sobreposição utilizam apenas a sequência relativa "
        "codificada em `surgery_year` + `surgery_quarter` — os valores absolutos dos anos não têm "
        "significado clínico."
    )


# ---------------------------------------------------------------------------
# Year-quarter timestamp helpers (private)
# ---------------------------------------------------------------------------

def _yq_to_timestamp(yq: str) -> Optional[pd.Timestamp]:
    """Convert ``'2024-Q2'`` or ``'2024'`` to a pd.Timestamp (start of period)."""
    if not yq or yq == "Unknown":
        return None
    try:
        if "-Q" in yq:
            parts = yq.split("-Q")
            year = int(parts[0])
            month = _QUARTER_TO_MONTH.get(f"Q{parts[1]}", 1)
            return pd.Timestamp(year=year, month=month, day=1)
        return pd.Timestamp(year=int(yq), month=1, day=1)
    except Exception:
        return None


def _yq_to_end_timestamp(yq: str) -> Optional[pd.Timestamp]:
    """Convert ``'2024-Q2'`` to end-of-quarter Timestamp."""
    if not yq or yq == "Unknown":
        return None
    try:
        if "-Q" in yq:
            parts = yq.split("-Q")
            year = int(parts[0])
            q = int(parts[1])
            # End of quarter: Q1→Mar31, Q2→Jun30, Q3→Sep30, Q4→Dec31
            end_months = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
            m, d = end_months.get(q, (12, 31))
            return pd.Timestamp(year=year, month=m, day=d)
        return pd.Timestamp(year=int(yq), month=12, day=31)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_year_quarter_range(data: pd.DataFrame) -> tuple:
    """Return (start, end) strings from surgery_year / surgery_quarter columns.

    Format: ``"2024-Q1"`` or ``"Unknown"`` when columns are absent.
    """
    if "surgery_year" not in data.columns:
        return ("Unknown", "Unknown")

    years = pd.to_numeric(data["surgery_year"], errors="coerce").dropna()
    if years.empty:
        return ("Unknown", "Unknown")

    if "surgery_quarter" in data.columns:
        temp = data[["surgery_year", "surgery_quarter"]].dropna()
        temp = temp.copy()
        temp["_y"] = pd.to_numeric(temp["surgery_year"], errors="coerce")
        temp["_q"] = temp["surgery_quarter"].map(_QUARTER_ORDER)
        temp = temp.dropna(subset=["_y", "_q"])
        if not temp.empty:
            temp["_sort"] = temp["_y"] * 10 + temp["_q"]
            earliest = temp.loc[temp["_sort"].idxmin()]
            latest = temp.loc[temp["_sort"].idxmax()]
            start = f"{int(earliest['_y'])}-{earliest['surgery_quarter']}"
            end = f"{int(latest['_y'])}-{latest['surgery_quarter']}"
            return (start, end)

    # Fallback: year only
    return (str(int(years.min())), str(int(years.max())))


# Backward-compat alias — preserves any caller using the private-style name.
_extract_year_quarter_range = extract_year_quarter_range


def check_temporal_overlap(
    training_start: str,
    training_end: str,
    validation_start: str,
    validation_end: str,
) -> dict:
    """Compare training and validation temporal ranges for overlap.

    Accepts year-quarter strings (``"2024-Q1"``) or plain years (``"2024"``).
    Returns dict with keys:

    overlap, status, severity, message_en, message_pt, surrogate_timeline.

    ``surrogate_timeline`` is True when one or both ranges contain years above
    ``SURROGATE_YEAR_THRESHOLD`` — the UI should show a de-identification
    disclaimer and not interpret absolute year values clinically.
    """
    # Detect de-identified surrogate years in either range.
    def _max_year(yq: str) -> Optional[int]:
        if not yq or yq == "Unknown":
            return None
        try:
            return int(yq.split("-")[0])
        except Exception:
            return None

    _years = [_max_year(y) for y in [training_start, training_end, validation_start, validation_end]]
    surrogate = any(y is not None and y > SURROGATE_YEAR_THRESHOLD for y in _years)

    result = {
        "training_range": (training_start, training_end),
        "validation_range": (validation_start, validation_end),
        "overlap": False,
        "status": "unknown",
        "severity": "info",
        "message_en": "",
        "message_pt": "",
        "surrogate_timeline": surrogate,
    }
    t_start = _yq_to_timestamp(training_start)
    t_end = _yq_to_end_timestamp(training_end)
    v_start = _yq_to_timestamp(validation_start)
    v_end = _yq_to_end_timestamp(validation_end)

    if t_start is None or t_end is None or v_start is None or v_end is None:
        result["status"] = "unknown"
        result["severity"] = "warning"
        result["message_en"] = "Could not parse temporal ranges — overlap check skipped."
        result["message_pt"] = "Não foi possível interpretar os períodos — verificação de sobreposição ignorada."
        return result

    # Surrogate-aware labels: clarify that absolute years are de-identified.
    _surr_note_en = " (de-identified surrogate years — not real clinical dates)" if surrogate else ""
    _surr_note_pt = " (anos substitutos desidentificados — não são datas clínicas reais)" if surrogate else ""

    if v_end < t_start:
        result["status"] = "validation_before_training"
        result["severity"] = "error"
        result["message_en"] = (
            f"The validation cohort surrogate range ({validation_start} \u2014 {validation_end}){_surr_note_en} "
            f"is entirely BEFORE the training cohort surrogate range ({training_start} \u2014 {training_end}). "
            "This is NOT temporal validation \u2014 it is retrograde validation and "
            "severely compromises methodological validity."
        )
        result["message_pt"] = (
            f"O intervalo substituto da coorte de validação ({validation_start} \u2014 {validation_end}){_surr_note_pt} "
            f"é inteiramente ANTERIOR ao intervalo substituto da coorte de treinamento ({training_start} \u2014 {training_end}). "
            "Isso NÃO é validação temporal \u2014 é validação retrógrada e "
            "compromete gravemente a validade metodológica."
        )
    elif v_start > t_end:
        result["status"] = "no_overlap"
        result["severity"] = "success"
        result["message_en"] = (
            f"No temporal overlap detected. "
            f"Validation cohort surrogate range ({validation_start} \u2014 {validation_end}) "
            f"is strictly after training cohort surrogate range ({training_start} \u2014 {training_end})"
            f"{_surr_note_en} \u2014 ideal for temporal validation."
        )
        result["message_pt"] = (
            f"Sem sobreposição temporal detectada. "
            f"Intervalo substituto da coorte de validação ({validation_start} \u2014 {validation_end}) "
            f"é estritamente posterior ao da coorte de treinamento ({training_start} \u2014 {training_end})"
            f"{_surr_note_pt} \u2014 ideal para validação temporal."
        )
    else:
        result["overlap"] = True
        result["status"] = "overlap"
        result["severity"] = "warning"
        result["message_en"] = (
            f"Temporal overlap detected between training ({training_start} \u2014 {training_end}) "
            f"and validation ({validation_start} \u2014 {validation_end}){_surr_note_en}. "
            "Patients in the overlapping period may have been used for training, "
            "which weakens the validity of temporal validation."
        )
        result["message_pt"] = (
            f"Sobreposição temporal detectada entre treinamento ({training_start} \u2014 {training_end}) "
            f"e validação ({validation_start} \u2014 {validation_end}){_surr_note_pt}. "
            "Pacientes no período sobreposto podem ter sido usados no treinamento, "
            "o que enfraquece a validade da validação temporal."
        )

    return result


# ---------------------------------------------------------------------------
# Chronological state — explicit four-way classification
# ---------------------------------------------------------------------------

# Canonical state names returned by ``check_temporal_overlap`` under the
# ``status`` key.  Exposing them as constants keeps call-sites type-safe and
# lets UI code dispatch on a closed set rather than on free-form strings.
CHRONO_STATE_NO_OVERLAP: str = "no_overlap"
CHRONO_STATE_OVERLAP: str = "overlap"
CHRONO_STATE_RETROGRADE: str = "validation_before_training"
CHRONO_STATE_UNKNOWN: str = "unknown"

CHRONO_STATES: tuple = (
    CHRONO_STATE_NO_OVERLAP,
    CHRONO_STATE_OVERLAP,
    CHRONO_STATE_RETROGRADE,
    CHRONO_STATE_UNKNOWN,
)


def chronological_state_label(status: str, language: str = "English") -> str:
    """Human-readable label for a chronological state.

    Given one of the canonical status strings produced by
    ``check_temporal_overlap`` (``no_overlap``, ``overlap``,
    ``validation_before_training``, ``unknown``), return a short label suited
    for UI display.  Unknown inputs fall back to the ``unknown`` label so the
    UI never silently hides an unrecognised state.
    """
    en = {
        CHRONO_STATE_NO_OVERLAP: "No overlap",
        CHRONO_STATE_OVERLAP: "Partial overlap",
        CHRONO_STATE_RETROGRADE: "Retrograde validation",
        CHRONO_STATE_UNKNOWN: "Unknown chronology",
    }
    pt = {
        CHRONO_STATE_NO_OVERLAP: "Sem sobreposição",
        CHRONO_STATE_OVERLAP: "Sobreposição parcial",
        CHRONO_STATE_RETROGRADE: "Validação retrógrada",
        CHRONO_STATE_UNKNOWN: "Cronologia desconhecida",
    }
    table = en if language == "English" else pt
    return table.get(status, table[CHRONO_STATE_UNKNOWN])


def format_locked_model_for_display(
    metadata: dict,
    language: str = "English",
) -> pd.DataFrame:
    """Format locked model metadata for the Temporal Validation tab."""
    def _tr(en, pt):
        return en if language == "English" else pt

    def _fmt_ts(ts):
        if not ts or ts == "Unknown":
            return "Unknown"
        return ts[:19].replace("T", " ")

    t_start = metadata.get("training_start_date", "Unknown")
    t_end = metadata.get("training_end_date", "Unknown")
    date_range = f"{t_start} — {t_end}" if t_start != "Unknown" and t_end != "Unknown" else "Unknown"

    calib = metadata.get("calibration", {})
    calib_method = calib.get("method", "N/A") if isinstance(calib, dict) else str(calib)
    locked_thr = metadata.get("locked_threshold", 0.08)

    rows = [
        (_tr("Model version", "Versão do modelo"), metadata.get("model_version", "N/A")),
        (_tr("Bundle saved at", "Bundle salvo em"), _fmt_ts(metadata.get("bundle_saved_at"))),
        (_tr("Training data source", "Fonte de dados do treino"), metadata.get("training_source_file", "N/A")),
        (_tr("Best model", "Melhor modelo"), metadata.get("best_model", "N/A")),
        (_tr("Patients in training", "Pacientes no treinamento"), str(metadata.get("n_patients", "N/A"))),
        (_tr("Events in training", "Eventos no treinamento"), str(metadata.get("n_events", "N/A"))),
        (_tr("Event rate in training", "Taxa de eventos no treinamento"), f"{metadata.get('event_rate', 0):.1%}"),
        (_tr("Number of features", "Número de variáveis"), str(metadata.get("n_features", "N/A"))),
        (_tr("CV strategy", "Estratégia de CV"), metadata.get("cv_strategy", "N/A")),
        (_tr("CV splits", "Folds de CV"), str(metadata.get("cv_splits", "N/A"))),
        (_tr("Calibration method", "Método de calibração"), calib_method),
        (_tr("Training date range", "Período do treinamento"), date_range),
        (_tr("Locked clinical threshold", "Limiar clínico bloqueado"), f"{locked_thr:.0%}"),
        (_tr("Lock status", "Status de bloqueio"), _tr("Locked — no retraining allowed", "Bloqueado — sem retreinamento")),
    ]
    value_col = _tr("Value", "Valor")
    out = pd.DataFrame(rows, columns=[_tr("Property", "Propriedade"), value_col])
    # Force the Value column to string dtype — rows mix Python types (ints,
    # strings like "RandomForest", formatted percentages like "14.98%"),
    # and pyarrow's object-column type inference inside Streamlit picks
    # int64 from early numeric rows and then raises ArrowInvalid on the
    # string rows.  The column is rendered as plain text anyway.
    out[value_col] = out[value_col].astype(str)
    return out


def build_temporal_validation_summary(
    cohort_summary: dict,
    performance_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    risk_category_df: pd.DataFrame,
    metadata: dict,
    threshold: float,
    language: str = "English",
    sts_availability: Optional[dict] = None,
    normalization_report=None,
    sts_accounting: Optional[dict] = None,
    common_cohort_perf: Optional[pd.DataFrame] = None,
    n_common: int = 0,
) -> str:
    """Build Markdown summary for temporal validation results.

    The report uses PDF-friendly table layouts throughout:

    * ``performance_df`` (16 raw columns) is split into two narrow sub-tables
      so every column fits without overlap in the rendered PDF:

        - Table A — Discrimination: Score, n, AUC, AUC lo/hi, AUPRC lo/hi, Brier
        - Table B — Calibration & Classification:
          Score, Cal.Int., Cal.Slp., HL p, Sens., Spec., PPV, NPV

    * ``pairwise_df`` drops the ``DeLong_skip_reason`` column from the table
      and renders it as a footnote caption below instead.

    * All float values are formatted to controlled decimal precision
      (3 d.p. for most metrics, 4 d.p. for p-values) to avoid
      machine-precision strings like ``0.7714285714285715``.

    Raw CSV/XLSX exports are generated directly from the unmodified DataFrames
    and are not affected by these presentation changes.
    """
    def _tr(en, pt):
        return en if language == "English" else pt

    # ── Formatting helpers ────────────────────────────────────────────────

    def _fv(val, decimals: int = 3) -> str:
        """Format a single value for report display.

        * floats → fixed decimal notation
        * NaN / None → "N/A"
        * everything else → str()
        """
        try:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "N/A"
            if isinstance(val, float):
                return f"{val:.{decimals}f}"
        except (TypeError, ValueError):
            pass
        return str(val)

    def _build_sub_table(
        src: pd.DataFrame,
        col_map: "list[tuple[str, str, int]]",
    ) -> "list[str]":
        """Return Markdown table lines for a subset of *src* columns.

        *col_map* is a list of ``(src_col, display_label, decimals)`` triples.
        Columns absent from *src* are silently skipped so the function is
        robust against DataFrames with varying column sets.
        """
        # Keep only columns that actually exist in src
        present = [(sc, lbl, dec) for sc, lbl, dec in col_map if sc in src.columns]
        if not present:
            return []
        labels = [lbl for _, lbl, _ in present]
        hdr = " | ".join(labels)
        sep = "|" + "|".join("--:" if i > 0 else ":--" for i in range(len(present))) + "|"
        out = [f"| {hdr} |", sep]
        for _, row in src.iterrows():
            cells = " | ".join(_fv(row[sc], dec) for sc, _, dec in present)
            out.append(f"| {cells} |")
        return out

    # ── Column maps for performance sub-tables ────────────────────────────
    #
    # Each entry: (raw_column_name, display_label_for_report, decimal_places)
    #
    # Table A — Discrimination / global performance (9 columns)
    _PERF_A: "list[tuple[str,str,int]]" = [
        ("Score",          _tr("Score",    "Score"),    0),
        ("n",              "n",                         0),
        ("AUC",            "AUC",                       3),
        ("AUC_IC95_inf",   _tr("AUC lo",  "AUC inf"),  3),
        ("AUC_IC95_sup",   _tr("AUC hi",  "AUC sup"),  3),
        ("AUPRC",          "AUPRC",                     3),
        ("AUPRC_IC95_inf", _tr("AUPRC lo","AUPRC inf"), 3),
        ("AUPRC_IC95_sup", _tr("AUPRC hi","AUPRC sup"), 3),
        ("Brier",          "Brier",                     3),
    ]

    # Table B — Calibration & classification (8 columns)
    _PERF_B: "list[tuple[str,str,int]]" = [
        ("Score",                  _tr("Score",      "Score"),      0),
        ("Calibration_Intercept",  _tr("Cal.Int.",   "Int.Cal."),   3),
        ("Calibration_Slope",      _tr("Cal.Slp.",   "Inc.Cal."),   3),
        ("HL_p",                   "HL p",                          4),
        ("Sensitivity",            _tr("Sens.",      "Sens."),      3),
        ("Specificity",            _tr("Spec.",      "Espec."),     3),
        ("PPV",                    "PPV",                           3),
        ("NPV",                    "NPV",                           3),
    ]

    # Column map for pairwise table (DeLong_skip_reason excluded — shown as footnote)
    _PAIR: "list[tuple[str,str,int]]" = [
        ("Comparison",          _tr("Comparison",  "Comparação"), 0),
        ("n",                   "n",                               0),
        ("Delta_AUC",           _tr("dAUC",        "dAUC"),       3),
        ("Delta_AUC_IC95_inf",  _tr("dAUC lo",     "dAUC inf"),   3),
        ("Delta_AUC_IC95_sup",  _tr("dAUC hi",     "dAUC sup"),   3),
        ("Bootstrap_p",         _tr("Boot.p",      "p Boot."),    4),
        ("DeLong_p",            _tr("DeLong p",    "p DeLong"),   4),
        ("NRI",                 "NRI",                             3),
        ("IDI",                 "IDI",                             3),
    ]

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cs = cohort_summary
    sts_note = sts_availability or {}
    sts_status = sts_note.get("status")

    lines = [
        f"# {_tr('Temporal Validation Report', 'Relatório de Validação Temporal')}",
        "",
        f"**{_tr('Generated', 'Gerado em')}:** {now}",
        f"**{_tr('Model version', 'Versão do modelo')}:** {metadata.get('model_version', 'N/A')}",
        f"**{_tr('Locked threshold', 'Limiar bloqueado')}:** {threshold:.0%}",
        f"**{_tr('Training cohort', 'Coorte de treinamento')}:** n = {metadata.get('n_patients', 'N/A')}, "
        f"{_tr('events', 'eventos')} = {metadata.get('n_events', 'N/A')} "
        f"({metadata.get('event_rate', 0):.1%})",
        f"**{_tr('Validation cohort', 'Coorte de validação')}:** n = {cs.get('n_total', 0)}, "
        f"{_tr('events', 'eventos')} = {cs.get('n_events', 0)} "
        f"({cs.get('event_rate', 0):.1%})",
        "",
        f"**{_tr('Methodological note', 'Nota metodológica')}:** "
        + _tr(
            "This report applies a previously locked model to an independent temporal cohort. "
            "No retraining, recalibration, or threshold adjustment was performed.",
            "Este relatório aplica um modelo previamente congelado a uma coorte temporal independente. "
            "Não houve retreinamento, recalibração ou ajuste de limiar.",
        ),
        "",
    ]

    # ── Dataset normalization summary (external CSV/Parquet only) ─────────
    if normalization_report is not None:
        _nr = normalization_report
        lines.append(f"## {_tr('Dataset Normalization', 'Normalização do Dataset')}")
        lines.append("")
        _rm = getattr(_nr, "read_meta", None)
        if _rm is not None:
            lines.append(f"| {_tr('Property', 'Propriedade')} | {_tr('Value', 'Valor')} |")
            lines.append("|:--|:--|")
            _src = getattr(_nr, "source_name", None) or "N/A"
            lines.append(f"| {_tr('Source', 'Fonte')} | {_src} |")
            lines.append(f"| {_tr('Encoding', 'Codificação')} | {_rm.encoding_used} |")
            lines.append(f"| {_tr('Delimiter', 'Delimitador')} | {_rm.delimiter!r} |")
            lines.append(f"| {_tr('Rows loaded', 'Linhas carregadas')} | {_rm.rows_loaded} |")
            lines.append(
                f"| {_tr('Columns loaded', 'Colunas carregadas')} | {_rm.columns_loaded} |"
            )
            lines.append("")
        _u = getattr(_nr, "unit_summary", {})
        if _u.get("height_converted") or _u.get("weight_converted"):
            lines.append(
                f"**{_tr('Unit conversions applied', 'Conversões de unidade aplicadas')}:**"
            )
            if _u.get("height_converted"):
                lines.append(
                    f"- {_tr('Height', 'Altura')}: {_u['n_height_converted']}"
                    f" {_tr('values converted from inches to cm', 'valores convertidos de polegadas para cm')}"
                    f" ({_tr('original median', 'mediana original')}: {_u['height_original_median']} in)"
                )
            if _u.get("weight_converted"):
                lines.append(
                    f"- {_tr('Weight', 'Peso')}: {_u['n_weight_converted']}"
                    f" {_tr('values converted from lb to kg', 'valores convertidos de lb para kg')}"
                    f" ({_tr('original median', 'mediana original')}: {_u['weight_original_median']} lb)"
                )
            lines.append("")
            lines.append(
                f"> **{_tr('Note', 'Nota')}:** "
                + _tr(
                    "Anthropometric units were automatically detected and converted before"
                    " any model inference. Original values are not stored — verify source"
                    " data if unit detection is unexpected.",
                    "Unidades antropométricas foram detectadas e convertidas automaticamente"
                    " antes de qualquer inferência do modelo. Os valores originais não são"
                    " armazenados — verifique os dados de origem se a detecção de unidade"
                    " for inesperada.",
                )
            )
            lines.append("")
        _sc = getattr(_nr, "scope_summary", {})
        _sr = getattr(_nr, "sts_readiness_summary", {})
        if _sc or _sr:
            lines.append(
                f"**{_tr('STS scope and preflight readiness', 'Escopo e prontidão pré-voo STS')}:**"
            )
            if _sc.get("n_pediatric", 0) > 0:
                lines.append(
                    f"- {_sc['n_pediatric']}"
                    f" {_tr('pediatric patient(s) (age < 18) excluded from adult STS scope', 'paciente(s) pediátrico(s) (idade < 18) excluídos do escopo STS adulto')}"
                )
            if _sc.get("n_sts_scope_excluded", 0) > 0:
                lines.append(
                    f"- {_sc['n_sts_scope_excluded']}"
                    f" {_tr('surgery(ies) outside STS ACSD scope', 'cirurgia(s) fora do escopo STS ACSD')}"
                    f" ({_tr('dissection / aneurysm / Bentall / etc.', 'dissecção / aneurisma / Bentall / etc.')})"
                )
            if _sr:
                lines.append(
                    f"- {_tr('STS-ready rows after normalization', 'Linhas STS-prontas após normalização')}:"
                    f" {_sr.get('n_ready', 0)}/{_sr.get('n_total', 0)}"
                    f" ({_sr.get('n_ready_pct', 0):.1f}%)"
                )
            lines.append("")
        _nw = getattr(_nr, "warnings", [])
        if _nw:
            lines.append(
                f"**{_tr('Normalization warnings', 'Avisos de normalização')}:**"
            )
            for _w in _nw:
                lines.append(f"- {_w}")
            lines.append("")

    # ── Cohort summary table ──────────────────────────────────────────────
    if sts_status in STS_AVAILABILITY_STATES:
        lines.append(
            f"**{_tr('STS availability note', 'Nota de disponibilidade do STS')}:** "
            f"{sts_note.get('report_note', '')}"
        )
        lines.append("")
        if sts_note.get("comparator_scope_note"):
            lines.append(f"> {sts_note['comparator_scope_note']}")
            lines.append("")

    lines.append(f"## {_tr('Cohort Summary', 'Resumo da Coorte')}")
    lines.append("")
    lines.append(f"| {_tr('Property', 'Propriedade')} | {_tr('Value', 'Valor')} |")
    lines.append("|:--|:--|")
    lines.append(f"| {_tr('Total patients', 'Total de pacientes')} | {cs.get('n_total', 0)} |")
    lines.append(f"| {_tr('Events (30-day mortality)', 'Eventos (mortalidade 30 dias)')} | {cs.get('n_events', 0)} |")
    lines.append(f"| {_tr('Event rate', 'Taxa de eventos')} | {cs.get('event_rate', 0):.1%} |")
    lines.append(f"| {_tr('Date range', 'Período')} | {cs.get('date_range', 'Unknown')} |")
    for level_key, level_label_en, level_label_pt in [
        ("n_complete", "Complete data",       "Dados completos"),
        ("n_adequate", "Adequate",             "Adequados"),
        ("n_partial",  "Partially imputed",    "Parcialmente imputados"),
        ("n_low",      "Heavily imputed",      "Muito imputados"),
    ]:
        n = cs.get(level_key, 0)
        pct = n / cs["n_total"] * 100 if cs.get("n_total", 0) > 0 else 0
        lines.append(f"| {_tr(level_label_en, level_label_pt)} | {n} ({pct:.1f}%) |")
    lines.append("")

    # ── STS pipeline accounting table ─────────────────────────────────────
    if sts_accounting is not None and sts_accounting.get("n_supported", 0) > 0:
        lines.append(f"## {_tr('STS Pipeline Accounting', 'Accounting do Pipeline STS')}")
        lines.append("")
        _acct = sts_accounting
        _acct_n_sup    = int(_acct.get("n_supported", 0))
        _acct_n_usable = int(_acct.get("n_final_usable", 0))
        _acct_cov      = (_acct_n_usable / _acct_n_sup * 100.0) if _acct_n_sup > 0 else 0.0
        _acct_no_score = max(0, _acct_n_sup - _acct_n_usable)
        lines.append(f"| {_tr('Category', 'Categoria')} | {_tr('Count', 'Contagem')} | {_tr('Coverage', 'Cobertura')} |")
        lines.append("|:--|--:|:--|")
        lines.append(f"| {_tr('Total cohort', 'Total da coorte')} | {_acct.get('n_total', 0)} | |")
        lines.append(f"| {_tr('Not supported (out of STS scope)', 'Não suportado (fora do escopo STS)')} | {_acct.get('n_not_supported', 0)} | |")
        lines.append(f"| {_tr('Uncertain (ambiguous surgery type)', 'Incerto (tipo de cirurgia ambíguo)')} | {_acct.get('n_uncertain', 0)} | |")
        lines.append(f"| {_tr('Supported (STS-eligible)', 'Suportado (STS-elegível)')} | {_acct_n_sup} | |")
        lines.append(f"| {_tr('Final usable STS score', 'Escore STS final utilizável')} | {_acct_n_usable} | {_acct_cov:.1f}% {_tr('of supported', 'dos suportados')} |")
        lines.append(f"| {_tr('Supported but no final STS score', 'Suportado sem escore STS final')} | {_acct_no_score} | |")
        lines.append("")

    # ── Performance — split into two narrow sub-tables ────────────────────
    if not performance_df.empty:
        lines.append(f"## {_tr('Discrimination and Calibration', 'Discriminação e Calibração')}")
        lines.append("")
        if sts_status == STS_AVAILABILITY_PARTIAL and sts_note.get("subset_note"):
            lines.append(f"*{sts_note['subset_note']}*")
            lines.append("")
        elif sts_status == STS_AVAILABILITY_UNAVAILABLE and sts_note.get("suppressed_note"):
            lines.append(f"*{sts_note['suppressed_note']}*")
            lines.append("")

        # Sub-table A: discrimination / global performance
        tbl_a = _build_sub_table(performance_df, _PERF_A)
        if tbl_a:
            lines.append(f"### {_tr('Discrimination', 'Discriminação')}")
            lines.append("")
            lines.extend(tbl_a)
            lines.append("")

        # Sub-table B: calibration & classification
        tbl_b = _build_sub_table(performance_df, _PERF_B)
        if tbl_b:
            lines.append(f"### {_tr('Calibration and Classification', 'Calibração e Classificação')}")
            lines.append("")
            lines.extend(tbl_b)
            lines.append("")

    # ── Pairwise comparison ───────────────────────────────────────────────
    if not pairwise_df.empty:
        lines.append(f"## {_tr('Pairwise Comparison', 'Comparação Pareada')}")
        lines.append("")

        tbl_pair = _build_sub_table(pairwise_df, _PAIR)
        if tbl_pair:
            lines.extend(tbl_pair)
            lines.append("")

        # DeLong skip-reason footnotes (rendered below table, not as a column)
        if "DeLong_skip_reason" in pairwise_df.columns:
            notes = [
                (str(row.get("Comparison", "?")), str(row["DeLong_skip_reason"]))
                for _, row in pairwise_df.iterrows()
                if row["DeLong_skip_reason"] and str(row["DeLong_skip_reason"]) not in ("", "None", "nan")
            ]
            if notes:
                lines.append(
                    f"*{_tr('DeLong note', 'Nota DeLong')}: "
                    + "; ".join(f"{comp}: {reason}" for comp, reason in notes)
                    + "*"
                )
                lines.append("")

    # ── Risk category distribution ────────────────────────────────────────
    if not risk_category_df.empty:
        lines.append(f"## {_tr('Risk Category Distribution', 'Distribuição por Classe de Risco')}")
        lines.append("")
        if sts_status in (STS_AVAILABILITY_PARTIAL, STS_AVAILABILITY_UNAVAILABLE) and sts_note.get("risk_category_note"):
            lines.append(f"*{sts_note['risk_category_note']}*")
            lines.append("")
        # Risk category table is already narrow (≤5 columns); render as-is
        # but format any float columns to avoid machine-precision noise.
        hdr = " | ".join(str(c) for c in risk_category_df.columns)
        lines.append(f"| {hdr} |")
        lines.append("|" + "|".join(":--" for _ in risk_category_df.columns) + "|")
        for _, row in risk_category_df.iterrows():
            cells = []
            for v in row.values:
                if isinstance(v, float) and not pd.isna(v):
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(str(v) if not (isinstance(v, float) and pd.isna(v)) else "N/A")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    if (
        sts_status == STS_AVAILABILITY_UNAVAILABLE
        and sts_note.get("risk_category_note")
        and risk_category_df.empty
    ):
        lines.append(f"## {_tr('Risk Category Distribution', 'DistribuiÃ§Ã£o por Classe de Risco')}")
        lines.append("")
        lines.append(f"*{sts_note['risk_category_note']}*")
        lines.append("")

    # ── Common cohort comparison ───────────────────────────────────────────
    if common_cohort_perf is not None and not common_cohort_perf.empty and n_common > 0:
        lines.append(f"## {_tr('Common STS-Available Cohort Comparison', 'Comparação da Coorte Comum com STS Disponível')}")
        lines.append("")
        lines.append(_tr(
            f"Metrics below use only the {n_common} rows where **all three** models "
            "(AI Risk, EuroSCORE II, and STS Score) have non-missing predictions. "
            "This common cohort is a strict STS-available subset and should not be used "
            "to replace the full-cohort primary results.",
            f"As métricas abaixo usam apenas as {n_common} linhas em que os **três** modelos "
            "(AI Risk, EuroSCORE II e STS Score) têm predições não ausentes. "
            "Esta coorte comum é um subconjunto restrito com STS disponível e não deve substituir "
            "os resultados primários da coorte completa.",
        ))
        lines.append("")
        tbl_cc_a = _build_sub_table(common_cohort_perf, _PERF_A)
        if tbl_cc_a:
            lines.append(f"### {_tr('Discrimination (Common Cohort)', 'Discriminação (Coorte Comum)')}")
            lines.append("")
            lines.extend(tbl_cc_a)
            lines.append("")
        tbl_cc_b = _build_sub_table(common_cohort_perf, _PERF_B)
        if tbl_cc_b:
            lines.append(f"### {_tr('Calibration and Classification (Common Cohort)', 'Calibração e Classificação (Coorte Comum)')}")
            lines.append("")
            lines.extend(tbl_cc_b)
            lines.append("")

    lines.append("---")
    lines.append(
        f"*{_tr('Generated by AI Risk — Temporal Validation Module', 'Gerado pelo AI Risk — Módulo de Validação Temporal')}*"
    )

    report = "\n".join(lines)
    report = report.replace("DistribuiÃ§Ã£o por Classe de Risco", "Distribuição por Classe de Risco")
    report = report.replace("DistribuiÃƒÂ§ÃƒÂ£o por Classe de Risco", "Distribuição por Classe de Risco")
    return report


# ---------------------------------------------------------------------------
# Exploratory helpers — post-hoc recalibration and threshold analysis
# ---------------------------------------------------------------------------

_RECAL_METHOD_LABELS: Dict[str, Dict[str, str]] = {
    "original":       {"English": "Original (no recalibration)", "Portuguese": "Original (sem recalibração)"},
    "intercept_only": {"English": "Intercept-only (slope = 1)",  "Portuguese": "Apenas intercepto (slope = 1)"},
    "logistic":       {"English": "Intercept + Slope (logistic)", "Portuguese": "Intercepto + Slope (logístico)"},
    "isotonic":       {"English": "Isotonic regression",          "Portuguese": "Regressão isotônica"},
}

_THRESHOLD_TYPE_LABELS: Dict[str, Dict[str, str]] = {
    "locked":  {"English": "Locked — operational",     "Portuguese": "Bloqueado — operacional"},
    "youden":  {"English": "Youden J — exploratory",   "Portuguese": "Youden J — exploratório"},
    "fixed":   {"English": "Fixed clinical",           "Portuguese": "Limiar clínico fixo"},
}


def build_exploratory_recalibration_summary(
    recal_data: Dict[str, Dict[str, dict]],
    rename: Dict[str, str],
    language: str = "English",
) -> dict:
    """Format pre-computed recalibration results into a report-ready structure.

    Parameters
    ----------
    recal_data:
        Mapping ``{score_col: {method_key: result_dict}}`` where ``method_key``
        is one of ``"intercept_only"``, ``"logistic"``, ``"isotonic"`` and
        ``result_dict`` is the dict returned by the corresponding
        ``recalibrate_*`` function in ``stats_compare``.
    rename:
        Display-name mapping ``{col -> label}`` (same dict used by the tab).
    language:
        ``"English"`` or ``"Portuguese"``.

    Returns
    -------
    dict with keys:
        ``"table"``     — :class:`pd.DataFrame` with one row per (score, method).
        ``"available"`` — ``bool``, ``True`` when at least one score had data.
    """
    _lang = "Portuguese" if language not in (None, "", "English") else "English"

    rows = []
    for score_col, methods in recal_data.items():
        label = rename.get(score_col, score_col)
        # Pull "before" stats from any available method (all share the same original).
        _any = next(iter(methods.values()), {}) if methods else {}
        brier_before = _any.get("brier_before")
        cal_int_before = _any.get("cal_intercept_before")
        cal_slp_before = _any.get("cal_slope_before")

        # Original row (no recalibration).
        rows.append({
            "Score": label,
            "Method": _RECAL_METHOD_LABELS["original"][_lang],
            "Brier_Before": brier_before,
            "Brier_After": brier_before,
            "Cal_Intercept_Before": cal_int_before,
            "Cal_Intercept_After": cal_int_before,
            "Cal_Slope_Before": cal_slp_before,
            "Cal_Slope_After": cal_slp_before,
        })

        for method_key in ("intercept_only", "logistic", "isotonic"):
            res = methods.get(method_key)
            if res is None:
                continue
            rows.append({
                "Score": label,
                "Method": _RECAL_METHOD_LABELS[method_key][_lang],
                "Brier_Before": res.get("brier_before"),
                "Brier_After": res.get("brier_after"),
                "Cal_Intercept_Before": res.get("cal_intercept_before"),
                "Cal_Intercept_After": res.get("cal_intercept_after"),
                "Cal_Slope_Before": res.get("cal_slope_before"),
                "Cal_Slope_After": res.get("cal_slope_after"),
            })

    if not rows:
        return {"table": pd.DataFrame(), "available": False}

    cols = [
        "Score", "Method",
        "Brier_Before", "Brier_After",
        "Cal_Intercept_Before", "Cal_Intercept_After",
        "Cal_Slope_Before", "Cal_Slope_After",
    ]
    return {"table": pd.DataFrame(rows)[cols], "available": True}


def build_exploratory_threshold_summary(
    threshold_tables: Dict[str, pd.DataFrame],
    locked_threshold: float,
    youden_thresholds: Dict[str, Tuple[float, float]],
    rename: Dict[str, str],
    language: str = "English",
) -> dict:
    """Format pre-computed threshold-analysis tables into a report-ready structure.

    Parameters
    ----------
    threshold_tables:
        Mapping ``{score_col: df}`` where ``df`` comes from
        ``threshold_analysis_table()`` in ``stats_compare``.  Must contain
        columns ``Threshold``, ``Sensitivity``, ``Specificity``, ``PPV``,
        ``NPV``, ``TP``, ``FP``, ``TN``, ``FN``, ``N_Flagged``, ``Flag_Rate_pct``.
    locked_threshold:
        The operationally locked threshold (float, e.g. ``0.08``).
    youden_thresholds:
        Mapping ``{score_col: (threshold, j_score)}``.
    rename:
        Display-name mapping ``{col -> label}``.
    language:
        ``"English"`` or ``"Portuguese"``.

    Returns
    -------
    dict with keys:
        ``"table"``       — :class:`pd.DataFrame` with one row per (score, threshold).
        ``"note_locked"`` — ``str`` explanatory note about the locked threshold.
        ``"note_youden"`` — ``str`` explanatory note about Youden's J.
        ``"available"``   — ``bool``.
    """
    _lang = "Portuguese" if language not in (None, "", "English") else "English"

    def _type_label(thr: float, score_col: str) -> str:
        if abs(thr - locked_threshold) < 1e-9:
            return _THRESHOLD_TYPE_LABELS["locked"][_lang]
        if score_col in youden_thresholds and abs(thr - youden_thresholds[score_col][0]) < 1e-9:
            return _THRESHOLD_TYPE_LABELS["youden"][_lang]
        return _THRESHOLD_TYPE_LABELS["fixed"][_lang]

    rows = []
    for score_col, df in threshold_tables.items():
        if df.empty:
            continue
        label = rename.get(score_col, score_col)
        for _, row in df.iterrows():
            thr = float(row["Threshold"])
            tp = float(row.get("TP", 0))
            fp = float(row.get("FP", 0))
            fn = float(row.get("FN", 0))
            tn = float(row.get("TN", 0))
            era = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            erb = fn / (fn + tn) if (fn + tn) > 0 else float("nan")
            rows.append({
                "Score": label,
                "Threshold": thr,
                "Type": _type_label(thr, score_col),
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "Sensitivity": row.get("Sensitivity"),
                "Specificity": row.get("Specificity"),
                "PPV": row.get("PPV"),
                "NPV": row.get("NPV"),
                "N_Flagged": row.get("N_Flagged"),
                "Flag_Rate_pct": row.get("Flag_Rate_pct"),
                "Event_Rate_Above": era,
                "Event_Rate_Below": erb,
            })

    if not rows:
        return {
            "table": pd.DataFrame(),
            "note_locked": "",
            "note_youden": "",
            "available": False,
        }

    cols = [
        "Score", "Threshold", "Type",
        "TP", "FP", "TN", "FN",
        "Sensitivity", "Specificity", "PPV", "NPV",
        "N_Flagged", "Flag_Rate_pct",
        "Event_Rate_Above", "Event_Rate_Below",
    ]
    if _lang == "Portuguese":
        note_locked = (
            f"**Limiar bloqueado ({locked_threshold:.1%}):** O limiar operacional pré-especificado "
            f"do estudo primário. Permanece inalterado. Todas as métricas primárias de "
            f"sensibilidade/especificidade reportadas em outros locais utilizam este limiar."
        )
        note_youden = (
            "**Ótimo de Youden J:** Derivado da coorte de validação maximizando "
            "Sensibilidade + Especificidade − 1. Trata-se de um ótimo post-hoc orientado pelos "
            "dados e não deve ser utilizado como limiar operacional primário. Uso apenas "
            "exploratório/suplementar."
        )
    else:
        note_locked = (
            f"**Locked threshold ({locked_threshold:.1%}):** The prespecified operational "
            f"threshold from the primary study. Remains unchanged. All primary "
            f"sensitivity/specificity metrics reported elsewhere use this threshold."
        )
        note_youden = (
            "**Youden's J optimum:** Derived from the validation cohort by maximising "
            "Sensitivity + Specificity − 1. This is a data-driven, post-hoc optimum and must "
            "not be used as the primary operating threshold. Exploratory/supplementary use only."
        )

    return {
        "table": pd.DataFrame(rows)[cols],
        "note_locked": note_locked,
        "note_youden": note_youden,
        "available": True,
    }


def build_exploratory_temporal_validation_section(
    recal_summary: Optional[dict],
    threshold_summary: Optional[dict],
    language: str = "English",
) -> str:
    """Build the Markdown string for the exploratory appendix section.

    This section is appended AFTER the main temporal-validation report and is
    clearly labelled as exploratory.  It must not alter the primary metrics.

    Parameters
    ----------
    recal_summary:
        Output of :func:`build_exploratory_recalibration_summary`, or ``None``.
    threshold_summary:
        Output of :func:`build_exploratory_threshold_summary`, or ``None``.
    language:
        ``"English"`` or ``"Portuguese"``.

    Returns
    -------
    str — Markdown text (may be empty string when both inputs are ``None``).
    """
    _lang = "Portuguese" if language not in (None, "", "English") else "English"

    def _tr(en: str, pt: str) -> str:
        return en if _lang == "English" else pt

    def _fv(val, decimals: int = 3) -> str:
        try:
            if val is None:
                return "N/A"
            if isinstance(val, float) and (val != val):  # NaN check without numpy
                return "N/A"
            if isinstance(val, float):
                return f"{val:.{decimals}f}"
            return str(val)
        except Exception:
            return "N/A"

    def _pct(val) -> str:
        try:
            if val is None:
                return "N/A"
            if isinstance(val, float) and (val != val):
                return "N/A"
            return f"{float(val):.1%}"
        except Exception:
            return "N/A"

    has_recal = recal_summary is not None and recal_summary.get("available", False)
    has_thresh = threshold_summary is not None and threshold_summary.get("available", False)

    if not has_recal and not has_thresh:
        return ""

    lines: List[str] = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## {_tr('Appendix: Exploratory Post-hoc Recalibration and Threshold Analysis', 'Apêndice: Análise Exploratória de Recalibração Pós-hoc e Limiar')}")
    lines.append("")
    lines.append(
        f"> **{_tr('Exploratory only', 'Apenas exploratório')}.**"
        f" {_tr('Recalibration was applied post-hoc to the validation cohort itself.', 'A recalibração foi aplicada post-hoc à própria coorte de validação.')}"
        f" {_tr('These results were not clinically validated and must not be used to report the primary model performance or guide clinical decisions.', 'Estes resultados não foram validados clinicamente e não devem ser usados para reportar o desempenho primário do modelo nem orientar decisões clínicas.')}"
        f" {_tr('The frozen model and locked threshold remain unchanged.', 'O modelo congelado e o limiar bloqueado permanecem inalterados.')}"
    )
    lines.append("")

    # ── Recalibration section ────────────────────────────────────────────────
    if has_recal:
        lines.append(f"### {_tr('Post-hoc Recalibration Summary', 'Resumo da Recalibração Pós-hoc')}")
        lines.append("")
        lines.append(_tr(
            "Four conditions are compared: Original (no recalibration), Intercept-only "
            "(slope fixed at 1), Intercept + Slope (logistic), and Isotonic Regression. "
            "All methods are applied to the same validation cohort.",
            "Quatro condições são comparadas: Original (sem recalibração), Apenas intercepto "
            "(slope fixo em 1), Intercepto + Slope (logístico) e Regressão isotônica. "
            "Todos os métodos são aplicados à mesma coorte de validação.",
        ))
        lines.append("")

        tbl = recal_summary["table"]
        # Build Markdown table
        headers = [
            _tr("Score", "Escore"),
            _tr("Method", "Método"),
            _tr("Brier Before", "Brier Antes"),
            _tr("Brier After", "Brier Depois"),
            _tr("Cal.Int. Before", "Int.Cal. Antes"),
            _tr("Cal.Int. After", "Int.Cal. Depois"),
            _tr("Cal.Slope Before", "Slope.Cal. Antes"),
            _tr("Cal.Slope After", "Slope.Cal. Depois"),
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in tbl.iterrows():
            cells = [
                str(row.get("Score", "")),
                str(row.get("Method", "")),
                _fv(row.get("Brier_Before"), 4),
                _fv(row.get("Brier_After"), 4),
                _fv(row.get("Cal_Intercept_Before"), 3),
                _fv(row.get("Cal_Intercept_After"), 3),
                _fv(row.get("Cal_Slope_Before"), 3),
                _fv(row.get("Cal_Slope_After"), 3),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # ── Threshold comparison section ─────────────────────────────────────────
    if has_thresh:
        lines.append(f"### {_tr('Threshold Comparison', 'Comparação de Limiares')}")
        lines.append("")
        lines.append(_tr(
            "The table below compares the prespecified locked threshold with the Youden J "
            "optimum and optional fixed clinical thresholds. "
            "**Only the locked threshold governs primary analysis.**",
            "A tabela abaixo compara o limiar bloqueado pré-especificado com o ótimo de "
            "Youden J e limiares clínicos fixos opcionais. "
            "**Apenas o limiar bloqueado rege a análise primária.**",
        ))
        lines.append("")

        ttbl = threshold_summary["table"]
        headers = [
            _tr("Score", "Escore"),
            _tr("Threshold", "Limiar"),
            _tr("Type", "Tipo"),
            "TP", "FP", "TN", "FN",
            _tr("Sensitivity", "Sensibilidade"),
            _tr("Specificity", "Especificidade"),
            "PPV",
            "NPV",
            _tr("N Flagged", "N Sinalizados"),
            _tr("Flag Rate %", "Taxa sinalizados %"),
            _tr("Event Rate (Above)", "Taxa evento (acima)"),
            _tr("Event Rate (Below)", "Taxa evento (abaixo)"),
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in ttbl.iterrows():
            cells = [
                str(row.get("Score", "")),
                _pct(row.get("Threshold")),
                str(row.get("Type", "")),
                str(int(row["TP"]) if row.get("TP") is not None else "N/A"),
                str(int(row["FP"]) if row.get("FP") is not None else "N/A"),
                str(int(row["TN"]) if row.get("TN") is not None else "N/A"),
                str(int(row["FN"]) if row.get("FN") is not None else "N/A"),
                _fv(row.get("Sensitivity"), 3),
                _fv(row.get("Specificity"), 3),
                _fv(row.get("PPV"), 3),
                _fv(row.get("NPV"), 3),
                str(int(row["N_Flagged"]) if row.get("N_Flagged") is not None else "N/A"),
                _fv(row.get("Flag_Rate_pct"), 1),
                _pct(row.get("Event_Rate_Above")),
                _pct(row.get("Event_Rate_Below")),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        if threshold_summary.get("note_locked"):
            lines.append(f"> {threshold_summary['note_locked']}")
            lines.append("")
        if threshold_summary.get("note_youden"):
            lines.append(f"> {threshold_summary['note_youden']}")
            lines.append("")

    lines.append("---")
    lines.append(
        f"*{_tr('Generated by AI Risk — Exploratory Module', 'Gerado pelo AI Risk — Módulo Exploratório')}*"
    )
    return "\n".join(lines)
