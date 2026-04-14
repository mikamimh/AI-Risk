"""Temporal validation helpers for AI Risk.

Extracted from model_metadata.py to isolate temporal-cohort logic from core
model metadata functions.

Provides:
- extract_year_quarter_range  — parse year-quarter range from a DataFrame
- check_temporal_overlap      — compare training vs. validation date ranges
- format_locked_model_for_display — tabular locked-model summary (Temporal Validation tab)
- build_temporal_validation_summary — Markdown report for temporal validation results

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
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Internal year-quarter constants
# ---------------------------------------------------------------------------

_QUARTER_TO_MONTH = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
_QUARTER_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

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
) -> str:
    """Build Markdown summary for temporal validation results."""
    def _tr(en, pt):
        return en if language == "English" else pt

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cs = cohort_summary

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

    # Cohort summary table
    lines.append(f"## {_tr('Cohort Summary', 'Resumo da Coorte')}")
    lines.append("")
    lines.append(f"| {_tr('Property', 'Propriedade')} | {_tr('Value', 'Valor')} |")
    lines.append("|:--|:--|")
    lines.append(f"| {_tr('Total patients', 'Total de pacientes')} | {cs.get('n_total', 0)} |")
    lines.append(f"| {_tr('Events (30-day mortality)', 'Eventos (mortalidade 30 dias)')} | {cs.get('n_events', 0)} |")
    lines.append(f"| {_tr('Event rate', 'Taxa de eventos')} | {cs.get('event_rate', 0):.1%} |")
    lines.append(f"| {_tr('Date range', 'Período')} | {cs.get('date_range', 'Unknown')} |")
    for level_key, level_label_en, level_label_pt in [
        ("n_complete", "Complete data", "Dados completos"),
        ("n_adequate", "Adequate", "Adequados"),
        ("n_partial", "Partially imputed", "Parcialmente imputados"),
        ("n_low", "Heavily imputed", "Muito imputados"),
    ]:
        n = cs.get(level_key, 0)
        pct = n / cs["n_total"] * 100 if cs.get("n_total", 0) > 0 else 0
        lines.append(f"| {_tr(level_label_en, level_label_pt)} | {n} ({pct:.1f}%) |")
    lines.append("")

    # Performance table
    if not performance_df.empty:
        lines.append(f"## {_tr('Discrimination and Calibration', 'Discriminação e Calibração')}")
        lines.append("")
        hdr = " | ".join(performance_df.columns)
        lines.append(f"| {hdr} |")
        lines.append("|" + "|".join(":--" for _ in performance_df.columns) + "|")
        for _, row in performance_df.iterrows():
            cells = " | ".join(str(v) for v in row.values)
            lines.append(f"| {cells} |")
        lines.append("")

    # Pairwise comparison
    if not pairwise_df.empty:
        lines.append(f"## {_tr('Pairwise Comparison', 'Comparação Pareada')}")
        lines.append("")
        hdr = " | ".join(pairwise_df.columns)
        lines.append(f"| {hdr} |")
        lines.append("|" + "|".join(":--" for _ in pairwise_df.columns) + "|")
        for _, row in pairwise_df.iterrows():
            cells = " | ".join(str(v) for v in row.values)
            lines.append(f"| {cells} |")
        lines.append("")

    # Risk categories
    if not risk_category_df.empty:
        lines.append(f"## {_tr('Risk Category Distribution', 'Distribuição por Classe de Risco')}")
        lines.append("")
        hdr = " | ".join(risk_category_df.columns)
        lines.append(f"| {hdr} |")
        lines.append("|" + "|".join(":--" for _ in risk_category_df.columns) + "|")
        for _, row in risk_category_df.iterrows():
            cells = " | ".join(str(v) for v in row.values)
            lines.append(f"| {cells} |")
        lines.append("")

    lines.append("---")
    lines.append(f"*{_tr('Generated by AI Risk — Temporal Validation Module', 'Gerado pelo AI Risk — Módulo de Validação Temporal')}*")

    return "\n".join(lines)
