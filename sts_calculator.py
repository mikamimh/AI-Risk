"""STS Adult Cardiac Surgery Risk Calculator — WebSocket client.

Queries the official STS Risk Calculator web application at
https://acsdriskcalc.research.sts.org/ via its Shiny WebSocket interface.

Returns all STS risk endpoints:
  - Operative Mortality (predmort)
  - Morbidity & Mortality (predmm)
  - Stroke (predstro)
  - Renal Failure (predrenf)
  - Reoperation (predreop)
  - Prolonged Ventilation (predvent)
  - Deep Sternal Wound Infection (preddeep)
  - Long Hospital Stay >14 days (pred14d)
  - Short Hospital Stay <6 days (pred6d)
"""

import asyncio
import json
import re
import html as html_mod
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

# Phase 2: persistent STS Score cache with revalidation and transparency.
# The cache layer is optional — if it fails to import, calculate_sts and
# calculate_sts_batch fall back to the legacy (uncached) behavior so the
# app remains runnable.
try:
    import sts_cache as _sts_cache
    HAS_STS_CACHE = True
except Exception:  # pragma: no cover - fall back silently
    _sts_cache = None  # type: ignore[assignment]
    HAS_STS_CACHE = False

# Phase 3: in-memory/session STS Score cache.
# Lookup order within calculate_sts_batch:
#   1. in-memory  (this dict)
#   2. persistent disk cache  (sts_cache.py)
#   3. network fetch via WebSocket
# Keyed by input_hash; stores the full entry dict so make_cache_hit_record
# works without modification.  Process-local — does NOT persist across server
# restarts, but DOES survive tab navigation within the same Streamlit session.
_sts_memory_cache: Dict[str, dict] = {}

_sts_log = logging.getLogger("sts_score.calculator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WS_URL = "wss://acsdriskcalc.research.sts.org/websocket/"

WS_HEADERS = [
    ("Origin", "https://acsdriskcalc.research.sts.org"),
    ("Referer", "https://acsdriskcalc.research.sts.org/"),
    ("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"),
]

# Maximum wall-clock seconds for a single-patient STS query (connect + all messages).
# The inner message loop allows up to 80 × 30 s = 2 400 s without this cap —
# this guard prevents multi-hour stalls for unreachable/unresponsive endpoints.
STS_PER_PATIENT_TIMEOUT_S: int = 90

# Maximum consecutive failures before calculate_sts_batch aborts the web-query
# phase and returns partial results.  Prevents infinite stalls when the STS
# endpoint is down entirely.
# Set to 10 (was 5): with chunk_size=1 each "failure" is a single patient.
# A transient network blip at batch-start would abort the whole run too eagerly
# at 5; 10 gives the endpoint time to recover across the inter-failure backoff.
STS_MAX_CONSECUTIVE_FAILURES: int = 10

# Seconds to wait after each consecutive failure before retrying the next
# patient.  Progressive: wait = min(BACKOFF_BASE * n, BACKOFF_MAX_S) where n
# is the current consecutive-failure count.  Zero on success (counter resets).
STS_CONSECUTIVE_FAILURE_BACKOFF_BASE_S: int = 5
STS_CONSECUTIVE_FAILURE_BACKOFF_MAX_S: int = 30

# Procedures NOT covered by the STS ACSD web calculator.
# Keywords are matched (substring, upper-case) against surgery_pre / Surgery.
_STS_UNSUPPORTED_KEYWORDS: frozenset = frozenset([
    "DISSECTION",
    "ANEURISM",
    "ANEURYSM",
    "BENTALL",
    "AORTIC ROOT REPLACEMENT",
    "AORTIC ROOT REPAIR",
    "AORTIC REPAIR",
    "AORTIC RECONSTRUCTION",
    "AORTA REPAIR",
])

# Surgical-priority values that cannot be safely mapped to an STS urgency field.
_STS_UNMAPPABLE_PRIORITY: frozenset = frozenset([
    "OBSERVATION ADMIT",
    "OBSERVATION",
])

PROCID_TO_PROC = {
    1: "Isolated CABG",
    2: "Isolated AVR",
    3: "Isolated MVR",
    4: "AVR + CABG",
    5: "MVR + CABG",
    7: "MV Repair",
    8: "MV Repair + CABG",
}

# All STS output keys
STS_RESULT_KEYS = {
    "Operative Mortality": "predmort",
    "Morbidity & Mortality": "predmm",
    "Stroke": "predstro",
    "Renal Failure": "predrenf",
    "Reoperation": "predreop",
    "Prolonged Ventilation": "predvent",
    "Deep Sternal Wound Infection": "preddeep",
    "Long Hospital Stay (>14 days)": "pred14d",
    "Short Hospital Stay (<6 days)": "pred6d",
}

# Human-readable labels for display
STS_LABELS = {
    "predmort": "Operative Mortality",
    "predmm": "Morbidity & Mortality",
    "predstro": "Stroke",
    "predrenf": "Renal Failure",
    "predreop": "Reoperation",
    "predvent": "Prolonged Ventilation",
    "preddeep": "Deep Sternal Wound Infection",
    "pred14d": "Long Hospital Stay (>14 days)",
    "pred6d": "Short Hospital Stay (<6 days)",
}

BOOLEAN_FIELDS = [
    "medacei48", "medgp", "medinotr", "medster", "medadp5days", "fhcad",
    "hypertn", "liverdis", "mediastrad", "unrespstat", "dialysis", "cancer",
    "syncope", "immsupp", "pneumonia", "slpapn", "hmo2", "pvd",
    "cvdpcarsurg", "carshock", "resusc", "stenleftmain",
    "vdstena", "vdstenm",
]

PRESENCE_BOOLEAN_FIELDS = [
    "cvdstenrt", "cvdstenlft", "laddiststenpercent", "vdaoprimet",
]

ARRAY_FIELD_MAPPINGS = {
    "diabetes": "diabetes",
    "endocarditis": "endocarditis",
    "ivdrugab": "ivdrugab",
    "alcohol": "alcohol",
    "tobaccouse": "tobaccouse",
    "chrlungd": "chrlungd",
    "heartfailtmg": "heartfailtmg",
    "classnyh": "classnyh",
    "cardsymptimeofadm": "cardsymptimeofadm",
    "miwhen": "miwhen",
    "numdisv": "numdisv",
    "vdinsufa": "vdinsufa",
    "vdinsufm": "vdinsufm",
    "vdinsuft": "vdinsuft",
    "arrhythatrfib": "arrhythatrfib",
    "arrhythafib": "arrhythafib",
    "arrhythaflutter": "arrhythaflutter",
    "arrhythvv": "arrhythvv",
    "arrhythsss": "arrhythsss",
    "arrhythsecond": "arrhythsecond",
    "arrhyththird": "arrhyththird",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val, default=None):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    s = str(val).strip()
    if s in ("", "-", "Unknown", "None", "nan"):
        return default
    # Handle Brazilian decimal format: "1,29" → "1.29"
    # If there's a comma but no dot, and ≤2 digits after comma → decimal separator
    if "," in s and "." not in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s and "." in s:
        # "1.000,29" → "1000.29" (European) or "1,000.29" → "1000.29" (US)
        if s.rindex(",") > s.rindex("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return default


def _safe_int(val, default=None):
    f = _safe_float(val)
    return int(f) if f is not None else default


# ---------------------------------------------------------------------------
# CSV / DataFrame row → STS field mapping
# ---------------------------------------------------------------------------

def _map_surgery_to_procid(surgery_str: str) -> int:
    s = str(surgery_str).strip().upper()
    has_cabg = "CABG" in s or "OPCAB" in s
    has_avr = "AVR" in s
    has_mvr = "MVR" in s
    has_mv_repair = "MV REPAIR" in s or "MITRAL REPAIR" in s or "PLASTIA MITRAL" in s

    if has_avr and has_cabg:
        return 4
    if has_mvr and has_cabg:
        return 5
    if has_mv_repair and has_cabg:
        return 8
    if has_avr:
        return 2
    if has_mvr:
        return 3
    if has_mv_repair:
        return 7
    if has_cabg:
        return 1
    return 2  # default to AVR


def _map_coronary_symptom(val) -> str:
    v = str(val).strip() if val else ""
    mapping = {
        "None": "No Symptoms",
        "No coronary symptoms": "No Symptoms",
        "Sem sintomas coronarianos": "No Symptoms",
        "Stable Angina": "Stable Angina",
        "Angina estável": "Stable Angina",
        "Unstable Angina": "Unstable Angina",
        "Angina instável": "Unstable Angina",
        "STEMI": "ST Elevation MI (STEMI)",
        "IAM com supra de ST": "ST Elevation MI (STEMI)",
        "NSTEMI": "Non-ST Elevation MI (Non-STEMI)",
        "Non-STEMI": "Non-ST Elevation MI (Non-STEMI)",
        "IAM sem supra de ST": "Non-ST Elevation MI (Non-STEMI)",
    }
    return mapping.get(v, "")


def _map_nyha(nyha) -> str:
    mapping = {"I": "Class I", "II": "Class II", "III": "Class III", "IV": "Class IV"}
    return mapping.get(str(nyha).strip(), "")


def _map_num_vessels(val) -> str:
    mapping = {"0": "None", "1": "One", "2": "Two", "3": "Three"}
    return mapping.get(str(val).strip() if val else "", "")


def _map_valve_severity(val) -> str:
    v = str(val).strip() if val else ""
    mapping = {
        "None": "None",
        "Trivial": "Trivial/Trace",
        "Trace": "Trivial/Trace",
        "Mild": "Mild",
        "Moderate": "Moderate",
        "Severe": "Severe",
    }
    return mapping.get(v, "")


def _map_incidence(prev_surgery) -> str:
    v = str(prev_surgery).strip() if prev_surgery else ""
    if v in ("None", "No", "", "nan"):
        return "First CV surgery"
    return "ReOp#1 CV surgery"


def _map_heart_failure(val) -> str:
    v = str(val).strip() if val else ""
    if v == "Chronic":
        return "Yes - Chronic"
    if v == "Acute":
        return "Yes - Acute"
    if v == "Both":
        return "Yes - Both"
    return ""


def _map_diabetes(val) -> str:
    v = str(val).strip() if val else ""
    if v == "Insulin":
        return "Yes, Insulin"
    if v == "Oral":
        return "Yes, Oral"
    if v == "Yes":
        return "Yes, Unknown"
    return ""


def _map_copd(val) -> str:
    v = str(val).strip() if val else ""
    if v == "Yes":
        return "Severity Unknown"
    if v in ("Mild", "Moderate", "Severe"):
        return v
    return ""


def _map_urgency(val) -> str:
    """Map a surgical-priority string to the STS urgency field value.

    DESIGN NOTE — OBSERVATION ADMIT
    --------------------------------
    ``OBSERVATION ADMIT`` (and similar observation-admission strings) is an
    *admission type*, not a surgical urgency level.  It has **no** defined
    mapping to the STS ACSD urgency field and must NOT reach this function.
    ``classify_sts_eligibility()`` is the single authoritative gatekeeper:
    it classifies any row whose ``surgical_priority`` is ``OBSERVATION ADMIT``
    as ``"uncertain"`` and the caller is expected to skip such rows before
    ever calling ``build_sts_input_from_row``.

    Unrecognised values that slip through despite the gatekeeper fall back to
    ``"Elective"`` (the most conservative STS default) so the batch does not
    crash — but they will also have been flagged as ``uncertain`` and skipped
    in the temporal-validation flow.
    """
    v = str(val).strip() if val else ""
    mapping = {
        "Elective": "Elective",
        "Eletiva": "Elective",
        "Urgent": "Urgent",
        "Urgente": "Urgent",
        "Emergency": "Emergent",
        "Emergent": "Emergent",
        "Emergência": "Emergent",
        "Salvage": "Emergent Salvage",
        "Salvamento": "Emergent Salvage",
        "Emergent Salvage": "Emergent Salvage",
    }
    result = mapping.get(v)
    if result is None:
        _sts_log.debug(
            "_map_urgency: unrecognised priority %r — defaulting to Elective. "
            "Row should have been caught by classify_sts_eligibility().",
            v,
        )
        return "Elective"
    return result


def classify_sts_eligibility(row: dict) -> tuple:
    """Classify a patient row for STS ACSD web-calculator eligibility.

    Returns:
        (status, reason) where *status* is one of:

        ``"supported"``
            Procedure is a standard cardiac operation covered by the STS ACSD
            calculator (CABG, AVR, MVR, MV Repair, and combinations).

        ``"not_supported"``
            Procedure is outside STS ACSD scope — aortic dissection repair,
            aortic aneurysm repair, Bentall-de Bono procedure, or similar.
            These cases should be skipped before calling the web calculator.

        ``"uncertain"``
            Procedure mapping could not be confirmed (empty field, unrecognised
            string, or unmappable priority).  The caller may choose to attempt
            the query or skip conservatively.
    """
    surgery = str(row.get("surgery_pre") or row.get("Surgery") or "").strip().upper()
    priority = str(row.get("surgical_priority") or row.get("Surgical Priority") or "").strip().upper()

    # --- hard exclusions ---
    for kw in _STS_UNSUPPORTED_KEYWORDS:
        if kw in surgery:
            return ("not_supported", f"procedure outside STS ACSD scope: {surgery!r}")

    # --- priority edge cases ---
    if priority in _STS_UNMAPPABLE_PRIORITY:
        return ("uncertain", f"surgical priority not directly mappable to STS urgency: {priority!r}")

    # --- known-supported procedure families ---
    has_cabg = "CABG" in surgery or "OPCAB" in surgery
    has_avr = "AVR" in surgery
    has_mvr = "MVR" in surgery
    has_mv_repair = "MV REPAIR" in surgery or "MITRAL REPAIR" in surgery or "PLASTIA MITRAL" in surgery

    if has_cabg or has_avr or has_mvr or has_mv_repair:
        return ("supported", "standard cardiac procedure covered by STS ACSD")

    if not surgery:
        return ("uncertain", "surgery type not specified")

    return ("uncertain", f"procedure not confirmed as STS-supported: {surgery!r}")


def build_sts_input_from_row(row: dict) -> dict:
    """Build STS-compatible input dict from a data row (CSV or form_map).

    Accepts both raw CSV column names (snake_case) and app display names.
    """
    def g(csv_key, app_key=None):
        """Get value from row, trying CSV key first then app key."""
        v = row.get(csv_key)
        if (v is None or (isinstance(v, float) and pd.isna(v))) and app_key:
            v = row.get(app_key)
        if isinstance(v, float) and pd.isna(v):
            return None
        return v

    d = {}

    # Procedure
    surgery = g("surgery_pre", "Surgery") or ""
    d["procid"] = str(_map_surgery_to_procid(surgery))

    # Demographics
    d["age"] = str(g("age_years", "Age (years)") or "")
    sex = str(g("sex", "Sex") or "")
    d["gender"] = "Female" if sex.strip().upper() == "F" else "Male"
    d["heightcm"] = str(g("height_cm", "Height (cm)") or "")
    d["weightkg"] = str(g("weight_kg", "Weight (kg)") or "")

    # Race
    race = str(g("race", "Race") or "")
    d["raceblack"] = "Yes" if "Black" in race else ""
    d["raceasian"] = "Yes" if "Asian" in race else ""
    d["racenativeam"] = ""
    d["racnativepacific"] = ""
    d["ethnicity"] = ""

    # Status / urgency
    d["status"] = _map_urgency(g("surgical_priority", "Surgical Priority"))

    # Incidence (previous surgery)
    d["incidenc"] = _map_incidence(g("previous_surgery", "Previous surgery"))

    # NYHA
    d["classnyh"] = _map_nyha(g("nyha_pre", "Preoperative NYHA") or "")

    # Heart failure
    d["heartfailtmg"] = _map_heart_failure(g("heart_failure", "HF"))

    # LVEF
    lvef = _safe_float(g("lvef_pre_pct", "LVEF, %"))
    d["hdef"] = str(lvef) if lvef is not None else ""

    # Labs
    creat = _safe_float(g("creatinine_pre_mg_dl", "Creatinine (mg/dL)"))
    d["creatlst"] = str(creat) if creat is not None else ""

    hct = _safe_float(g("hematocrit_pre_pct", "Hematocrit (%)"))
    d["hct"] = str(hct) if hct is not None else ""

    wbc = _safe_float(g("wbc_count_pre_10e3_ul", "WBC Count (10³/μL)"))
    d["wbc"] = str(wbc) if wbc is not None else ""

    platelets = _safe_int(g("platelet_count_pre_cells_ul", "Platelet Count (cells/μL)"))
    d["platelets"] = str(platelets) if platelets is not None else ""

    # Comorbidities
    d["diabetes"] = _map_diabetes(g("diabetes", "Diabetes"))
    d["hypertn"] = "Yes" if str(g("hypertension", "Hypertension") or "").strip() == "Yes" else ""
    d["dialysis"] = "Yes" if str(g("dialysis_pre", "Dialysis") or "").strip() == "Yes" else ""
    d["pvd"] = "Yes" if str(g("peripheral_vascular_disease", "PVD") or "").strip() == "Yes" else ""
    d["cva"] = "Yes" if str(g("stroke_history", "CVA") or "").strip() == "Yes" else ""
    d["infendo"] = "Yes" if str(g("infective_endocarditis_pre", "IE") or "").strip() == "Yes" else ""
    d["chrlungd"] = _map_copd(g("copd", "Chronic Lung Disease"))
    d["immsupp"] = ""
    d["cancer"] = ""

    # Coronary symptoms
    d["cardsymptimeofadm"] = _map_coronary_symptom(g("coronary_symptom", "Coronary Symptom"))

    # MI timing
    symptom = str(g("coronary_symptom", "Coronary Symptom") or "")
    d["miwhen"] = "1 to 7 Days" if "STEMI" in symptom or "IAM" in symptom else ""

    # Coronary anatomy
    d["numdisv"] = _map_num_vessels(g("diseased_vessels_count", "No. of Diseased Vessels"))
    d["stenleftmain"] = "Yes" if str(g("left_main_stenosis_ge_50_pct", "Left Main Stenosis ≥ 50%") or "").strip() == "Yes" else ""

    # Arrhythmia
    remote = str(g("arrhythmia_remote", "Arrhythmia Remote") or "").strip()
    recent = str(g("arrhythmia_recent", "Arrhythmia Recent") or "").strip()

    d["arrhythatrfib"] = ""
    d["arrhythafib"] = ""
    d["arrhythaflutter"] = ""
    d["arrhyththird"] = ""
    d["arrhythsecond"] = ""
    d["arrhythsss"] = ""
    d["arrhythvv"] = ""

    if remote not in ("", "None", "nan"):
        if "Fibrillation" in remote or "AF" in remote or "FA" in remote:
            d["arrhythatrfib"] = "Remote"
        elif "Flutter" in remote:
            d["arrhythaflutter"] = "Remote"
        elif "3" in remote or "Third" in remote:
            d["arrhyththird"] = "Remote"
        else:
            d["arrhythatrfib"] = "Remote"

    if recent not in ("", "None", "nan"):
        if "Fibrillation" in recent or "AF" in recent or "FA" in recent:
            d["arrhythatrfib"] = "Recent"
        elif "Flutter" in recent:
            d["arrhythaflutter"] = "Recent"
        elif "Block" in recent or "3" in recent or "Third" in recent:
            d["arrhyththird"] = "Recent"
        else:
            d["arrhythatrfib"] = "Recent"

    # Valve disease
    as_val = str(g("aortic_stenosis_pre", "Aortic Stenosis") or "").strip()
    d["vdstena"] = "Yes" if as_val and as_val not in ("None", "-", "", "nan") else ""

    ms_val = str(g("mitral_stenosis_pre", "Mitral Stenosis") or "").strip()
    d["vdstenm"] = "Yes" if ms_val and ms_val not in ("None", "-", "", "nan") else ""

    d["vdinsufa"] = _map_valve_severity(g("aortic_regurgitation_pre", "Aortic Regurgitation"))
    d["vdinsufm"] = _map_valve_severity(g("mitral_regurgitation_pre", "Mitral Regurgitation"))
    d["vdinsuft"] = _map_valve_severity(g("tricuspid_regurgitation_pre", "Tricuspid Regurgitation"))

    # Medications (limited info from CSV)
    meds = str(g("preoperative_medications", "Preoperative Medications") or "")
    d["medacei48"] = "Yes" if "ACE" in meds or "ARB" in meds else ""
    d["medgp"] = ""
    d["medbeta"] = ""
    d["medster"] = ""
    d["medadp5days"] = ""
    d["medinotr"] = ""

    # Defaults for unavailable fields
    d["carshock"] = ""
    d["resusc"] = ""
    d["iabpwhen"] = ""
    d["cathbasassistwhen"] = ""
    d["ecmowhen"] = ""
    d["prcab"] = ""
    d["prvalve"] = ""
    d["poc"] = ""
    d["pocpci"] = ""
    d["pocpciwhen"] = ""
    d["pocpciin"] = ""
    d["prcvint"] = ""
    d["tobaccouse"] = ""
    d["alcohol"] = ""
    d["ivdrugab"] = ""
    d["pneumonia"] = ""
    d["hmo2"] = ""
    d["slpapn"] = ""

    return d


# ---------------------------------------------------------------------------
# Pre-flight payload validation
# ---------------------------------------------------------------------------

_VALID_PROCIDS = {"1", "2", "3", "4", "5", "6", "7", "8"}
_VALID_GENDERS = {"Male", "Female"}
_VALID_STATUSES = {"Elective", "Urgent", "Emergent", "Emergent Salvage"}


# ---------------------------------------------------------------------------
# Typed STS query exception hierarchy
# ---------------------------------------------------------------------------

class StsQueryError(Exception):
    """Base class for all STS WebSocket query failures.

    Subclasses carry a precise failure category so callers can decide whether
    a failure is endpoint-level (connection/timeout) or application-level
    (empty response / parse failure) without inspecting the error message.
    Only endpoint-level failures count toward the batch-abort consecutive-
    failure threshold; application-level failures indicate the endpoint is
    reachable and should not penalise the abort counter.
    """


class StsEndpointUnreachableError(StsQueryError):
    """TCP/TLS connection to the STS endpoint could not be established.

    Covers: connection refused, DNS resolution failure, TLS handshake error,
    ``websockets.exceptions.WebSocketException`` at connect time, and any
    unclassified transport-level error.  A persistent run of these errors
    reliably indicates the endpoint is down or blocked.
    """


class StsSessionTimeoutError(StsQueryError):
    """WebSocket session or per-message timeout.

    Fires when:
    * ``asyncio.wait_for(ws.recv(), 30)`` expires mid-session, or
    * the outer per-patient hard timeout (``STS_PER_PATIENT_TIMEOUT_S``)
      fires before any result arrives.

    Likely causes: severe endpoint overload, rate limiting, or network
    packet loss on an established connection.  May recover if load drops.
    Counts toward the abort counter because the endpoint is functionally
    unreachable from the caller's perspective.
    """


class StsEmptyResponseError(StsQueryError):
    """The endpoint was reachable and responded, but no valid result arrived.

    Fires after all 80 ``recv()`` rounds without encountering an HTML block
    that contains a parseable STS result.  The server returned data for the
    session, but none matched the expected result format.  Does **not**
    count toward the abort counter — the endpoint is reachable.
    """


class StsParseError(StsQueryError):
    """An HTML block containing 'Mortality' was received but could not be parsed.

    The STS service returned a response fragment matching the expected
    pattern for a risk table, but ``_parse_html_response`` could not extract
    a numeric ``predmort`` value from it.  Does **not** count toward the abort
    counter — the transport and Shiny session both succeeded.
    """


class StsConnectError(StsEndpointUnreachableError):
    """TCP/TLS/DNS failure before the WebSocket handshake begins.

    Covers: ConnectionRefusedError, ConnectionResetError, OSError, and any
    other transport-layer exception raised while ``websockets.connect()`` is
    establishing the underlying TCP connection.  A persistent run of these
    errors reliably indicates the host is unreachable or the port is not
    listening.  Counts toward the abort counter.
    """


class StsHandshakeError(StsEndpointUnreachableError):
    """WebSocket upgrade or session initialisation failed.

    Covers: ``websockets.exceptions.InvalidHandshake``, unexpected HTTP
    responses during the WebSocket upgrade, and any
    ``websockets.exceptions.WebSocketException`` raised during the opening
    handshake (before the first application message).  Counts toward the
    abort counter.
    """


class StsConnectionClosedError(StsEndpointUnreachableError):
    """The WebSocket connection was closed by the server mid-session.

    Fires when ``websockets.exceptions.ConnectionClosed`` (or a subclass) is
    raised *after* the handshake succeeded and at least one message exchange
    has begun.  Distinct from ``StsConnectError`` (the handshake never
    completed) and ``StsEmptyResponseError`` (80 messages received normally,
    no result).  Counts toward the abort counter.
    """


# ---------------------------------------------------------------------------
# websockets exception class aliases (safe under HAS_WEBSOCKETS = False)
# ---------------------------------------------------------------------------
# Except-clause class expressions are evaluated at runtime, so referencing
# websockets.exceptions.* directly would raise NameError when the library is
# not installed.  We alias them here to a harmless sentinel so the except
# clauses in _query_sts_ws_inner compile and run regardless of availability.

if HAS_WEBSOCKETS:
    # The exception classes are exported from the top-level websockets namespace.
    # websockets.exceptions also re-exports them, but the top-level attributes
    # are more reliably available across websockets versions.
    _WS_ConnectionClosed: type = websockets.ConnectionClosed          # type: ignore[attr-defined]
    _WS_WebSocketException: type = websockets.WebSocketException      # type: ignore[attr-defined]
    _WS_InvalidHandshake: type = websockets.InvalidHandshake          # type: ignore[attr-defined]
else:
    # Unreachable sentinels — _query_sts_ws_inner is never called without websockets.
    _WS_ConnectionClosed = type("_WS_ConnectionClosed_sentinel", (), {})      # type: ignore[assignment]
    _WS_WebSocketException = type("_WS_WebSocketException_sentinel", (), {})  # type: ignore[assignment]
    _WS_InvalidHandshake = type("_WS_InvalidHandshake_sentinel", (), {})      # type: ignore[assignment]


def validate_sts_input(d: dict) -> list:
    """Validate an STS input dict produced by ``build_sts_input_from_row``.

    Returns a list of human-readable error strings.  An empty list means the
    payload is ready to send.  All four checks are always evaluated so the
    caller sees the full set of problems at once.

    Required fields checked:
    * ``age``   — must be a number in [1, 110]
    * ``procid``— must be one of "1"–"8"
    * ``gender``— must be "Male" or "Female"
    * ``status``— must be one of the four STS urgency values
    """
    errors: list = []

    # age: non-empty and numeric in plausible human range
    age_raw = d.get("age", "")
    if not age_raw:
        errors.append("age: missing")
    else:
        try:
            age_val = float(age_raw)
            if not (1 <= age_val <= 110):
                errors.append(f"age: out of range ({age_raw!r}); expected 1–110")
        except (ValueError, TypeError):
            errors.append(f"age: not numeric ({age_raw!r})")

    # procid: must be one of the eight STS procedure codes
    procid = d.get("procid", "")
    if procid not in _VALID_PROCIDS:
        errors.append(
            f"procid: invalid value ({procid!r}); expected one of {sorted(_VALID_PROCIDS)}"
        )

    # gender: STS accepts exactly two values
    gender = d.get("gender", "")
    if gender not in _VALID_GENDERS:
        errors.append(
            f"gender: invalid value ({gender!r}); expected 'Male' or 'Female'"
        )

    # status: must map to a known STS urgency level
    status = d.get("status", "")
    if status not in _VALID_STATUSES:
        errors.append(
            f"status: invalid value ({status!r}); expected one of {sorted(_VALID_STATUSES)}"
        )

    return errors


# ---------------------------------------------------------------------------
# WebSocket communication
# ---------------------------------------------------------------------------

def _create_websocket_init_data() -> dict:
    return {
        "prcvint": [],
        "Proc": [],
        "incidenc": [],
        "status": [],
        "gender": [],
        "racemulti": [],
        "payordata": [],
        "diabetes": [],
        "endocarditis": [],
        "ivdrugab": [],
        "alcohol": [],
        "tobaccouse": [],
        "chrlungd": [],
        "cvd": [],
        "heartfailtmg": [],
        "classnyh": [],
        "mcs": [],
        "cardsymptimeofadm": [],
        "miwhen": [],
        "numdisv": [],
        "vdinsufa": [],
        "vdinsufm": [],
        "vdinsuft": [],
        "arrhythatrfib": [],
        "arrhythafib": [],
        "arrhythaflutter": [],
        "arrhythvv": [],
        "arrhythsss": [],
        "arrhythsecond": [],
        "arrhyththird": [],
        "prvalveproc": [],
        "pocpci": [],
        "pocint": [],
        "tab": "Clinical Summary",
        "decline:shiny.action": 0,
        "reset:shiny.action": 0,
        "copybuttonestimates:shiny.action": 0,
        "copybuttonsummary:shiny.action": 0,
        "vstrpr": False,
        **{field: False for field in BOOLEAN_FIELDS},
        **{field: False for field in PRESENCE_BOOLEAN_FIELDS},
        "ageN:shiny.number": None,
        "heightN:shiny.number": None,
        "weightN:shiny.number": None,
        "BMI:shiny.number": None,
        "creatlstN:shiny.number": None,
        "hctN:shiny.number": None,
        "wbcN:shiny.number": None,
        "plateletsN:shiny.number": None,
        "medadpidis:shiny.number": None,
        "hdef:shiny.number": None,
        ".clientdata_output_errorMessage_hidden": False,
        ".clientdata_output_text2_hidden": False,
        ".clientdata_output_summary_hidden": False,
        ".clientdata_pixelratio": 1,
        ".clientdata_url_protocol": "https:",
        ".clientdata_url_hostname": "acsdriskcalc.research.sts.org",
        ".clientdata_url_port": "",
        ".clientdata_url_pathname": "/",
        ".clientdata_url_search": "",
        ".clientdata_url_hash_initial": "",
        ".clientdata_url_hash": "",
        ".clientdata_singletons": "add739c82ab207ed2c80be4b7e4b181525eb7a75",
    }


def _prepare_ws_messages(sts_dict: dict) -> tuple:
    init_data = _create_websocket_init_data()
    update_data = {}

    # Procedure
    procid = sts_dict.get("procid", "")
    if procid and int(procid) in PROCID_TO_PROC:
        update_data["Proc"] = [PROCID_TO_PROC[int(procid)]]

    # Demographics
    if sts_dict.get("age"):
        try:
            update_data["ageN:shiny.number"] = int(float(sts_dict["age"]))
        except (ValueError, TypeError):
            pass
    if sts_dict.get("gender"):
        update_data["gender"] = [sts_dict["gender"]]
    if sts_dict.get("status"):
        update_data["status"] = [sts_dict["status"]]
    if sts_dict.get("incidenc"):
        update_data["incidenc"] = [sts_dict["incidenc"]]

    # Biometrics
    h_val = _safe_float(sts_dict.get("heightcm"))
    w_val = _safe_float(sts_dict.get("weightkg"))
    if h_val:
        update_data["heightN:shiny.number"] = h_val
    if w_val:
        update_data["weightN:shiny.number"] = w_val
    if h_val and w_val:
        update_data["BMI:shiny.number"] = round(w_val / ((h_val / 100) ** 2), 2)

    # Labs
    lab_map = {
        "creatlst": ("creatlstN:shiny.number", float),
        "hct": ("hctN:shiny.number", int),
        "wbc": ("wbcN:shiny.number", float),
        "platelets": ("plateletsN:shiny.number", int),
        "hdef": ("hdef:shiny.number", float),
    }
    for sts_field, (ws_field, converter) in lab_map.items():
        val = sts_dict.get(sts_field, "")
        if val:
            try:
                update_data[ws_field] = converter(float(val))
            except (ValueError, TypeError):
                pass

    # Boolean fields
    for field in BOOLEAN_FIELDS:
        if sts_dict.get(field) == "Yes":
            update_data[field] = True

    for field in PRESENCE_BOOLEAN_FIELDS:
        if sts_dict.get(field):
            update_data[field] = True

    # Array fields
    for sts_field, ws_field in ARRAY_FIELD_MAPPINGS.items():
        val = sts_dict.get(sts_field, "")
        if val:
            update_data[ws_field] = [val]

    # Race
    race_items = []
    if sts_dict.get("raceblack") == "Yes":
        race_items.append("Black/African American")
    if sts_dict.get("raceasian") == "Yes":
        race_items.append("Asian")
    if race_items:
        update_data["racemulti"] = race_items

    # CVD
    cvd_items = []
    if sts_dict.get("cva") == "Yes":
        cvd_items.append("CVA")
    if cvd_items:
        update_data["cvd"] = cvd_items

    # Endocarditis
    if sts_dict.get("infendo") == "Yes":
        update_data["endocarditis"] = ["Active"]

    init_msg = '{"method":"init","data":' + json.dumps(init_data) + '}'
    update_msg = '{"method":"update","data":' + json.dumps(update_data) + '}'
    return init_msg, update_msg


def _parse_html_response(html_content: str) -> Dict[str, float]:
    result = {}
    td_contents = re.findall(r'<td[^>]*>(.*?)</td>', html_content, re.DOTALL)

    i = 0
    while i < len(td_contents) - 1:
        label = html_mod.unescape(re.sub(r'<[^>]+>', '', td_contents[i])).strip()

        for known_label, key in STS_RESULT_KEYS.items():
            if known_label in label:
                value_cell = re.sub(r'<[^>]+>', '', td_contents[i + 1]).strip()
                pct_match = re.search(r'([\d.]+)\s*%', value_cell)
                if pct_match:
                    result[key] = float(pct_match.group(1)) / 100.0
                i += 2
                break
        else:
            i += 1

    return result


def _extract_text2_html(msg_data: dict) -> Optional[str]:
    """Extract the text2 HTML value from a Shiny WebSocket message.

    Handles two protocol variants:
      • Shiny ≤1.6 (legacy): ``{"values": {"text2": "<html>"}}``
      • Shiny ≥1.7 / modern: ``{"method": "upd", "data": {"output": {"text2": {"html": "..."}}}}``
    Also handles the rare case where text2 is a plain string at the top level.
    """
    # Modern Shiny: method=upd, data.output.text2
    method = msg_data.get("method")
    if method == "upd":
        output = msg_data.get("data", {}).get("output", {})
        text2 = output.get("text2")
        if isinstance(text2, dict):
            return text2.get("html")
        if isinstance(text2, str):
            return text2

    # Legacy Shiny: values.text2
    values = msg_data.get("values", {})
    if isinstance(values, dict):
        text2 = values.get("text2")
        if isinstance(text2, dict):
            return text2.get("html")
        if isinstance(text2, str):
            return text2

    # Top-level text2 (uncommon but seen in some Shiny proxies)
    text2 = msg_data.get("text2")
    if isinstance(text2, dict):
        return text2.get("html")
    if isinstance(text2, str):
        return text2

    return None


async def _query_sts_ws_inner(sts_input: dict) -> Dict[str, float]:
    """Core WebSocket query — called exclusively through _query_sts_ws.

    Returns a dict with STS risk fields on success.

    Raises:
        StsConnectError: TCP/DNS failure before the WebSocket handshake.
        StsHandshakeError: WebSocket upgrade or session initialisation failed.
        StsConnectionClosedError: Server closed the connection mid-session.
        StsEndpointUnreachableError: Other unclassified transport error.
        StsSessionTimeoutError: ``ws.recv()`` timed out mid-session.
        StsEmptyResponseError: 80 messages received without a parseable result.
        StsParseError: HTML result block received but ``predmort`` not extractable.
    """
    init_msg, update_msg = _prepare_ws_messages(sts_input)

    # Phase tracker: updated as the connection progresses so the outer except
    # clauses can raise the most specific subclass.
    # Values: "connect" → "pre_send" → "recv"
    _phase = "connect"

    try:
        async with websockets.connect(
            WS_URL,
            additional_headers=WS_HEADERS,
            open_timeout=30,
            close_timeout=10,
        ) as ws:
            _phase = "pre_send"
            await ws.send(init_msg)
            await asyncio.sleep(1.0)
            await ws.send(update_msg)
            _phase = "recv"

            _seen_methods: set = set()
            for _ in range(80):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                except asyncio.TimeoutError:
                    _sts_log.debug(
                        "STS Score ws_timeout (StsSessionTimeoutError) — methods seen: %s",
                        sorted(_seen_methods) or "(none)",
                    )
                    raise StsSessionTimeoutError(
                        f"recv timeout after 30 s; methods seen: "
                        f"{sorted(_seen_methods) or '(none)'}"
                    )

                try:
                    msg_data = json.loads(msg)
                except (json.JSONDecodeError, ValueError):
                    continue

                _seen_methods.add(msg_data.get("method", "(no method)"))

                try:
                    html_val = _extract_text2_html(msg_data)

                    if html_val and ("Operative Mortality" in html_val or "Mortality" in html_val):
                        result = _parse_html_response(html_val)
                        if result and "predmort" in result:
                            return result
                        # HTML received but predmort not extractable — raise for classification
                        _sts_log.warning(
                            "STS Score html_parse_failure (StsParseError): "
                            "found Mortality HTML but got keys=%s; snippet: %.120s",
                            list(result.keys()),
                            html_val[:120],
                        )
                        raise StsParseError(
                            f"html_parse_failure: keys={list(result.keys())}; "
                            f"snippet={html_val[:80]!r}"
                        )
                except StsQueryError:
                    raise  # propagate typed errors; do not suppress
                except (AttributeError, KeyError):
                    continue

            _sts_log.debug(
                "STS Score ws_no_result (StsEmptyResponseError) — methods seen: %s",
                sorted(_seen_methods) or "(none)",
            )
            raise StsEmptyResponseError(
                f"80 messages received without a valid result; "
                f"methods seen: {sorted(_seen_methods) or '(none)'}"
            )

    except StsQueryError:
        raise  # propagate typed exceptions unchanged
    except _WS_ConnectionClosed as e:
        # Server closed the WebSocket after the handshake completed.
        _sts_log.debug(
            "STS Score ws_closed (StsConnectionClosedError) phase=%s: %s: %s",
            _phase, type(e).__name__, e,
        )
        raise StsConnectionClosedError(
            f"ConnectionClosed during {_phase}: {type(e).__name__}: {e}"
        ) from e
    except _WS_InvalidHandshake as e:
        _sts_log.debug(
            "STS Score ws_handshake_failed (StsHandshakeError): %s: %s",
            type(e).__name__, e,
        )
        raise StsHandshakeError(f"{type(e).__name__}: {e}") from e
    except (OSError, ConnectionRefusedError, ConnectionResetError) as e:
        _sts_log.debug(
            "STS Score connect_failed (StsConnectError) phase=%s: %s: %s",
            _phase, type(e).__name__, e,
        )
        raise StsConnectError(
            f"{type(e).__name__} at {_phase}: {e}"
        ) from e
    except Exception as e:
        # Unclassified transport error: use phase to pick the best subclass.
        if _phase == "connect":
            _sts_log.debug(
                "STS Score connect_failed (StsConnectError) unclassified: %s: %s",
                type(e).__name__, e,
            )
            raise StsConnectError(f"{type(e).__name__}: {e}") from e
        _sts_log.debug(
            "STS Score endpoint_unreachable (StsEndpointUnreachableError) "
            "phase=%s: %s: %s",
            _phase, type(e).__name__, e,
        )
        raise StsEndpointUnreachableError(
            f"{type(e).__name__} at {_phase}: {e}"
        ) from e


async def _query_sts_ws(sts_input: dict) -> Dict[str, float]:
    """Query the STS Score WebSocket with a hard per-patient timeout.

    Wraps ``_query_sts_ws_inner`` in ``asyncio.wait_for`` so a single patient
    can never stall for more than ``STS_PER_PATIENT_TIMEOUT_S`` seconds regardless
    of how many WebSocket messages the server sends.

    Raises:
        StsQueryError subclasses from ``_query_sts_ws_inner``, plus
        StsSessionTimeoutError when the outer hard timeout fires.
    """
    try:
        return await asyncio.wait_for(
            _query_sts_ws_inner(sts_input),
            timeout=STS_PER_PATIENT_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        _sts_log.warning(
            "STS Score per-patient hard timeout (%ds) exceeded — "
            "raising StsSessionTimeoutError",
            STS_PER_PATIENT_TIMEOUT_S,
        )
        raise StsSessionTimeoutError(
            f"per-patient hard timeout ({STS_PER_PATIENT_TIMEOUT_S}s) exceeded"
        )
    except StsQueryError:
        raise  # propagate typed exceptions unchanged


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine, handling the case where an event loop already exists (e.g. Streamlit)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop (e.g. Streamlit) — use a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def _sts_fetch_once(sts_input: dict) -> Dict[str, float]:
    """Single fetch attempt against the STS Score WebSocket endpoint.

    Returns an empty dict on any failure. Used as the ``fetch_func`` for
    the STS Score cache layer, which owns retries and validation.
    """
    if not HAS_WEBSOCKETS:
        return {}
    try:
        return _run_async(_query_sts_ws(sts_input)) or {}
    except Exception:
        return {}


def calculate_sts(
    row: dict,
    patient_id: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, float]:
    """Calculate STS Score risk endpoints for a single patient row.

    Phase 2: by default this routes through the persistent STS Score
    cache (``sts_cache``). The cache handles:
      - input hashing (clinically relevant STS Score fields only)
      - 14-day TTL revalidation
      - integration-version invalidation
      - response validation + retry
      - stale fallback to a previous valid entry
      - structured execution logging

    The return value remains the plain result dict so legacy callers
    continue to work. The full structured ``ExecutionRecord`` is
    appended to ``calculate_sts.last_execution_log`` for inspection.

    Args:
        row: dict with CSV column names or app display names.
        patient_id: stable identifier for the patient (used for the
            patient-level stale fallback index). May be None.
        use_cache: if False, bypass the STS Score cache entirely
            and perform a one-shot fetch (legacy behavior).

    Returns:
        Dict with keys like 'predmort', 'predmm', etc.
        Values are decimals (0.0646 = 6.46%).
        Empty dict if the STS Score query failed and no fallback exists.
    """
    if not HAS_WEBSOCKETS:
        return {}

    try:
        sts_input = build_sts_input_from_row(row)
    except Exception as e:
        _sts_log.warning("STS Score mapping_failure patient=%s: %s", patient_id, e)
        if HAS_STS_CACHE:
            calculate_sts.last_execution_log.append(
                _sts_cache.make_failed_record(
                    str(patient_id) if patient_id is not None else None,
                    None,
                    "build_input",
                    f"mapping_failure: {e}",
                )
            )
        return {}

    if not (use_cache and HAS_STS_CACHE):
        try:
            return _run_async(_query_sts_ws(sts_input))
        except Exception:
            return {}

    record = _sts_cache.get_cached_or_fetch(
        sts_input,
        patient_id=patient_id,
        fetch_func=_sts_fetch_once,
        max_retries=4,
    )
    calculate_sts.last_execution_log.append(record)
    return record.result


# Execution log shared across calls to calculate_sts().
# Callers (e.g. app.py) can read and clear this list as needed.
calculate_sts.last_execution_log = []  # type: ignore[attr-defined]


async def _calculate_sts_chunk_async(
    rows_with_indices: list,
    max_concurrent: int = 2,
    max_retries: int = 2,
    failure_log: list = None,
    patient_timeout_s: Optional[int] = None,
) -> dict:
    """Calculate STS for a chunk of rows. Returns {index: result_dict}.

    Args:
        patient_timeout_s: if set, the TOTAL wall-clock budget for a single
            patient across ALL retry attempts.  Each individual ``_query_sts_ws``
            call already carries its own per-attempt timeout; this outer cap
            ensures that the combined time (query + retries + back-off sleeps)
            never exceeds ``patient_timeout_s`` seconds.  When this fires the
            patient is recorded as failed in ``failure_log``.  Pass
            ``STS_PER_PATIENT_TIMEOUT_S`` to make "90 s per patient" literally
            true regardless of retry count.
    """
    sem = asyncio.Semaphore(max_concurrent)
    results: dict = {}

    async def _calc_one(idx: int, row: dict):
        patient_name = row.get("Name", row.get("Nome", f"Row {idx+1}")) if row else f"Row {idx+1}"
        surgery = row.get("Surgery", row.get("Cirurgia", "?")) if row else "?"
        if not row:
            results[idx] = {}
            if failure_log is not None:
                failure_log.append({"idx": idx, "name": patient_name, "reason": "empty row"})
            return
        try:
            sts_input = build_sts_input_from_row(row)
        except Exception as e:
            results[idx] = {}
            if failure_log is not None:
                failure_log.append({"idx": idx, "name": patient_name, "surgery": surgery, "reason": f"build_input error: {e}"})
            return
        last_err: Optional[str] = None
        last_exc_type: Optional[str] = None
        _failure_type = "endpoint"  # default; refined below on each caught exception
        _attempt_log: list = []
        for attempt in range(max_retries + 1):
            _t0 = time.monotonic()
            async with sem:
                try:
                    result = await _query_sts_ws(sts_input)
                    last_err = None
                    last_exc_type = None
                    _attempt_log.append({
                        "attempt": attempt,
                        "elapsed_s": round(time.monotonic() - _t0, 2),
                        "success": True,
                    })
                except StsParseError as e:
                    result = {}
                    last_err = str(e)
                    last_exc_type = "StsParseError"
                    _failure_type = "parse_error"
                    _attempt_log.append({
                        "attempt": attempt,
                        "elapsed_s": round(time.monotonic() - _t0, 2),
                        "success": False,
                        "exc_class": last_exc_type,
                        "failure_type": _failure_type,
                    })
                    # Parse failures are deterministic for this input — skip retries.
                    break
                except StsEmptyResponseError as e:
                    result = {}
                    last_err = str(e)
                    last_exc_type = "StsEmptyResponseError"
                    _failure_type = "empty_response"
                    _attempt_log.append({
                        "attempt": attempt,
                        "elapsed_s": round(time.monotonic() - _t0, 2),
                        "success": False,
                        "exc_class": last_exc_type,
                        "failure_type": _failure_type,
                    })
                except (StsEndpointUnreachableError, StsSessionTimeoutError) as e:
                    result = {}
                    last_err = str(e)
                    last_exc_type = type(e).__name__
                    _failure_type = "endpoint"
                    _attempt_log.append({
                        "attempt": attempt,
                        "elapsed_s": round(time.monotonic() - _t0, 2),
                        "success": False,
                        "exc_class": last_exc_type,
                        "failure_type": _failure_type,
                    })
                except Exception as e:
                    result = {}
                    last_err = str(e)
                    last_exc_type = type(e).__name__
                    _failure_type = "endpoint"  # unknown → conservative
                    _attempt_log.append({
                        "attempt": attempt,
                        "elapsed_s": round(time.monotonic() - _t0, 2),
                        "success": False,
                        "exc_class": last_exc_type,
                        "failure_type": _failure_type,
                    })
            if result and "predmort" in result:
                results[idx] = result
                return
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))
        results[idx] = result or {}
        if failure_log is not None:
            reason = f"no result after {max_retries+1} attempts"
            if last_err:
                reason += f" (last error: {last_err})"
            failure_log.append({
                "idx": idx, "name": patient_name, "surgery": surgery,
                "reason": reason,
                "exception_type": last_exc_type,
                "failure_type": _failure_type,
                "attempt_log": _attempt_log,
            })

    async def _calc_one_bounded(idx: int, row: dict):
        """Wraps _calc_one with a total per-patient wall-clock cap."""
        patient_name = row.get("Name", row.get("Nome", f"Row {idx+1}")) if row else f"Row {idx+1}"
        try:
            await asyncio.wait_for(_calc_one(idx, row), timeout=patient_timeout_s)
        except asyncio.TimeoutError:
            if idx not in results:
                results[idx] = {}
            if failure_log is not None:
                failure_log.append({
                    "idx": idx,
                    "name": patient_name,
                    "reason": (
                        f"per-patient timeout exceeded ({patient_timeout_s} s total "
                        "across all retry attempts)"
                    ),
                    "exception_type": "asyncio.TimeoutError",
                    "failure_type": "per_patient_timeout",
                })

    if patient_timeout_s is not None:
        tasks = [_calc_one_bounded(idx, row) for idx, row in rows_with_indices]
    else:
        tasks = [_calc_one(idx, row) for idx, row in rows_with_indices]
    await asyncio.gather(*tasks, return_exceptions=True)
    return results


def _calculate_sts_batch_legacy(
    rows: list,
    progress_callback=None,
    chunk_size: int = 5,
) -> tuple:
    """Legacy uncached batch path (preserved for use_cache=False).

    Returns (results, last_error, failure_log).
    """
    all_results: list = [{} for _ in rows]
    total = len(rows)
    done = 0
    last_error = None
    failure_log: list = []

    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk = [(i, rows[i]) for i in range(chunk_start, chunk_end)]
        try:
            chunk_results = _run_async(
                _calculate_sts_chunk_async(chunk, failure_log=failure_log)
            )
            for idx, result in chunk_results.items():
                all_results[idx] = result
        except Exception as e:
            last_error = e
        done = chunk_end
        if progress_callback:
            try:
                progress_callback(done, total)
            except Exception:
                pass

    return all_results, last_error, failure_log


def calculate_sts_batch(
    rows: list,
    progress_callback=None,
    phase_callback=None,
    chunk_size: int = 5,
    patient_ids: Optional[List[Optional[str]]] = None,
    use_cache: bool = True,
    abort_event=None,
    chunk_start_callback=None,
    chunk_done_callback=None,
) -> list:
    """Calculate STS Score risk scores for a batch of patient rows.

    Phase 2: by default this routes every row through the persistent
    STS Score cache. The flow is:

      1. Pre-check: compute each row's input hash and look it up first in the
         in-memory session cache, then on disk.  Rows already resolved are
         returned immediately as ``cached`` without touching the network.
      2. Fetch: the remaining rows are queried via the existing chunked
         async WebSocket path (preserving first-run performance).
      3. Post-process: each fetched result is validated; valid results
         are persisted as ``fresh`` (or ``refreshed`` if an expired prior
         entry existed). Invalid / missing results trigger the stale
         fallback policy and produce a ``stale_fallback`` or ``failed``
         execution record.

    All per-row statuses are appended to
    ``calculate_sts_batch.last_execution_log`` as ``ExecutionRecord``
    objects. The returned list preserves input order and contains the
    effective result dict for each row (possibly from a stale fallback,
    or empty on total failure), keeping the existing call-site contract.

    Args:
        rows: list of dicts, each with CSV column names or app display names.
        progress_callback: optional callable(current, total).
        phase_callback: optional callable(phase_num, phase_total, label, detail)
            fired at each major phase transition (1-based; phase_total == 4).
            Phases: 1=checking cache, 2=identifying misses, 3=querying web,
            4=validating and consolidating.
        chunk_size: number of patients per async chunk (fetch path only).
        patient_ids: optional list of stable per-row patient identifiers
            (same length as ``rows``). Used for cross-hash stale fallback.
        use_cache: if False, bypass the STS Score cache entirely and run
            the legacy chunked async path (no hashing, no persistence,
            no execution log).
        abort_event: optional ``threading.Event``.  When set (by an external
            thread or UI callback), the fetch loop stops after the current
            chunk completes and sets ``_batch_aborted = True``.  Already-fetched
            results and cache hits are preserved.  A ``None`` value means no
            cooperative cancellation is requested.
        chunk_start_callback: optional callable(patient_idx: int, total_pending: int,
            patient_id: str | None) fired in the worker thread immediately before
            the network query for each chunk/patient begins.  When chunk_size==1,
            this fires once per patient and can be used to record the patient-level
            start timestamp in a shared progress dict for UI display.  Exceptions
            in the callback are silently swallowed so they cannot abort the batch.
        chunk_done_callback: optional callable(patient_idx: int, total_pending: int,
            success: bool) fired immediately after each chunk/patient network query
            completes.  ``success`` is True when at least one row in the chunk
            returned a valid ``predmort`` value.  With chunk_size==1 this fires
            once per patient and lets the caller track running completed/failed
            counts in a shared dict for live UI display.  Exceptions silently
            swallowed.
    """
    # Reset public diagnostic attributes up front so the UI always sees
    # the current call's state even on early returns.
    calculate_sts_batch.last_error = None
    calculate_sts_batch.failure_log = []
    calculate_sts_batch.last_execution_log = []
    calculate_sts_batch._batch_aborted = False           # type: ignore[attr-defined]
    calculate_sts_batch._abort_before_query_count = 0    # type: ignore[attr-defined]
    calculate_sts_batch.chunk_log = []                    # type: ignore[attr-defined]

    if not HAS_WEBSOCKETS:
        return [{} for _ in rows]

    total = len(rows)
    if total == 0:
        return []

    # Normalize patient_ids to the same length as rows.
    pids: List[Optional[str]]
    if patient_ids is None:
        pids = [None] * total
    else:
        pids = list(patient_ids)
        if len(pids) < total:
            pids = pids + [None] * (total - len(pids))
        elif len(pids) > total:
            pids = pids[:total]

    # --- Legacy path ---
    if not (use_cache and HAS_STS_CACHE):
        results, last_error, failure_log = _calculate_sts_batch_legacy(
            rows, progress_callback=progress_callback, chunk_size=chunk_size
        )
        calculate_sts_batch.last_error = last_error
        calculate_sts_batch.failure_log = failure_log
        return results

    # --- Cached path ---
    results: list = [{} for _ in rows]
    # Pre-allocated and indexed by eligible position (0-based).  Every row
    # is assigned execution_log[i] = rec in Phase A or Phase C so that
    # exec_log[eli_pos] always returns the correct record for that patient.
    # Appending sequentially would interleave Phase-A resolutions (cache hits /
    # build failures) before Phase-C fetch records, breaking positional lookup.
    execution_log: list = [None] * total
    failure_log: list = []
    last_error: Optional[Exception] = None

    # Phase A: cache pre-check (no network).
    # Lookup order: 1. in-memory/session cache  2. persistent disk cache
    # Items in `pending` are (orig_idx, row, sts_input, input_hash, prior_entry, patient_id_str).
    if phase_callback is not None:
        try:
            phase_callback(1, 4, "checking cache", f"0/{total} checked")
        except Exception:
            pass
    pending: list = []
    for i, row in enumerate(rows):
        pid = pids[i]
        pid_str = str(pid) if pid is not None else None

        if not row:
            rec = _sts_cache.make_failed_record(
                pid_str, None, "build_input", "empty_row"
            )
            execution_log[i] = rec
            failure_log.append({
                "idx": i, "patient_id": pid_str, "status": rec.status,
                "stage": rec.stage, "reason": rec.reason,
                "retry_attempted": False, "used_previous_cache": False,
            })
            continue

        try:
            sts_input = build_sts_input_from_row(row)
        except Exception as e:
            rec = _sts_cache.make_failed_record(
                pid_str, None, "build_input", f"mapping_failure: {e}"
            )
            execution_log[i] = rec
            failure_log.append({
                "idx": i, "patient_id": pid_str, "status": rec.status,
                "stage": rec.stage, "reason": rec.reason,
                "retry_attempted": False, "used_previous_cache": False,
            })
            continue

        validation_errors = validate_sts_input(sts_input)
        if validation_errors:
            reason = "payload_invalid: " + "; ".join(validation_errors)
            rec = _sts_cache.make_failed_record(
                pid_str, None, "build_input", reason
            )
            execution_log[i] = rec
            failure_log.append({
                "idx": i, "patient_id": pid_str, "status": rec.status,
                "stage": rec.stage, "reason": rec.reason,
                "retry_attempted": False, "used_previous_cache": False,
            })
            continue

        try:
            input_hash = _sts_cache.compute_input_hash(sts_input)
        except Exception as e:
            rec = _sts_cache.make_failed_record(
                pid_str, None, "build_input", f"input_hash_error: {e}"
            )
            execution_log[i] = rec
            failure_log.append({
                "idx": i, "patient_id": pid_str, "status": rec.status,
                "stage": rec.stage, "reason": rec.reason,
                "retry_attempted": False, "used_previous_cache": False,
            })
            continue

        # 1. In-memory/session cache (fastest path — skips JSON file read).
        mem_entry = _sts_memory_cache.get(input_hash)
        if mem_entry is not None:
            rec = _sts_cache.make_cache_hit_record(mem_entry, pid_str, input_hash)
            execution_log[i] = rec
            if pid_str:
                _sts_cache.remember_patient_hash(pid_str, input_hash)
            results[i] = dict(mem_entry.get("result") or {})
            continue

        # 2. Persistent disk cache.
        entry = _sts_cache.load_entry(input_hash)
        entry_usable = (
            entry is not None
            and entry.get("integration_version") == _sts_cache.STS_SCORE_INTEGRATION_VERSION
            and _sts_cache.is_valid_result(entry.get("result"))
            and not _sts_cache.is_expired(entry)
        )
        if entry_usable:
            _sts_memory_cache[input_hash] = entry  # promote to in-memory cache
            rec = _sts_cache.make_cache_hit_record(entry, pid_str, input_hash)
            execution_log[i] = rec
            if pid_str:
                _sts_cache.remember_patient_hash(pid_str, input_hash)
            results[i] = dict(entry.get("result") or {})
            continue

        pending.append((i, row, sts_input, input_hash, entry, pid_str))

    # Report progress after the cache-only phase.
    done_base = total - len(pending)
    if phase_callback is not None:
        try:
            phase_callback(
                2, 4, "identifying cache misses",
                f"{done_base} cache hit{'s' if done_base != 1 else ''}, "
                f"{len(pending)} miss{'es' if len(pending) != 1 else ''}",
            )
        except Exception:
            pass
    if progress_callback:
        try:
            progress_callback(done_base, total)
        except Exception:
            pass

    # Phase B: fetch the pending rows using the existing chunked async path.
    if phase_callback is not None:
        try:
            phase_callback(
                3, 4, "querying web calculator",
                f"{len(pending)} patient{'s' if len(pending) != 1 else ''} to fetch",
            )
        except Exception:
            pass
    fetched: Dict[int, Dict[str, float]] = {}
    _batch_aborted = False  # set True if consecutive failures or user abort
    # ``_abort_before_query`` tracks which *local* (pending-list) indices were
    # marked as unqueried because the batch was aborted before their turn.
    # Phase C uses this set to distinguish "never reached" from "tried and failed".
    _abort_before_query: set = set()
    # Capture the consecutive-failure count that triggered the abort so Phase C
    # can embed it in the per-row failure reason.
    _abort_consecutive_failures: int = 0
    _abort_reason_type: str = ""  # "consecutive_failures" | "user_cancel"
    # Structured per-chunk execution log.  Each dict records the outcome of one
    # chunk (with chunk_size=1, one chunk == one patient).
    chunk_log: list = []
    if pending:
        local_pairs = [
            (local_i, pending[local_i][1]) for local_i in range(len(pending))
        ]
        inner_failure_log: list = []
        _consecutive_failures = 0
        _chunk_index = 0
        for chunk_start in range(0, len(local_pairs), chunk_size):
            # Check cooperative abort (user-requested cancellation).
            if abort_event is not None and abort_event.is_set():
                _sts_log.info(
                    "STS Score batch cooperative abort requested — "
                    "stopping after chunk %d; %d rows remain unqueried",
                    chunk_start // chunk_size,
                    len(local_pairs) - chunk_start,
                )
                _batch_aborted = True
                _abort_reason_type = "user_cancel"
                _abort_consecutive_failures = _consecutive_failures
                for local_i, _ in local_pairs[chunk_start:]:
                    fetched.setdefault(local_i, {})
                    _abort_before_query.add(local_i)
                break
            if _batch_aborted:
                # Fill remaining as empty so Phase C can handle them via fallback.
                for local_i, _ in local_pairs[chunk_start:]:
                    fetched.setdefault(local_i, {})
                    _abort_before_query.add(local_i)
                break
            chunk_end = min(chunk_start + chunk_size, len(local_pairs))
            chunk = local_pairs[chunk_start:chunk_end]
            chunk_patient_ids = [
                pending[local_i][5] for local_i, _ in chunk if local_i < len(pending)
            ]
            # Fire chunk_start_callback before the network query so the caller
            # (e.g. the temporal-validation polling handler) can record the
            # per-patient start timestamp.  With chunk_size==1 this fires once
            # per patient, making the "90 s per patient" timeout observable.
            if chunk_start_callback is not None:
                try:
                    _cb_local_i = local_pairs[chunk_start][0]
                    _cb_pid = pending[_cb_local_i][5] if _cb_local_i < len(pending) else None
                    chunk_start_callback(chunk_start, len(local_pairs), _cb_pid)
                except Exception:
                    pass
            chunk_had_success = False
            _chunk_exc_type: Optional[str] = None
            _chunk_exc_msg: Optional[str] = None
            # Record inner_failure_log cursor so we can isolate this chunk's entries.
            _inner_fl_start = len(inner_failure_log)
            try:
                chunk_results = _run_async(
                    _calculate_sts_chunk_async(
                        chunk,
                        max_retries=4,
                        failure_log=inner_failure_log,
                        # Global per-patient timeout: covers ALL retry attempts so
                        # "STS_PER_PATIENT_TIMEOUT_S seconds per patient" is
                        # literally true regardless of how many retries are attempted.
                        patient_timeout_s=STS_PER_PATIENT_TIMEOUT_S,
                    )
                )
                for local_i, result in chunk_results.items():
                    fetched[local_i] = result or {}
                    if result and "predmort" in result:
                        chunk_had_success = True
            except Exception as e:
                last_error = e
                _chunk_exc_type = type(e).__name__
                _chunk_exc_msg = str(e)
                _sts_log.warning("STS Score batch chunk error: %s", e)
                for local_i, _ in chunk:
                    fetched.setdefault(local_i, {})
            # Chunk success/failure counts for the structured log.
            _chunk_success = sum(
                1 for local_i, _ in chunk if fetched.get(local_i, {}).get("predmort") is not None
            )
            _chunk_failure = len(chunk) - _chunk_success
            # Fire chunk_done_callback so callers can track running
            # completed/failed counts for live progress display.
            if chunk_done_callback is not None:
                try:
                    chunk_done_callback(chunk_start, len(local_pairs), chunk_had_success)
                except Exception:
                    pass

            # ── Classify this chunk's failure type ──────────────────────────────
            # Endpoint-level failures (connection refused, timeouts) count toward
            # the consecutive-failure abort counter.  Application-level failures
            # (empty response, parse error) mean the endpoint IS reachable and must
            # NOT increment the counter — they will not benefit from aborting the batch.
            _chunk_entries = inner_failure_log[_inner_fl_start:]
            _chunk_endpoint_failure_count = sum(
                1 for _ifl in _chunk_entries
                if _ifl.get("failure_type", "endpoint") in ("endpoint", "per_patient_timeout")
            )
            # If inner_failure_log was not populated (either _run_async raised an outer
            # exception before _calculate_sts_chunk_async could append entries, or
            # _run_async was bypassed entirely — e.g. in tests), fall back to treating
            # all chunk members as endpoint failures (conservative default).
            if not chunk_had_success and not _chunk_entries:
                _chunk_endpoint_failure_count = len(chunk)
            # Primary failure type for the chunk log (None when any success).
            if chunk_had_success:
                _chunk_primary_failure_type: Optional[str] = None
            elif _chunk_endpoint_failure_count > 0:
                _chunk_primary_failure_type = "endpoint"
            elif _chunk_entries:
                _chunk_primary_failure_type = _chunk_entries[0].get("failure_type", "endpoint")
            else:
                _chunk_primary_failure_type = "endpoint"  # outer exception, no inner entries

            # ── Consecutive-failure abort logic ─────────────────────────────────
            _aborted_after_this_chunk = False
            _counted_toward_abort = False
            if not chunk_had_success:
                if _chunk_endpoint_failure_count > 0:
                    # Endpoint-level failure — advance the abort counter.
                    _consecutive_failures += 1
                    _counted_toward_abort = True
                    if _consecutive_failures >= STS_MAX_CONSECUTIVE_FAILURES:
                        _sts_log.warning(
                            "STS Score batch aborted after %d consecutive endpoint "
                            "chunk failures — endpoint may be unreachable; "
                            "%d rows remain unqueried",
                            _consecutive_failures,
                            len(local_pairs) - chunk_end,
                        )
                        _batch_aborted = True
                        _abort_reason_type = "consecutive_failures"
                        _abort_consecutive_failures = _consecutive_failures
                        _aborted_after_this_chunk = True
                    else:
                        # Progressive backoff: give a flapping endpoint time to recover.
                        # Runs in the worker thread so time.sleep is safe here.
                        import time as _tv_sleep_mod
                        _backoff_s = min(
                            STS_CONSECUTIVE_FAILURE_BACKOFF_BASE_S * _consecutive_failures,
                            STS_CONSECUTIVE_FAILURE_BACKOFF_MAX_S,
                        )
                        if _backoff_s > 0:
                            _sts_log.info(
                                "STS Score consecutive endpoint failure %d/%d — "
                                "waiting %ds before next patient",
                                _consecutive_failures,
                                STS_MAX_CONSECUTIVE_FAILURES,
                                _backoff_s,
                            )
                            _tv_sleep_mod.sleep(_backoff_s)
                else:
                    # Non-endpoint failure (empty response / parse error): the endpoint
                    # IS reachable, so reset the consecutive counter and do not backoff.
                    _sts_log.info(
                        "STS Score chunk %d: non-endpoint failure (type=%r) — "
                        "consecutive endpoint-failure counter reset (was %d)",
                        _chunk_index,
                        _chunk_primary_failure_type,
                        _consecutive_failures,
                    )
                    _consecutive_failures = 0
            else:
                _consecutive_failures = 0
            chunk_log.append({
                "chunk_index":              _chunk_index,
                "chunk_start":              chunk_start,
                "row_count":                len(chunk),
                "patient_ids":              chunk_patient_ids,
                "success_count":            _chunk_success,
                "failure_count":            _chunk_failure,
                "failure_type":             _chunk_primary_failure_type,
                "endpoint_failure_count":   _chunk_endpoint_failure_count,
                "counted_toward_abort":     _counted_toward_abort,
                "exception_type":           _chunk_exc_type,
                "exception_message":        _chunk_exc_msg,
                "aborted_after_this_chunk": _aborted_after_this_chunk,
            })
            _chunk_index += 1
            if progress_callback:
                try:
                    progress_callback(done_base + chunk_end, total)
                except Exception:
                    pass
    calculate_sts_batch._batch_aborted = _batch_aborted  # type: ignore[attr-defined]
    calculate_sts_batch._abort_before_query_count = len(_abort_before_query)  # type: ignore[attr-defined]
    calculate_sts_batch.chunk_log = chunk_log  # type: ignore[attr-defined]

    # Phase C: validate, persist, and classify each pending row.
    if phase_callback is not None:
        try:
            phase_callback(
                4, 4, "validating and consolidating results",
                f"{len(fetched)} result{'s' if len(fetched) != 1 else ''} fetched",
            )
        except Exception:
            pass
    for local_i, (orig_i, row, sts_input, input_hash, prior_entry, pid_str) in enumerate(pending):
        result = fetched.get(local_i, {}) or {}
        need_refresh = (
            prior_entry is not None
            and prior_entry.get("integration_version") == _sts_cache.STS_SCORE_INTEGRATION_VERSION
            and _sts_cache.is_valid_result(prior_entry.get("result"))
        )

        if _sts_cache.is_valid_result(result):
            _sts_cache.persist_fresh_result(sts_input, result, input_hash, pid_str)
            # Promote fresh result to in-memory cache for subsequent lookups
            # within the same session without a disk read.
            _sts_memory_cache[input_hash] = {
                "input_hash": input_hash,
                "integration_version": _sts_cache.STS_SCORE_INTEGRATION_VERSION,
                "result": dict(result),
                "created_ts": time.time(),
            }
            rec = _sts_cache.make_fresh_record(
                result, pid_str, input_hash,
                refreshed=need_refresh, retry_attempted=True,
            )
            execution_log[orig_i] = rec
            results[orig_i] = dict(result)
            continue

        # --- Batch-abort path ---------------------------------------------------
        # This row was never queried because the batch was aborted before reaching
        # it.  We still attempt a stale-cache fallback (requirement D: preserve
        # partial success) before marking it as unqueried.
        if local_i in _abort_before_query:
            if _abort_reason_type == "user_cancel":
                _abort_row_reason = (
                    "Batch cancelled by user; row not queried."
                )
            else:
                _abort_row_reason = (
                    f"Batch aborted after {_abort_consecutive_failures} consecutive "
                    f"chunk failures; row not queried."
                )
            fallback_entry = _sts_cache.find_stale_fallback(
                pid_str, input_hash, prior_entry
            )
            if fallback_entry is not None:
                # A stale result exists — use it so earlier cached data is not lost.
                rec = _sts_cache.make_stale_fallback_record(
                    fallback_entry, pid_str, input_hash,
                    f"batch_abort: {_abort_row_reason}", retry_attempted=False,
                )
                execution_log[orig_i] = rec
                failure_log.append({
                    "idx": orig_i, "patient_id": pid_str, "status": rec.status,
                    "stage": "batch_abort",
                    "reason": _abort_row_reason,
                    "retry_attempted": False, "used_previous_cache": True,
                })
                results[orig_i] = dict(fallback_entry.get("result") or {})
            else:
                rec = _sts_cache.make_failed_record(
                    pid_str, input_hash, "batch_abort",
                    _abort_row_reason, retry_attempted=False,
                )
                execution_log[orig_i] = rec
                failure_log.append({
                    "idx": orig_i, "patient_id": pid_str, "status": rec.status,
                    "stage": "batch_abort",
                    "reason": _abort_row_reason,
                    "retry_attempted": False, "used_previous_cache": False,
                })
                results[orig_i] = {}
            continue

        # --- Normal fetch-failure path ------------------------------------------
        fallback_entry = _sts_cache.find_stale_fallback(
            pid_str, input_hash, prior_entry
        )
        reason = "fetch_failed" if not result else "response_validation_failure"
        if fallback_entry is not None:
            rec = _sts_cache.make_stale_fallback_record(
                fallback_entry, pid_str, input_hash, reason, retry_attempted=True,
            )
            execution_log[orig_i] = rec
            failure_log.append({
                "idx": orig_i, "patient_id": pid_str, "status": rec.status,
                "stage": rec.stage, "reason": rec.reason,
                "retry_attempted": True, "used_previous_cache": True,
            })
            results[orig_i] = dict(fallback_entry.get("result") or {})
            continue

        rec = _sts_cache.make_failed_record(
            pid_str, input_hash, "fetch",
            f"{reason}; no fallback available", retry_attempted=True,
        )
        execution_log[orig_i] = rec
        failure_log.append({
            "idx": orig_i, "patient_id": pid_str, "status": rec.status,
            "stage": rec.stage, "reason": rec.reason,
            "retry_attempted": True, "used_previous_cache": False,
        })
        results[orig_i] = {}

    # ── Endpoint health summary (query-phase only; excludes Phase A cache hits) ──
    # Populated regardless of abort/success so the UI always has current counts.
    _n_queried_rows = len(pending) - len(_abort_before_query)
    _n_queried_with_score = sum(
        1 for local_i, (orig_i, *_rest) in enumerate(pending)
        if local_i not in _abort_before_query
        and (results[orig_i] or {}).get("predmort") is not None
    )
    # Count failures by exception subtype for detailed diagnostics.
    _failure_subtype_counts: Dict[str, int] = {}
    for _ifl in inner_failure_log if pending else []:
        _st = _ifl.get("exception_type") or "unknown"
        _failure_subtype_counts[_st] = _failure_subtype_counts.get(_st, 0) + 1

    calculate_sts_batch.endpoint_health_summary = {  # type: ignore[attr-defined]
        "n_eligible_for_fetch":      len(pending),
        "n_queried":                 _n_queried_rows,
        "n_queried_with_score":      _n_queried_with_score,
        "n_chunks_attempted":        len(chunk_log),
        "n_chunks_any_success":      sum(1 for cl in chunk_log if cl.get("success_count", 0) > 0),
        "n_chunks_all_failed":       sum(1 for cl in chunk_log if cl.get("success_count", 0) == 0),
        "n_chunks_endpoint_failure": sum(1 for cl in chunk_log if cl.get("endpoint_failure_count", 0) > 0),
        "n_rows_unqueried":          len(_abort_before_query),
        "abort_reason":              _abort_reason_type or None,
        "abort_endpoint_failures":   _abort_consecutive_failures if _batch_aborted else 0,
        "failure_subtype_counts":    _failure_subtype_counts,
    }

    calculate_sts_batch.last_error = last_error
    calculate_sts_batch.failure_log = failure_log
    calculate_sts_batch.last_execution_log = execution_log
    return results


calculate_sts_batch.last_error = None
calculate_sts_batch.failure_log = []
calculate_sts_batch.last_execution_log = []
calculate_sts_batch._batch_aborted = False          # type: ignore[attr-defined]
calculate_sts_batch._abort_before_query_count = 0   # type: ignore[attr-defined]
calculate_sts_batch.chunk_log = []                   # type: ignore[attr-defined]
calculate_sts_batch.endpoint_health_summary = {}     # type: ignore[attr-defined]
