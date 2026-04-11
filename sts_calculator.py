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
    return mapping.get(v, "Elective")


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


async def _query_sts_ws(sts_input: dict) -> Dict[str, float]:
    init_msg, update_msg = _prepare_ws_messages(sts_input)

    async with websockets.connect(
        WS_URL,
        additional_headers=WS_HEADERS,
        open_timeout=30,
        close_timeout=10,
    ) as ws:
        await ws.send(init_msg)
        await asyncio.sleep(1.0)
        await ws.send(update_msg)

        for _ in range(80):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                return {}

            try:
                msg_data = json.loads(msg)
                # Check both possible response structures
                values = msg_data.get("values", {})
                text2 = values.get("text2")
                html_val = None
                if isinstance(text2, dict):
                    html_val = text2.get("html")
                elif isinstance(text2, str):
                    html_val = text2

                if html_val and ("Operative Mortality" in html_val or "Mortality" in html_val):
                    result = _parse_html_response(html_val)
                    if result and "predmort" in result:
                        return result
            except (json.JSONDecodeError, AttributeError, KeyError):
                continue

    return {}


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
        max_retries=2,
    )
    calculate_sts.last_execution_log.append(record)
    return record.result


# Execution log shared across calls to calculate_sts().
# Callers (e.g. app.py) can read and clear this list as needed.
calculate_sts.last_execution_log = []  # type: ignore[attr-defined]


async def _calculate_sts_chunk_async(rows_with_indices: list, max_concurrent: int = 2, max_retries: int = 2, failure_log: list = None) -> dict:
    """Calculate STS for a chunk of rows. Returns {index: result_dict}."""
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
        last_err = None
        for attempt in range(max_retries + 1):
            async with sem:
                try:
                    result = await _query_sts_ws(sts_input)
                    last_err = None
                except Exception as e:
                    result = {}
                    last_err = str(e)
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
            failure_log.append({"idx": idx, "name": patient_name, "surgery": surgery, "reason": reason})

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
    chunk_size: int = 5,
    patient_ids: Optional[List[Optional[str]]] = None,
    use_cache: bool = True,
) -> list:
    """Calculate STS Score risk scores for a batch of patient rows.

    Phase 2: by default this routes every row through the persistent
    STS Score cache. The flow is:

      1. Pre-check: compute each row's input hash and look it up on disk.
         Rows with a valid, in-TTL, version-matching cached entry are
         resolved immediately as ``cached`` without touching the network.
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
        chunk_size: number of patients per async chunk (fetch path only).
        patient_ids: optional list of stable per-row patient identifiers
            (same length as ``rows``). Used for cross-hash stale fallback.
        use_cache: if False, bypass the STS Score cache entirely and run
            the legacy chunked async path (no hashing, no persistence,
            no execution log).
    """
    # Reset public diagnostic attributes up front so the UI always sees
    # the current call's state even on early returns.
    calculate_sts_batch.last_error = None
    calculate_sts_batch.failure_log = []
    calculate_sts_batch.last_execution_log = []

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
    execution_log: list = []
    failure_log: list = []
    last_error: Optional[Exception] = None

    # Phase A: cache pre-check (no network).
    # Items in `pending` are (orig_idx, row, sts_input, input_hash, prior_entry, patient_id_str).
    pending: list = []
    for i, row in enumerate(rows):
        pid = pids[i]
        pid_str = str(pid) if pid is not None else None

        if not row:
            rec = _sts_cache.make_failed_record(
                pid_str, None, "build_input", "empty_row"
            )
            execution_log.append(rec)
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
            execution_log.append(rec)
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
            execution_log.append(rec)
            failure_log.append({
                "idx": i, "patient_id": pid_str, "status": rec.status,
                "stage": rec.stage, "reason": rec.reason,
                "retry_attempted": False, "used_previous_cache": False,
            })
            continue

        entry = _sts_cache.load_entry(input_hash)
        entry_usable = (
            entry is not None
            and entry.get("integration_version") == _sts_cache.STS_SCORE_INTEGRATION_VERSION
            and _sts_cache.is_valid_result(entry.get("result"))
            and not _sts_cache.is_expired(entry)
        )
        if entry_usable:
            rec = _sts_cache.make_cache_hit_record(entry, pid_str, input_hash)
            execution_log.append(rec)
            if pid_str:
                _sts_cache.remember_patient_hash(pid_str, input_hash)
            results[i] = dict(entry.get("result") or {})
            continue

        pending.append((i, row, sts_input, input_hash, entry, pid_str))

    # Report progress after the cache-only phase.
    done_base = total - len(pending)
    if progress_callback:
        try:
            progress_callback(done_base, total)
        except Exception:
            pass

    # Phase B: fetch the pending rows using the existing chunked async path.
    fetched: Dict[int, Dict[str, float]] = {}
    if pending:
        local_pairs = [
            (local_i, pending[local_i][1]) for local_i in range(len(pending))
        ]
        inner_failure_log: list = []
        for chunk_start in range(0, len(local_pairs), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(local_pairs))
            chunk = local_pairs[chunk_start:chunk_end]
            try:
                chunk_results = _run_async(
                    _calculate_sts_chunk_async(chunk, failure_log=inner_failure_log)
                )
                for local_i, result in chunk_results.items():
                    fetched[local_i] = result or {}
            except Exception as e:
                last_error = e
                _sts_log.warning("STS Score batch chunk error: %s", e)
                for local_i, _ in chunk:
                    fetched.setdefault(local_i, {})
            if progress_callback:
                try:
                    progress_callback(done_base + chunk_end, total)
                except Exception:
                    pass

    # Phase C: validate, persist, and classify each pending row.
    for local_i, (orig_i, row, sts_input, input_hash, prior_entry, pid_str) in enumerate(pending):
        result = fetched.get(local_i, {}) or {}
        need_refresh = (
            prior_entry is not None
            and prior_entry.get("integration_version") == _sts_cache.STS_SCORE_INTEGRATION_VERSION
            and _sts_cache.is_valid_result(prior_entry.get("result"))
        )

        if _sts_cache.is_valid_result(result):
            _sts_cache.persist_fresh_result(sts_input, result, input_hash, pid_str)
            rec = _sts_cache.make_fresh_record(
                result, pid_str, input_hash,
                refreshed=need_refresh, retry_attempted=True,
            )
            execution_log.append(rec)
            results[orig_i] = dict(result)
            continue

        # Fetch failed -> stale fallback or total failure.
        fallback_entry = _sts_cache.find_stale_fallback(
            pid_str, input_hash, prior_entry
        )
        reason = "fetch_failed" if not result else "response_validation_failure"
        if fallback_entry is not None:
            rec = _sts_cache.make_stale_fallback_record(
                fallback_entry, pid_str, input_hash, reason, retry_attempted=True,
            )
            execution_log.append(rec)
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
        execution_log.append(rec)
        failure_log.append({
            "idx": orig_i, "patient_id": pid_str, "status": rec.status,
            "stage": rec.stage, "reason": rec.reason,
            "retry_attempted": True, "used_previous_cache": False,
        })
        results[orig_i] = {}

    calculate_sts_batch.last_error = last_error
    calculate_sts_batch.failure_log = failure_log
    calculate_sts_batch.last_execution_log = execution_log
    return results


calculate_sts_batch.last_error = None
calculate_sts_batch.failure_log = []
calculate_sts_batch.last_execution_log = []
