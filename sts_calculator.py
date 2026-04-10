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
from typing import Dict, Optional

import pandas as pd

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

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


def calculate_sts(row: dict) -> Dict[str, float]:
    """Calculate STS risk scores for a single patient row.

    Args:
        row: dict with CSV column names or app display names.

    Returns:
        Dict with keys like 'predmort', 'predmm', etc.
        Values are decimals (0.0646 = 6.46%).
        Empty dict if the web calculator query fails.
    """
    if not HAS_WEBSOCKETS:
        return {}
    sts_input = build_sts_input_from_row(row)
    try:
        return _run_async(_query_sts_ws(sts_input))
    except Exception:
        return {}


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


def calculate_sts_batch(rows: list, progress_callback=None, chunk_size: int = 5) -> list:
    """Calculate STS risk scores for a batch of patient rows.

    Processes in small chunks so progress can be reported from the main thread
    (avoids deadlock when Streamlit progress callbacks run from async threads).

    Args:
        rows: list of dicts, each with CSV column names or app display names.
        progress_callback: optional callable(current, total) for progress tracking.
        chunk_size: number of patients per async chunk.

    Returns:
        List of result dicts (same order as input).
        If errors occur, the last error is stored in the `last_error` attribute.
    """
    if not HAS_WEBSOCKETS:
        return [{} for _ in rows]

    all_results: list = [{} for _ in rows]
    total = len(rows)
    done = 0
    last_error = None
    failure_log: list = []

    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk = [(i, rows[i]) for i in range(chunk_start, chunk_end)]
        try:
            chunk_results = _run_async(_calculate_sts_chunk_async(chunk, failure_log=failure_log))
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

    # Store diagnostics for display
    calculate_sts_batch.last_error = last_error
    calculate_sts_batch.failure_log = failure_log
    return all_results

calculate_sts_batch.last_error = None
calculate_sts_batch.failure_log = []
