import pandas as pd
import pytest

from ai_risk_inference import _build_input_row
import risk_data
from risk_data import (
    BLANK_MEANS_NO_COLUMNS,
    normalize_arrhythmia_recent_value,
    normalize_coronary_symptom_value,
    parse_suspension_anticoagulation_days,
    prepare_flat_dataset,
)


def test_row_inference_applies_blank_means_no_only_to_explicit_history_columns():
    feature_columns = sorted(BLANK_MEANS_NO_COLUMNS | {"Hypertension"})
    form = {col: "" for col in feature_columns}

    row = _build_input_row(feature_columns, form)

    for col in BLANK_MEANS_NO_COLUMNS:
        assert row.at[0, col] == "No"
    assert row.at[0, "Previous surgery"] == "No"
    assert row.at[0, "Hypertension"] != "No"


def test_row_inference_treats_absent_explicit_history_columns_as_no():
    feature_columns = sorted(BLANK_MEANS_NO_COLUMNS)

    row = _build_input_row(feature_columns, {})

    for col in BLANK_MEANS_NO_COLUMNS:
        assert row.at[0, col] == "No"


def test_row_inference_canonicalizes_literal_none_coronary_symptom():
    row = _build_input_row(["Coronary Symptom"], {"Coronary Symptom": "None"})

    assert row.at[0, "Coronary Symptom"] == "No coronary symptoms"


def test_row_inference_preserves_arrhythmia_recent_none_as_valid_value():
    row = _build_input_row(["Arrhythmia Recent"], {"Arrhythmia Recent": "None"})

    assert row.at[0, "Arrhythmia Recent"] == "None"


def test_row_inference_keeps_arrhythmia_recent_blank_missing():
    row = _build_input_row(["Arrhythmia Recent"], {"Arrhythmia Recent": ""})

    assert pd.isna(row.at[0, "Arrhythmia Recent"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("None", "No coronary symptoms"),
        ("No Symptoms", "No coronary symptoms"),
        ("NSTEMI", "Non-STEMI"),
        ("Stable angina", "Stable Angina"),
        ("", ""),
        ("Unknown", "Unknown"),
    ],
)
def test_coronary_symptom_canonicalizer_is_narrow(raw, expected):
    assert normalize_coronary_symptom_value(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("None", "None"),
        ("No", "None"),
        ("Atrial Fibrillation", "Atrial Fibrillation"),
        ("AF", "Atrial Fibrillation"),
        ("", pd.NA),
        ("Unknown", pd.NA),
    ],
)
def test_arrhythmia_recent_canonicalizer_is_narrow(raw, expected):
    result = normalize_arrhythmia_recent_value(raw)
    if pd.isna(expected):
        assert pd.isna(result)
    else:
        assert result == expected


@pytest.mark.parametrize("missing_token", ["", "-", "nan", "N/A", "not informed"])
def test_row_inference_treats_missing_tokens_as_no_for_explicit_history_columns(missing_token):
    feature_columns = sorted(BLANK_MEANS_NO_COLUMNS)
    form = {col: missing_token for col in feature_columns}

    row = _build_input_row(feature_columns, form)

    for col in BLANK_MEANS_NO_COLUMNS:
        assert row.at[0, col] == "No"


@pytest.mark.parametrize("value", ["", "-", "nan", "N/A", "not informed", "abc"])
def test_suspension_of_anticoagulation_blank_or_invalid_stays_missing(value):
    col = "Suspension of Anticoagulation (day)"

    row = _build_input_row([col], {col: value})

    assert pd.isna(row.at[0, col])


def test_suspension_of_anticoagulation_numeric_string_is_preserved():
    col = "Suspension of Anticoagulation (day)"

    row = _build_input_row([col], {col: "> 5"})

    assert row.at[0, col] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("> 5", 5.0),
        ("5 days", 5.0),
        ("2d", 2.0),
        ("3 dias", 3.0),
    ],
)
def test_suspension_of_anticoagulation_recoverable_text_is_parsed(raw, expected):
    col = "Suspension of Anticoagulation (day)"

    row = _build_input_row([col], {col: raw})

    assert row.at[0, col] == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["5-7 days", "several days", "unknown duration"])
def test_suspension_of_anticoagulation_ambiguous_text_stays_missing(raw):
    assert pd.isna(parse_suspension_anticoagulation_days(raw))


def test_flat_dataset_path_preserves_narrow_blank_means_no_semantics(monkeypatch):
    source_df = pd.DataFrame(
        [
            {
                "Name": "P1",
                "Surgery": "CABG",
                "Death": "No",
                "Previous surgery": "",
                "HF": "",
                "Arrhythmia Remote": "",
                "Arrhythmia Recent": "",
                "Family Hx of CAD": "",
                "Anticoagulation/ Antiaggregation": "",
                "Suspension of Anticoagulation (day)": "",
                "Creatinine (mg/dL)": "",
                "Coronary Symptom": "",
                "Aortic Stenosis": "",
                "Aortic Regurgitation": "",
            },
            {
                "Name": "P2",
                "Surgery": "AVR",
                "Death": "Yes",
                "Previous surgery": "Yes",
                "HF": "Yes",
                "Arrhythmia Remote": "No",
                "Arrhythmia Recent": "Yes",
                "Family Hx of CAD": "Yes",
                "Anticoagulation/ Antiaggregation": "Yes",
                "Suspension of Anticoagulation (day)": "7",
                "Creatinine (mg/dL)": "1.2",
                "Coronary Symptom": "Stable Angina",
                "Aortic Stenosis": "None",
                "Aortic Regurgitation": "Mild",
            },
            {
                "Name": "P3",
                "Surgery": "AVR",
                "Death": "No",
                "Previous surgery": "No",
                "HF": "No",
                "Arrhythmia Remote": "No",
                "Arrhythmia Recent": "No",
                "Family Hx of CAD": "No",
                "Anticoagulation/ Antiaggregation": "No",
                "Suspension of Anticoagulation (day)": "",
                "Creatinine (mg/dL)": "1.0",
                "Coronary Symptom": "Unknown",
                "Aortic Stenosis": "Unknown",
                "Aortic Regurgitation": "",
            },
            {
                "Name": "P4",
                "Surgery": "CABG",
                "Death": "No",
                "Previous surgery": "No",
                "HF": "No",
                "Arrhythmia Remote": "No",
                "Arrhythmia Recent": "No",
                "Family Hx of CAD": "No",
                "Anticoagulation/ Antiaggregation": "No",
                "Suspension of Anticoagulation (day)": "",
                "Creatinine (mg/dL)": "1.0",
                "Coronary Symptom": "None",
                "Aortic Stenosis": "None",
                "Aortic Regurgitation": "",
            },
        ]
    )
    monkeypatch.setattr(risk_data, "_read_csv_auto", lambda _path: source_df.copy())

    prepared = prepare_flat_dataset("synthetic.csv")
    data = prepared.data.set_index("Name")

    for col in BLANK_MEANS_NO_COLUMNS:
        assert data.at["P1", col] == "No"
    assert pd.isna(data.at["P1", "Arrhythmia Recent"])
    assert pd.isna(data.at["P1", "Suspension of Anticoagulation (day)"])
    assert pd.isna(data.at["P1", "Creatinine (mg/dL)"])
    assert pd.isna(data.at["P1", "Coronary Symptom"])
    assert data.at["P1", "Aortic Stenosis"] == "None"
    assert pd.isna(data.at["P1", "Aortic Regurgitation"])
    assert pd.isna(data.at["P3", "Coronary Symptom"])
    assert pd.isna(data.at["P3", "Aortic Stenosis"])
    assert data.at["P3", "Arrhythmia Recent"] == "None"
    assert data.at["P4", "Coronary Symptom"] == "No coronary symptoms"
    assert data.at["P4", "Arrhythmia Recent"] == "None"
