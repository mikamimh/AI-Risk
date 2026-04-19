import pandas as pd
import pytest

from ai_risk_inference import _build_input_row
import risk_data
from risk_data import BLANK_MEANS_NO_COLUMNS, prepare_flat_dataset


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
                "Aortic Stenosis": "",
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
                "Aortic Stenosis": "None",
            },
        ]
    )
    monkeypatch.setattr(risk_data, "_read_csv_auto", lambda _path: source_df.copy())

    prepared = prepare_flat_dataset("synthetic.csv")
    data = prepared.data.set_index("Name")

    for col in BLANK_MEANS_NO_COLUMNS:
        assert data.at["P1", col] == "No"
    assert pd.isna(data.at["P1", "Suspension of Anticoagulation (day)"])
    assert pd.isna(data.at["P1", "Creatinine (mg/dL)"])
    assert data.at["P1", "Aortic Stenosis"] != "No"
