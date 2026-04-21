import numpy as np
import pandas as pd

from ai_risk_inference import _build_input_row
from risk_data import add_missingness_indicators


def test_panel_missingness_indicators_are_derived_before_imputation():
    df = pd.DataFrame({
        "Creatinine (mg/dL)": [1.0, np.nan, 0.9],
        "Cr clearance, ml/min *": [80.0, 70.0, np.nan],
        "Hematocrit (%)": [40.0, 38.0, 37.0],
        "WBC Count (10³/μL)": [7.1, np.nan, 6.8],
        "Platelet Count (cells/μL)": [200000, 180000, 150000],
        "INR": [1.1, 1.0, np.nan],
        "PTT": [30.0, 31.0, 29.0],
    })

    out = add_missingness_indicators(df)

    assert out["missing_renal_labs"].tolist() == [0, 1, 1]
    assert out["missing_cbc_labs"].tolist() == [0, 1, 0]
    assert out["missing_coagulation_labs"].tolist() == [0, 0, 1]


def test_single_row_inference_populates_missingness_indicators():
    feature_columns = [
        "Creatinine (mg/dL)",
        "Cr clearance, ml/min *",
        "Hematocrit (%)",
        "WBC Count (10³/μL)",
        "Platelet Count (cells/μL)",
        "INR",
        "PTT",
        "Surgery",
        "missing_renal_labs",
        "missing_cbc_labs",
        "missing_coagulation_labs",
    ]
    form = {
        "Creatinine (mg/dL)": 1.0,
        "Cr clearance, ml/min *": "",
        "Hematocrit (%)": 40.0,
        "WBC Count (10³/μL)": 7.1,
        "Platelet Count (cells/μL)": 200000,
        "INR": 1.1,
        "PTT": 30.0,
        "Surgery": "CABG",
    }

    row = _build_input_row(feature_columns, form)

    assert int(row.loc[0, "missing_renal_labs"]) == 1
    assert int(row.loc[0, "missing_cbc_labs"]) == 0
    assert int(row.loc[0, "missing_coagulation_labs"]) == 0
