from io import BytesIO

import pandas as pd

from app_data_dictionary import (
    build_dictionary_xlsx_bytes,
    get_app_reading_dictionary_dataframe,
    get_reading_aliases_dataframe,
    get_reading_rules_dataframe,
)
from risk_data import FLAT_ALIAS_TO_APP_COLUMNS


def test_dictionary_exposes_current_flat_aliases():
    df = get_app_reading_dictionary_dataframe("English")
    row = df.loc[df["Variable"].eq("Age (years)")].iloc[0]

    assert FLAT_ALIAS_TO_APP_COLUMNS["age_years"] == "Age (years)"
    assert "age_years" in row["Accepted flat aliases"]


def test_dictionary_marks_active_model_features_from_argument():
    df = get_app_reading_dictionary_dataframe(
        "English",
        model_feature_columns=["Age (years)"],
    )
    row = df.loc[df["Variable"].eq("Age (years)")].iloc[0]

    assert row["Current app/model usage"] == "Predictor in active model"


def test_dictionary_xlsx_contains_dicionario_aliases_and_rules_sheets():
    dictionary_df = get_app_reading_dictionary_dataframe("English")
    aliases_df = get_reading_aliases_dataframe("English")
    rules_df = get_reading_rules_dataframe("English")

    xlsx_bytes = build_dictionary_xlsx_bytes(dictionary_df, aliases_df, rules_df)
    xls = pd.ExcelFile(BytesIO(xlsx_bytes))

    assert xls.sheet_names == ["Dicionario", "Aliases_CSV", "Regras_de_Leitura"]
    exported = pd.read_excel(BytesIO(xlsx_bytes), sheet_name="Dicionario")
    assert "Age (years)" in exported["Variable"].values
