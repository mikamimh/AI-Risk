"""Tests for compute_case_uid and its integration in prepare_* datasets."""

import pytest
from risk_data import compute_case_uid, EXCLUDED_METADATA_COLUMNS


def test_determinism():
    uid1 = compute_case_uid("JOAO SILVA", "2025-03-15", "CABG")
    uid2 = compute_case_uid("JOAO SILVA", "2025-03-15", "CABG")
    assert uid1 == uid2


def test_length_16():
    uid = compute_case_uid("PATIENT A", "2025-01-01", "AVR")
    assert len(uid) == 16


def test_hex_string():
    uid = compute_case_uid("X", "2020-12-31", "CABG")
    assert all(c in "0123456789abcdef" for c in uid)


def test_different_surgery_different_uid():
    uid_cabg = compute_case_uid("MARIA SOUZA", "2025-06-10", "CABG")
    uid_avr = compute_case_uid("MARIA SOUZA", "2025-06-10", "AVR")
    assert uid_cabg != uid_avr


def test_different_patient_different_uid():
    uid_a = compute_case_uid("PACIENTE A", "2025-06-10", "CABG")
    uid_b = compute_case_uid("PACIENTE B", "2025-06-10", "CABG")
    assert uid_a != uid_b


def test_different_date_different_uid():
    uid1 = compute_case_uid("JOAO", "2025-01-01", "CABG")
    uid2 = compute_case_uid("JOAO", "2025-06-01", "CABG")
    assert uid1 != uid2


def test_case_insensitive_patient_key():
    uid_upper = compute_case_uid("JOAO SILVA", "2025-01-01", "CABG")
    uid_lower = compute_case_uid("joao silva", "2025-01-01", "CABG")
    assert uid_upper == uid_lower


def test_case_insensitive_surgery():
    uid_upper = compute_case_uid("JOAO", "2025-01-01", "CABG")
    uid_lower = compute_case_uid("JOAO", "2025-01-01", "cabg")
    assert uid_upper == uid_lower


def test_whitespace_stripped_surgery():
    uid_plain = compute_case_uid("JOAO", "2025-01-01", "CABG")
    uid_padded = compute_case_uid("JOAO", "2025-01-01", "  CABG  ")
    assert uid_plain == uid_padded


def test_case_uid_in_excluded_metadata():
    assert "case_uid" in EXCLUDED_METADATA_COLUMNS


def test_prepare_master_dataset_has_case_uid(tmp_path):
    """Integration: case_uid column appears in prepare_master_dataset output."""
    import pandas as pd
    from risk_data import prepare_master_dataset

    # Build a minimal multi-sheet XLSX fixture
    pre_df = pd.DataFrame({
        "Name": ["Paciente Teste"],
        "Procedure Date": ["2025-03-15"],
        "Surgery": ["AVR"],
        "Sex": ["Male"],
        "Age (years)": [65],
    })
    post_df = pd.DataFrame({
        "Patient": ["Paciente Teste"],
        "Procedure Date": ["2025-03-15"],
        "Death": [0],
    })
    eco_df = pd.DataFrame({
        "Patient": ["Paciente Teste"],
        "Exam date": ["2025-02-01"],
        "Pré-LVEF, %": [60],
    })

    xlsx_path = str(tmp_path / "test_fixture.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pre_df.to_excel(writer, sheet_name="Preoperative", index=False)
        post_df.to_excel(writer, sheet_name="Postoperative", index=False)
        eco_df.to_excel(writer, sheet_name="Pre-Echocardiogram", index=False)

    prepared = prepare_master_dataset(xlsx_path)
    assert "case_uid" in prepared.data.columns
    uid_val = prepared.data["case_uid"].iloc[0]
    assert isinstance(uid_val, str)
    assert len(uid_val) == 16
    assert "case_uid" not in prepared.feature_columns


def test_prepare_flat_dataset_has_case_uid(tmp_path):
    """Integration: case_uid column appears in prepare_flat_dataset output."""
    import pandas as pd
    from risk_data import prepare_flat_dataset

    df = pd.DataFrame({
        "Name": ["Paciente A", "Paciente B"],
        "Procedure Date": ["2025-01-10", "2025-02-20"],
        "Surgery": ["CABG", "AVR"],
        "morte_30d": [0, 1],
        "Sex": ["Male", "Female"],
        "Age (years)": [60, 70],
    })
    csv_path = str(tmp_path / "flat.csv")
    df.to_csv(csv_path, index=False)

    prepared = prepare_flat_dataset(csv_path)
    assert "case_uid" in prepared.data.columns
    # Two different patients → two different UIDs
    uids = prepared.data["case_uid"].tolist()
    assert uids[0] != uids[1]
    assert "case_uid" not in prepared.feature_columns
