"""Tests for echo-to-surgery interval calculation and echo_stale flag."""

import datetime
import pandas as pd
import pytest
from risk_data import prepare_master_dataset, EXCLUDED_METADATA_COLUMNS


def _make_xlsx(tmp_path, echo_date: str, proc_date: str = "2025-06-01"):
    pre_df = pd.DataFrame({
        "Name": ["Paciente Echo"],
        "Procedure Date": [proc_date],
        "Surgery": ["AVR"],
        "Sex": ["Male"],
        "Age (years)": [70],
    })
    post_df = pd.DataFrame({
        "Patient": ["Paciente Echo"],
        "Procedure Date": [proc_date],
        "Death": [0],
    })
    eco_df = pd.DataFrame({
        "Patient": ["Paciente Echo"],
        "Exam date": [echo_date],
        "Pré-LVEF, %": [60],
    })
    xlsx_path = str(tmp_path / "echo_test.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pre_df.to_excel(writer, sheet_name="Preoperative", index=False)
        post_df.to_excel(writer, sheet_name="Postoperative", index=False)
        eco_df.to_excel(writer, sheet_name="Pre-Echocardiogram", index=False)
    return xlsx_path


def test_days_pre_echo_column_present(tmp_path):
    xlsx = _make_xlsx(tmp_path, echo_date="2025-05-01", proc_date="2025-06-01")
    prepared = prepare_master_dataset(xlsx)
    assert "days_pre_echo_to_surgery" in prepared.data.columns


def test_days_pre_echo_correct_value(tmp_path):
    # Echo: 2025-04-01, Surgery: 2025-06-01 → 61 days
    xlsx = _make_xlsx(tmp_path, echo_date="2025-04-01", proc_date="2025-06-01")
    prepared = prepare_master_dataset(xlsx)
    days = prepared.data["days_pre_echo_to_surgery"].iloc[0]
    assert days == 61


def test_echo_stale_false_for_recent_echo(tmp_path):
    # 30 days before surgery → not stale
    xlsx = _make_xlsx(tmp_path, echo_date="2025-05-02", proc_date="2025-06-01")
    prepared = prepare_master_dataset(xlsx)
    assert prepared.data["echo_stale"].iloc[0] is False or prepared.data["echo_stale"].iloc[0] == False


def test_echo_stale_true_for_old_echo(tmp_path):
    # 200 days before surgery → stale
    xlsx = _make_xlsx(tmp_path, echo_date="2024-11-13", proc_date="2025-06-01")
    prepared = prepare_master_dataset(xlsx)
    assert prepared.data["echo_stale"].iloc[0] is True or prepared.data["echo_stale"].iloc[0] == True


def test_echo_stale_boundary_180_days(tmp_path):
    # Exactly 180 days: not stale (> 180 is stale)
    import datetime
    proc = datetime.date(2025, 6, 1)
    echo = proc - datetime.timedelta(days=180)
    xlsx = _make_xlsx(tmp_path, echo_date=str(echo), proc_date=str(proc))
    prepared = prepare_master_dataset(xlsx)
    assert prepared.data["echo_stale"].iloc[0] == False


def test_echo_stale_boundary_181_days(tmp_path):
    import datetime
    proc = datetime.date(2025, 6, 1)
    echo = proc - datetime.timedelta(days=181)
    xlsx = _make_xlsx(tmp_path, echo_date=str(echo), proc_date=str(proc))
    prepared = prepare_master_dataset(xlsx)
    assert prepared.data["echo_stale"].iloc[0] == True


def test_days_pre_echo_not_in_feature_columns(tmp_path):
    xlsx = _make_xlsx(tmp_path, echo_date="2025-05-01", proc_date="2025-06-01")
    prepared = prepare_master_dataset(xlsx)
    assert "days_pre_echo_to_surgery" not in prepared.feature_columns
    assert "echo_stale" not in prepared.feature_columns


def test_days_pre_echo_in_excluded_metadata():
    assert "days_pre_echo_to_surgery" in EXCLUDED_METADATA_COLUMNS
    assert "echo_stale" in EXCLUDED_METADATA_COLUMNS


def test_info_contains_echo_interval_stats(tmp_path):
    xlsx = _make_xlsx(tmp_path, echo_date="2025-04-01", proc_date="2025-06-01")
    prepared = prepare_master_dataset(xlsx)
    assert "echo_interval_median_days" in prepared.info
    assert "echo_interval_p75_days" in prepared.info
    assert "echo_stale_count" in prepared.info
    assert prepared.info["echo_stale_count"] == 0
