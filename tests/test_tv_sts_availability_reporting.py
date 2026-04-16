"""STS availability reporting tests for temporal validation.

These tests cover the presentation-layer policy for incomplete STS coverage:
  - explicit complete / partial / unavailable classification
  - exact-count warning text
  - report/PDF Markdown notes for partial and unavailable coverage
  - subset-only STS labelling without altering AI Risk / EuroSCORE II labels
"""

from __future__ import annotations

import pandas as pd

from temporal_validation import (
    STS_AVAILABILITY_COMPLETE,
    STS_AVAILABILITY_PARTIAL,
    STS_AVAILABILITY_UNAVAILABLE,
    build_sts_availability_summary,
    build_temporal_validation_summary,
    classify_sts_availability,
)
from tabs.temporal_validation import _sts_availability_details_caption


def _cohort_summary() -> dict:
    return {
        "n_total": 26,
        "n_events": 5,
        "event_rate": 5 / 26,
        "date_range": "2022-Q1 — 2024-Q4",
        "n_complete": 20,
        "n_adequate": 4,
        "n_partial": 2,
        "n_low": 0,
    }


def _metadata() -> dict:
    return {
        "model_version": "v1.0",
        "n_patients": 200,
        "n_events": 40,
        "event_rate": 0.20,
    }


def _performance_df(sts_label: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Score": "AI Risk",
            "n": 26,
            "AUC": 0.771,
            "AUC_IC95_inf": 0.524,
            "AUC_IC95_sup": 0.952,
            "AUPRC": 0.624,
            "AUPRC_IC95_inf": 0.301,
            "AUPRC_IC95_sup": 0.883,
            "Brier": 0.124,
            "Calibration_Intercept": -0.314,
            "Calibration_Slope": 0.900,
            "HL_p": 0.4568,
            "Sensitivity": 0.800,
            "Specificity": 0.750,
            "PPV": 0.571,
            "NPV": 0.900,
        },
        {
            "Score": "EuroSCORE II",
            "n": 26,
            "AUC": 0.690,
            "AUC_IC95_inf": 0.429,
            "AUC_IC95_sup": 0.905,
            "AUPRC": 0.501,
            "AUPRC_IC95_inf": 0.250,
            "AUPRC_IC95_sup": 0.810,
            "Brier": 0.150,
            "Calibration_Intercept": 0.123,
            "Calibration_Slope": 1.050,
            "HL_p": 0.3211,
            "Sensitivity": 0.714,
            "Specificity": 0.700,
            "PPV": 0.500,
            "NPV": 0.875,
        },
        {
            "Score": sts_label,
            "n": 6,
            "AUC": 0.650,
            "AUC_IC95_inf": 0.420,
            "AUC_IC95_sup": 0.880,
            "AUPRC": 0.400,
            "AUPRC_IC95_inf": 0.200,
            "AUPRC_IC95_sup": 0.700,
            "Brier": 0.180,
            "Calibration_Intercept": -0.100,
            "Calibration_Slope": 0.800,
            "HL_p": 0.5000,
            "Sensitivity": 0.500,
            "Specificity": 0.667,
            "PPV": 0.333,
            "NPV": 0.800,
        },
    ])


def _risk_category_df(sts_label: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"Score": "AI Risk", "Category": "Low (<5%)", "n": 10, "%": 38.5, "Observed_mortality": 0.10},
        {"Score": "EuroSCORE II", "Category": "Intermediate (5-15%)", "n": 8, "%": 30.8, "Observed_mortality": 0.25},
        {"Score": sts_label, "Category": "High (>15%)", "n": 2, "%": 33.3, "Observed_mortality": 0.50},
    ])


def test_classify_sts_availability_complete_partial_unavailable():
    assert classify_sts_availability(22, 22) == STS_AVAILABILITY_COMPLETE
    assert classify_sts_availability(22, 6) == STS_AVAILABILITY_PARTIAL
    assert classify_sts_availability(22, 0) == STS_AVAILABILITY_UNAVAILABLE


def test_complete_availability_has_no_partial_or_unavailable_banner():
    summary = build_sts_availability_summary(22, 22)
    assert summary["status"] == STS_AVAILABILITY_COMPLETE
    assert summary["banner_text"] == ""
    assert summary["score_label"] == "STS Score"


def test_partial_availability_warning_uses_exact_counts():
    summary = build_sts_availability_summary(22, 6)
    assert summary["status"] == STS_AVAILABILITY_PARTIAL
    assert "STS availability: PARTIAL." in summary["banner_text"]
    assert "6 of 22 eligible rows" in summary["banner_text"]
    assert "subset" in summary["banner_text"].lower()
    assert "27.3%" in summary["coverage_text"]


def test_unavailable_availability_warning_uses_exact_counts():
    summary = build_sts_availability_summary(22, 0)
    assert summary["status"] == STS_AVAILABILITY_UNAVAILABLE
    assert "STS availability: UNAVAILABLE." in summary["banner_text"]
    assert "No eligible rows produced a usable final STS score." in summary["banner_text"]
    assert "0/22" in summary["coverage_text"]


def test_execution_details_text_complete_partial_and_unavailable():
    assert _sts_availability_details_caption("complete", 22, 22, "English") == "STS availability: complete (22/22 eligible)"
    assert _sts_availability_details_caption("partial", 6, 22, "English") == "STS availability: partial (6/22 eligible)"
    assert _sts_availability_details_caption("unavailable", 0, 22, "English") == "STS availability: unavailable (0/22 eligible)"


def test_execution_details_text_restore_path_can_reuse_same_caption_helper():
    assert _sts_availability_details_caption("partial", 6, 22, "Portuguese") == "Disponibilidade do STS: parcial (6/22 elegíveis)"
    assert _sts_availability_details_caption("unavailable", 0, 22, "Portuguese") == "Disponibilidade do STS: indisponível (0/22 elegíveis)"


def test_report_note_reflects_partial_status_and_subset_label():
    sts_summary = build_sts_availability_summary(22, 6)
    md = build_temporal_validation_summary(
        cohort_summary=_cohort_summary(),
        performance_df=_performance_df(sts_summary["score_label"]),
        pairwise_df=pd.DataFrame(),
        calibration_df=pd.DataFrame(),
        risk_category_df=_risk_category_df(sts_summary["score_label"]),
        metadata=_metadata(),
        threshold=0.08,
        sts_availability=sts_summary,
    )

    assert "STS availability note" in md
    assert "PARTIAL" in md
    assert "6 of 22 eligible rows" in md
    assert "27.3% coverage" in md
    assert sts_summary["score_label"] in md
    assert sts_summary["subset_note"] in md
    assert sts_summary["risk_category_note"] in md


def test_report_note_reflects_unavailable_status():
    sts_summary = build_sts_availability_summary(22, 0)
    md = build_temporal_validation_summary(
        cohort_summary=_cohort_summary(),
        performance_df=_performance_df("STS Score").query("Score != 'STS Score'").reset_index(drop=True),
        pairwise_df=pd.DataFrame(),
        calibration_df=pd.DataFrame(),
        risk_category_df=pd.DataFrame(),
        metadata=_metadata(),
        threshold=0.08,
        sts_availability=sts_summary,
    )

    assert "STS availability note" in md
    assert "UNAVAILABLE" in md
    assert "0 of 22 eligible rows" in md
    assert sts_summary["risk_category_note"] in md


def test_risk_category_section_gets_local_subset_note_when_partial():
    sts_summary = build_sts_availability_summary(6, 2)
    md = build_temporal_validation_summary(
        cohort_summary=_cohort_summary(),
        performance_df=_performance_df(sts_summary["score_label"]),
        pairwise_df=pd.DataFrame(),
        calibration_df=pd.DataFrame(),
        risk_category_df=_risk_category_df(sts_summary["score_label"]),
        metadata=_metadata(),
        threshold=0.08,
        sts_availability=sts_summary,
    )

    risk_section = md.split("## Risk Category Distribution", 1)[1]
    assert sts_summary["risk_category_note"] in risk_section


def test_risk_category_section_gets_local_omission_note_when_unavailable_but_other_scores_exist():
    sts_summary = build_sts_availability_summary(22, 0)
    risk_df = pd.DataFrame([
        {"Score": "AI Risk", "Category": "Low (<5%)", "n": 10, "%": 38.5, "Observed_mortality": 0.10},
        {"Score": "EuroSCORE II", "Category": "Intermediate (5-15%)", "n": 8, "%": 30.8, "Observed_mortality": 0.25},
    ])
    md = build_temporal_validation_summary(
        cohort_summary=_cohort_summary(),
        performance_df=_performance_df("STS Score").query("Score != 'STS Score'").reset_index(drop=True),
        pairwise_df=pd.DataFrame(),
        calibration_df=pd.DataFrame(),
        risk_category_df=risk_df,
        metadata=_metadata(),
        threshold=0.08,
        sts_availability=sts_summary,
    )

    risk_section = md.split("## Risk Category Distribution", 1)[1]
    assert sts_summary["risk_category_note"] in risk_section
    assert "| AI Risk |" in risk_section
    assert "| EuroSCORE II |" in risk_section
    assert "STS Score |" not in risk_section


def test_ai_risk_and_euroscore_labels_remain_unchanged_when_sts_is_partial():
    sts_summary = build_sts_availability_summary(22, 6)
    md = build_temporal_validation_summary(
        cohort_summary=_cohort_summary(),
        performance_df=_performance_df(sts_summary["score_label"]),
        pairwise_df=pd.DataFrame(),
        calibration_df=pd.DataFrame(),
        risk_category_df=_risk_category_df(sts_summary["score_label"]),
        metadata=_metadata(),
        threshold=0.08,
        sts_availability=sts_summary,
    )

    assert "| AI Risk |" in md
    assert "| EuroSCORE II |" in md
    assert "AI Risk (available for" not in md
    assert "EuroSCORE II (available for" not in md


def test_report_builder_portuguese_output_has_no_mojibake_in_risk_category_heading():
    sts_summary = build_sts_availability_summary(22, 0, language="Portuguese")
    md = build_temporal_validation_summary(
        cohort_summary=_cohort_summary(),
        performance_df=_performance_df("STS Score").query("Score != 'STS Score'").reset_index(drop=True),
        pairwise_df=pd.DataFrame(),
        calibration_df=pd.DataFrame(),
        risk_category_df=pd.DataFrame(),
        metadata=_metadata(),
        threshold=0.08,
        language="Portuguese",
        sts_availability=sts_summary,
    )

    assert "Distribuição por Classe de Risco" in md
    assert "DistribuiÃ" not in md
