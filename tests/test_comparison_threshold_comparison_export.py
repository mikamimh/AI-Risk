from __future__ import annotations

from io import BytesIO
import zipfile

import pandas as pd

from export_helpers import build_comparison_full_package, build_comparison_xlsx


def _minimal_triple_ci() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Score": "AI Risk",
                "n": 120,
                "AUC": 0.81,
                "AUC_IC95_inf": 0.74,
                "AUC_IC95_sup": 0.87,
                "AUPRC": 0.42,
                "AUPRC_IC95_inf": 0.31,
                "AUPRC_IC95_sup": 0.52,
                "Brier": 0.10,
                "Brier_IC95_inf": 0.08,
                "Brier_IC95_sup": 0.12,
            }
        ]
    )


def _minimal_threshold_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Score": "AI Risk",
                "n": 120,
                "AUC": 0.81,
                "AUPRC": 0.42,
                "Brier": 0.10,
                "Sensitivity": 0.72,
                "Specificity": 0.74,
                "PPV": 0.39,
                "NPV": 0.92,
            }
        ]
    )


def _threshold_comparison_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Threshold label": "5%",
                "Threshold": 0.05,
                "Sensitivity": 0.85,
                "Specificity": 0.60,
                "PPV": 0.30,
                "NPV": 0.95,
                "Accuracy": 0.66,
                "TP": 17,
                "FP": 40,
                "TN": 60,
                "FN": 3,
                "High risk (%)": 47.5,
                "High risk (n)": 57,
                "n": 120,
            },
            {
                "Threshold label": "8%",
                "Threshold": 0.08,
                "Sensitivity": 0.72,
                "Specificity": 0.74,
                "PPV": 0.39,
                "NPV": 0.92,
                "Accuracy": 0.74,
                "TP": 14,
                "FP": 24,
                "TN": 76,
                "FN": 6,
                "High risk (%)": 31.7,
                "High risk (n)": 38,
                "n": 120,
            },
            {
                "Threshold label": "10%",
                "Threshold": 0.10,
                "Sensitivity": 0.66,
                "Specificity": 0.79,
                "PPV": 0.42,
                "NPV": 0.91,
                "Accuracy": 0.77,
                "TP": 13,
                "FP": 20,
                "TN": 80,
                "FN": 7,
                "High risk (%)": 27.5,
                "High risk (n)": 33,
                "n": 120,
            },
            {
                "Threshold label": "15%",
                "Threshold": 0.15,
                "Sensitivity": 0.51,
                "Specificity": 0.88,
                "PPV": 0.49,
                "NPV": 0.89,
                "Accuracy": 0.82,
                "TP": 10,
                "FP": 10,
                "TN": 90,
                "FN": 10,
                "High risk (%)": 16.7,
                "High risk (n)": 20,
                "n": 120,
            },
            {
                "Threshold label": "Youden",
                "Threshold": 0.087,
                "Sensitivity": 0.74,
                "Specificity": 0.73,
                "PPV": 0.40,
                "NPV": 0.92,
                "Accuracy": 0.74,
                "TP": 14,
                "FP": 25,
                "TN": 75,
                "FN": 5,
                "High risk (%)": 32.5,
                "High risk (n)": 39,
                "n": 120,
            },
        ]
    )


def _roc_plot_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"score": "AI Risk", "fpr": 0.0, "tpr": 0.0, "threshold": float("inf"), "cohort": "triple"},
            {"score": "AI Risk", "fpr": 0.1, "tpr": 0.7, "threshold": 0.2, "cohort": "triple"},
            {"score": "AI Risk", "fpr": 1.0, "tpr": 1.0, "threshold": 0.0, "cohort": "triple"},
            {"score": "EuroSCORE II", "fpr": 0.0, "tpr": 0.0, "threshold": float("inf"), "cohort": "triple"},
            {"score": "EuroSCORE II", "fpr": 0.2, "tpr": 0.6, "threshold": 0.2, "cohort": "triple"},
            {"score": "EuroSCORE II", "fpr": 1.0, "tpr": 1.0, "threshold": 0.0, "cohort": "triple"},
        ]
    )


def _calibration_plot_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"score": "AI Risk", "bin": 1, "mean_predicted_risk": 0.04, "observed_event_rate": 0.05, "n_in_bin": 40, "cohort": "triple"},
            {"score": "AI Risk", "bin": 2, "mean_predicted_risk": 0.12, "observed_event_rate": 0.11, "n_in_bin": 40, "cohort": "triple"},
            {"score": "AI Risk", "bin": 3, "mean_predicted_risk": 0.30, "observed_event_rate": 0.28, "n_in_bin": 40, "cohort": "triple"},
            {"score": "EuroSCORE II", "bin": 1, "mean_predicted_risk": 0.05, "observed_event_rate": 0.04, "n_in_bin": 40, "cohort": "triple"},
            {"score": "EuroSCORE II", "bin": 2, "mean_predicted_risk": 0.15, "observed_event_rate": 0.10, "n_in_bin": 40, "cohort": "triple"},
            {"score": "EuroSCORE II", "bin": 3, "mean_predicted_risk": 0.35, "observed_event_rate": 0.25, "n_in_bin": 40, "cohort": "triple"},
        ]
    )


def _dca_plot_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"score": "AI Risk", "threshold": 0.05, "net_benefit": 0.10, "strategy": "AI Risk", "cohort": "triple"},
            {"score": "AI Risk", "threshold": 0.10, "net_benefit": 0.08, "strategy": "AI Risk", "cohort": "triple"},
            {"score": "Treat all", "threshold": 0.05, "net_benefit": 0.04, "strategy": "Treat all", "cohort": "triple"},
            {"score": "Treat all", "threshold": 0.10, "net_benefit": 0.02, "strategy": "Treat all", "cohort": "triple"},
        ]
    )


def test_full_package_contains_threshold_comparison_md_and_xlsx_sheet():
    payload = build_comparison_full_package(
        triple_ci=_minimal_triple_ci(),
        calib_df=pd.DataFrame(),
        formal_df=pd.DataFrame(),
        delong_df=pd.DataFrame(),
        reclass_df=pd.DataFrame(),
        threshold_metrics=_minimal_threshold_metrics(),
        threshold=0.08,
        n_triple=120,
        model_version="test-v1",
        language="English",
        threshold_comparison_df=_threshold_comparison_df(),
        roc_plot_df=_roc_plot_df(),
        calibration_plot_df=_calibration_plot_df(),
        dca_plot_df=_dca_plot_df(),
    )

    with zipfile.ZipFile(BytesIO(payload), "r") as zf:
        names = set(zf.namelist())
        assert "comparison_full_report.md" in names
        assert "comparison_tables.xlsx" in names
        assert "figures/roc.png" in names
        assert "figures/calibration.png" in names
        assert "figures/dca.png" in names
        assert "comparison_roc_data.csv" in names
        assert "comparison_calibration_data.csv" in names
        assert "comparison_dca_data.csv" in names

        md = zf.read("comparison_full_report.md").decode("utf-8")
        assert "Threshold Performance Across Candidate Cutoffs" in md
        assert "primary" in md.lower() and "threshold" in md.lower()

        xlsx_bytes = zf.read("comparison_tables.xlsx")
        xls = pd.ExcelFile(BytesIO(xlsx_bytes))
        assert "10_THRESHOLD_COMPARISON" in xls.sheet_names
        assert "11_FIG_ROC_DATA" in xls.sheet_names
        assert "12_FIG_CALIBRATION_DATA" in xls.sheet_names
        assert "13_FIG_DCA_DATA" in xls.sheet_names

        thr_df = pd.read_excel(BytesIO(xlsx_bytes), sheet_name="10_THRESHOLD_COMPARISON")
        assert "Threshold label" in thr_df.columns
        assert "Accuracy" in thr_df.columns
        assert "High risk (%)" in thr_df.columns
        assert {"5%", "8%", "10%", "15%", "Youden"}.issubset(
            set(thr_df["Threshold label"].astype(str))
        )


def test_structured_xlsx_writes_threshold_comparison_sheet():
    xlsx_bytes = build_comparison_xlsx(
        triple_ci=_minimal_triple_ci(),
        calib_df=pd.DataFrame(),
        formal_df=pd.DataFrame(),
        delong_df=pd.DataFrame(),
        reclass_df=pd.DataFrame(),
        threshold_metrics=_minimal_threshold_metrics(),
        threshold=0.08,
        n_triple=120,
        model_version="test-v1",
        language="English",
        threshold_comparison_df=_threshold_comparison_df(),
        roc_plot_df=_roc_plot_df(),
        calibration_plot_df=_calibration_plot_df(),
        dca_plot_df=_dca_plot_df(),
    )
    xls = pd.ExcelFile(BytesIO(xlsx_bytes))
    assert "10_THRESHOLD_COMPARISON" in xls.sheet_names
    assert "11_FIG_ROC_DATA" in xls.sheet_names
    assert "12_FIG_CALIBRATION_DATA" in xls.sheet_names
    assert "13_FIG_DCA_DATA" in xls.sheet_names
