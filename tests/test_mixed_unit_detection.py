"""Priority 4 tests — Mixed-unit heterogeneity detection in normalize_external_units.

Covers:
  - Clearly imperial dataset → no mixed-unit warning
  - Clearly metric dataset → no mixed-unit warning
  - Suspicious height mix (cm + inches rows) → warning
  - Suspicious weight mix (kg + lb rows) → warning
  - Mixed-unit warning appears in normalization_report.warnings via pipeline
  - Small datasets (< 10 rows) do not trigger spurious warnings
"""
import numpy as np
import pandas as pd
import pytest

from risk_data import normalize_external_units, normalize_external_dataset, ExternalReadMeta


# ──────────────────────────────────────────────────────────────────────────────
# normalize_external_units — height heterogeneity
# ──────────────────────────────────────────────────────────────────────────────

class TestMixedHeightDetection:
    def test_all_imperial_no_mixed_warning(self):
        """All-inches heights: no mixed warning, conversion applied."""
        heights = [65.0, 67.0, 70.0, 68.0, 66.0, 64.0, 72.0, 63.0, 69.0, 71.0]
        df = pd.DataFrame({"Height (cm)": heights})
        _, summary = normalize_external_units(df)
        mixed = [w for w in summary["warnings"] if "Mixed height" in w]
        assert len(mixed) == 0, f"Unexpected mixed warning: {mixed}"
        assert summary["height_converted"] is True

    def test_all_metric_no_mixed_warning(self):
        """All-cm heights: no mixed warning, no conversion."""
        heights = [165.0, 170.0, 175.0, 168.0, 172.0, 180.0, 160.0, 158.0, 183.0, 176.0]
        df = pd.DataFrame({"Height (cm)": heights})
        _, summary = normalize_external_units(df)
        mixed = [w for w in summary["warnings"] if "Mixed height" in w]
        assert len(mixed) == 0
        assert summary["height_converted"] is False

    def test_suspicious_height_mix_triggers_warning(self):
        """Some rows in inches range, some in cm range → warning."""
        # 6 metric-range rows (150–180 cm) + 4 imperial-range rows (62–70 in)
        heights = [165.0, 170.0, 175.0, 168.0, 160.0, 172.0,  # metric-range
                   65.0, 68.0, 62.0, 70.0]                       # imperial-range
        df = pd.DataFrame({"Height (cm)": heights})
        _, summary = normalize_external_units(df)
        mixed = [w for w in summary["warnings"] if "Mixed height" in w]
        assert len(mixed) == 1, f"Expected 1 mixed warning, got: {summary['warnings']}"
        assert "45–90" in mixed[0]
        assert "130–250" in mixed[0]

    def test_mixed_warning_cites_column_name(self):
        df = pd.DataFrame({"height_cm": [170.0, 65.0, 168.0, 67.0, 172.0,
                                          160.0, 66.0, 175.0, 64.0, 180.0]})
        _, summary = normalize_external_units(df)
        mixed = [w for w in summary["warnings"] if "Mixed height" in w]
        assert len(mixed) == 1
        assert "height_cm" in mixed[0]

    def test_small_dataset_no_spurious_warning(self):
        """Fewer than 10 valid rows → no mixed-unit check."""
        heights = [165.0, 67.0, 170.0]  # mixed-looking but < 10 rows
        df = pd.DataFrame({"Height (cm)": heights})
        _, summary = normalize_external_units(df)
        mixed = [w for w in summary["warnings"] if "Mixed height" in w]
        assert len(mixed) == 0


# ──────────────────────────────────────────────────────────────────────────────
# normalize_external_units — weight heterogeneity
# ──────────────────────────────────────────────────────────────────────────────

class TestMixedWeightDetection:
    def test_all_lbs_no_mixed_warning(self):
        """All-lb weights: no mixed warning."""
        weights = [185.0, 200.0, 165.0, 220.0, 175.0, 190.0, 155.0, 210.0, 195.0, 170.0]
        df = pd.DataFrame({"Weight (kg)": weights})
        _, summary = normalize_external_units(df)
        mixed = [w for w in summary["warnings"] if "Mixed weight" in w]
        assert len(mixed) == 0

    def test_all_kg_no_mixed_warning(self):
        """All-kg weights: no mixed warning, no conversion."""
        weights = [70.0, 75.0, 80.0, 65.0, 85.0, 90.0, 60.0, 78.0, 72.0, 68.0]
        df = pd.DataFrame({"Weight (kg)": weights})
        _, summary = normalize_external_units(df)
        mixed = [w for w in summary["warnings"] if "Mixed weight" in w]
        assert len(mixed) == 0
        assert summary["weight_converted"] is False

    def test_suspicious_weight_mix_triggers_warning(self):
        """Some rows > 200 (lb) and some < 100 (kg) → warning."""
        # 6 >200-lb-range rows + 4 <100-kg-range rows
        weights = [210.0, 220.0, 215.0, 205.0, 230.0, 250.0,  # lb range
                   75.0, 80.0, 70.0, 85.0]                      # kg range
        df = pd.DataFrame({"Weight (kg)": weights})
        _, summary = normalize_external_units(df)
        mixed = [w for w in summary["warnings"] if "Mixed weight" in w]
        assert len(mixed) == 1, f"Expected 1 mixed warning, got: {summary['warnings']}"
        assert "> 200" in mixed[0] or ">200" in mixed[0]
        assert "< 100" in mixed[0] or "<100" in mixed[0]

    def test_small_weight_dataset_no_spurious_warning(self):
        weights = [210.0, 75.0, 220.0]  # < 10 rows
        df = pd.DataFrame({"Weight (kg)": weights})
        _, summary = normalize_external_units(df)
        mixed = [w for w in summary["warnings"] if "Mixed weight" in w]
        assert len(mixed) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline integration — mixed warnings bubble up to ExternalNormalizationReport
# ──────────────────────────────────────────────────────────────────────────────

class TestMixedUnitWarningInReport:
    def test_mixed_height_warning_in_report(self):
        """Mixed height warning propagates to ExternalNormalizationReport.warnings."""
        heights = [165.0, 170.0, 175.0, 168.0, 160.0, 172.0,
                   65.0, 68.0, 62.0, 70.0]
        df = pd.DataFrame({
            "Height (cm)": heights,
            "age_years": [60.0] * 10,
            "Sex": ["Male"] * 10,
        })
        read_meta = ExternalReadMeta(
            encoding_used="utf-8", delimiter=",", rows_loaded=10, columns_loaded=3
        )
        _, report = normalize_external_dataset(df, source_name="test.csv", read_meta=read_meta)
        mixed = [w for w in report.warnings if "Mixed height" in w]
        assert len(mixed) >= 1

    def test_clean_dataset_no_mixed_warning_in_report(self):
        """No mixed warning for a clearly metric dataset."""
        heights = [165.0, 170.0, 175.0, 168.0, 172.0, 160.0, 180.0, 158.0, 183.0, 176.0]
        df = pd.DataFrame({
            "Height (cm)": heights,
            "age_years": [60.0] * 10,
        })
        read_meta = ExternalReadMeta(
            encoding_used="utf-8", delimiter=",", rows_loaded=10, columns_loaded=2
        )
        _, report = normalize_external_dataset(df, source_name="test.csv", read_meta=read_meta)
        mixed = [w for w in report.warnings if "Mixed height" in w or "Mixed weight" in w]
        assert len(mixed) == 0
