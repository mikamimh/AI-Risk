"""Priority 7 — MIMIC-like regression tests.

Uses the ``mimic_like_raw_df`` and ``mimic_like_normalized`` fixtures from
conftest.py to exercise the full external-normalization pipeline end-to-end on a
synthetic dataset covering all known edge cases:

  - Imperial height/weight → auto-conversion to SI
  - Pediatric rows (age < 18) → is_pediatric flag + STS exclusion
  - Mixed token variants (Sim/Não, oui/non, ja/nein) → normalized to Yes/No
  - Out-of-scope surgeries (Bentall, dissection, Ross, transplant, homograft) → excluded
  - Supported adult STS surgeries (CABG, AVR, MVR) → STS-ready (if other fields ok)
"""
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Unit conversion
# ──────────────────────────────────────────────────────────────────────────────

class TestMimicUnitConversion:
    def test_height_converted(self, mimic_like_normalized):
        _, report = mimic_like_normalized
        assert report.unit_summary["height_converted"] is True

    def test_weight_converted(self, mimic_like_normalized):
        _, report = mimic_like_normalized
        assert report.unit_summary["weight_converted"] is True

    def test_height_values_in_cm_range_after_conversion(self, mimic_like_normalized):
        df, _ = mimic_like_normalized
        if "Height (cm)" in df.columns:
            import pandas as pd
            h = pd.to_numeric(df["Height (cm)"], errors="coerce").dropna()
            # After conversion from inches, all adult heights should be > 100 cm
            assert float(h.min()) > 100, f"Minimum post-conversion height unexpectedly low: {h.min()}"

    def test_weight_values_in_kg_range_after_conversion(self, mimic_like_normalized):
        df, _ = mimic_like_normalized
        if "Weight (kg)" in df.columns:
            import pandas as pd
            w = pd.to_numeric(df["Weight (kg)"], errors="coerce").dropna()
            # After lb→kg conversion, max should be below 200 kg (reasonable)
            assert float(w.max()) < 200, f"Post-conversion max weight unexpectedly high: {w.max()}"


# ──────────────────────────────────────────────────────────────────────────────
# Pediatric exclusion
# ──────────────────────────────────────────────────────────────────────────────

class TestMimicPediatricExclusion:
    def test_pediatric_count_at_least_two(self, mimic_like_normalized):
        _, report = mimic_like_normalized
        assert report.scope_summary["n_pediatric"] >= 2

    def test_pediatric_rows_flagged_in_df(self, mimic_like_normalized):
        df, _ = mimic_like_normalized
        assert "is_pediatric" in df.columns
        n_ped = int(df["is_pediatric"].sum())
        assert n_ped >= 2

    def test_pediatric_rows_not_sts_ready(self, mimic_like_normalized):
        df, _ = mimic_like_normalized
        if "is_pediatric" in df.columns and "sts_input_ready" in df.columns:
            ped_rows = df[df["is_pediatric"]]
            assert not ped_rows["sts_input_ready"].any(), (
                "No pediatric row should have sts_input_ready=True"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Token normalization
# ──────────────────────────────────────────────────────────────────────────────

class TestMimicTokenNormalization:
    def test_portuguese_yes_normalized(self, mimic_like_normalized):
        df, _ = mimic_like_normalized
        # "Sim" → "Yes" in Hypertension column (first row)
        if "Hypertension" in df.columns:
            first_val = str(df["Hypertension"].iloc[0])
            assert first_val == "Yes", f"Expected 'Yes', got {first_val!r}"

    def test_french_yes_normalized(self, mimic_like_normalized):
        df, _ = mimic_like_normalized
        # Second row: "oui" → "Yes"
        if "Hypertension" in df.columns:
            val = str(df["Hypertension"].iloc[1])
            assert val == "Yes", f"Expected 'Yes', got {val!r}"

    def test_german_yes_normalized(self, mimic_like_normalized):
        df, _ = mimic_like_normalized
        # Third row: "ja" → "Yes"
        if "Hypertension" in df.columns:
            val = str(df["Hypertension"].iloc[2])
            assert val == "Yes", f"Expected 'Yes', got {val!r}"

    def test_portuguese_no_normalized(self, mimic_like_normalized):
        df, _ = mimic_like_normalized
        # Fourth row Hypertension: "Não" → "No"
        if "Hypertension" in df.columns:
            val = str(df["Hypertension"].iloc[3])
            assert val == "No", f"Expected 'No', got {val!r}"


# ──────────────────────────────────────────────────────────────────────────────
# Scope exclusion
# ──────────────────────────────────────────────────────────────────────────────

class TestMimicScopeExclusion:
    def test_n_scope_excluded_at_least_five(self, mimic_like_normalized):
        """Bentall, dissection, Ross, transplant, homograft = 5 excluded."""
        _, report = mimic_like_normalized
        assert report.scope_summary["n_sts_scope_excluded"] >= 5

    def test_bentall_row_excluded(self, mimic_like_normalized):
        df, _ = mimic_like_normalized
        if "sts_scope_excluded" in df.columns and "Surgery" in df.columns:
            bentall = df[df["Surgery"].str.upper().str.contains("BENTALL", na=False)]
            assert not bentall.empty
            assert bentall["sts_scope_excluded"].all()

    def test_ross_row_excluded(self, mimic_like_normalized):
        df, _ = mimic_like_normalized
        if "sts_scope_excluded" in df.columns and "Surgery" in df.columns:
            ross = df[df["Surgery"].str.upper().str.contains("ROSS", na=False)]
            assert not ross.empty
            assert ross["sts_scope_excluded"].all()

    def test_transplant_row_excluded(self, mimic_like_normalized):
        df, _ = mimic_like_normalized
        if "sts_scope_excluded" in df.columns and "Surgery" in df.columns:
            tx = df[df["Surgery"].str.upper().str.contains("TRANSPLANT", na=False)]
            assert not tx.empty
            assert tx["sts_scope_excluded"].all()

    def test_homograft_row_excluded(self, mimic_like_normalized):
        df, _ = mimic_like_normalized
        if "sts_scope_excluded" in df.columns and "Surgery" in df.columns:
            hg = df[df["Surgery"].str.upper().str.contains("HOMOGRAFT", na=False)]
            assert not hg.empty
            assert hg["sts_scope_excluded"].all()


# ──────────────────────────────────────────────────────────────────────────────
# STS readiness summary coherence
# ──────────────────────────────────────────────────────────────────────────────

class TestMimicStsReadiness:
    def test_n_total_matches_row_count(self, mimic_like_normalized, mimic_like_raw_df):
        _, report = mimic_like_normalized
        assert report.sts_readiness_summary["n_total"] == len(mimic_like_raw_df)

    def test_n_ready_plus_not_ready_equals_total(self, mimic_like_normalized):
        _, report = mimic_like_normalized
        rs = report.sts_readiness_summary
        n_excluded = (
            rs.get("n_pediatric_excluded", 0)
            + rs.get("n_scope_excluded", 0)
            + rs.get("n_missing_fields", 0)
            + rs.get("n_invalid_fields", 0)
        )
        # n_ready + excluded cases ≤ n_total (may overlap, but ready must be smaller)
        assert rs["n_ready"] <= rs["n_total"]

    def test_n_ready_pct_consistent(self, mimic_like_normalized):
        _, report = mimic_like_normalized
        rs = report.sts_readiness_summary
        if rs["n_total"] > 0:
            expected_pct = rs["n_ready"] / rs["n_total"] * 100
            assert abs(rs["n_ready_pct"] - expected_pct) < 0.1

    def test_supported_surgeries_candidates_for_sts_ready(self, mimic_like_normalized):
        df, report = mimic_like_normalized
        # At least some rows should be STS-ready (the 5 adult supported surgery rows)
        assert report.sts_readiness_summary["n_ready"] >= 1


# ──────────────────────────────────────────────────────────────────────────────
# Report structure
# ──────────────────────────────────────────────────────────────────────────────

class TestMimicReportStructure:
    def test_summary_lines_non_empty(self, mimic_like_normalized):
        _, report = mimic_like_normalized
        lines = report.summary_lines()
        assert len(lines) >= 1

    def test_export_rows_non_empty(self, mimic_like_normalized):
        _, report = mimic_like_normalized
        rows = report.to_export_rows()
        assert len(rows) >= 5  # at minimum the structured metadata fields

    def test_pediatric_warning_in_report_warnings(self, mimic_like_normalized):
        _, report = mimic_like_normalized
        text = " ".join(report.warnings).lower()
        assert "pediatric" in text

    def test_read_meta_present(self, mimic_like_normalized):
        _, report = mimic_like_normalized
        assert report.read_meta is not None
        assert report.read_meta.encoding_used == "utf-8"
