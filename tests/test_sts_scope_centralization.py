"""Priority 1 regression tests — centralized STS scope exclusion keyword source.

Verifies that classify_sts_eligibility (runtime STS gate) and
apply_external_scope_rules (normalization pipeline) both draw from the single
canonical STS_UNSUPPORTED_SURGERY_KEYWORDS constant in sts_calculator.py and
classify procedures identically.
"""
import pandas as pd
import pytest

from sts_calculator import STS_UNSUPPORTED_SURGERY_KEYWORDS, classify_sts_eligibility
from risk_data import apply_external_scope_rules


# ──────────────────────────────────────────────────────────────────────────────
# STS_UNSUPPORTED_SURGERY_KEYWORDS content
# ──────────────────────────────────────────────────────────────────────────────

class TestStsUnsupportedSurgeryKeywords:
    def test_is_frozenset(self):
        assert isinstance(STS_UNSUPPORTED_SURGERY_KEYWORDS, frozenset)

    def test_contains_core_keywords(self):
        for kw in ("DISSECTION", "ANEURISM", "ANEURYSM", "BENTALL",
                   "AORTIC ROOT REPLACEMENT", "AORTIC ROOT REPAIR",
                   "AORTIC REPAIR", "AORTIC RECONSTRUCTION", "AORTA REPAIR"):
            assert kw in STS_UNSUPPORTED_SURGERY_KEYWORDS, f"Missing keyword: {kw}"

    def test_contains_extended_keywords(self):
        """ROSS, HOMOGRAFT, TRANSPLANT must be in the canonical set."""
        for kw in ("ROSS", "HOMOGRAFT", "TRANSPLANT"):
            assert kw in STS_UNSUPPORTED_SURGERY_KEYWORDS, f"Missing extended keyword: {kw}"

    def test_minimum_size(self):
        """At least 12 keywords (original 9 + ROSS + HOMOGRAFT + TRANSPLANT)."""
        assert len(STS_UNSUPPORTED_SURGERY_KEYWORDS) >= 12


# ──────────────────────────────────────────────────────────────────────────────
# classify_sts_eligibility uses the canonical set
# ──────────────────────────────────────────────────────────────────────────────

class TestClassifyStsEligibilityUsesCanonicalKeywords:
    @pytest.mark.parametrize("surgery,kw", [
        ("AORTIC DISSECTION REPAIR", "DISSECTION"),
        ("AORTIC ANEURYSM REPAIR", "ANEURYSM"),
        ("BENTALL PROCEDURE", "BENTALL"),
        ("ROSS PROCEDURE", "ROSS"),
        ("HOMOGRAFT REPLACEMENT", "HOMOGRAFT"),
        ("CARDIAC TRANSPLANT", "TRANSPLANT"),
    ])
    def test_excluded_surgery_is_not_supported(self, surgery, kw):
        status, reason = classify_sts_eligibility({"surgery_pre": surgery})
        assert status == "not_supported", (
            f"Expected not_supported for {surgery!r} (keyword {kw!r}), got {status!r}"
        )

    @pytest.mark.parametrize("surgery", ["ISOLATED CABG", "AVR", "MVR", "CABG + AVR"])
    def test_supported_surgery_not_excluded(self, surgery):
        status, _ = classify_sts_eligibility({
            "surgery_pre": surgery,
            "surgical_priority": "Elective",
            "sex": "Male",
            "age_years": 65,
        })
        assert status != "not_supported", f"Supported surgery {surgery!r} was excluded"


# ──────────────────────────────────────────────────────────────────────────────
# apply_external_scope_rules uses the canonical set (via imported constant)
# ──────────────────────────────────────────────────────────────────────────────

class TestApplyExternalScopeRulesUsesCanonicalKeywords:
    @pytest.mark.parametrize("surgery,kw", [
        ("AORTIC DISSECTION REPAIR", "DISSECTION"),
        ("AORTIC ANEURYSM", "ANEURYSM"),
        ("BENTALL PROCEDURE", "BENTALL"),
        ("ROSS PROCEDURE", "ROSS"),
        ("HOMOGRAFT REPLACEMENT", "HOMOGRAFT"),
        ("CARDIAC TRANSPLANT", "TRANSPLANT"),
    ])
    def test_excluded_surgery_flagged(self, surgery, kw):
        df = pd.DataFrame([
            {"age_years": 65, "surgery_pre": surgery},
        ])
        result, summary = apply_external_scope_rules(df)
        assert result["sts_scope_excluded"].iloc[0], (
            f"Expected sts_scope_excluded=True for {surgery!r} (keyword {kw!r})"
        )
        assert summary["n_sts_scope_excluded"] >= 1

    @pytest.mark.parametrize("surgery", ["ISOLATED CABG", "AVR", "MVR"])
    def test_supported_surgery_not_flagged(self, surgery):
        df = pd.DataFrame([
            {"age_years": 65, "surgery_pre": surgery},
        ])
        result, summary = apply_external_scope_rules(df)
        assert not result["sts_scope_excluded"].iloc[0], (
            f"Supported surgery {surgery!r} was incorrectly excluded"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Cross-path consistency: both paths agree on every keyword
# ──────────────────────────────────────────────────────────────────────────────

class TestCrossPathConsistency:
    """Both classify_sts_eligibility and apply_external_scope_rules must agree
    for every keyword in STS_UNSUPPORTED_SURGERY_KEYWORDS."""

    @pytest.mark.parametrize("kw", sorted(STS_UNSUPPORTED_SURGERY_KEYWORDS))
    def test_both_paths_exclude_same_surgery(self, kw):
        surgery = f"TEST {kw} PROCEDURE"

        # classify_sts_eligibility path
        sts_status, _ = classify_sts_eligibility({"surgery_pre": surgery})

        # apply_external_scope_rules path
        df = pd.DataFrame([{"age_years": 65, "surgery_pre": surgery}])
        result, _ = apply_external_scope_rules(df)
        scope_flag = bool(result["sts_scope_excluded"].iloc[0])

        assert sts_status == "not_supported", (
            f"classify_sts_eligibility did not exclude surgery containing {kw!r}"
        )
        assert scope_flag, (
            f"apply_external_scope_rules did not exclude surgery containing {kw!r}"
        )
