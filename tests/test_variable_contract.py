"""Tests for variable_contract.py and derived frozensets in risk_data.py."""

import pytest
from variable_contract import VARIABLE_CONTRACT
from risk_data import (
    FLAT_PREOP_ALLOWED_COLUMNS,
    BLANK_MEANS_NO_COLUMNS,
    BLANK_MEANS_NONE_COLUMNS,
    NONE_IS_VALID_COLUMNS,
    LITERAL_NONE_IS_VALID_COLUMNS,
)


# ── Contract completeness ─────────────────────────────────────────────────────

def test_all_flat_preop_allowed_in_contract():
    """Every canonical preop column should have a contract entry."""
    # A few are intentionally excluded (noise/removed from features):
    exclude_from_check = {
        "Classification of Heart Failure According to Ejection Fraction",
        "Preoperative Medications",
    }
    missing = [
        col for col in FLAT_PREOP_ALLOWED_COLUMNS
        if col not in VARIABLE_CONTRACT and col not in exclude_from_check
    ]
    assert missing == [], f"Variables in FLAT_PREOP_ALLOWED_COLUMNS missing from contract: {missing}"


def test_contract_has_required_fields():
    required = {"dtype", "parse_mode", "blank_semantics", "none_is_valid"}
    for col, spec in VARIABLE_CONTRACT.items():
        missing = required - set(spec.keys())
        assert not missing, f"Contract entry '{col}' missing fields: {missing}"


def test_contract_dtype_values():
    valid_dtypes = {"numeric", "binary", "ordinal", "categorical", "text"}
    for col, spec in VARIABLE_CONTRACT.items():
        assert spec["dtype"] in valid_dtypes, f"'{col}' has invalid dtype '{spec['dtype']}'"


def test_contract_parse_mode_values():
    valid_modes = {"strict", "tolerant", "categorical"}
    for col, spec in VARIABLE_CONTRACT.items():
        assert spec["parse_mode"] in valid_modes, (
            f"'{col}' has invalid parse_mode '{spec['parse_mode']}'"
        )


def test_contract_blank_semantics_values():
    valid = {"absent", "unknown", "not_applicable"}
    for col, spec in VARIABLE_CONTRACT.items():
        assert spec["blank_semantics"] in valid, (
            f"'{col}' has invalid blank_semantics '{spec['blank_semantics']}'"
        )


# ── Derived frozensets match originals ───────────────────────────────────────

def test_blank_means_no_matches_original():
    """BLANK_MEANS_NO_COLUMNS derived from contract == original hardcoded set."""
    expected = frozenset({"Family Hx of CAD", "Anticoagulation/ Antiaggregation"})
    assert BLANK_MEANS_NO_COLUMNS == expected, (
        f"Got {sorted(BLANK_MEANS_NO_COLUMNS)}, expected {sorted(expected)}"
    )


def test_blank_means_none_matches_original():
    """BLANK_MEANS_NONE_COLUMNS derived from contract == original hardcoded set."""
    expected = frozenset({"Aortic Stenosis", "Arrhythmia Remote", "HF", "Previous surgery"})
    assert BLANK_MEANS_NONE_COLUMNS == expected, (
        f"Got {sorted(BLANK_MEANS_NONE_COLUMNS)}, expected {sorted(expected)}"
    )


def test_none_is_valid_canonical_matches_original():
    """Canonical (non-alias) members of NONE_IS_VALID_COLUMNS match original valve set."""
    expected_canonical = frozenset({
        "Aortic Stenosis", "Aortic Regurgitation",
        "Mitral Stenosis", "Mitral Regurgitation",
        "Tricuspid Regurgitation",
    })
    canonical_in_set = frozenset(
        k for k in NONE_IS_VALID_COLUMNS
        if not any(k.startswith(pfx) for pfx in ("aortic_", "mitral_", "tricuspid_"))
    )
    assert canonical_in_set == expected_canonical


def test_none_is_valid_aliases_present():
    """Internal flat-CSV aliases must be in NONE_IS_VALID_COLUMNS."""
    aliases = {
        "aortic_stenosis_pre", "aortic_regurgitation_pre",
        "mitral_stenosis_pre", "mitral_regurgitation_pre",
        "tricuspid_regurgitation_pre",
        "aortic_stenosis_post", "aortic_regurgitation_post",
    }
    assert aliases.issubset(NONE_IS_VALID_COLUMNS)


def test_literal_none_is_valid_contains_none_is_valid():
    """LITERAL_NONE_IS_VALID_COLUMNS must be a superset of NONE_IS_VALID_COLUMNS."""
    assert NONE_IS_VALID_COLUMNS.issubset(LITERAL_NONE_IS_VALID_COLUMNS)


def test_literal_none_is_valid_extra_columns():
    """LITERAL extends NONE_IS_VALID with the 6 expected extra columns."""
    extra = LITERAL_NONE_IS_VALID_COLUMNS - NONE_IS_VALID_COLUMNS
    expected_extra = frozenset({
        "Arrhythmia Recent", "Arrhythmia Remote",
        "Aortic Root Abscess", "HF",
        "Preoperative Medications", "Previous surgery",
    })
    assert extra == expected_extra, f"Got extra: {sorted(extra)}, expected: {sorted(expected_extra)}"


# ── Plausibility ranges ───────────────────────────────────────────────────────

def test_plausible_range_only_on_numeric():
    """Only numeric variables should have a plausible_range."""
    for col, spec in VARIABLE_CONTRACT.items():
        if "plausible_range" in spec:
            assert spec["dtype"] == "numeric", (
                f"'{col}' has plausible_range but dtype='{spec['dtype']}'"
            )


def test_plausible_range_tuple_two_elements():
    """plausible_range must be a 2-tuple (lo, hi)."""
    for col, spec in VARIABLE_CONTRACT.items():
        if "plausible_range" in spec:
            r = spec["plausible_range"]
            assert len(r) == 2, f"'{col}' plausible_range is not length-2: {r}"
            assert r[0] < r[1], f"'{col}' plausible_range lo >= hi: {r}"


def test_clinical_plausibility_ranges_derived_from_contract():
    """_CLINICAL_PLAUSIBILITY_RANGES must be derivable from the contract."""
    from risk_data import _CLINICAL_PLAUSIBILITY_RANGES
    expected = {
        k: v["plausible_range"]
        for k, v in VARIABLE_CONTRACT.items()
        if "plausible_range" in v and v.get("dtype") == "numeric"
    }
    assert _CLINICAL_PLAUSIBILITY_RANGES == expected


# ── Strict-parse variables covered ───────────────────────────────────────────

def test_strict_parse_columns_in_contract():
    """The strict-parse variables from the plan must be in the contract."""
    strict_cols = [
        "Age (years)", "Creatinine (mg/dL)", "Cr clearance, ml/min *",
        "Hematocrit (%)", "INR", "PTT",
        "Pré-LVEF, %", "BSA, m2", "Height (cm)", "Weight (kg)",
        "PSAP", "TAPSE",
    ]
    for col in strict_cols:
        assert col in VARIABLE_CONTRACT, f"'{col}' not found in VARIABLE_CONTRACT"
        assert VARIABLE_CONTRACT[col]["parse_mode"] == "strict", (
            f"'{col}' expected parse_mode='strict', got '{VARIABLE_CONTRACT[col]['parse_mode']}'"
        )


# ── Dtype correctness ─────────────────────────────────────────────────────────

def test_multi_category_variables_not_binary():
    """Variables with >2 clinical categories must not be dtype='binary'."""
    multi_category = {
        "Diabetes",                      # No/Oral/Insulin/Diet Only/No Control Method
        "CVA",                           # No/TIA/≤30d/≥30d
        "IE",                            # No/Yes/Possible
        "Cancer ≤ 5 yrs",               # No + specific cancer types
        "Anticoagulation/ Antiaggregation",  # No + medication regimens
        "Pneumonia",                     # No/Treated/Under treatment
    }
    for name in multi_category:
        spec = VARIABLE_CONTRACT.get(name)
        assert spec is not None, f"'{name}' not in VARIABLE_CONTRACT"
        assert spec["dtype"] != "binary", (
            f"'{name}' has multi-level clinical categories but dtype='{spec['dtype']}'. "
            "Should be 'categorical'."
        )


def test_blank_impute_no_flag_consistency():
    """Every column in BLANK_MEANS_NO_COLUMNS must have blank_impute_no=True or be legacy-binary."""
    for name in BLANK_MEANS_NO_COLUMNS:
        spec = VARIABLE_CONTRACT.get(name)
        assert spec is not None, f"'{name}' in BLANK_MEANS_NO_COLUMNS but not in contract"
        has_flag = spec.get("blank_impute_no", False)
        is_legacy = spec.get("blank_semantics") == "absent" and spec.get("dtype") == "binary"
        assert has_flag or is_legacy, (
            f"'{name}' is in BLANK_MEANS_NO_COLUMNS but has neither blank_impute_no=True "
            "nor the legacy (blank_semantics=absent AND dtype=binary) combination"
        )
