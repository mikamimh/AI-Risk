"""Guard tests for binary direct encoding.

The set `_BINARY_DIRECT_ENCODE_COLS` in `modeling.py` bypasses TargetEncoder
for truly binary Yes/No clinical variables. Adding a multi-level variable
to that set silently collapses clinically informative categories to NaN
(e.g. Diabetes "Oral"/"Insulin"/"Diet Only" would all be lost).

Ablation V2 (20 seeds) showed that when 7 multi-level columns were
mistakenly treated as binary, AUC dropped from 0.744 to 0.733. This guard
prevents that regression.

See ADR-002 in docs/decisions/ for the full encoding policy.
"""

import pytest

from modeling import _BINARY_DIRECT_ENCODE_COLS, _POSITIVE_TOKENS, _NEGATIVE_TOKENS


# Columns with clinically informative multi-level categories.
# These MUST be routed through TargetEncoder, never through
# binary direct encoding.
#
# Reference values observed in Dataset_2025.xlsx:
#   Diabetes                   -> No | Oral | Insulin | Diet Only | No Control Method
#   CVA                        -> No | TIA | <= 30 days | >= 30 days
#   IE                         -> No | Yes | Possible
#   Cancer <= 5 yrs            -> No | Bowel | Breast | (other tumor types)
#   Anticoagulation/Antiagg.   -> No | AAS | Clopidogrel | (other regimens)
#   Pneumonia                  -> No | Treated | Under treatment
#   Family Hx of CAD           -> Yes | No | blank (blank_means_no applies,
#                                kept categorical for borderline values)
_MULTILEVEL_COLUMNS_FORBIDDEN_FROM_BINARY = frozenset({
    "Diabetes",
    "CVA",
    "IE",
    "Cancer ≤ 5 yrs",
    "Anticoagulation/ Antiaggregation",
    "Pneumonia",
    "Family Hx of CAD",
})


def test_multilevel_columns_are_not_binary_direct_encoded():
    """Fail if any multi-level clinical column is in _BINARY_DIRECT_ENCODE_COLS.

    Adding a multi-level variable to the binary set would silently
    collapse clinically informative categories to NaN after
    `_encode_binary_direct()` and degrade model performance.
    """
    overlap = _BINARY_DIRECT_ENCODE_COLS & _MULTILEVEL_COLUMNS_FORBIDDEN_FROM_BINARY
    assert not overlap, (
        f"Clinically multi-level columns must be routed through TargetEncoder, "
        f"not binary direct encoding. Offending columns: {sorted(overlap)}. "
        f"See ADR-002 in docs/decisions/ and the ablation V2 results."
    )


def test_binary_tokens_do_not_include_multilevel_artifacts():
    """Fail if tokens specific to multi-level variables are in the binary sets.

    Tokens like 'possible' (IE), 'treated' / 'active' (Pneumonia) belong to
    multi-level categorical variables and must not be part of the binary
    Yes/No token vocabulary. Their presence would be a sign that someone
    is considering re-adding those variables to _BINARY_DIRECT_ENCODE_COLS,
    which would be a regression.
    """
    forbidden_positive_tokens = frozenset({"possible", "active", "treated"})
    leaked = _POSITIVE_TOKENS & forbidden_positive_tokens
    assert not leaked, (
        f"These tokens belong to multi-level variables (IE: 'possible'; "
        f"Pneumonia: 'active', 'treated') and must not be in _POSITIVE_TOKENS. "
        f"Leaked tokens: {sorted(leaked)}. "
        f"See ADR-002 in docs/decisions/."
    )


def test_binary_token_sets_are_disjoint():
    """Sanity: positive and negative token vocabularies must not overlap."""
    overlap = _POSITIVE_TOKENS & _NEGATIVE_TOKENS
    assert not overlap, (
        f"_POSITIVE_TOKENS and _NEGATIVE_TOKENS must be disjoint. "
        f"Overlap: {sorted(overlap)}"
    )


def test_binary_direct_encode_cols_is_nonempty():
    """Sanity: binary direct encoding list must not be empty.

    Empties would silently revert all binary variables to TargetEncoder
    and re-introduce the encoding noise that ablation V2 eliminated.
    """
    assert len(_BINARY_DIRECT_ENCODE_COLS) > 0, (
        "_BINARY_DIRECT_ENCODE_COLS is empty. Binary Yes/No variables would "
        "revert to TargetEncoder, re-introducing encoding noise. "
        "Expected 11 columns as of v14 (2026-04-23). See ADR-002."
    )


def test_expected_binary_columns_are_present():
    """Lock in the 11 columns validated by ablation V2 over 20 seeds.

    If this test fails, the binary list was modified. Update the expected
    set below AND document the change in the bundle changelog and in
    ADR-002. Do not blindly update this test.
    """
    expected = frozenset({
        "Left Main Stenosis ≥ 50%",
        "Proximal LAD Stenosis ≥ 70%",
        "CCS4",
        "Hypertension",
        "Dyslipidemia",
        "PVD",
        "Alcohol",
        "Dialysis",
        "Chronic Lung Disease",
        "Critical preoperative state",
        "Poor mobility",
    })
    missing = expected - _BINARY_DIRECT_ENCODE_COLS
    extra = _BINARY_DIRECT_ENCODE_COLS - expected
    assert not missing and not extra, (
        f"_BINARY_DIRECT_ENCODE_COLS does not match the v14 baseline. "
        f"Missing: {sorted(missing)}. Extra: {sorted(extra)}. "
        f"If this change is intentional, update this test and document "
        f"in ADR-002 with a new ablation."
    )
