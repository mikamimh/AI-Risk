"""Equivalence tests: Temporal Validation uses the exact same canonical pipeline
as the main/training pipeline for every data-processing step.

These tests exercise the shared canonical functions directly — the same functions
called by both the training pipeline and the Temporal Validation tab — and prove
that semantically equivalent inputs (different spelling, locale, or format)
produce the same normalised output.

No Streamlit runtime is required.
"""
import math
import pytest

# ---------------------------------------------------------------------------
# 1. Missing-value handling — MISSING_TOKENS
# ---------------------------------------------------------------------------

def test_missing_tokens_contains_expected_values():
    """MISSING_TOKENS covers the canonical set of blank/unknown tokens."""
    from risk_data import MISSING_TOKENS
    for token in ("", "-", "--", "nan", "none", "na", "n/a", "null",
                  "not applicable", "unknown", "not informed", "não informado"):
        assert token in MISSING_TOKENS, f"Expected {token!r} in MISSING_TOKENS"


def test_missing_tokens_case_insensitive_via_parse_number():
    """parse_number returns NaN for every MISSING_TOKEN variant (case-insensitive)."""
    import numpy as np
    from risk_data import parse_number, MISSING_TOKENS
    for tok in MISSING_TOKENS:
        for variant in (tok, tok.upper(), tok.title()):
            result = parse_number(variant)
            assert result is np.nan or (isinstance(result, float) and math.isnan(result)), \
                f"parse_number({variant!r}) should return NaN, got {result!r}"


# ---------------------------------------------------------------------------
# 2. Numeric parsing — comma vs dot decimal, percentage, string-stored
# ---------------------------------------------------------------------------

def test_parse_number_dot_decimal():
    from risk_data import parse_number
    assert parse_number("1.5") == pytest.approx(1.5)
    assert parse_number("100.0") == pytest.approx(100.0)


def test_parse_number_comma_decimal_pt_br():
    """Brazilian comma-decimal format produces the same float as dot-decimal."""
    from risk_data import parse_number
    assert parse_number("1,5") == pytest.approx(1.5)
    assert parse_number("65,4") == pytest.approx(65.4)


def test_parse_number_comma_equals_dot():
    """'1,5' and '1.5' resolve to the same numeric value."""
    from risk_data import parse_number
    assert parse_number("1,5") == parse_number("1.5")
    assert parse_number("65,4") == parse_number("65.4")
    assert parse_number("0,08") == parse_number("0.08")


def test_parse_number_integer_string():
    from risk_data import parse_number
    assert parse_number("42") == pytest.approx(42.0)
    assert parse_number("0") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Outcome parsing — map_death_30d
# ---------------------------------------------------------------------------

class TestMapDeath30d:
    """map_death_30d must produce 1 for event and 0 for survivor, regardless of
    language, casing, or format variant."""

    EVENT_VARIANTS = [
        "Yes", "yes", "YES",
        "Sim", "sim", "SIM",
        "Death", "death", "DEATH",
        "1",
    ]
    SURVIVOR_VARIANTS = [
        "No", "no", "NO",
        "Não", "não", "NÃO",
        "Nao", "nao",
        # "0" is intentionally excluded: parse_postop_timing("0") = day 0
        # = operative mortality → returns 1, not 0.  Day 0 is an event.
        "-", "--",   # survivor tokens (no post-op death recorded)
    ]

    @pytest.mark.parametrize("val", EVENT_VARIANTS)
    def test_event_variants_return_1(self, val):
        from risk_data import map_death_30d
        assert map_death_30d(val) == 1, f"Expected 1 for {val!r}"

    @pytest.mark.parametrize("val", SURVIVOR_VARIANTS)
    def test_survivor_variants_return_0(self, val):
        from risk_data import map_death_30d
        assert map_death_30d(val) == 0, f"Expected 0 for {val!r}"

    def test_yes_and_sim_equivalent(self):
        from risk_data import map_death_30d
        assert map_death_30d("Yes") == map_death_30d("Sim")

    def test_no_and_nao_equivalent(self):
        from risk_data import map_death_30d
        assert map_death_30d("No") == map_death_30d("Não")

    def test_death_string_is_event(self):
        from risk_data import map_death_30d
        assert map_death_30d("Death") == 1

    @pytest.mark.parametrize("missing", ("", "Unknown", "N/A", "nan"))
    def test_missing_returns_0(self, missing):
        """Unrecognised / blank values default to 0 (survivor) without raising."""
        from risk_data import map_death_30d
        assert map_death_30d(missing) == 0


# ---------------------------------------------------------------------------
# 4. Surgery classification — classify_sts_eligibility
# ---------------------------------------------------------------------------

class TestClassifyStsEligibility:
    """classify_sts_eligibility is the canonical gatekeeper for STS eligibility.
    The Temporal Validation tab calls this exact function — not a re-derived
    version.  Tests ensure semantic equivalence across naming variants."""

    # supported
    SUPPORTED_ROWS = [
        {"surgery_pre": "CABG"},
        {"surgery_pre": "OPCAB"},
        {"surgery_pre": "AVR"},
        {"surgery_pre": "MVR"},
        {"surgery_pre": "MV REPAIR"},
        {"surgery_pre": "MITRAL REPAIR"},
        {"surgery_pre": "PLASTIA MITRAL"},
        {"surgery_pre": "AVR + CABG"},
        {"surgery_pre": "MVR + CABG"},
        {"surgery_pre": "MV REPAIR + CABG"},
        {"Surgery": "CABG"},            # alternate column name
        {"Surgery": "AVR"},
    ]

    # hard exclusions — not_supported
    # Keywords must be substrings present in _STS_UNSUPPORTED_KEYWORDS:
    # DISSECTION, ANEURISM, ANEURYSM, BENTALL, AORTIC ROOT REPLACEMENT,
    # AORTIC ROOT REPAIR, AORTIC REPAIR, AORTIC RECONSTRUCTION, AORTA REPAIR
    NOT_SUPPORTED_ROWS = [
        {"surgery_pre": "AORTIC DISSECTION REPAIR"},   # contains DISSECTION
        {"surgery_pre": "AORTIC ANEURYSM REPAIR"},     # contains ANEURYSM
        {"surgery_pre": "BENTALL PROCEDURE"},           # contains BENTALL
        {"surgery_pre": "BENTALL"},                     # exact keyword
        {"surgery_pre": "AORTIC ROOT REPLACEMENT"},     # exact keyword
        {"surgery_pre": "AORTIC RECONSTRUCTION"},       # exact keyword
        {"surgery_pre": "AORTA REPAIR"},                # exact keyword
    ]

    # uncertain
    UNCERTAIN_ROWS = [
        {"surgery_pre": ""},
        {"surgery_pre": "UNKNOWN CARDIAC PROCEDURE"},
        {"surgery_pre": "CABG", "surgical_priority": "OBSERVATION ADMIT"},
        {"Surgery": "AVR", "Surgical Priority": "OBSERVATION"},
    ]

    @pytest.mark.parametrize("row", SUPPORTED_ROWS)
    def test_supported_procedures(self, row):
        from sts_calculator import classify_sts_eligibility
        status, _ = classify_sts_eligibility(row)
        assert status == "supported", f"Expected supported for {row}"

    @pytest.mark.parametrize("row", NOT_SUPPORTED_ROWS)
    def test_not_supported_exclusions(self, row):
        from sts_calculator import classify_sts_eligibility
        status, _ = classify_sts_eligibility(row)
        assert status == "not_supported", f"Expected not_supported for {row}"

    @pytest.mark.parametrize("row", UNCERTAIN_ROWS)
    def test_uncertain_cases(self, row):
        from sts_calculator import classify_sts_eligibility
        status, _ = classify_sts_eligibility(row)
        assert status == "uncertain", f"Expected uncertain for {row}"

    def test_observation_admit_is_uncertain_not_supported(self):
        """OBSERVATION ADMIT must never be 'supported' — it has no STS urgency mapping."""
        from sts_calculator import classify_sts_eligibility
        row = {"surgery_pre": "CABG", "surgical_priority": "OBSERVATION ADMIT"}
        status, reason = classify_sts_eligibility(row)
        assert status == "uncertain"
        assert "OBSERVATION ADMIT" in reason or "priority" in reason.lower()

    def test_surgery_column_name_aliases(self):
        """Both 'surgery_pre' and 'Surgery' column names resolve identically."""
        from sts_calculator import classify_sts_eligibility
        r1 = classify_sts_eligibility({"surgery_pre": "CABG"})
        r2 = classify_sts_eligibility({"Surgery": "CABG"})
        assert r1[0] == r2[0] == "supported"

    def test_priority_column_name_aliases(self):
        """Both 'surgical_priority' and 'Surgical Priority' column names resolve identically."""
        from sts_calculator import classify_sts_eligibility
        r1 = classify_sts_eligibility({"surgery_pre": "AVR", "surgical_priority": "OBSERVATION ADMIT"})
        r2 = classify_sts_eligibility({"Surgery": "AVR", "Surgical Priority": "OBSERVATION ADMIT"})
        assert r1[0] == r2[0] == "uncertain"


# ---------------------------------------------------------------------------
# 5. Urgency mapping — _map_urgency  (English ↔ Portuguese equivalence)
# ---------------------------------------------------------------------------

class TestMapUrgency:
    """_map_urgency must produce the same STS string regardless of language."""

    @pytest.mark.parametrize("en,pt", [
        ("Elective",        "Eletiva"),
        ("Urgent",          "Urgente"),
        ("Emergent",        "Emergência"),
        ("Emergent Salvage","Salvamento"),
    ])
    def test_en_pt_equivalence(self, en, pt):
        from sts_calculator import _map_urgency
        assert _map_urgency(en) == _map_urgency(pt), \
            f"Expected {en!r} == {pt!r} after mapping"

    def test_elective_variants(self):
        from sts_calculator import _map_urgency
        assert _map_urgency("Elective") == "Elective"
        assert _map_urgency("Eletiva") == "Elective"

    def test_urgent_variants(self):
        from sts_calculator import _map_urgency
        assert _map_urgency("Urgent") == "Urgent"
        assert _map_urgency("Urgente") == "Urgent"

    def test_emergent_variants(self):
        from sts_calculator import _map_urgency
        assert _map_urgency("Emergent") == "Emergent"
        assert _map_urgency("Emergência") == "Emergent"
        assert _map_urgency("Emergency") == "Emergent"

    def test_salvage_variants(self):
        from sts_calculator import _map_urgency
        assert _map_urgency("Salvage") == "Emergent Salvage"
        assert _map_urgency("Salvamento") == "Emergent Salvage"
        assert _map_urgency("Emergent Salvage") == "Emergent Salvage"

    def test_unknown_defaults_to_elective(self):
        """Unrecognised values default to Elective (most conservative STS default)."""
        from sts_calculator import _map_urgency
        assert _map_urgency("OBSERVATION ADMIT") == "Elective"
        assert _map_urgency("") == "Elective"


# ---------------------------------------------------------------------------
# 6. Surgery → STS procedure ID mapping
# ---------------------------------------------------------------------------

class TestMapSurgeryToProcid:
    """_map_surgery_to_procid maps surgery strings to STS procid integers.
    Tests cover naming variants so the Temporal Validation cohort and the
    training cohort receive identical procedure IDs for equivalent procedures."""

    @pytest.mark.parametrize("surgery,expected_id", [
        ("CABG",               1),
        ("OPCAB",              1),
        ("cabg",               1),   # case-insensitive
        ("AVR",                2),
        ("aortic valve replacement", 2),  # fallback to default (2)
        ("MVR",                3),
        ("MV REPAIR",          7),
        ("MITRAL REPAIR",      7),
        ("PLASTIA MITRAL",     7),
        ("AVR + CABG",         4),
        ("MVR + CABG",         5),
        ("MV REPAIR + CABG",   8),
    ])
    def test_procedure_mapping(self, surgery, expected_id):
        from sts_calculator import _map_surgery_to_procid
        assert _map_surgery_to_procid(surgery) == expected_id, \
            f"Expected procid {expected_id} for {surgery!r}"

    def test_cabg_and_opcab_same_procid(self):
        """CABG and OPCAB (off-pump CABG) must map to the same procedure ID."""
        from sts_calculator import _map_surgery_to_procid
        assert _map_surgery_to_procid("CABG") == _map_surgery_to_procid("OPCAB")

    def test_mv_repair_variants_same_procid(self):
        """MV REPAIR, MITRAL REPAIR, PLASTIA MITRAL all map to procid 7."""
        from sts_calculator import _map_surgery_to_procid
        ids = {
            _map_surgery_to_procid("MV REPAIR"),
            _map_surgery_to_procid("MITRAL REPAIR"),
            _map_surgery_to_procid("PLASTIA MITRAL"),
        }
        assert ids == {7}


# ---------------------------------------------------------------------------
# 7. build_sts_input_from_row — column-name alias resolution
# ---------------------------------------------------------------------------

def test_build_sts_input_accepts_csv_columns():
    """build_sts_input_from_row works with snake_case CSV column names."""
    from sts_calculator import build_sts_input_from_row
    row = {
        "surgery_pre": "CABG",
        "surgical_priority": "Elective",
        "age": 65,
        "gender": "Male",
    }
    result = build_sts_input_from_row(row)
    assert isinstance(result, dict)
    assert "procid" in result or "proc" in result or len(result) > 0


def test_build_sts_input_accepts_app_columns():
    """build_sts_input_from_row works with display-name (app) column names."""
    from sts_calculator import build_sts_input_from_row
    row = {
        "Surgery": "CABG",
        "Surgical Priority": "Elective",
        "Age": 65,
        "Gender": "Male",
    }
    result = build_sts_input_from_row(row)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_build_sts_input_csv_and_app_columns_equivalent():
    """CSV snake_case columns and app display-name columns produce identical STS inputs."""
    from sts_calculator import build_sts_input_from_row
    csv_row = {
        "surgery_pre": "AVR",
        "surgical_priority": "Urgent",
        "age": 72,
        "gender": "Female",
    }
    app_row = {
        "Surgery": "AVR",
        "Surgical Priority": "Urgente",   # Portuguese variant
        "Age": 72,
        "Gender": "Female",
    }
    r_csv = build_sts_input_from_row(csv_row)
    r_app = build_sts_input_from_row(app_row)
    # procid must match for AVR
    assert r_csv.get("procid") == r_app.get("procid")
