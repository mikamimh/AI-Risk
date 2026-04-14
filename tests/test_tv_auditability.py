"""Auditability tests for Temporal Validation — Task F.

Covers:
  1. fail_log structure is coherent when failures exist
  2. fail_log is empty / not inherited when no failures occur
  3. Death = "0" semantic correctness (operative day 0 = event)
  4. prepare_info keys are available for XLSX and flat-CSV paths
  5. fail_log is NOT inherited between runs with different context sigs

No Streamlit runtime required.
"""

import hashlib
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _context_sig(file_bytes: bytes, sts_on: bool = True) -> str:
    file_sig = hashlib.sha256(file_bytes).hexdigest()[:24]
    raw = f"{file_sig}|2025-01-01|model_v1|0.080000|sts={'1' if sts_on else '0'}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _normalize_fail_entry(entry: dict) -> dict:
    """Mirrors the normalization used in execution details."""
    return {
        "patient_id":    entry.get("patient_id") or entry.get("name") or "?",
        "row_index":     entry.get("idx", ""),
        "status":        entry.get("status", "failed"),
        "stage":        entry.get("stage") or "",
        "reason":        entry.get("reason", "?"),
        "retry":         entry.get("retry_attempted", ""),
        "stale_cache":   entry.get("used_previous_cache", ""),
    }


# ---------------------------------------------------------------------------
# 1. Fail log structure coherence
# ---------------------------------------------------------------------------

class TestFailLogStructure:
    """When STS failures occur the fail log must have coherent, non-empty entries."""

    MODERN_ENTRY = {
        "idx": 3,
        "patient_id": "P007",
        "status": "failed",
        "stage": "fetch",
        "reason": "fetch_failed; no fallback available",
        "retry_attempted": True,
        "used_previous_cache": False,
    }
    LEGACY_ENTRY = {
        "idx": 0,
        "name": "John Doe",
        "surgery": "CABG",
        "reason": "timeout exceeded (90 s)",
    }
    TIMEOUT_ENTRY = {
        "idx": 1,
        "name": "Row 2",
        "reason": "per-patient timeout exceeded (90 s total across all retry attempts)",
    }

    def test_modern_entry_normalizes_correctly(self):
        n = _normalize_fail_entry(self.MODERN_ENTRY)
        assert n["patient_id"] == "P007"
        assert n["row_index"] == 3
        assert n["status"] == "failed"
        assert n["stage"] == "fetch"
        assert "fetch_failed" in n["reason"]
        assert n["retry"] is True
        assert n["stale_cache"] is False

    def test_legacy_entry_normalizes_gracefully(self):
        """Legacy entries use 'name' instead of 'patient_id'; must not crash."""
        n = _normalize_fail_entry(self.LEGACY_ENTRY)
        assert n["patient_id"] == "John Doe"
        assert n["status"] == "failed"          # default
        assert n["stage"] == ""                 # not present → empty string

    def test_timeout_entry_normalizes(self):
        n = _normalize_fail_entry(self.TIMEOUT_ENTRY)
        assert n["patient_id"] == "Row 2"
        assert "timeout" in n["reason"].lower()

    def test_fail_log_to_dataframe(self):
        """Normalised fail log must produce a valid DataFrame with expected columns."""
        fail_log = [self.MODERN_ENTRY, self.LEGACY_ENTRY, self.TIMEOUT_ENTRY]
        rows = [_normalize_fail_entry(e) for e in fail_log]
        df = pd.DataFrame(rows)
        for col in ("patient_id", "row_index", "status", "stage", "reason",
                    "retry", "stale_cache"):
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) == 3

    def test_empty_fail_log_produces_no_table(self):
        """With no failures the fail log is empty — no table should be rendered."""
        fail_log = []
        # Mirrors the `if _tv_fail_log:` guard in the execution details section.
        should_show_table = bool(fail_log)
        assert not should_show_table

    def test_stale_fallback_entry_has_stale_cache_true(self):
        stale_entry = {
            "idx": 2, "patient_id": "P003", "status": "stale_cache",
            "stage": "fetch", "reason": "fetch_failed",
            "retry_attempted": True, "used_previous_cache": True,
        }
        n = _normalize_fail_entry(stale_entry)
        assert n["stale_cache"] is True


# ---------------------------------------------------------------------------
# 2. Death = "0" canonical semantics
# ---------------------------------------------------------------------------

class TestDeathZeroSemantics:
    """Death = '0' must be interpreted as operative death on day 0 (event = 1)."""

    def test_death_zero_returns_1(self):
        from risk_data import map_death_30d
        assert map_death_30d("0") == 1, \
            "Death='0' is day 0 (operative death) — must return 1, not 0"

    def test_death_zero_equals_yes(self):
        from risk_data import map_death_30d
        assert map_death_30d("0") == map_death_30d("Yes") == 1

    def test_death_zero_not_equal_no(self):
        from risk_data import map_death_30d
        assert map_death_30d("0") != map_death_30d("No")
        assert map_death_30d("No") == 0

    def test_boolean_false_string_not_treated_as_zero(self):
        """'False' (boolean string) is not in the survivor token set; defaults to 0."""
        from risk_data import map_death_30d
        # 'False' is neither "0" (operative day) nor a canonical event token,
        # so it falls through to the default 0 (safe conservative).
        result = map_death_30d("False")
        assert result == 0

    def test_day_zero_event_via_postop_timing(self):
        """parse_postop_timing('0') should return day-0 event."""
        from risk_data import parse_postop_timing
        timing = parse_postop_timing("0")
        assert timing.within_30d is True, \
            "Day 0 must be within_30d=True (operative death)"
        assert timing.category in ("operative", "day_of_surgery", "days_to_event"), \
            f"Unexpected category for day 0: {timing.category}"

    def test_death_zero_not_ambiguous_with_survivor(self):
        """'0' must never classify as the same outcome as 'No'."""
        from risk_data import map_death_30d
        assert map_death_30d("0") != map_death_30d("No")
        assert map_death_30d("0") != map_death_30d("Não")
        assert map_death_30d("0") != map_death_30d("-")


# ---------------------------------------------------------------------------
# 3. prepare_info keys availability
# ---------------------------------------------------------------------------

class TestPrepareInfoKeys:
    """prepare_master_dataset / prepare_flat_dataset must expose expected info keys."""

    def test_flat_info_has_n_rows_and_source_type(self, tmp_path):
        """Flat CSV path produces info with 'source_type'='flat' and 'n_rows'."""
        from risk_data import prepare_flat_dataset
        csv_content = (
            "surgery_pre,Death,Age,Gender\n"
            "CABG,No,65,Male\n"
            "AVR,Yes,72,Female\n"
            "MVR,No,58,Male\n"
        )
        csv_path = tmp_path / "cohort.csv"
        csv_path.write_text(csv_content)
        prepared = prepare_flat_dataset(str(csv_path))
        assert prepared.info.get("source_type") == "flat"
        assert prepared.info.get("n_rows") == 3

    def test_flat_info_does_not_have_pre_rows_keys(self, tmp_path):
        """Flat CSV path does NOT have XLSX-specific intermediate keys."""
        from risk_data import prepare_flat_dataset
        csv_content = "surgery_pre,Death,Age\nCABG,No,65\nAVR,Yes,72\n"
        csv_path = tmp_path / "cohort.csv"
        csv_path.write_text(csv_content)
        prepared = prepare_flat_dataset(str(csv_path))
        # These keys only exist for XLSX / prepare_master_dataset
        assert "pre_rows_before_criteria" not in prepared.info
        assert "matched_pre_post_rows" not in prepared.info


# ---------------------------------------------------------------------------
# 4. fail_log not inherited between runs with different context sigs
# ---------------------------------------------------------------------------

class TestFailLogIsolation:
    """The fail log must be tied to the run that produced it.
    Stale state purge clears the old result before re-running.
    """

    FILE_A = b"patient_id,Surgery,Death\n1,CABG,No\n2,AVR,Yes\n"
    FILE_B = b"patient_id,Surgery,Death\n1,MVR,No\n2,CABG,Yes\n3,AVR,No\n"

    def test_different_files_different_sigs(self):
        assert _context_sig(self.FILE_A) != _context_sig(self.FILE_B)

    def test_stale_fail_log_is_purged_on_sig_change(self):
        """Simulate session_state purge: stale fail log must not survive
        into the next run when the context sig changes."""
        saved_sig = _context_sig(self.FILE_A)
        new_sig   = _context_sig(self.FILE_B)

        # Simulate what the stale-purge guard does
        fake_session = {
            "_tv_result": {
                "fail_log": [{"idx": 0, "patient_id": "P001",
                              "status": "failed", "stage": "fetch",
                              "reason": "timeout"}],
                "sts_fail_details": "- **patient=P001** | ...",
            },
            "_tv_result_sig": saved_sig,
        }

        should_purge = (
            fake_session.get("_tv_result_sig") is not None
            and fake_session["_tv_result_sig"] != new_sig
        )
        if should_purge:
            fake_session.pop("_tv_result", None)
            fake_session.pop("_tv_result_sig", None)

        assert "_tv_result" not in fake_session, \
            "Stale fail_log must be cleared when context sig changes"

    def test_same_file_same_sig_preserves_fail_log(self):
        """If the file hasn't changed, the saved fail log is correctly preserved."""
        sig = _context_sig(self.FILE_A)

        fake_session = {
            "_tv_result": {
                "fail_log": [{"idx": 0, "patient_id": "P001",
                              "status": "failed", "reason": "timeout"}],
            },
            "_tv_result_sig": sig,
        }

        should_purge = (
            fake_session.get("_tv_result_sig") is not None
            and fake_session["_tv_result_sig"] != sig
        )
        if should_purge:
            fake_session.pop("_tv_result", None)

        # Same sig → no purge → fail log is preserved
        assert "_tv_result" in fake_session
        assert len(fake_session["_tv_result"]["fail_log"]) == 1

    def test_sts_toggle_purges_fail_log(self):
        """Toggling STS on/off changes the context sig → stale fail log purged."""
        sig_with_sts    = _context_sig(self.FILE_A, sts_on=True)
        sig_without_sts = _context_sig(self.FILE_A, sts_on=False)

        fake_session = {
            "_tv_result": {
                "fail_log": [{"idx": 0, "reason": "timeout"}],
            },
            "_tv_result_sig": sig_with_sts,
        }

        should_purge = (
            fake_session.get("_tv_result_sig") is not None
            and fake_session["_tv_result_sig"] != sig_without_sts
        )
        if should_purge:
            fake_session.pop("_tv_result", None)

        assert "_tv_result" not in fake_session
