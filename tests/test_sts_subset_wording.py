"""Priority 6 tests — STS subset comparability wording.

Verifies that when STS n < full cohort n (PARTIAL availability):
  - comparator_scope_note is populated and mentions subset-only
  - comparator_scope_note explicitly mentions AI Risk model and EuroSCORE II at cohort level
  - The note appears in the PDF/Markdown report
  - COMPLETE status has no comparator_scope_note
  - UNAVAILABLE status has no comparator_scope_note
  - Portuguese translation is present for PARTIAL status
"""
import pandas as pd
import pytest

from tv_helpers import (
    build_sts_availability_summary,
    build_temporal_validation_summary,
    STS_AVAILABILITY_COMPLETE,
    STS_AVAILABILITY_PARTIAL,
    STS_AVAILABILITY_UNAVAILABLE,
)


# ──────────────────────────────────────────────────────────────────────────────
# build_sts_availability_summary — comparator_scope_note
# ──────────────────────────────────────────────────────────────────────────────

class TestComparatorScopeNote:
    def test_partial_has_comparator_scope_note(self):
        note = build_sts_availability_summary(n_eligible=100, n_score=70)
        assert note["comparator_scope_note"], "Expected non-empty comparator_scope_note for PARTIAL"

    def test_partial_note_mentions_subset_only(self):
        note = build_sts_availability_summary(n_eligible=100, n_score=70)
        text = note["comparator_scope_note"].lower()
        assert "subset" in text

    def test_partial_note_mentions_ai_risk(self):
        note = build_sts_availability_summary(n_eligible=100, n_score=70)
        text = note["comparator_scope_note"]
        assert "AI Risk" in text

    def test_partial_note_mentions_euroscore(self):
        note = build_sts_availability_summary(n_eligible=100, n_score=70)
        text = note["comparator_scope_note"]
        assert "EuroSCORE" in text

    def test_partial_note_mentions_cohort_level(self):
        note = build_sts_availability_summary(n_eligible=100, n_score=70)
        text = note["comparator_scope_note"].lower()
        assert "cohort" in text

    def test_partial_note_contains_n_score_and_n_eligible(self):
        note = build_sts_availability_summary(n_eligible=100, n_score=70)
        text = note["comparator_scope_note"]
        assert "70" in text
        assert "100" in text

    def test_complete_no_comparator_scope_note(self):
        note = build_sts_availability_summary(n_eligible=50, n_score=50)
        assert note["status"] == STS_AVAILABILITY_COMPLETE
        assert not note.get("comparator_scope_note", ""), (
            "COMPLETE status should not have comparator_scope_note"
        )

    def test_unavailable_no_comparator_scope_note(self):
        note = build_sts_availability_summary(n_eligible=50, n_score=0)
        assert note["status"] == STS_AVAILABILITY_UNAVAILABLE
        assert not note.get("comparator_scope_note", ""), (
            "UNAVAILABLE status should not have comparator_scope_note"
        )

    def test_partial_portuguese_comparator_note(self):
        note = build_sts_availability_summary(n_eligible=100, n_score=70, language="Portuguese")
        text = note["comparator_scope_note"]
        assert "subconjunto" in text.lower() or "subset" in text.lower()
        assert "EuroSCORE" in text

    def test_score_label_partial_says_subset(self):
        """score_label for PARTIAL should contain 'subset' keyword."""
        note = build_sts_availability_summary(n_eligible=100, n_score=70)
        label = note["score_label"].lower()
        assert "subset" in label


# ──────────────────────────────────────────────────────────────────────────────
# Report / PDF rendering
# ──────────────────────────────────────────────────────────────────────────────

def _call_summary(sts_availability=None, language="English"):
    return build_temporal_validation_summary(
        cohort_summary={
            "n_total": 100, "n_events": 10, "event_rate": 0.10,
            "date_range": "2024-Q1 — 2024-Q4",
            "n_complete": 60, "n_adequate": 20, "n_partial": 10, "n_low": 10,
        },
        performance_df=pd.DataFrame(),
        pairwise_df=pd.DataFrame(),
        calibration_df=pd.DataFrame(),
        risk_category_df=pd.DataFrame(),
        metadata={
            "model_version": "1.0", "n_patients": 200, "n_events": 20,
            "event_rate": 0.10, "best_model": "RandomForest", "locked_threshold": 0.08,
        },
        threshold=0.08,
        language=language,
        sts_availability=sts_availability,
    )


class TestComparatorScopeNoteInReport:
    def test_scope_note_in_report_for_partial(self):
        avail = build_sts_availability_summary(n_eligible=100, n_score=70)
        report = _call_summary(sts_availability=avail)
        assert "AI Risk" in report or "subset" in report.lower()

    def test_euroscore_mentioned_in_report_for_partial(self):
        avail = build_sts_availability_summary(n_eligible=100, n_score=70)
        report = _call_summary(sts_availability=avail)
        assert "EuroSCORE" in report

    def test_scope_note_as_blockquote_in_report(self):
        """comparator_scope_note must be rendered as a Markdown blockquote."""
        avail = build_sts_availability_summary(n_eligible=100, n_score=70)
        report = _call_summary(sts_availability=avail)
        lines_starting_gt = [l for l in report.splitlines() if l.strip().startswith(">")]
        assert len(lines_starting_gt) >= 1, "Expected at least one blockquote line"

    def test_no_scope_note_in_report_for_complete(self):
        avail = build_sts_availability_summary(n_eligible=50, n_score=50)
        report = _call_summary(sts_availability=avail)
        assert "EuroSCORE II metrics are computed at full cohort level" not in report

    def test_portuguese_scope_note_in_report(self):
        avail = build_sts_availability_summary(n_eligible=100, n_score=70, language="Portuguese")
        report = _call_summary(sts_availability=avail, language="Portuguese")
        assert "EuroSCORE" in report
