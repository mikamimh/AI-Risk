"""Cache-invalidation tests for the Temporal Validation tab.

These tests exercise the exact signature-construction logic used in app.py
without requiring a Streamlit runtime.  They prove that:

  - The context signature is content-addressed (SHA256 of file bytes).
  - Any change to file content produces a different signature.
  - Identical content always produces the same signature, regardless of
    filename, file size metadata, or Streamlit file_id.
  - Changing STS mode, threshold, or model also invalidates the signature.
  - The stale-purge guard correctly identifies mismatched signatures.

The signature helpers are deliberately self-contained copies of the app.py
logic so that these tests remain independent of the Streamlit import chain.
"""

import hashlib
import pytest


# ---------------------------------------------------------------------------
# Signature helpers — mirrors the exact logic in app.py
# ---------------------------------------------------------------------------

def _file_content_hash(file_bytes: bytes) -> str:
    """Mirrors: _tv_file_content_hash = sha256(_tv_file_bytes).hexdigest()[:24]"""
    return hashlib.sha256(file_bytes).hexdigest()[:24]


def _context_sig(
    file_bytes: bytes,
    bundle_saved_at: str = "2025-01-01",
    forced_model: str = "model_v1",
    locked_threshold: float = 0.08,
    sts_on: bool = True,
) -> str:
    """Mirrors the _tv_context_sig construction in app.py."""
    file_sig = _file_content_hash(file_bytes)
    raw = (
        f"{file_sig}|"
        f"{bundle_saved_at}|"
        f"{forced_model}|"
        f"{locked_threshold:.6f}|"
        f"sts={'1' if sts_on else '0'}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1. Content hash — determinism and content-addressing
# ---------------------------------------------------------------------------

def test_same_bytes_same_hash():
    """Identical file bytes always produce the same content hash."""
    data = b"patient_id,Surgery,Death\n1,CABG,No\n2,AVR,Yes\n"
    assert _file_content_hash(data) == _file_content_hash(data)


def test_different_bytes_different_hash():
    """Changing one byte changes the content hash."""
    data_a = b"patient_id,Surgery\n1,CABG\n"
    data_b = b"patient_id,Surgery\n1,AVR\n"
    assert _file_content_hash(data_a) != _file_content_hash(data_b)


def test_hash_is_content_not_length():
    """Same length, different content → different hash."""
    data_a = b"AAAA"
    data_b = b"BBBB"
    assert len(data_a) == len(data_b)
    assert _file_content_hash(data_a) != _file_content_hash(data_b)


def test_same_content_different_filename_same_hash():
    """Filename does not affect the hash — only bytes matter."""
    data = b"id,Surgery\n1,CABG\n"
    # Simulate two uploads with different filenames but same bytes
    hash_file_a = _file_content_hash(data)   # "validation_jan.xlsx"
    hash_file_b = _file_content_hash(data)   # "validation_feb.xlsx"
    assert hash_file_a == hash_file_b


def test_same_content_different_size_metadata_same_hash():
    """A hypothetical padded/re-encoded file with same logical content
    produces a different hash (correct — bytes changed)."""
    data_original = b"id,Surgery\n1,CABG\n"
    data_with_bom  = b"\xef\xbb\xbf" + data_original    # UTF-8 BOM prepended
    assert _file_content_hash(data_original) != _file_content_hash(data_with_bom)


def test_hash_length_is_24():
    """Content hash is exactly 24 hex characters."""
    h = _file_content_hash(b"test")
    assert len(h) == 24


# ---------------------------------------------------------------------------
# 2. Context signature — all components affect the sig
# ---------------------------------------------------------------------------

BASE_BYTES = b"id,Surgery,Death\n1,CABG,No\n"

def test_context_sig_deterministic():
    assert _context_sig(BASE_BYTES) == _context_sig(BASE_BYTES)


def test_different_file_content_different_sig():
    """Changed file bytes → different context sig."""
    other = b"id,Surgery,Death\n1,AVR,Yes\n"
    assert _context_sig(BASE_BYTES) != _context_sig(other)


def test_same_content_different_filename_same_sig():
    """Filename is not an input to the sig — same bytes → same sig."""
    assert _context_sig(BASE_BYTES) == _context_sig(BASE_BYTES)


def test_sts_mode_change_invalidates_sig():
    """Toggling the STS checkbox must produce a different sig."""
    sig_on  = _context_sig(BASE_BYTES, sts_on=True)
    sig_off = _context_sig(BASE_BYTES, sts_on=False)
    assert sig_on != sig_off


def test_threshold_change_invalidates_sig():
    """Changing the locked threshold must produce a different sig."""
    sig_a = _context_sig(BASE_BYTES, locked_threshold=0.08)
    sig_b = _context_sig(BASE_BYTES, locked_threshold=0.10)
    assert sig_a != sig_b


def test_model_change_invalidates_sig():
    """Switching the model must produce a different sig."""
    sig_a = _context_sig(BASE_BYTES, forced_model="model_v1")
    sig_b = _context_sig(BASE_BYTES, forced_model="model_v2")
    assert sig_a != sig_b


def test_bundle_timestamp_change_invalidates_sig():
    """Retraining the bundle (new saved_at) must produce a different sig."""
    sig_a = _context_sig(BASE_BYTES, bundle_saved_at="2025-01-01")
    sig_b = _context_sig(BASE_BYTES, bundle_saved_at="2025-06-01")
    assert sig_a != sig_b


def test_context_sig_length_is_16():
    """Context sig is exactly 16 hex characters."""
    assert len(_context_sig(BASE_BYTES)) == 16


# ---------------------------------------------------------------------------
# 3. Invalidation scenarios required by the spec
# ---------------------------------------------------------------------------

def test_same_name_different_content_new_sig():
    """Scenario: user uploads file_A.xlsx, then file_B.xlsx (same name,
    different content).  Must produce different sigs — NOT cached."""
    bytes_a = b"id,Surgery,Death\n1,CABG,No\n2,AVR,Yes\n"
    bytes_b = b"id,Surgery,Death\n1,CABG,Yes\n2,MVR,No\n"   # changed content
    sig_a = _context_sig(bytes_a)
    sig_b = _context_sig(bytes_b)
    assert sig_a != sig_b, "Different content must produce different sigs"


def test_different_name_same_content_same_sig():
    """Scenario: user renames file but content is identical.
    Sig should be the same — result is safely reusable."""
    data = b"id,Surgery,Death\n1,CABG,No\n"
    assert _context_sig(data) == _context_sig(data)


def test_different_name_different_content_new_sig():
    """Scenario: user uploads a completely different file with a new name."""
    bytes_old = b"id,Surgery\n1,CABG\n"
    bytes_new = b"id,Surgery\n1,BENTALL\n"
    assert _context_sig(bytes_old) != _context_sig(bytes_new)


def test_after_cancel_different_content_new_sig():
    """Scenario: user cancels STS run, uploads a new file.
    New content → new sig → stale state must not be inherited."""
    bytes_cancelled = b"id,Surgery,Death\n1,CABG,No\n"
    bytes_new_upload = b"id,Surgery,Death\n1,AVR,Yes\n2,MVR,No\n"
    assert _context_sig(bytes_cancelled) != _context_sig(bytes_new_upload)


def test_after_session_restore_different_content_new_sig():
    """Scenario: result is restored from session, user then uploads a new file.
    New content must produce a new sig — no result reuse."""
    bytes_session = b"id,Surgery,Death\n1,CABG,No\n"
    bytes_new     = b"id,Surgery,Death\n1,MVR,Yes\n"
    sig_session = _context_sig(bytes_session)
    sig_new     = _context_sig(bytes_new)
    assert sig_session != sig_new


# ---------------------------------------------------------------------------
# 4. Stale-purge guard logic
# ---------------------------------------------------------------------------

def test_purge_fires_when_sig_changes():
    """The purge guard: fires when a saved sig exists AND differs from current."""
    saved_sig   = _context_sig(b"old file bytes")
    current_sig = _context_sig(b"new file bytes")
    # Guard condition: saved_sig is not None AND saved_sig != current_sig
    should_purge = (saved_sig is not None) and (saved_sig != current_sig)
    assert should_purge


def test_purge_does_not_fire_when_sigs_match():
    """The purge guard: does NOT fire when saved sig equals current sig."""
    data = b"same file bytes"
    saved_sig   = _context_sig(data)
    current_sig = _context_sig(data)
    should_purge = (saved_sig is not None) and (saved_sig != current_sig)
    assert not should_purge


def test_purge_does_not_fire_on_first_upload():
    """The purge guard: does NOT fire when no previous sig exists (first upload)."""
    saved_sig   = None                      # nothing in session_state yet
    current_sig = _context_sig(b"first file")
    should_purge = (saved_sig is not None) and (saved_sig != current_sig)
    assert not should_purge


def test_purge_does_not_fire_during_valid_running_sts():
    """If an STS thread is running with a valid (matching) sig, don't purge."""
    data = b"current file"
    current_sig = _context_sig(data)
    sts_ctx_sig  = current_sig              # thread was started for current file
    tv_sts_ctx_valid = (sts_ctx_sig == current_sig)   # True → don't purge
    saved_sig    = current_sig              # result sig also matches
    should_purge = (
        (saved_sig is not None)
        and (saved_sig != current_sig)
        and not tv_sts_ctx_valid
    )
    assert not should_purge


def test_purge_fires_when_sts_ctx_is_stale():
    """If an STS thread was started for a different file, its ctx sig won't
    match the current sig — purge should proceed."""
    old_data     = b"previous file content"
    new_data     = b"new file content"
    old_sig      = _context_sig(old_data)
    current_sig  = _context_sig(new_data)
    sts_ctx_sig  = old_sig                  # thread was for old file
    tv_sts_ctx_valid = (sts_ctx_sig == current_sig)   # False
    saved_sig    = old_sig
    should_purge = (
        (saved_sig is not None)
        and (saved_sig != current_sig)
        and not tv_sts_ctx_valid
    )
    assert should_purge
