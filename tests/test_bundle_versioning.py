"""Guard test: bundle model_version must match the current MODEL_VERSION constant.

Catches the v14→v15 class of incident: code version bumped (new keywords,
new features, new guardrails) but the AI Risk bundle was never retrained,
so comparisons run on a stale model that doesn't reflect the current pipeline.

Tests are skipped when no bundle file exists (CI environment without a
trained model) but enforced when the bundle is present locally.
"""

import pytest
from pathlib import Path
from config.base_config import AppConfig


_BUNDLE_PATH = Path("ia_risk_bundle.joblib")


def _load_payload():
    """Load raw bundle payload dict (same path as app.py)."""
    import joblib
    import bundle_io
    payload = joblib.load(str(_BUNDLE_PATH))
    return bundle_io.normalize_payload(payload)


@pytest.mark.skipif(
    not _BUNDLE_PATH.exists(),
    reason="No bundle file — skipped in CI environments without trained model",
)
def test_bundle_version_matches_code():
    """Bundle model_version must match current AppConfig.MODEL_VERSION.

    Fails if MODEL_VERSION was bumped in config but the bundle was not
    retrained. Fix: retrain the model after any MODEL_VERSION bump.
    """
    payload = _load_payload()
    artifacts = payload.get("artifacts")
    manifest = getattr(artifacts, "training_manifest", None)

    assert manifest is not None, (
        "Bundle has no training_manifest — was it trained with an old pipeline? "
        "Retrain to get a manifest-aware bundle."
    )

    bundle_version = manifest.get("model_version")
    code_version = AppConfig.MODEL_VERSION
    assert bundle_version == code_version, (
        f"Bundle version {bundle_version!r} does not match code version "
        f"{code_version!r}. Retrain the model after bumping MODEL_VERSION. "
        "Running comparisons on a stale bundle produces misleading results."
    )


@pytest.mark.skipif(
    not _BUNDLE_PATH.exists(),
    reason="No bundle file — skipped in CI environments without trained model",
)
def test_bundle_manifest_has_required_fields():
    """training_manifest must contain the provenance fields written at training time."""
    payload = _load_payload()
    artifacts = payload.get("artifacts")
    manifest = getattr(artifacts, "training_manifest", None)

    if manifest is None:
        pytest.skip("Bundle has no training_manifest (pre-manifest bundle)")

    required = {"n_rows", "n_events", "best_model", "model_version", "generated_at"}
    missing = required - set(manifest.keys())
    assert not missing, f"training_manifest missing fields: {missing}"
