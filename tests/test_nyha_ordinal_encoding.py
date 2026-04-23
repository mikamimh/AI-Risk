"""NYHA ordinal encoding tests — SKIPPED.

Ordinal encoding (I<II<III<IV) was tested and reverted:
  AUC -0.007, Sensitivity @8% -0.044 vs TargetEncoder baseline.
  TargetEncoder captures the non-linear mortality jump III→IV more accurately.

Tests are kept for history; all are skipped so they do not block CI.
"""

import pytest

SKIP_REASON = (
    "Reverted: NYHA ordinal encoding degraded AUC -0.007 and sensitivity -0.044 "
    "vs TargetEncoder. Non-linear class distances favour target encoding."
)


@pytest.mark.skip(reason=SKIP_REASON)
def test_nyha_has_own_ordinal_transformer():
    pass


@pytest.mark.skip(reason=SKIP_REASON)
def test_nyha_not_in_categorical_transformer():
    pass


@pytest.mark.skip(reason=SKIP_REASON)
def test_nyha_absent_column_skipped():
    pass


@pytest.mark.skip(reason=SKIP_REASON)
def test_nyha_constants():
    pass
