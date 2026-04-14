# Bundle compatibility

Policy for versioning and evolving the persisted training bundle
(`ia_risk_bundle.joblib`).

This document describes contracts only. For the implementation, see
[`bundle_io.py`](../bundle_io.py); for how the cache is read and written,
see `load_train_bundle` / `load_cached_bundle_only` in [`app.py`](../app.py).

---

## What the schema version covers

The schema version applies to the **persisted payload** — the top-level dict
written by `joblib.dump`, not the inner ML artifacts. A payload has the
shape:

```python
{
    "bundle_schema_version": <int>,   # since v1
    "signature": {...},                # cache key (file stat + model version)
    "bundle": {                        # inner ML content
        "prepared":   {...},           # dataclass → dict
        "artifacts":  {...},           # dataclass → dict
        "run_report": {...},           # optional, since Phase 3
    },
    "saved_at":        "<iso-utc>",
    "training_source": "<filename>",
}
```

- `BUNDLE_SCHEMA_VERSION` (current: **1**) is defined in
  [`bundle_io.py`](../bundle_io.py) and stamped by the write path in
  `load_train_bundle`.
- `LEGACY_BUNDLE_SCHEMA_VERSION = 0` is the implicit version of any payload
  that predates this field.

## How legacy bundles are handled

Bundles written before versioning was introduced lack the
`bundle_schema_version` key. `read_payload_schema_version` interprets this
absence (and any malformed value) as version `0`. `normalize_payload` then:

1. Validates the payload's essential structure (see "Required vs. optional"
   below).
2. Applies the migration ladder (`if source_version < N: ...`). For `v0 → v1`
   the migration is **intentionally a no-op**: the tolerance to missing
   optional artifact fields that existed in v0 was already a de-facto
   contract via `.get()` inside `deserialize_bundle`, and v1 only makes that
   tolerance explicit.
3. Stamps `bundle_schema_version = BUNDLE_SCHEMA_VERSION` on the returned
   (shallow-copied) payload.
4. Attaches `_loaded_schema_version` with the **original** version read from
   disk. This key is **diagnostic only** and must not be re-persisted.

**Legacy bundles are never rewritten just to "clean up".** The upgrade
happens in memory on every load. The on-disk file is only rewritten by the
normal retrain flow, which naturally produces a current-version payload.

## `schema_version` vs. `loaded_schema_version` in `bundle_info`

Both keys are surfaced in the `bundle_info` dict returned from the load
paths and propagated through `TabContext.bundle_info`:

| Key                     | Meaning                                              |
|:------------------------|:-----------------------------------------------------|
| `schema_version`        | Current code's schema version (always matches `BUNDLE_SCHEMA_VERSION`). |
| `loaded_schema_version` | Version actually read from disk. Equal to `schema_version` for freshly written bundles; `0` for legacy bundles. |

Consumers access these via `.get()`. No UI currently displays them — they
exist for diagnostics, telemetry, or an optional future compat notice.

## Required vs. optional fields

`validate_payload` enforces a deliberately small required set. Everything
else is tolerated by `deserialize_bundle` via `.get()` with sensible
defaults.

**Required (raises `BundleSchemaError`):**
- The payload itself is a `dict`.
- `payload["bundle"]` is a `dict`.
- `bundle["prepared"]` and `bundle["artifacts"]` are present.

**Optional (missing is fine):**
- `bundle_schema_version` — missing means legacy (v0).
- `signature`, `saved_at`, `training_source` — missing degrades to
  `"Unknown"` in `bundle_info`.
- `bundle["run_report"]` — missing means no RunReport (Phase-3 feature).
- Inside `bundle["artifacts"]`: `oof_raw`, `calibration_method`,
  `youden_thresholds`, `best_youden_threshold` — missing values default per
  the `TrainedArtifacts` dataclass.

A `BundleSchemaError` raised during load is caught and converted to the
status string `"Cache corrompido"`; the caller receives
`(None, "Cache corrompido", empty_info)` and the app prompts a retrain.

---

## Bumping the schema version

Follow this checklist when a change to the persisted shape requires a new
version:

1. Increment `BUNDLE_SCHEMA_VERSION` in [`bundle_io.py`](../bundle_io.py).
2. Add an `if source_version < N:` branch to the migration ladder in
   `normalize_payload`. Keep the branch focused: transform only the fields
   the bump affects; forward-compatible defaults are preferable to raises.
3. Preserve backward compatibility when feasible — reading an older payload
   should still produce a working bundle. Break compatibility only when
   there is no defensible migration (and document the reason in the branch).
4. Add tests covering `v(N-1) → vN`: a round-trip of a current-version
   payload, a load of a version-(N-1) payload, and the relevant
   missing/optional-field cases. Extend
   [`tests/test_bundle_schema_version.py`](../tests/test_bundle_schema_version.py).
5. Update the "Version history" comment in `bundle_io.py` and the shape
   diagram at the top of this document.

## What not to do

- **Do not** introduce a migration framework, registry, or plugin system
  until there is a concrete second migration that demands it. A linear `if`
  ladder is easier to read and audit.
- **Do not** rewrite legacy bundles on load. Normalize in memory; let the
  retrain flow produce the new version naturally.
- **Do not** move the schema version inside the inner `bundle` dict. The
  version describes the persisted artifact's shape, and the inner
  `serialize_bundle` / `deserialize_bundle` pair is intentionally kept
  unaware of it.
- **Do not** persist `_loaded_schema_version`. It is a diagnostic attached
  by `normalize_payload`; the leading underscore signals this intent.
