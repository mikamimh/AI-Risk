# AI Risk — Architecture & Design

## System Overview

AI Risk is a Streamlit-based research tool for 30-day mortality risk stratification in cardiac surgery patients. It combines a locally trained machine learning model with established clinical scores (EuroSCORE II and STS Score). It is not a clinical decision-support system.

The system runs entirely locally; no patient data leaves the machine except for STS Score queries to the official STS web calculator.

```
┌──────────────────────────────────────────────────────────┐
│  User Interface                                           │
│  app.py — orchestration, tab routing, Streamlit UI        │
├──────────────────────────────────────────────────────────┤
│  Inference Layer                                          │
│  ai_risk_inference.py  — frozen-model inference core     │
├──────────────────────────────────────────────────────────┤
│  Analytics & Scoring                                      │
│  risk_data.py        | Data prep, normalization           │
│  modeling.py         | ML training & selection            │
│  euroscore.py        | EuroSCORE II formula               │
│  explainability.py   | SHAP explanations                  │
│  stats_compare.py    | Statistical evaluation             │
│  subgroups.py        | Subgroup assignment & metrics      │
├──────────────────────────────────────────────────────────┤
│  STS Score Subsystem                                      │
│  sts_calculator.py   | WebSocket transport                │
│  sts_cache.py        | Cache & revalidation policy        │
├──────────────────────────────────────────────────────────┤
│  Observability & Bundle                                   │
│  observability.py    | RunReport/RunStep, renderers       │
│  bundle_io.py        | Bundle serialization/deserialization│
│  report_text.py      | Methods/Results text builders      │
├──────────────────────────────────────────────────────────┤
│  Metadata / Export / Temporal                             │
│  model_metadata.py      | Versioning, audit, completeness │
│  export_helpers.py      | Stats report export             │
│  temporal_validation.py | Temporal cohort helpers         │
│  variable_dictionary.py | Variable reference              │
├──────────────────────────────────────────────────────────┤
│  Configuration                                            │
│  config/base_config.py  | AppConfig                       │
│  config/model_config.py | ML hyperparameters              │
├──────────────────────────────────────────────────────────┤
│  Storage                                                  │
│  ia_risk_bundle.joblib   | Serialized training bundle     │
│  .ia_risk_cache/         | STS Score disk cache           │
└──────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

### app.py
**Main application — UI routing and orchestration**

app.py owns the Streamlit tab layout, sidebar controls, and session-state wiring. Logic extraction across phases has significantly reduced its scope: it is now primarily a caller of the modules below rather than an owner of computation.

What remains in app.py:
- Streamlit tab layout (10 tabs) and sidebar controls (language, data source)
- File upload and cache-key comparison for training-trigger decisions
- Session-state management for the loaded bundle
- Calls to `observability` renderers (`render_run_report_compact`, `render_run_report`)
- Calls to `bundle_io` for serialize/deserialize
- Calls to `sts_cache.get_cached_or_fetch` (passing `sts_calculator` fetch functions)
- Result visualization and download buttons

What has been extracted out of app.py:
- Execution report data structures and renderers → `observability.py`
- Bundle serialization/deserialization → `bundle_io.py`
- Frozen AI Risk inference → `ai_risk_inference.py`
- STS Score cache and revalidation policy → `sts_cache.py`
- Subgroup assignment and metrics → `subgroups.py`
- Methods/Results manuscript text → `report_text.py`
- Statistical summary export → `export_helpers.py`
- Temporal cohort helpers → `temporal_validation.py`

### risk_data.py
**Data integration and preparation**
- Multi-format loading: `.xlsx`, `.xls`, `.db`, `.sqlite`, `.csv`, `.parquet`
- Column normalization: Brazilian/English numeric conventions, valve severity encoding (`None < Trivial < Mild < Moderate < Severe`)
- Patient matching across sheets: Name + Procedure Date
- Eligibility filtering (Surgery and Procedure Date non-null)
- Feature engineering: BSA (DuBois formula), combined-surgery flags, procedure weight, thoracic-aorta flag
- STS Score variable mapping
- Produces an `IngestionReport` dataclass consumed by `observability.build_step_ingestion()`

### modeling.py
**ML pipeline: training, selection, and preprocessing**
- Feature preprocessing:
  - Numeric: median imputation + StandardScaler
  - Valve severity: OrdinalEncoder with clinical order
  - Other categorical: mode imputation + TargetEncoder
- Candidate model instantiation: LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost, StackingEnsemble
- StratifiedGroupKFold cross-validation (same patient never in both train and test folds)
- Per-model calibration strategy applied inside each CV fold:
  - RandomForest: sigmoid (Platt scaling), inner cv ≤ 5
  - LightGBM, CatBoost: isotonic, inner cv ≤ 5
  - XGBoost: isotonic, inner cv ≤ 3
  - LogisticRegression, StackingEnsemble: uncalibrated
- Auto-selection with clinical-usability guardrails (AUC floor, Brier baseline, dynamic range, threshold coverage)
- `clean_features()` — shared feature-cleaning helper used by both training and inference
- Produces `TrainedArtifacts` dataclass; serialized via `bundle_io.serialize_bundle()`

### euroscore.py
**EuroSCORE II calculation**
- Published logistic regression formula (Nashef et al., *Eur J Cardiothorac Surg*, 2012): 18 risk factors, 27 coefficients
- Variable mapping from the available dataset; approximations documented in the Analysis Guide tab

### sts_calculator.py
**STS Score WebSocket transport**
- Builds the canonical STS Score input dict from a patient row (`build_sts_input_from_row`)
- Establishes a WebSocket connection to the official STS Risk Calculator (`acsdriskcalc.research.sts.org`) — not a documented public API
- Sends patient data and receives the predicted mortality probability
- Does not know about caching, retries, or fallback — those are the responsibility of `sts_cache.py`

### sts_cache.py
**STS Score cache and revalidation policy**

This module owns all persistence, revalidation, and fallback logic for STS Score results. `sts_calculator.py` only performs the WebSocket fetch; `sts_cache.py` decides when a fetch is needed and what to do on failure.

Cache validation policy — a cached entry is returned unchanged only when ALL hold:
1. The clinically relevant input fields hash to the same SHA-256 as the cached entry
2. The cached entry's `integration_version` matches `STS_SCORE_INTEGRATION_VERSION`
3. The cached result dict passes validation (has a non-NaN `predmort`)
4. The entry is within the TTL window (default: 14 days, from `AppConfig.STS_SCORE_CACHE_TTL_DAYS`)

If any condition fails, the STS Score web calculator is queried via `sts_calculator`. On failure:
- Retry up to `max_retries` attempts with back-off
- Stale fallback — preference order:
  1. Same-hash expired/invalid on-disk entry (TTL fallback)
  2. Patient's previous-hash entry via the patient index (cross-hash fallback)
- If no fallback is available, return an explicit `failed` record

Five result statuses surfaced per patient:

| Status | Meaning |
|:--|:--|
| `fresh` | No prior entry; fetched and cached successfully |
| `cached` | Valid entry within TTL; returned without a network query |
| `refreshed` | Entry was expired or invalid; re-fetched successfully |
| `stale_fallback` | Fetch failed; returned a previous valid entry |
| `failed` | Fetch failed and no fallback was available |

Key public interface:
- `get_cached_or_fetch(sts_input, patient_id, fetch_func)` → `ExecutionRecord`
- `summarise_execution_log(records)` → `{status: count}` — consumed by `observability.build_step_sts_score()`
- `ExecutionRecord` dataclass — structured per-patient result including status, result dict, hash, age, and reason

### ai_risk_inference.py
**Frozen-model inference core — no Streamlit dependencies**
- Single unified inference path shared by individual prediction, batch processing, and temporal validation
- `_run_ai_risk_inference_row()`: assembles a single-row DataFrame from raw input, aligns dtypes to training schema, cleans features, runs `predict_proba`, assesses completeness, and returns a structured result dict with a `None` or `{patient_id, stage, reason}` incident
- `apply_frozen_model_to_temporal_cohort()`: row-level inference loop over a temporal cohort DataFrame with optional `progress_callback` and structured incident collection
- Carries no Streamlit state and no `tr()` closure; independently testable

### explainability.py
**SHAP-based model interpretation**
- Global feature importance across the dataset
- Local per-patient SHAP explanation (waterfall / bar)
- Dependence plots and beeswarm plots

### stats_compare.py
**Statistical analysis and model comparison**
- Discrimination: AUC, AUPRC with bootstrap 95% CI
- Calibration: Brier score, Hosmer-Lemeshow test, calibration intercept and slope
- Head-to-head comparison: DeLong test, bootstrap AUC difference
- Reclassification: NRI, IDI
- Decision Curve Analysis (DCA)
- Threshold-specific metrics: sensitivity, specificity, PPV, NPV

### subgroups.py
**Subgroup assignment and per-subgroup metrics — no Streamlit dependencies**

Extracted from app.py during Phase 4. The bilingual `tr(en, pt)` helper is passed in explicitly.

- `surgery_family()` — three-way classification (coronary / valve / mixed / other) from Surgery text
- `surgery_type_group()` — finer-grained surgery-type label for the Subgroups tab
- `lvef_group()` — ejection-fraction bucket (preserved / mildly reduced / reduced / unknown)
- `renal_group()` — renal-function bucket from creatinine clearance; Cockcroft-Gault fallback when clearance is missing
- `evaluate_subgroup()` — runs `evaluate_scores_with_threshold` across groupby segments with bootstrap 95% CI

### observability.py
**Execution report — data structures and renderers**

This module owns the entire observability layer. app.py is a caller, not an owner.

Data structures:
- `RunStep` dataclass — one execution phase: `name`, `status`, `summary`, `counters`, `details`, `incidents`, `audit_records`
- `RunReport` dataclass — aggregates `RunStep` objects; `overall_status()` returns the worst status across steps; `to_dict()` / `from_dict()` for bundle persistence via `bundle_io`

Builder functions (pure transformers, no Streamlit imports):
- `build_step_ingestion(ingestion_report, feature_columns)` — from `risk_data.IngestionReport`; partitions warnings into predictor-relevant vs. informational
- `build_step_eligibility(info)` — from `prepared.info`
- `build_step_training(leaderboard, best_model_name, n_features, prevalence)` — from `modeling` leaderboard
- `build_step_sts_score(execution_log)` — from `sts_cache` execution log; calls `sts_cache.summarise_execution_log()`

Streamlit renderers (Streamlit imported lazily inside each function):
- `render_run_report_compact(report)` — compact top-of-page chip row; one colored chip per phase; surfaces blocking errors as `st.error` banners
- `render_run_report(report)` — expandable per-step panels at bottom of page; errors auto-expand, warnings stay collapsed
- `render_sts_score_incidents(execution_log)` — per-patient STS Score incident table; used by temporal validation to mirror the batch flow

### bundle_io.py
**Bundle serialization and deserialization — no Streamlit dependencies**

Extracted from app.py during Phase 4. Solves Streamlit's module-reload problem: when Streamlit reloads modules between runs, the class identity of `PreparedData` and `TrainedArtifacts` changes, causing `joblib` to fail to unpickle instances. `bundle_io` converts dataclasses to plain dicts before saving and back after loading.

- `bundle_signature(xlsx_path)` — stable cache key (file path + mtime_ns + size + model version) used by app.py to decide whether retraining is needed
- `serialize_bundle(bundle)` — converts `PreparedData`, `TrainedArtifacts`, and `RunReport` to plain dicts before `joblib.dump`
- `deserialize_bundle(bundle)` — reverses the above; reloads `modeling` and `observability` via `importlib.reload` to use the currently live class definitions; reconstructs `RunReport` via `RunReport.from_dict()`

### report_text.py
**Manuscript-ready Methods and Results text builders — no Streamlit dependencies**

Extracted from app.py during Phase 4. The bilingual `tr(en, pt)` helper and the selected `language` are passed in as explicit arguments.

- `build_methods_text(mode, language, tr)` — Methods section text in two verbosity levels (Summary / Detailed) and two languages (English / Portuguese)
- `build_results_text(mode, context, language, tr)` — Results section text driven by a context dict of computed metrics

### model_metadata.py
**Model versioning, audit trail, and metadata**
- `build_model_metadata()` — serializes training metadata into the joblib bundle
- `assess_input_completeness()` — classifies input quality: complete / adequate / partial / low
- `format_metadata_for_display()` — structured display DataFrame for the Models tab
- `generate_individual_report()`, `generate_clinical_explanation()` — per-patient Markdown reports
- `log_analysis()`, `read_audit_log()` — append-only audit trail
- `check_temporal_overlap()`, `check_validation_readiness()` — temporal validation guards
- `export_model_bundle_metadata()` — exports bundle metadata as JSON/CSV
- Re-exports all public symbols from `export_helpers` and `temporal_validation` for backward compatibility

### export_helpers.py
**Statistical summary export — pure transformers, no project state**
- `build_statistical_summary()` — assembles a Markdown statistical report from input DataFrames
- `statistical_summary_to_dataframes()` — parses Markdown tables into a `dict[str, DataFrame]`
- `statistical_summary_to_xlsx()`, `statistical_summary_to_csv()`, `statistical_summary_to_pdf()` — format converters (XLSX via openpyxl; PDF via fpdf2)
- Zero project-level imports; only stdlib, numpy, pandas, openpyxl, fpdf

### temporal_validation.py
**Temporal cohort helpers**
- `extract_year_quarter_range()` — derives (start, end) year-quarter strings from a DataFrame's `surgery_year` / `surgery_quarter` columns
- `check_temporal_overlap()` — compares training vs. validation date ranges; returns structured dict with `status`, `severity`, and bilingual `message` fields
- `format_locked_model_for_display()` — tabular summary of locked-model metadata for the Temporal Validation tab
- `build_temporal_validation_summary()` — Markdown report for temporal validation results

### variable_dictionary.py
**Variable reference**
- `get_dictionary_dataframe()` — structured DataFrame of all variables, types, and descriptions
- `get_dictionary_by_domain()` — filtered view by clinical domain

### config/
**Centralized configuration**
- `base_config.py` — `AppConfig`: paths, CV folds, decision threshold, random seed, UI settings, STS Score cache settings
- `model_config.py` — candidate model hyperparameters; `get_model_params()`, `list_available_models()`

---

## Architectural Flows

### 1. AI Risk Training

```
File upload (XLSX / CSV / DB / Parquet)
  └─ risk_data.prepare_master_dataset()
       ├─ Multi-format load
       ├─ Column normalization & mapping (produces IngestionReport)
       ├─ Patient matching (Name + Procedure Date)
       ├─ Eligibility filter (Surgery + Procedure Date non-null)
       ├─ Feature engineering (BSA, combined-surgery flags, procedure weight)
       └─ Returns PreparedData dataclass + IngestionReport

  └─ observability.build_step_ingestion(ingestion_report, feature_columns)
       └─ Produces RunStep for "Ingestion & normalization"

  └─ modeling.train_and_select_model(prepared)
       ├─ clean_features()
       ├─ StratifiedGroupKFold (same patient never in both folds)
       ├─ Per-fold: preprocess → fit candidate → calibrate → OOF predict
       ├─ Candidates: LR, RF, XGB, LGBM, CatBoost, StackingEnsemble
       ├─ Auto-select with clinical-usability guardrails
       └─ Returns TrainedArtifacts dataclass

  └─ observability.build_step_training(leaderboard, ...)
       └─ Produces RunStep for "Model training"

  └─ RunReport assembled from all RunSteps

  └─ bundle_io.serialize_bundle({prepared, artifacts, run_report, ...})
       └─ Converts dataclasses → plain dicts (Streamlit module-reload safety)

  └─ joblib.dump → ia_risk_bundle.joblib
```

### 2. Unified AI Risk Frozen Inference

All three inference flows (individual prediction, batch, temporal validation) share a single implementation in `ai_risk_inference.py`. No retraining or recalibration ever occurs after the bundle is saved.

```
row_dict  (form data or DataFrame row)
  └─ _build_input_row(feature_columns, row_dict)
       ├─ Fill NaN for missing features
       ├─ Prefix-matching for truncated column names (Excel export artefact)
       ├─ Derived flags: cirurgia_combinada, peso_procedimento, thoracic_aorta_flag
       └─ Categorical defaults for boolean comorbidities

  └─ _align_input_to_training_schema(input_row, reference_df)
       └─ Coerce dtypes to match training-time schema (numeric / categorical)

  └─ clean_features(model_input, numeric_columns=numeric_cols)
       └─ Final object-dtype safety coercion pass

  └─ model_pipeline.predict_proba(model_input)[:, 1]
       └─ Frozen estimator — identical to the pipeline saved at training time

  └─ assess_input_completeness(feature_columns, input_row, language)
       └─ Level: complete / adequate / partial / low

Returns: {probability, completeness, incident, model_input}
  └─ incident = None on success
  └─ incident = {patient_id, stage, reason} on failure
```

Callers:
- **Individual** (Prediction tab): calls `_run_ai_risk_inference_row` once per manual form submission
- **Batch** (Batch & Export tab): iterates uploaded rows; AI Risk incidents collected and surfaced via `observability`
- **Temporal** (Temporal Validation tab): `apply_frozen_model_to_temporal_cohort` wraps the row loop with an optional `progress_callback`

### 3. STS Score Transport and Cache/Revalidation

The STS Score flow is split across two modules with distinct responsibilities:

```
For each patient in the cohort:

  sts_cache.get_cached_or_fetch(sts_input, patient_id, fetch_func)
    │
    ├─ Stage: build_input
    │    └─ compute_input_hash(sts_input) — SHA-256 of canonical fields + integration version
    │
    ├─ Stage: cache_lookup
    │    └─ load_entry(input_hash)
    │         ├─ Cache HIT (version OK, result valid, within 14-day TTL)
    │         │    └─ return ExecutionRecord(status="cached")
    │         └─ Cache MISS or expired
    │              → proceed to fetch
    │
    ├─ Stage: fetch  [calls fetch_func = sts_calculator WebSocket transport]
    │    ├─ sts_calculator.build_sts_input_from_row(row)
    │    │    └─ Maps patient variables to STS Score input format
    │    ├─ WebSocket connection to acsdriskcalc.research.sts.org
    │    ├─ Retry: up to max_retries attempts with back-off
    │    └─ On success → persist_fresh_result() → cache written to disk
    │         └─ return ExecutionRecord(status="fresh" | "refreshed")
    │
    └─ Stage: stale fallback  [fetch failed]
         ├─ find_stale_fallback(patient_id, input_hash, same_hash_entry)
         │    ├─ Option 1: same-hash expired entry (TTL fallback)
         │    └─ Option 2: patient's previous-hash via patient index (cross-hash fallback)
         ├─ Fallback found → return ExecutionRecord(status="stale_fallback")
         └─ No fallback  → return ExecutionRecord(status="failed")

Post-loop:
  └─ sts_cache.summarise_execution_log(records) → {status: count}
  └─ observability.build_step_sts_score(execution_log)
       ├─ n_usable == 0 or fail_ratio ≥ 0.5  → STATUS_ERROR
       ├─ n_failed > 0 (partial)              → STATUS_WARNING
       └─ all succeeded                        → STATUS_OK
```

### 4. Observability / Execution Report

`observability.py` owns the data model and both renderers. app.py is only the wiring layer.

```
After each major phase, app.py calls:
  └─ observability.build_step_ingestion(ingestion_report, feature_columns)
  └─ observability.build_step_eligibility(prepared.info)
  └─ observability.build_step_training(leaderboard, ...)
  └─ observability.build_step_sts_score(sts_execution_log)

RunReport assembled:
  └─ RunReport.add(step) for each RunStep
  └─ RunReport.overall_status() → worst status across all steps

Persisted in bundle:
  └─ bundle_io.serialize_bundle includes RunReport.to_dict()
  └─ bundle_io.deserialize_bundle reconstructs RunReport.from_dict()

Rendered in UI:
  └─ observability.render_run_report_compact(report, tr=tr)
       └─ Top-of-page chip row: one colored ● per phase
       └─ st.error banner if any step is STATUS_ERROR

  └─ observability.render_run_report(report, tr=tr)
       └─ Per-step expandable panels at bottom of Overview tab
       └─ ERROR steps auto-expand; warnings stay collapsed
       └─ Counters table + details bullets + incidents dataframe per step
       └─ Ingestion step: downloadable normalization audit CSV
```

### 5. Temporal Validation

```
User uploads a temporal cohort (post-training period)
  └─ risk_data.prepare_master_dataset() → temporal_data

  └─ temporal_validation.extract_year_quarter_range(temporal_data)
       └─ Returns (start_yq, end_yq) — e.g., ("2023-Q1", "2024-Q4")

  └─ temporal_validation.check_temporal_overlap(
         training_start, training_end, validation_start, validation_end)
       └─ Returns {status, severity, message} — flags retrograde or overlapping cohorts

  └─ ai_risk_inference.apply_frozen_model_to_temporal_cohort(
         model_pipeline, feature_columns, reference_df, temporal_data)
       └─ Applies the frozen model row-by-row (no retraining, no recalibration)
       └─ Returns {probabilities, completeness, incidents, n_total, n_failed}

  └─ observability.render_sts_score_incidents(sts_execution_log)
       └─ Per-patient STS Score incidents in temporal validation mirror the batch flow

  └─ stats_compare: compute discrimination + calibration on (y_true, probabilities)
       └─ Compared against training-cohort performance

  └─ temporal_validation.build_temporal_validation_summary(...)
       └─ Markdown report available for download

  └─ temporal_validation.format_locked_model_for_display(metadata)
       └─ Tabular summary of locked-model properties shown in the tab
```

### 6. Export Helpers

```
Statistical comparison DataFrames
  └─ export_helpers.build_statistical_summary(...)
       └─ Markdown text with section headers and tables

  └─ Converters (available in the Statistical Comparison tab):
       ├─ statistical_summary_to_xlsx() → XLSX bytes (one sheet per section)
       ├─ statistical_summary_to_csv()  → CSV string (sections concatenated)
       └─ statistical_summary_to_pdf()  → PDF bytes (requires fpdf2)

All symbols re-exported from model_metadata for backward compatibility.
```

---

## Configuration Management

### base_config.py
```python
from config import AppConfig

# Core paths
AppConfig.MODEL_CACHE_FILE         # Path to ia_risk_bundle.joblib
AppConfig.LOCAL_DATA_DIR           # Input data directory
AppConfig.TEMP_DATA_DIR            # Temporary files

# Training
AppConfig.CV_SPLITS                # CV folds (default 5)
AppConfig.RANDOM_SEED              # Reproducibility seed (default 42)
AppConfig.MODEL_VERSION            # Version string embedded in bundles

# STS Score cache
AppConfig.STS_SCORE_CACHE_DIR      # Disk cache directory
AppConfig.STS_SCORE_CACHE_TTL_DAYS # Cache TTL (default 14 days)
AppConfig.STS_SCORE_INTEGRATION_VERSION  # Bumping this invalidates all prior cache entries

# UI
AppConfig.PAGE_TITLE               # Streamlit page title
AppConfig.LANGUAGES                # Supported UI languages
AppConfig.MIN_SAMPLE_SIZE          # Minimum patients for analysis
AppConfig.MISSING_TOKENS           # Strings treated as missing values
```

### model_config.py
```python
from config import get_model_params, list_available_models

models = list_available_models()
# ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'StackingEnsemble']

params = get_model_params("XGBoost")
```

---

## Extension Points

### Adding a New Candidate Model
1. Add hyperparameters to `config/model_config.py`
2. Update `modeling.py`'s `_build_candidates()` to instantiate the new model
3. Add a calibration strategy entry to the per-model calibration selector in `modeling.py`

### Adding a New Feature
1. Map the column alias in `risk_data.py`'s alias table
2. Add feature engineering logic if needed
3. Update `docs/DATA_FORMAT.md`

### Adding a Statistical Test
1. Implement in `stats_compare.py`
2. Call from the relevant section in `app.py`
3. If the result should appear in the export, update `export_helpers.build_statistical_summary()`

### Invalidating the STS Score Cache
Bump `AppConfig.STS_SCORE_INTEGRATION_VERSION` in `config/base_config.py`. Every prior cache entry has the old version string baked into its hash and will not pass the version check — all patients will be re-fetched on the next run.

---

## Security & Privacy

- Patient names and dates are used only for internal cross-sheet matching and are never stored in the model cache or exported results
- The STS Score disk cache stores risk probabilities keyed by canonicalized clinical payload — no patient names
- `ia_risk_bundle.joblib` contains only model weights, feature schemas, aggregate training metadata, and the last `RunReport` — no patient-level data
- Recommended: run on a local network, not internet-facing
