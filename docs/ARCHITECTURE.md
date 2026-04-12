# AI Risk — Architecture & Design

## System Overview

AI Risk is a Streamlit-based research tool for 30-day mortality risk stratification in cardiac surgery patients. It combines a locally trained machine learning model with established clinical scores (EuroSCORE II and STS Score). It is not a clinical decision-support system.

The system runs entirely locally; no patient data leaves the machine except for STS Score queries to the official STS web calculator.

```
┌──────────────────────────────────────────────────┐
│          User Interface (Streamlit)               │
│          app.py — orchestration, UI, 10 tabs      │
├──────────────────────────────────────────────────┤
│  Inference Layer                                  │
│  ai_risk_inference.py  — frozen-model core        │
├──────────────────────────────────────────────────┤
│  Core Processing Layer                            │
│  risk_data.py        | Data preparation           │
│  modeling.py         | ML training & selection    │
│  euroscore.py        | EuroSCORE II formula       │
│  sts_calculator.py   | STS Score via WebSocket    │
│  explainability.py   | SHAP explanations          │
│  stats_compare.py    | Statistical evaluation     │
├──────────────────────────────────────────────────┤
│  Metadata / Export / Temporal Layer               │
│  model_metadata.py      | Versioning, audit       │
│  export_helpers.py      | Stats report export     │
│  temporal_validation.py | Temporal cohort helpers │
│  variable_dictionary.py | Variable reference      │
├──────────────────────────────────────────────────┤
│  Configuration Layer                              │
│  config/base_config.py  | App config              │
│  config/model_config.py | ML hyperparameters      │
├──────────────────────────────────────────────────┤
│  Data Storage                                     │
│  Data/                   | User datasets          │
│  ia_risk_bundle.joblib   | Serialized model bundle│
│  .ia_risk_cache/         | STS Score disk cache   │
└──────────────────────────────────────────────────┘
```

---

## Module Responsibilities

### app.py
**Main application — UI and orchestration**
- Streamlit tab layout (10 tabs) and sidebar controls (language, data source)
- File upload and data source management
- AI Risk model training trigger and model-selection controls
- Result visualization and export (XLSX, CSV, PDF, Markdown)
- Structured execution report: compact status row + detailed per-step expander

### risk_data.py
**Data integration and preparation**
- Multi-format loading: `.xlsx`, `.xls`, `.db`, `.sqlite`, `.csv`, `.parquet`
- Column normalization: Brazilian/English numeric conventions, valve severity encoding (`None < Trivial < Mild < Moderate < Severe`)
- Patient matching across sheets: Name + Procedure Date
- Eligibility filtering (Surgery and Procedure Date non-null)
- Feature engineering: BSA (DuBois formula), combined-surgery flags, procedure weight, thoracic-aorta flag
- STS Score variable mapping

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

### euroscore.py
**EuroSCORE II calculation**
- Published logistic regression formula (Nashef et al., *Eur J Cardiothorac Surg*, 2012): 18 risk factors, 27 coefficients
- Variable mapping from the available dataset; approximations documented in the Analysis Guide tab

### sts_calculator.py
**STS Score acquisition via WebSocket automation**
- Automated querying of the official STS Risk Calculator (`acsdriskcalc.research.sts.org`) via its WebSocket/Shiny interface — not a documented public API
- Disk cache keyed by canonicalized patient payload; patients with a fresh cache hit skip the network query entirely
- Per-patient retry logic: up to 4 attempts with back-off before marking as failed
- Stale fallback: if all retries fail but a prior cached value exists, reuses it and flags as `stale_fallback` in the execution record
- Severity classification: partial failures (some patients fail, usable results remain) → warning; `n_usable == 0` or `fail_ratio ≥ 0.5` → blocking error

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
- `base_config.py` — `AppConfig`: paths, CV folds, decision threshold, random seed, UI settings
- `model_config.py` — candidate model hyperparameters; `get_model_params()`, `list_available_models()`

---

## Architectural Flows

### 1. AI Risk Training

```
File upload (XLSX / CSV / DB / Parquet)
  └─ risk_data.prepare_master_dataset()
       ├─ Multi-format load
       ├─ Column normalization & mapping
       ├─ Patient matching (Name + Procedure Date)
       ├─ Eligibility filter (Surgery + Procedure Date non-null)
       ├─ Feature engineering (BSA, combined-surgery flags, procedure weight)
       └─ Returns PreparedData dataclass

  └─ modeling.train_and_select_model(prepared)
       ├─ clean_features()
       ├─ StratifiedGroupKFold (same patient never in both folds)
       ├─ Per-fold: preprocess → fit candidate → calibrate → OOF predict
       ├─ Candidates: LR, RF, XGB, LGBM, CatBoost, StackingEnsemble
       ├─ Auto-select with clinical-usability guardrails
       └─ Returns TrainedArtifacts dataclass

  └─ Bundle saved: ia_risk_bundle.joblib
       ├─ PreparedData  (feature_columns, reference_df)
       └─ TrainedArtifacts  (best model pipeline, leaderboard, OOF predictions)
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
- **Batch** (Batch & Export tab): iterates uploaded rows; AI Risk incidents are collected in `_batch_ai_incidents` and surfaced in the execution report
- **Temporal** (Temporal Validation tab): `apply_frozen_model_to_temporal_cohort` wraps the row loop with an optional `progress_callback`

### 3. STS Score Cache / Revalidation Flow

```
For each patient in the cohort:
  └─ Canonicalize patient payload → cache key

  If cache hit (fresh):
    └─ Return cached probability immediately (no network query)

  Else:
    └─ WebSocket connection to acsdriskcalc.research.sts.org
         ├─ Map variables to STS input format
         ├─ Retry: up to 4 attempts with back-off
         └─ On success  → write to disk cache, return probability
         └─ On all retries failed:
              ├─ Stale fallback: if a prior cached entry exists → return it, flag as stale_fallback
              └─ Else → mark patient as failed

Post-loop severity:
  ├─ All patients succeeded              → status: OK
  ├─ Some failed, fail_ratio < 0.5      → status: warning
  └─ n_usable == 0 or fail_ratio ≥ 0.5  → status: blocking error
```

### 4. Observability / Execution Report

Every run produces a structured execution record with status, counters, and per-patient incident lists. Surfaced in two places:

- **Compact status row** (sidebar/header): one chip per phase — ✅ OK / ⚠️ warning / ❌ blocking error. A blocking error also displays an inline banner.
- **Execution report expander** (bottom of the Overview tab): per-step breakdown with raw counters (rows in/out, imputed, excluded, failed) and per-patient incident lists for audit.

Phases covered:

| Phase | What is tracked |
|:--|:--|
| Ingestion & normalization | rows loaded, file format, exclusions, imputation counts |
| Cohort eligibility | patients matched, excluded, reason codes |
| AI Risk training | model selected, CV folds, feature count, OOF metrics |
| STS Score execution | fetched, cached, stale, failed, fail ratio |

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

AppConfig.MODEL_CACHE_FILE    # Path to ia_risk_bundle.joblib
AppConfig.LOCAL_DATA_DIR      # Input data directory
AppConfig.TEMP_DATA_DIR       # Temporary / STS cache files
AppConfig.CV_SPLITS           # CV folds (default 5)
AppConfig.RANDOM_SEED         # Reproducibility seed (default 42)
AppConfig.PAGE_TITLE          # Streamlit page title
AppConfig.LANGUAGES           # Supported UI languages
AppConfig.MIN_SAMPLE_SIZE     # Minimum patients for analysis
AppConfig.MISSING_TOKENS      # Strings treated as missing values
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

---

## Security & Privacy

- Patient names and dates are used only for internal cross-sheet matching and are never stored in the model cache or exported results
- The STS Score disk cache stores risk probabilities keyed by canonicalized clinical payload — no patient names
- `ia_risk_bundle.joblib` contains only model weights, feature schemas, and aggregate training metadata — no patient-level data
- Recommended: run on a local network, not internet-facing
