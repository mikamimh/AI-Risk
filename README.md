# AI Risk — Cardiac Surgery Risk Stratification

[![CI](https://github.com/mikamimh/AI-Risk/actions/workflows/ci.yml/badge.svg)](https://github.com/mikamimh/AI-Risk/actions/workflows/ci.yml)

Research tool developed as part of a master's dissertation on risk stratification in cardiovascular surgery using artificial intelligence.

The training source can be a canonical multi-sheet workbook/database, a flat CSV/Parquet table, or a single-sheet Excel export. Single-sheet Excel files with default names such as `Planilha1` are treated like flat datasets.

## What this app does

The system reads structured clinical data from hospital electronic records, builds an analytical dataset with one row per surgery, and calculates three risk scores for predicting 30-day or in-hospital mortality:

| Score | Method | Source |
|:--|:--|:--|
| **AI Risk** | Machine learning model trained on the local dataset | Computed locally |
| **EuroSCORE II** | Published logistic equation (Nashef et al., 2012) | Computed locally from coefficients |
| **STS Score** | STS Adult Cardiac Surgery Risk Calculator (Predicted Risk of Mortality) | Obtained via automated queries to the official web calculator |

The app then compares the three scores statistically and provides explainability tools for research and clinical discussion. The current model is developed on a local single-center cohort; external generalizability is not assumed and should be evaluated with the Temporal Validation workflow before broader interpretation.

### Important: this is a research tool

This application is designed to support academic research in risk stratification and structured review of model outputs. It is **not** a clinical decision-support system and should not be used for autonomous clinical decision-making.

## Interface flow

The app is organised visually as ten tabs in the following order:

| # | Tab | Role |
|:--|:--|:--|
| 1 | **Overview** | Organised in five fixed snapshot blocks: **Cohort Snapshot** (n, event rate, surgery profile), **Model Snapshot** (model, version, threshold, calibration method), **Performance Snapshot** (leaderboard with calibrated OOF metrics and per-model Youden thresholds), **Operational Snapshot** (score availability), **Audit Snapshot** (eligibility flow and execution report) |
| 2 | **Prediction** | Single-patient scoring: AI Risk, EuroSCORE II, and STS Score for one case with per-variable contributions; individual patient report export (Markdown / PDF / XLSX / CSV) covering scores, input completeness, clinical interpretation, risk factors, and methodological notes |
| 3 | **Batch** | Two workflows in one tab. **Research Export** exports the active research cohort with app-calculated AI Risk, EuroSCORE II, STS Score, classes, and OOF predictions (CSV/XLSX). **Batch Prediction** accepts new patient files, reports compatibility KPIs, computes AI Risk + EuroSCORE II and optional STS Score, shows a short preview plus full table in an expander, and exports CSV/XLSX/Markdown/PDF. |
| 4 | **Comparison** | Head-to-head statistical comparison of AI Risk, EuroSCORE II, and STS Score. Layout: **Operational threshold** → **Main Result — Matched Triple Cohort** → **Calibration at a Glance** → **Threshold Comparison** → probability distributions/diagnostics → **Overall Comparison — Available Data** → **Supplementary Pairwise Comparisons** → **Supplementary Clinical Utility** → **Interpretation & Export**. |
| 5 | **Temporal Validation** | Apply the frozen trained model to an independently uploaded cohort. Layout: **Locked Model** → **Cohort Integrity** (upload, chronological check, normalization, STS availability) → **Main Validation Result** → **Supplementary / Exploratory** (common cohort when available, recalibration, threshold analysis, case-level predictions) → exports. |
| 6 | **Data Quality** | Exception-oriented quality panel: top metrics, issues block when missingness/readiness warnings exist, score availability, validation readiness, and long detailed tables inside expanders. |
| 7 | **Models** | Plain-language guide to how each candidate algorithm works — how it behaves, its strengths and limitations, and when it typically fails. Also includes selected-model predictor/explainability views. |
| 8 | **Subgroups** | Performance stratified by clinically relevant subgroups with controls, best-subgroup insight, compact table, full table expander, caution flags for small n/events, and CSV/PDF/XLSX exports. |
| 9 | **Guide** | Methodological notes, variable mapping, EuroSCORE II approximations; Methods text download (.txt). |
| 10 | **Dictionary** | Live data dictionary generated from the current ingestion code. Shows source/app variables, aliases, missing/blank rules, derived fields, active model usage, and exports CSV plus an XLSX workbook with `Dicionario`, `Aliases_CSV`, and `Regras_de_Leitura`. |

**Model force-selection** is done via the sidebar selectbox (not in a dedicated tab). The sidebar also controls the display language (English / Portuguese).

## Export surfaces

Exports are intentionally grouped by use case:

- **Prediction** exports one individual report in Markdown, PDF, XLSX, and CSV. The report includes the selected AI model probability, EuroSCORE II, STS Score when available, input completeness, imputation detail, clinical interpretation, and audit metadata.
- **Batch — Research Export** exports the current analytical cohort as CSV/XLSX, including app-calculated scores, risk classes, and OOF predictions from all AI candidate models. The on-screen preview is short; the complete table is inside an expander and in the downloads.
- **Batch — Batch Prediction** exports new-patient predictions as CSV/XLSX/Markdown/PDF. CSV/XLSX always include the full result table and all AI model prediction columns, even when the UI preview hides per-model columns.
- **Comparison** has two primary on-demand buttons. **Summary Report** builds a curated PDF with main performance, Calibration at a Glance, threshold classification, and pairwise comparison. **Full Package** builds a ZIP containing `comparison_summary.pdf`, `comparison_full_report.pdf` when PDF support is available, `comparison_full_report.md`, `comparison_tables.xlsx`, `comparison_metrics.csv`, figure data CSVs, and PNG figures for ROC, calibration, and DCA when source data are available. The structured XLSX may include `00_README`, `01_EXECUTIVE_SUMMARY`, `02_MAIN_METRICS`, `03_THRESHOLD_PERF`, `04_CALIBRATION`, `05_PAIRWISE`, `06_RECLASSIFICATION`, `07_CLINICAL_UTILITY`, `08_OVERALL_COMPARE`, `09_ALLPAIRS_FULL`, `10_THRESHOLD_COMPARISON`, and figure-data sheets `11`-`13` depending on data availability.
- **Temporal Validation** exports `ai_risk_temporal_{version}_{date}_summary.xlsx`, `_predictions.csv`, `_report.pdf`, and `_report.md`. The summary workbook includes cohort summary, performance, pairwise comparison, risk categories, calibration, case-level predictions, common-cohort results when applicable, normalization summary when applicable, and exploratory recalibration/threshold sheets when available.
- **Subgroups** exports the current subgroup panel as CSV, a summary PDF, and a lazy-built consolidated XLSX (`subgroup_all_panels.xlsx`) with README, summary, compact subgroup metrics, full subgroup metrics, and caution flags.
- **Dictionary** exports CSV plus `data_dictionary.xlsx` with the live `Dicionario`, accepted flat-file aliases, and current reading rules.

Large exports in Comparison, Temporal Validation, and Subgroups are generated lazily: the first click prepares the file and replaces the same visual slot with the download button. Cached export bytes are invalidated when the relevant model/source/threshold context changes.

## How each score is computed

### AI Risk (machine learning)

- Trained on preoperative data only (clinical, laboratory, echocardiographic)
- Postoperative complications are never used as predictors
- Candidate algorithms compared in each run: LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost, StackingEnsemble
- Validated via StratifiedGroupKFold cross-validation (same patient never appears in both train and test folds)
- Candidate models are compared by cross-validated, calibrated out-of-fold performance (discrimination and calibration). Automatic selection applies explicit clinical-usability guardrails to the calibrated OOF distribution — the auto-selected default must (a) produce at least some predictions below the 8% clinical threshold, (b) have AUC above a minimum floor, (c) have a Brier score lower than the prevalence baseline, and (d) have a non-degenerate dynamic range. Models that fail any guardrail remain visible in the leaderboard and can still be force-selected manually.
- The current operationally appropriate default is **RandomForest**: its calibrated OOF probabilities cross below 8%, its calibration intercept is near zero and slope near one, and it behaves as a high-sensitivity triage rule at the fixed 8% threshold
- Calibration is applied inside each CV fold for honest OOF evaluation, using a per-model strategy: RandomForest uses sigmoid (Platt scaling) with inner cv≤5; LightGBM and CatBoost use isotonic with inner cv≤5; XGBoost uses isotonic with inner cv≤3. LogisticRegression and StackingEnsemble are used uncalibrated. Only a numerical-stability epsilon clip (1e-6) is applied; the calibrated probability is the clinical output
- Individual predictions include a disagreement safeguard: when candidate AI models differ by more than 10 percentage points for the same patient, the selected primary model remains unchanged, but the UI displays a caution banner with the candidate-model minimum, median, maximum, and range. This is an interpretation safeguard only; it does not average models, change calibration, or alter thresholds.
- Numeric variables: median imputation + StandardScaler
- Valve severity variables: OrdinalEncoder with clinical order (None < Trivial < Mild < Moderate < Severe) — "None" means no disease, not missing data
- Other categorical variables: mode imputation + TargetEncoder (smooth="auto")

The same inference core (`ai_risk_inference.py`) is used by all three scoring contexts — individual prediction, batch new-patient prediction, and temporal validation — so the probability computed is identical regardless of how a patient reaches the model.

### EuroSCORE II

- Calculated locally using the published logistic regression formula with 18 risk factors and 27 coefficients
- Reference: Nashef et al., *Eur J Cardiothorac Surg*, 2012
- Variables mapped from the available dataset; some approximations are documented in the Guide tab

### STS Score (STS Predicted Risk of Mortality)

The STS Score is obtained by **automated interaction with the official STS Risk Calculator web application** hosted at `acsdriskcalc.research.sts.org`. This is done by:

1. Mapping patient variables to the STS input format
2. Establishing a WebSocket connection to the STS Risk Calculator's Shiny server
3. Sending patient data and receiving the calculated risk

**This is not a documented, public REST API.** The STS does not publish an official programmatic API for external consumption. The approach used here automates the same web calculator that clinicians use manually, via its WebSocket interface. This distinction is methodologically important:

| Aspect | This implementation | Official STS API |
|:--|:--|:--|
| Data source | Same calculator used by clinicians | No public API exists |
| Method | WebSocket automation of the Shiny interface | N/A |
| Coefficients | Proprietary (server-side) | Not publicly available |
| Availability | Requires internet; may break if STS updates the interface | N/A |
| Validation | Results are consistent with the web calculator for supported procedures | N/A |

**Supported procedures:** Isolated CABG, Isolated AVR, Isolated MVR, AVR+CABG, MVR+CABG, MV Repair, MV Repair+CABG.

**STS eligibility pre-classification:** Before any network query is attempted, each patient row is passed through `classify_sts_eligibility()`, the single authoritative gatekeeper for STS admission. It assigns one of three statuses:

| Status | Meaning | Action |
|:--|:--|:--|
| `supported` | CABG, AVR, MVR, MV Repair, or combinations thereof | Queried normally |
| `not_supported` | Aortic dissection, aneurism, aneurysm, Bentall, aortic root replacement, or similar unsupported variants | Skipped — STS calculator does not support these procedures |
| `uncertain` | OBSERVATION ADMIT, blank surgery string, or other unmappable priority | Always skipped — never silently mapped to any urgency category |

**OBSERVATION ADMIT rule:** This admission type is classified as `uncertain` and skipped by `classify_sts_eligibility()`. It is **never** silently mapped to `Elective` or any other urgency category. `classify_sts_eligibility()` is the sole authority on whether a row reaches the STS calculator.

A compact eligibility summary is shown before querying begins: `STS eligibility: N supported · N not supported (skipped) · N uncertain — OBSERVATION ADMIT or unmapped priority (skipped)`.

**Per-patient timeout:** Each STS query is bounded by a 90-second hard cap (via `asyncio.wait_for`). Before this safeguard was added, the inner WebSocket loop could run up to 80 message cycles × 30 s each = 2 400 s per patient, multiplied by up to 4 retries, making large cohorts non-interactive for hours. Patients that time out return an empty result (`{}`) and are recorded as failures in the execution log.

**Cooperative cancellation (Temporal Validation):** While the STS Score batch is running in the Temporal Validation tab, a **Cancel STS run** button is shown alongside the live progress indicator. Clicking it sets a `threading.Event` abort flag; the batch stops after the current chunk completes (cooperative, not preemptive). Partial results computed before cancellation are preserved and used for analysis. Each row in the final result carries a status label: `completed`, `failed`, `skipped`, or `cancelled_remaining`.

**Limitations:**
- ~1–3% of patients may fail due to ambiguous procedure mapping
- Results may differ slightly from manual entry due to field mapping approximations
- The web interface may change without notice, potentially breaking the automation
- For the dissertation, this should be described as "automated querying of the STS web calculator" rather than "official API"

**Robustness of the STS Score fetch path:**

The STS Score lookup uses a three-tier cache before making a network request:

1. **In-memory cache** — a process-local dictionary keyed by canonicalised patient payload. Checked first on every call. Populated from disk hits (promotion) and successful fresh fetches. Survives tab navigation within the same Streamlit session; cleared on server restart. This eliminates redundant disk reads for patients already seen in the current session.
2. **Persistent disk cache** — JSON files keyed by the same payload hash (TTL = 14 days). Survives server restarts. Successful disk hits are promoted to the in-memory layer.
3. **Network fetch** — WebSocket connection to the STS calculator. Only reached when both cache layers miss.

Additional reliability mechanisms:
- **Retries:** each fetch is retried up to 4 times with back-off before being marked as failed (transient WebSocket errors account for most failures)
- **Stale fallback:** if a fresh fetch fails but a previously cached value exists for the same payload, that stale value is reused and flagged as `stale_fallback` in the execution record
- **Severity classification:** partial failures (one or more patients fail but usable results remain) are reported as **warnings**; only `n_usable == 0` or `fail_ratio ≥ 0.5` are reported as **blocking errors**
- **Dual Shiny protocol support:** the WebSocket parser handles both the legacy Shiny message format (`values.text2`) and the modern format (`method=upd / data.output.text2`), so the fetch path remains functional if the STS calculator's Shiny version changes

After each STS Score batch run, a compact summary line is shown (Cache hits / Misses / Refreshed / Stale fallback / Failed) followed by a collapsible "View STS Score execution details" expander with the last phase label and any per-patient incidents.

**Batch-abort protection and audit trail:** If `STS_MAX_CONSECUTIVE_FAILURES` (10) chunks in a row produce no valid result, the batch is aborted automatically — this prevents an unreachable endpoint from blocking the UI for the full cohort duration. Rows that were never queried due to the abort are classified as `batch_abort` in the failure log (distinct from `fetch` failures that were actively attempted). Partial results computed before the abort are preserved. The UI surfaces:
- An execution summary block (counts by outcome: cached / fresh+refreshed / stale\_fallback / query\_failed / build\_input\_failed / unqueried\_abort)
- A `st.warning` banner when the batch was aborted, showing how many rows were not queried
- A per-chunk log expander (auto-expanded on abort) with success/failure counts per chunk and the exact chunk that triggered the abort

These attributes are exposed on `calculate_sts_batch` for programmatic access: `failure_log`, `last_execution_log`, `chunk_log`, `_batch_aborted`, `_abort_before_query_count`.

## How to run

1. Install dependencies:

```bash
pip install -r requirements.txt -c constraints.txt
```

`requirements.txt` declares the direct dependencies (`>=` lower bounds); `constraints.txt` pins them to known-good versions for reproducible installs.

2. Run the test suite:

```bash
python -m pytest -q
```

3. Start the app:

```bash
streamlit run app.py
```

4. Your browser should open automatically.

### Local Windows launcher (AI Risk.exe)

`AI Risk.exe` is a small compiled launcher (not a bundled app — Python and all dependencies remain in the normal environment). Double-click it instead of the `.bat` file to start the app.

> **Note:** `AI Risk.exe` is no longer versioned in the repository. Build it locally by running `build_exe.bat`.

**Python resolution order:**
1. `.venv\Scripts\python.exe` — project virtual environment (preferred)
2. `venv\Scripts\python.exe` — alternate venv name
3. `python` / `python3` on the system `PATH`

The launcher verifies that Streamlit is available in the resolved environment before starting. If it fails, the console window stays open with a message explaining what was found and what to fix.

**To rebuild the exe** after changing `launcher.py`, run `build_exe.bat`.

## Expected data structure

The app accepts multiple data formats:

### Multi-sheet format (`.xlsx`, `.xls`, `.db`, `.sqlite`)

Use this format when the source is split across clinical tables. Excel/database inputs with multiple sheets/tables are expected to contain the required names below.

Required sheets/tables:
- `Preoperative` — patient demographics, comorbidities, clinical status
- `Pre-Echocardiogram` — echocardiographic data
- `Postoperative` — outcomes including death

Optional sheets/tables:
- `EuroSCORE II` — pre-calculated EuroSCORE II values for reference
- `EuroSCORE II Automático` — automatically calculated EuroSCORE II values
- `STS Score` — pre-calculated STS Score values for reference/fallback

### Flat format (`.csv`, `.parquet`, single-sheet `.xlsx`/`.xls`)

A single table with all variables. Excel exports with one worksheet, including default sheet names such as `Planilha1`, are handled as flat datasets. Must include a `morte_30d` or `Death` column for the outcome.

Format selection rule:
- `.csv` and `.parquet` are always treated as flat datasets.
- `.xlsx`/`.xls` with exactly one worksheet is treated as a flat dataset, regardless of the sheet name.
- `.xlsx`/`.xls` with multiple worksheets is treated as a multi-sheet source and must include `Preoperative`, `Pre-Echocardiogram`, and `Postoperative`.

For flat Excel, the app reads cell values, not visual formatting. Symbols stored as text (for example `<=`, `>=`, `≤`, `≥`, `³`, `μ`, accented text) are preserved better than in many CSV exports; icons, colors, conditional-formatting arrows, and inserted images are not interpreted as data.

Because XLSX preserves symbols more faithfully, training metrics may differ slightly from a CSV export of the same spreadsheet when the CSV encoding has collapsed distinct values into ambiguous text. For example, `≤ 30 days` and `≥ 30 days` can both become `? 30 days` in a lossy CSV export; in XLSX they remain distinct and are modeled as distinct categories.

**Accepted `Death` / `morte_30d` column values** — the canonical timing-based format is preferred, but boolean-style labels are also accepted as a fallback:

| Value(s) | Interpretation |
|:--|:--|
| `Operative`, `Death` | event = 1 (operative/in-hospital death) |
| `0` | event = 1 (death on day of surgery) |
| `1` … `30` | event = 1 (death within 30 days) |
| `31`, `> 30`, `>30` | event = 0 (death beyond 30-day window) |
| `-`, `--` | event = 0 (survivor, standard encoding) |
| `Yes`, `yes`, `Y`, `y`, `True`, `true`, `Sim`, `sim` | event = 1 (boolean fallback) |
| `No`, `no`, `N`, `n`, `False`, `false`, `Não`, `Nao`, `nao` | event = 0 (boolean fallback) |
| blank, `nan`, `none`, `-` | missing — treated as 0 with no warning |
| anything else | treated as 0, a warning is emitted — review those records |

Timing-based values always take precedence over boolean labels. Boolean labels are only applied when the timing parser cannot interpret the value.

### Ingestion & normalization

All loader paths (`.xlsx`, `.xls`, `.db`, `.sqlite`, `.csv`, `.parquet`) converge on a single normalization routine so that downstream code sees a consistent analytical dataset regardless of the input format:

- **Numeric normalization** — Brazilian and English numeric conventions are both accepted. Strings like `"1,24%"`, `"1.24%"`, `"1,24"`, and `"1.24"` are normalised to the same float; trailing percent signs are stripped and the value is rescaled when appropriate.
- **Valve severity** — valve variables accept the ordered set `None < Trivial < Mild < Moderate < Severe`. The literal value **"None" means *no disease*, not missing data** — it is treated as the lowest level of the ordinal scale, not as an NA.
- **Aortic stenosis blank convention** — for the source `Aortic Stenosis` severity field only, a truly blank cell is normalised to `"None"` because this local echo field records positive stenosis grades explicitly while blank denotes no aortic stenosis. Textual unknown tokens such as `"Unknown"`, `"-"`, or `"N/A"` remain missing. This rule is not applied globally to echo/lab fields or to other valve fields.
- **Coronary presentation convention** — `Coronary Symptom` is treated as a single coronary-presentation field because it feeds both the local model and established score mappings. It may contain no symptoms, angina presentations, and ACS/MI labels (`Non-STEMI`, `STEMI`). The literal value `"None"` is canonicalized to `"No coronary symptoms"` before generic missing-token handling. True blanks and textual unknown tokens remain missing.
- **Recent arrhythmia convention** — `Arrhythmia Recent` is a categorical recent-arrhythmia field. The literal value `"None"` is a valid clinical category meaning no recent arrhythmia and is not treated as missing. `"No"` is canonicalized to `"None"` for compatibility with older UI/source entries. True blanks and textual unknown tokens remain missing.
- **Suspension of anticoagulation days** — `Suspension of Anticoagulation (day)` is a conditional numeric field. Blank, not-informed, and not-applicable values remain missing and are never filled with `0`. Simple recoverable text such as `"> 5"`, `"5 days"`, or `"2d"` is parsed to the numeric day value; ambiguous free text remains missing.
- **Binary history columns — implicit negative rule** — a narrow set of binary yes/no history flags follows the cardiac surgery registry convention that *if the condition was present, it was documented; if blank, the condition is absent*. After all standard missing-token normalization (empty string, `"-"`, `"nan"`, etc. → NaN), remaining NaN values in these columns are filled with `"No"` — not by a global rule, but by an explicit named constant (`BLANK_MEANS_NO_COLUMNS`) applied only to the listed variables:

  | Variable | Rationale |
  |:--|:--|
  | `Previous surgery` | Prior cardiac surgery is always charted when present. |
  | `HF` | Heart failure is assessed in every pre-op evaluation; blank = no HF. |
  | `Arrhythmia Remote` | Past arrhythmia is documented when present; blank = no known remote arrhythmia. |
  | `Family Hx of CAD` | Family history of CAD is documented when positive; blank = negative. |
  | `Anticoagulation/ Antiaggregation` | Critical for surgical planning — always documented if in use; blank = not on treatment. |

  **Explicitly excluded from this rule:** `Suspension of Anticoagulation (day)` — this is a numeric conditional field (applicable only when `Anticoagulation=Yes`); blank means N/A or not documented, not zero days. `Arrhythmia Recent` is also excluded because `"None"` is a valid categorical value and blank remains missing.

- **Column exclusion** — columns with **>95% missing** are dropped before modelling so that degenerate columns never reach the preprocessor or the leaderboard.
- **Patient matching** — patient name and procedure date are used exclusively for cross-sheet matching and are never passed to the model as predictors.

### External-dataset normalization pipeline

When an external CSV or Parquet file is ingested (e.g. for Temporal Validation), a dedicated preprocessing pipeline runs **before** the data reaches `prepare_master_dataset`, the STS workflow, or temporal validation. Every action is logged in a structured `ExternalNormalizationReport`; nothing is changed silently.

| Stage | Function | What it does |
|:--|:--|:--|
| 1 — Encoding | `read_external_table_with_fallback` | Tries encodings in order: `utf-8-sig` → `utf-8` → `cp1252` → `latin-1`. Records the encoding used, detected delimiter, and shape. |
| 2 — Column aliases | `canonicalize_external_columns` | Strips whitespace, collapses internal spaces, and maps known snake_case aliases to canonical display names (e.g. `age_years` → `Age (years)`). |
| 3 — Token variants | `normalize_external_tokens` | Maps linguistic Yes/No variants to canonical English: `Sim` → `Yes`, `Não`/`nao` → `No`, `oui` → `Yes`, `nein` → `No`, etc. Applied only to columns where ≥ 50% of values look binary. |
| 4 — Anthropometric units | `normalize_external_units` | Detects probable unit mismatches: height median < 100 → suspect inches → convert × 2.54; weight median > 140 and max > 250 → suspect lbs → convert ÷ 2.205. An optional BSA cross-check flags divergences > 20%. |
| 5 — Clinical scope | `apply_external_scope_rules` | Flags `is_pediatric = True` for age < 18 (outside adult STS ACSD scope). Cleans surgery text and sets `sts_scope_excluded = True` for procedures containing unsupported keywords (dissection / aneurysm / Bentall / Ross / transplant / homograft). |
| 6 — STS preflight | `build_sts_readiness_flags` | Per-row: checks pediatric flag, scope exclusion, and presence/validity of minimum required STS fields (age, sex, surgery, surgical priority). Sets `sts_input_ready`, `sts_missing_required_fields`, `sts_readiness_reason`. |
| 7 — Orchestration | `normalize_external_dataset` | Runs all stages in order; returns `(normalized_df, ExternalNormalizationReport)`. |

**What is auto-corrected (with logging):** encoding fallback, column alias mapping, linguistic token variants, anthropometric unit conversion.

**What is flagged only (not auto-corrected):** pediatric patients, out-of-scope STS procedures, missing/invalid required STS fields.

The normalization summary is surfaced in:
- A collapsible **"Dataset normalization summary"** expander in the Temporal Validation tab
- The **PDF/Markdown report** (a "Dataset Normalization" section with encoding, unit conversions, scope exclusions, and STS-ready row count)
- The `ExternalNormalizationReport.summary_lines()` method for programmatic access

## Clinical notes

- Patient name and procedure date are used only for internal matching across sheets — never as predictors
- AI Risk uses exclusively preoperative predictors, preventing temporal leakage
- **All scores displayed are computed by the app** — EuroSCORE II from the published equation, STS Score via automated query to the web calculator. Values pre-existing in the input file are retained only as optional reference, not as the primary source
- Missing data is handled via imputation; the app tracks and displays which variables were imputed
- Input completeness is classified in 4 levels (complete, adequate, partial, low) based on clinical relevance of missing variables
- Detailed valve measurements (AVA, MVA, PHT, gradients, Vena contracta) are optional and do not penalize the completeness indicator
- BSA (body surface area) is auto-calculated using the DuBois formula
- Model version, training metadata, and imputation details are recorded for auditability

## Project structure

| File | Role |
|:--|:--|
| `app.py` | Streamlit application shell: sidebar/source selection, train/retrain orchestration, shared UI helpers, tab routing, Overview, Prediction, Models, Subgroups, Data Quality, and Dictionary rendering |
| `ai_risk_inference.py` | Frozen-model inference core shared by individual, batch, and temporal flows |
| `risk_data.py` | Data loading, normalization, patient matching, feature engineering |
| `modeling.py` | ML pipeline: preprocessing, training, candidate selection |
| `euroscore.py` | EuroSCORE II formula implementation |
| `sts_calculator.py` | STS Score WebSocket transport — connects to the STS calculator website; holds the process-local in-memory cache |
| `sts_cache.py` | STS Score persistent disk cache: TTL, versioning, stale fallback, patient index, ExecutionRecord |
| `observability.py` | Execution report: RunReport/RunStep data structures, builders, and Streamlit renderers |
| `bundle_io.py` | Bundle serialization/deserialization (Streamlit module-reload safety) |
| `tabs/` | Extracted Streamlit tab modules for Batch & Export, Comparison, and Temporal Validation; each receives shared state through `TabContext` |
| `subgroups.py` | Subgroup assignment (surgery type, LVEF, renal function) and per-subgroup metrics |
| `report_text.py` | Manuscript-ready Methods and Results text builders |
| `explainability.py` | SHAP-based model explainability |
| `stats_compare.py` | Statistical evaluation and model comparison |
| `model_metadata.py` | Model versioning, audit trail, individual reports; re-exports from export_helpers and temporal_validation |
| `export_helpers.py` | Statistical and report export helpers. Builds Comparison Summary PDF, Full PDF, Full Package ZIP, structured comparison XLSX with optional DCA/overall/all-pairs/threshold/figure-data sheets, Markdown/CSV/PDF report conversions, and PNG figure exports. Used by Comparison, Prediction, Batch, and Temporal Validation downloads. |
| `temporal_validation.py` | Temporal validation helper functions: chronology checks, STS availability wording, locked-model display, common-cohort/exploratory summaries, and Markdown report sections |
| `variable_dictionary.py` | Base clinical variable definitions |
| `app_data_dictionary.py` | Live app-reading dictionary generated from ingestion constants, aliases, missing-value rules, plausibility ranges, derived variables, and active model features |
| `config/` | Centralized configuration and hyperparameters |

## Execution reporting

Each run is instrumented with a structured execution report surfaced in two places:

- **Compact status row** (near the top of the page, below the tab bar) — one chip per phase (ingestion & normalization, cohort eligibility, model training, STS Score execution) with a single-glyph status: ✅ OK, ⚠️ warning, ❌ blocking error. A blocking error also displays an inline banner.
- **Detailed expander in the Overview tab** ("Execution report") — per-step expanders with short summaries, raw counters (rows in / out, imputed, excluded, failed), and the list of incidents (e.g., patients for which the STS Score fetch failed) so the user can audit what happened in the current run.

Severity is classified independently of the Python exception layer: partial failures (e.g., some patients failing STS Score while usable results remain) are reported as **warnings** and do not block downstream analysis. Only `n_usable == 0` or `fail_ratio ≥ 0.5` is escalated to **blocking error**.

## Progress transparency

Long-running flows emit phased progress so the user always knows which stage is running. The display is hierarchical — one primary status, one optional secondary substatus:

- **Primary status** (`Phase N/N: label`) — the macro phase of the current flow, shown as a caption below the progress bar.
- **Secondary substatus** (`↳ STS Score subphase N/4: label`) — shown only when an STS Score query is in progress inside a larger flow; visually lighter (indented `↳` prefix) so it reads as a detail under the primary, not a competing status.
- **Progress bar** — advances through the full flow. During STS Score subphases the bar text shows `"STS Score…"` to avoid repeating the label already visible in the substatus line.
- **Compact summary** — after STS Score completes: `Cache hits: N | Misses: N | Refreshed: N | Stale fallback: N | Failed: N`.
- **Details expander** — collapsible, shows last phase, last subphase, and any per-patient incidents.

| Flow | Primary phases |
|:--|:--|
| Training / retraining | 1/5 loading and preparing dataset → 2/5 cohort eligibility → 3/5 training candidate models → 4/5 computing scores (EuroSCORE II / STS Score) → 5/5 building reports and bundle |
| Batch new-patient prediction — local phase | 1/2 applying AI Risk + EuroSCORE II → 2/2 consolidating results |
| Batch new-patient prediction — STS Score subphases | ↳ 1/4 checking cache → ↳ 2/4 identifying misses → ↳ 3/4 querying web calculator → ↳ 4/4 validating and consolidating |
| Temporal validation | 1/5 loading cohort → 2/5 applying AI Risk model → 3/5 computing EuroSCORE II → 4/5 querying STS Score web calculator → 5/5 computing metrics and consolidating |

Primary phase captions are cleared automatically when the flow completes. A collapsible execution-detail expander is shown after each major local flow completes, reporting the last phase, last subphase, and any relevant counts or incidents.

## Session reuse

Two results are preserved across tab navigation without recomputation:

**STS Score in-memory cache** — the process-local cache in `sts_calculator.py` persists for the lifetime of the server process. Patients queried once in any flow (training, batch, temporal validation) are resolved from memory on every subsequent call in the same session, skipping both the disk read and the network round-trip.

**Temporal validation session cache** — when the Temporal Validation tab has been run and the user navigates away and back, the app checks whether the current context (uploaded file identity, bundle save timestamp, selected model, threshold) matches the previously stored result via a 16-character SHA-256 context signature. If it matches, results are restored from `st.session_state` immediately without re-running AI Risk inference, EuroSCORE II, or STS Score queries. A notice is shown: "Temporal validation results restored from session — no recomputation performed. Click **Run temporal validation** to recompute." Results are recomputed when any context parameter changes.

## Temporal validation behavior

The Temporal Validation tab applies the frozen trained model to an **independently uploaded patient cohort**. It does not split the training dataset by date.

- The user uploads a separate patient file (the temporal cohort)
- The app checks that the temporal cohort does not chronologically overlap with the training cohort; if it does, the run button is disabled
- The locked threshold (8% clinical default) is applied from training and is not recalculated
- All three scores are computed: AI Risk (via the frozen model), EuroSCORE II, and STS Score (via the web calculator with the same cache path as other flows)
- Performance metrics are reported at the fixed 8% threshold
- Results persist in session state for tab-navigation reuse (see "Session reuse" above)
- Pairwise ROC comparison uses both bootstrap ΔAUC with 95% CI (always applicable) and the DeLong test. DeLong is **suppressed** when the validation cohort has fewer than 2 events or fewer than 2 non-events — its covariance estimate is mathematically undefined below that floor. In that case the report shows an em dash ('—') for the DeLong p-value with a short footnote explaining the skip, and the bootstrap ΔAUC remains the primary comparison statistic.

**Surrogate timeline detection:** If the `surgery_year` column contains values above 2050, the dataset is automatically identified as using a **de-identified surrogate timeline** (e.g., MIMIC-IV re-identifies dates as 2111–2195). In this case: (a) the chronological overlap check uses the surrogate years without comparing them to real training dates; (b) all date range displays include a notice — *"de-identified surrogate timeline — not real clinical dates"*; and (c) the range label changes from "Date range" to "Surrogate range". This prevents de-identified years from being misread as real clinical dates.

**STS mode selector:** A checkbox allows the user to include or exclude STS Score from the temporal validation run. When unchecked, STS queries are skipped entirely and the analysis proceeds with AI Risk and EuroSCORE II only. The checkbox defaults to enabled when `websockets` is installed.

**Partial STS availability policy:** Temporal validation now classifies STS coverage for the cohort as `complete`, `partial`, or `unavailable` based on usable final STS scores among STS-eligible rows. When coverage is partial, the UI and PDF report show an explicit warning with exact counts (for example, `6 of 22 eligible`) and label STS summaries as subset-only. When coverage is unavailable, STS-specific cohort-level summaries are omitted and replaced by a clear note. Raw CSV/XLSX exports still retain the STS columns.

### Primary analysis

The primary analysis is methodologically locked:

- The **frozen model** is applied as-is — no retraining, no recalibration.
- The **locked threshold (8%)** remains unchanged for all primary classification metrics.
- Comparisons with EuroSCORE II and STS use the same cohort and the same outcome definition.
- Metrics reported: AUC, AUPRC, Brier score, calibration intercept and slope, Hosmer–Lemeshow p, sensitivity, specificity, PPV, NPV (at the locked threshold), pairwise DeLong AUC comparisons, and risk-category distributions.

### Exploratory analysis

An optional **exploratory appendix** is appended to the report and clearly labelled as supplementary. It includes:

- **Post-hoc recalibration** — intercept-only, intercept + slope (logistic), and isotonic regression, applied to the validation cohort after the fact.
- **Threshold comparison** — fixed clinical thresholds (2%, 5%, 8%, 10%) and the **Youden's J optimum** per model.
- **Confusion matrix components** (TP, FP, TN, FN) for each threshold row.

> **Important:** The exploratory section does not alter the primary results. Recalibration is post-hoc and data-driven; Youden's J is optimised on the same cohort being evaluated. Neither should be used to report primary model performance or to replace the locked threshold in clinical decision-making.

### STS availability

STS scoring requires a network call to an external endpoint and is only valid for patients classified as **supported** (within STS ACSD surgical scope). Patients are classified as:

| Status | Description |
|---|---|
| `supported` | STS-eligible; query attempted |
| `not_supported` | Outside STS ACSD scope (e.g. Bentall, aortic dissection repair); skipped |
| `uncertain` | Ambiguous surgery type or missing fields; skipped |

The report includes a **STS pipeline accounting table** that shows total cohort size, counts per eligibility category, final usable STS scores, and coverage percentage among supported patients.

When STS coverage is **partial**, STS-based metrics reflect only the eligible subset and are not directly comparable to full-cohort AI Risk and EuroSCORE II metrics. A **common-cohort comparison** (all three models evaluated on the STS-available subset) is included in the report when applicable, enabling fair side-by-side comparison.

### Exports and artefacts

**Filename convention:** `ai_risk_temporal_{version}_{date}_{type}.{ext}` — for example `ai_risk_temporal_1.0.0_20260419_summary.xlsx`.

| Artefact | Format | Filename suffix | Contents |
|---|---|---|---|
| Full report | PDF / Markdown | `_report.pdf` / `_report.md` | Primary results + STS accounting + common cohort + exploratory appendix |
| Summary tables | XLSX (multi-sheet) | `_summary.xlsx` | `cohort_summary`, `performance`, `pairwise_comparison`, `risk_categories`, `calibration`, `case_level_predictions`, plus `common_cohort`, `Normalization_Summary`, `Exploratory_Recalibration`, and `Exploratory_Thresholds` when available |
| Case-level predictions | CSV | `_predictions.csv` | Per-patient scores and risk classes |
| STS eligibility log | XLSX | `sts_eligibility.xlsx` | Separate download shown in STS eligibility/detail sections when the eligibility table is available |
| STS patient audit | XLSX | `sts_patient_audit.xlsx` | Separate download shown with result details when per-patient STS audit records are available |

### Auditability

The **execution details** expander (visible after a run) exposes:

- Pipeline provenance: data loading, surgery classification, STS input builder, model inference steps.
- Content hash and context signature used for session caching.
- STS eligibility breakdown and per-chunk execution log.
- Detailed per-patient failure log when STS queries fail or the batch is aborted early.

### Performance and interpretation notes

**Performance.** Large cohorts, interactive charts, threshold sweeps, and exploratory recalibration can increase processing time. STS scoring time depends on endpoint response latency and cohort size. The STS batch will abort automatically after a configurable number of consecutive failures to avoid unbounded waits.

**Statistical interpretation.** In temporal cohorts with a different event rate from the training set, discrimination (AUC) may remain acceptable while calibration is shifted. This is expected when the baseline risk or case-mix has changed over time and does not by itself invalidate the model — it is one reason the exploratory recalibration module exists.

## Decision threshold

The **operational clinical threshold remains fixed at 8%**. This is the default used throughout the app for classification, clinical comparison, and temporal validation.

- **Asymmetric cost of errors:** In cardiac surgery, the cost of missing a high-risk patient (false negative) far outweighs the cost of an unnecessary alert (false positive). A missed at-risk patient may die without adequate team preparation; an unnecessary alert only means the team prepares more carefully — causing no harm.
- **Clinical consistency:** The average mortality rate in cardiac surgery ranges from 3–8% globally. The 8% threshold sits just above this range, meaning it does not flag most patients as high-risk, but is low enough to capture patients at real risk before it becomes clinically obvious.
- **Aligned with established scores:** This is consistent with EuroSCORE II stratification thresholds (low risk <3%, intermediate 3–8%, high risk >8%).
- **Operationally a high-sensitivity triage rule:** At 8% the current AI Risk configuration (RandomForest) behaves as a high-sensitivity triage threshold — it favors detecting at-risk patients at the cost of more false positives, which is the safer posture in surgical risk stratification.

The app additionally computes and displays a **per-model Youden threshold** (the OOF-optimal J cutoff) in the leaderboard as a complementary, model-specific reference. Youden is shown for auditability and balanced-classifier comparison; **it is not the default operational threshold**. In the Statistical Comparison tab the user can switch between the fixed 8% clinical mode and the stored Youden mode, but the fixed 8% remains the default.

The threshold slider and the Youden switch never modify the model, the calibrated probabilities, or any discrimination/calibration metric — AUC, AUPRC, and Brier score evaluate the full probability distribution and are unaffected by any threshold choice.

## Methodological transparency

This app follows TRIPOD/TRIPOD-AI reporting principles. Key methodological decisions:
- Internal validation only (cross-validation); no external validation yet
- Single-center data; generalizability not established
- Some EuroSCORE II variables are approximated from available fields
- STS Score obtained via web calculator automation, not a proprietary formula replication
- Post-hoc calibration is applied inside each outer CV fold using a per-model strategy (see the AI Risk section above for the exact method per model); the inner calibration CV uses StratifiedKFold and does not enforce patient grouping (sklearn limitation — risk is minor because the inner fits are 1- or 2-parameter)
- Automatic best-model selection applies explicit clinical-usability guardrails to the calibrated OOF distribution; models that fail any guardrail remain in the leaderboard and can still be force-selected manually
- Discrimination (AUC, AUPRC) and calibration (Brier, intercept, slope) are both reported and used jointly
- The operational clinical threshold is fixed at 8%; the per-model Youden threshold is shown as a complementary reference, not as the default
- AI Risk is a complementary analytical/research tool — it is not a clinical decision-support system and must not be used for autonomous clinical decision-making
- NRI and IDI are reported as complementary reclassification metrics, not as primary evidence of model superiority
- Risk of bias should be assessed across PROBAST domains
