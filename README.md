# AI Risk — Cardiac Surgery Risk Stratification

Research tool developed as part of a master's dissertation on risk stratification in cardiovascular surgery using artificial intelligence.

## What this app does

The system reads structured clinical data from hospital electronic records, builds an analytical dataset with one row per surgery, and calculates three risk scores for predicting 30-day or in-hospital mortality:

| Score | Method | Source |
|:--|:--|:--|
| **AI Risk** | Machine learning model trained on the local dataset | Computed locally |
| **EuroSCORE II** | Published logistic equation (Nashef et al., 2012) | Computed locally from coefficients |
| **STS Score** | STS Adult Cardiac Surgery Risk Calculator (Predicted Risk of Mortality) | Obtained via automated queries to the official web calculator |

The app then compares the three scores statistically and provides explainability tools for research and clinical discussion.

### Important: this is a research tool

This application is designed to support academic research in risk stratification. It is **not** a clinical decision-support system and should not be used for autonomous clinical decision-making.

## Interface flow

The app is organised as ten tabs in the following order:

| # | Tab | Role |
|:--|:--|:--|
| 1 | **Overview** | Cohort summary, grouped surgery profile (descriptive mortality by surgery category), input-completeness indicator, and execution report for the current run |
| 2 | **Prediction** | Single-patient scoring: AI Risk, EuroSCORE II and STS for one case with per-variable contributions |
| 3 | **Batch & Export** | Full dataset with all three scores plus OOF predictions; XLSX/CSV download |
| 4 | **Statistical Comparison** | Head-to-head comparison of AI Risk, EuroSCORE II and STS with bootstrap 95% CI, DeLong, DCA, NRI/IDI |
| 5 | **Temporal Validation** | Split by procedure date to evaluate stability over time at the fixed 8% threshold |
| 6 | **Data Quality** | Missing-data audit, imputation tracking, valve-severity coverage |
| 7 | **Models** | Leaderboard with calibrated OOF metrics and per-model Youden thresholds; force-select a candidate |
| 8 | **Subgroups** | Performance stratified by clinically relevant subgroups |
| 9 | **Analysis Guide** | Methodological notes, variable mapping, EuroSCORE II approximations |
| 10 | **Variable Dictionary** | Structured dictionary of every variable consumed by the app |

## How each score is computed

### AI Risk (machine learning)

- Trained on preoperative data only (clinical, laboratory, echocardiographic)
- Postoperative complications are never used as predictors
- Candidate algorithms compared in each run: LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost, StackingEnsemble
- Validated via StratifiedGroupKFold cross-validation (same patient never appears in both train and test folds)
- Candidate models are compared by cross-validated, calibrated out-of-fold performance (discrimination and calibration). Automatic selection applies explicit clinical-usability guardrails to the calibrated OOF distribution — the auto-selected default must (a) produce at least some predictions below the 8% clinical threshold, (b) have AUC above a minimum floor, (c) have a Brier score lower than the prevalence baseline, and (d) have a non-degenerate dynamic range. Models that fail any guardrail remain visible in the leaderboard and can still be force-selected manually.
- The current operationally appropriate default is **RandomForest**: its calibrated OOF probabilities cross below 8%, its calibration intercept is near zero and slope near one, and it behaves as a high-sensitivity triage rule at the fixed 8% threshold
- Calibration is applied inside each CV fold for honest OOF evaluation, using a per-model strategy: RandomForest uses sigmoid (Platt scaling) with inner cv≤5; LightGBM and CatBoost use isotonic with inner cv≤5; XGBoost uses isotonic with inner cv≤3. LogisticRegression and StackingEnsemble are used uncalibrated. Only a numerical-stability epsilon clip (1e-6) is applied; the calibrated probability is the clinical output
- Numeric variables: median imputation + StandardScaler
- Valve severity variables: OrdinalEncoder with clinical order (None < Trivial < Mild < Moderate < Severe) — "None" means no disease, not missing data
- Other categorical variables: mode imputation + TargetEncoder (smooth="auto")

### EuroSCORE II

- Calculated locally using the published logistic regression formula with 18 risk factors and 27 coefficients
- Reference: Nashef et al., *Eur J Cardiothorac Surg*, 2012
- Variables mapped from the available dataset; some approximations are documented in the Analysis Guide tab

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

**Limitations:**
- ~1–3% of patients may fail due to ambiguous procedure mapping
- Results may differ slightly from manual entry due to field mapping approximations
- The web interface may change without notice, potentially breaking the automation
- For the dissertation, this should be described as "automated querying of the STS web calculator" rather than "official API"

**Robustness of the STS fetch path:**

- **Cache:** every successful fetch is stored on disk keyed by the canonicalised patient payload, so reruns do not re-query the calculator for unchanged inputs
- **Retries:** each fetch is retried up to 4 times with back-off before being marked as failed (transient WebSocket errors account for most failures)
- **Stale fallback:** if a fresh fetch fails but a previously cached value exists for the same payload, that stale value is reused and flagged as `stale_fallback` in the execution record
- **Severity classification:** partial failures (one or more patients fail but usable results remain) are reported as **warnings**; only `n_usable == 0` or `fail_ratio ≥ 0.5` are reported as **blocking errors**

## How to run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
streamlit run app.py
```

3. Your browser should open automatically.

## Expected data structure

The app accepts multiple data formats:

### Multi-sheet format (`.xlsx`, `.xls`, `.db`, `.sqlite`)

Required sheets/tables:
- `Preoperative` — patient demographics, comorbidities, clinical status
- `Pre-Echocardiogram` — echocardiographic data
- `Postoperative` — outcomes including death

Optional sheets/tables:
- `EuroSCORE II` — pre-calculated EuroSCORE II values for reference
- `EuroSCORE II Automático` — automatically calculated EuroSCORE II values
- `STS Score` — pre-calculated STS values for reference/fallback

### Flat format (`.csv`, `.parquet`)

A single table with all variables. Must include a `morte_30d` or `Death` column for the outcome.

### Ingestion & normalization

All loader paths (`.xlsx`, `.xls`, `.db`, `.sqlite`, `.csv`, `.parquet`) converge on a single normalization routine so that downstream code sees a consistent analytical dataset regardless of the input format:

- **Numeric normalization** — Brazilian and English numeric conventions are both accepted. Strings like `"1,24%"`, `"1.24%"`, `"1,24"`, and `"1.24"` are normalised to the same float; trailing percent signs are stripped and the value is rescaled when appropriate.
- **Valve severity** — valve variables accept the ordered set `None < Trivial < Mild < Moderate < Severe`. The literal value **"None" means *no disease*, not missing data** — it is treated as the lowest level of the ordinal scale, not as an NA.
- **Column exclusion** — columns with **>95% missing** are dropped before modelling so that degenerate columns never reach the preprocessor or the leaderboard.
- **Patient matching** — patient name and procedure date are used exclusively for cross-sheet matching and are never passed to the model as predictors.

## Clinical notes

- Patient name and procedure date are used only for internal matching across sheets — never as predictors
- AI Risk uses exclusively preoperative predictors, preventing temporal leakage
- **All scores displayed are computed by the app** — EuroSCORE II from the published equation, STS via automated query to the web calculator. Values pre-existing in the input file are retained only as optional reference, not as the primary source
- Missing data is handled via imputation; the app tracks and displays which variables were imputed
- Input completeness is classified in 4 levels (complete, adequate, partial, low) based on clinical relevance of missing variables
- Detailed valve measurements (AVA, MVA, PHT, gradients, Vena contracta) are optional and do not penalize the completeness indicator
- BSA (body surface area) is auto-calculated using the DuBois formula
- Statistical summary can be exported in PDF, XLSX, CSV, or Markdown
- Model version, training metadata, and imputation details are recorded for auditability

## Project structure

| File | Role |
|:--|:--|
| `app.py` | Streamlit application (UI, tab routing, orchestration) |
| `ai_risk_inference.py` | Frozen-model inference core shared by individual, batch, and temporal flows |
| `risk_data.py` | Data loading, normalization, patient matching, feature engineering |
| `modeling.py` | ML pipeline: preprocessing, training, candidate selection |
| `euroscore.py` | EuroSCORE II formula implementation |
| `sts_calculator.py` | STS Score WebSocket transport — connects to the STS calculator website |
| `sts_cache.py` | STS Score cache and revalidation policy: TTL, versioning, stale fallback, patient index |
| `observability.py` | Execution report: RunReport/RunStep data structures, builders, and Streamlit renderers |
| `bundle_io.py` | Bundle serialization/deserialization (Streamlit module-reload safety) |
| `subgroups.py` | Subgroup assignment (surgery type, LVEF, renal function) and per-subgroup metrics |
| `report_text.py` | Manuscript-ready Methods and Results text builders |
| `explainability.py` | SHAP-based model explainability |
| `stats_compare.py` | Statistical evaluation and model comparison |
| `model_metadata.py` | Model versioning, audit trail, individual reports; re-exports from export_helpers and temporal_validation |
| `export_helpers.py` | Statistical summary export: Markdown → XLSX / CSV / PDF |
| `temporal_validation.py` | Temporal cohort helpers: year-quarter range, overlap check, locked-model display, Markdown summary |
| `variable_dictionary.py` | Structured variable dictionary for documentation |
| `config/` | Centralized configuration and hyperparameters |

## Execution reporting

Each run is instrumented with a structured execution report surfaced in two places:

- **Top compact status row** (sidebar / header) — one chip per phase (ingestion & normalization, cohort eligibility, model training, STS Score execution) with a single-glyph status: ✅ OK, ⚠️ warning, ❌ blocking error. A blocking error also displays an inline banner at the top of the page.
- **Bottom detailed expander** ("Execution report") — per-step expanders with short summaries, raw counters (rows in / out, imputed, excluded, failed), and the list of incidents (e.g., patients for which the STS fetch failed) so the user can audit what happened in the current run.

Severity is classified independently of the Python exception layer: partial failures (e.g., some patients failing STS while usable results remain) are reported as **warnings** and do not block downstream analysis. Only `n_usable == 0` or `fail_ratio ≥ 0.5` is escalated to **blocking error**.

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
