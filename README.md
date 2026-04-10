# AI Risk — Cardiac Surgery Risk Stratification

Research tool developed as part of a master's dissertation on risk stratification in cardiovascular surgery using artificial intelligence.

## What this app does

The system reads structured clinical data from hospital electronic records, builds an analytical dataset with one row per surgery, and calculates three risk scores for predicting 30-day or in-hospital mortality:

| Score | Method | Source |
|:--|:--|:--|
| **AI Risk** | Machine learning model trained on the local dataset | Computed locally |
| **EuroSCORE II** | Published logistic equation (Nashef et al., 2012) | Computed locally from coefficients |
| **STS PROM** | STS Adult Cardiac Surgery Risk Calculator | Obtained via automated queries to the official web calculator |

The app then compares the three scores statistically and provides explainability tools for research and clinical discussion.

### Important: this is a research tool

This application is designed to support academic research in risk stratification. It is **not** a clinical decision-support system and should not be used for autonomous clinical decision-making.

## How each score is computed

### AI Risk (machine learning)

- Trained on preoperative data only (clinical, laboratory, echocardiographic)
- Postoperative complications are never used as predictors
- Multiple candidate algorithms: LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost, MLP, StackingEnsemble
- Validated via StratifiedGroupKFold cross-validation (same patient never in both train and test)
- Best model selected by AUC on calibrated out-of-fold predictions
- Tree-based models calibrated via Platt scaling (sigmoid); calibration applied inside each CV fold for honest OOF evaluation
- Only numerical-stability epsilon clipping (1e-6) — the calibrated probability is the clinical output
- Numeric variables: median imputation + StandardScaler
- Valve severity variables: OrdinalEncoder with clinical order (None < Trivial < Mild < Moderate < Severe) — "None" means no disease, not missing data
- Other categorical variables: mode imputation + TargetEncoder (smooth="auto")

### EuroSCORE II

- Calculated locally using the published logistic regression formula with 18 risk factors and 27 coefficients
- Reference: Nashef et al., *Eur J Cardiothorac Surg*, 2012
- Variables mapped from the available dataset; some approximations are documented in the Analysis Guide tab

### STS Predicted Risk of Mortality (STS PROM)

The STS score is obtained by **automated interaction with the official STS Risk Calculator web application** hosted at `acsdriskcalc.research.sts.org`. This is done by:

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
| `app.py` | Streamlit application (UI, orchestration) |
| `risk_data.py` | Data loading, validation, matching, feature engineering |
| `modeling.py` | ML pipeline: preprocessing, training, model selection |
| `euroscore.py` | EuroSCORE II formula implementation |
| `sts_calculator.py` | STS web calculator automation via WebSocket |
| `explainability.py` | SHAP-based model explainability |
| `stats_compare.py` | Statistical evaluation and model comparison |
| `model_metadata.py` | Model versioning, audit trail, individual reports, export (PDF/XLSX/CSV) |
| `variable_dictionary.py` | Structured variable dictionary for documentation |
| `config/` | Centralized configuration and hyperparameters |

## Decision threshold

The default decision threshold is **8%**. This is a conservative threshold chosen for the cardiac surgery context:

- **Asymmetric cost of errors:** In cardiac surgery, the cost of missing a high-risk patient (false negative) far outweighs the cost of an unnecessary alert (false positive). A missed at-risk patient may die without adequate team preparation; an unnecessary alert only means the team prepares more carefully — causing no harm.
- **Clinical consistency:** The average mortality rate in cardiac surgery ranges from 3–8% globally. The 8% threshold sits just above this range, meaning it does not flag most patients as high-risk, but is low enough to capture patients at real risk before it becomes clinically obvious.
- **Aligned with established scores:** This is consistent with EuroSCORE II stratification thresholds (low risk <3%, intermediate 3–8%, high risk >8%).
- **Sensitivity-oriented:** The threshold favors higher sensitivity (detecting more at-risk patients) at the cost of more false positives — the safer posture in surgical risk stratification.

The threshold is user-adjustable in the app via a slider. AUC, AUPRC, and Brier score are not affected by the threshold — they evaluate the full probability distribution.

## Methodological transparency

This app follows TRIPOD/TRIPOD-AI reporting principles. Key methodological decisions:
- Internal validation only (cross-validation); no external validation yet
- Single-center data; generalizability not established
- Some EuroSCORE II variables are approximated from available fields
- STS obtained via web calculator automation, not a proprietary formula replication
- Post-hoc calibration (Platt scaling) is applied inside each outer CV fold; however, the inner calibration CV uses StratifiedKFold and does not enforce patient grouping (sklearn limitation — risk is minor because calibration fits only 2 parameters)
- NRI and IDI are reported as complementary reclassification metrics, not as primary evidence of model superiority
- Risk of bias should be assessed across PROBAST domains
