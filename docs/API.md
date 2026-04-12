# AI Risk API Reference

## Core Functions

### risk_data.py

#### `prepare_master_dataset(source) -> PreparedData`
Load and prepare data from any supported source format.

**Parameters:**
- `source`: Path to an `.xlsx`, `.xls`, `.db`, `.sqlite`, `.csv`, or `.parquet` file

**Returns:**
- `PreparedData` dataclass containing:
  - `data`: Prepared DataFrame with matched records
  - `feature_columns`: List of feature column names
  - `info`: Dictionary with preparation statistics

**Raises:**
- `FileNotFoundError`: If file does not exist
- `ValueError`: If required sheets are missing or no matching records are found

**Example:**
```python
from risk_data import prepare_master_dataset

prepared = prepare_master_dataset("patient_data.xlsx")
print(f"n={len(prepared.data)}, features={len(prepared.feature_columns)}")
```

---

#### `procedure_weight(surgery_str: str) -> int`
Calculate procedure complexity weight from surgery description.

**Parameters:**
- `surgery_str` (str): Comma-separated surgery list

**Returns:**
- Weight (1, 2, or 3+)

**Example:**
```python
weight = procedure_weight("CABG, mitral valve repair")  # Returns 2
```

---

### modeling.py

#### `train_and_select_model(df: pd.DataFrame, feature_columns: List[str]) -> TrainedArtifacts`
Train candidate models and select best performer using grouped cross-validation.

**Parameters:**
- `df`: Master dataset with features and target
- `feature_columns`: List of features to use
- `y_col` (default "morte_30d"): Target variable
- `group_col` (default "_patient_key"): Patient grouping column

**Returns:**
- `TrainedArtifacts` containing:
  - `model`: Best sklearn Pipeline
  - `leaderboard`: DataFrame ranking all candidates
  - `oof_predictions`: Out-of-fold predictions
  - `feature_columns`: Final features used
  - `best_model_name`: Name of selected model

**Example:**
```python
from modeling import train_and_select_model

artifacts = train_and_select_model(df, feature_cols)
print(artifacts.leaderboard)  # See ranking of all models
```

---

#### `clean_features(df: pd.DataFrame) -> pd.DataFrame`
Clean and standardize feature data.

**Operations:**
- Strip whitespace from string columns
- Replace missing tokens (nan, unknown, etc.) with NaN
- Attempt numeric conversion
- Remove non-informative columns

**Example:**
```python
from modeling import clean_features

X_clean = clean_features(X_raw)
```

---

### euroscore.py

#### `euroscore_from_row(row: pd.Series) -> float`
Calculate EuroSCORE II probability from patient record.

**Parameters:**
- `row`: Patient data with required columns

**Returns:**
- Probability (0-1)

**Required columns:** age, sex, nyha, ccs4, diabetes, surgical_priority, etc.

**Example:**
```python
from euroscore import euroscore_from_row

prob = euroscore_from_row(patient_row)
print(f"EuroSCORE II: {prob:.1%}")
```

---

### stats_compare.py

#### `evaluate_scores(df: pd.DataFrame, y_col: str, score_cols: List[str]) -> pd.DataFrame`
Evaluate multiple risk scores on dataset.

**Parameters:**
- `df`: Data with outcomes and scores
- `y_col`: Outcome column name
- `score_cols`: List of score column names

**Returns:**
- DataFrame with metrics (AUC, AUPRC, Brier)

**Example:**
```python
from stats_compare import evaluate_scores

results = evaluate_scores(
    df,
    y_col="morte_30d",
    score_cols=["ia_risk_pred", "euroscore_calc", "sts_score"]
)
print(results)
```

---

#### `bootstrap_auc_diff(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, n_boot: int = 2000) -> dict`
Bootstrap confidence interval for AUC difference between two models.

**Parameters:**
- `y`: Binary outcome
- `p1`: Predictions from model 1
- `p2`: Predictions from model 2
- `n_boot`: Number of bootstrap samples

**Returns:**
- Dictionary with keys:
  - `delta_auc`: Difference in AUC
  - `ci_lower`: 95% CI lower bound
  - `ci_upper`: 95% CI upper bound
  - `p_value`: P-value

**Example:**
```python
from stats_compare import bootstrap_auc_diff

result = bootstrap_auc_diff(y_test, model1_probs, model2_probs)
print(f"Model 1 - Model 2 AUC difference: {result['delta_auc']:.3f}")
```

---

### explainability.py

#### `class ModelExplainer`
SHAP-based explainer for model predictions.

**Methods:**

##### `global_importance(X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame`
Compute feature importance across dataset.

**Returns:**
- DataFrame with columns: feature, mean_abs_shap, std_shap, mean_shap

**Example:**
```python
from explainability import ModelExplainer

explainer = ModelExplainer(model, X_train)
importance = explainer.global_importance(X_test, top_n=10)
print(importance)
```

---

##### `local_explanation(X: pd.DataFrame, idx: int) -> Tuple[plt.Figure, float]`
Generate SHAP explanation for single prediction.

**Parameters:**
- `X`: Data
- `idx`: Row index

**Returns:**
- Tuple of (matplotlib figure, probability)

**Example:**
```python
fig, prob = explainer.local_explanation(X_test, idx=0)
print(f"Prediction: {prob:.1%}")
plt.show()
```

---

##### `plot_importance(X: pd.DataFrame, top_n: int = 15) -> plt.Figure`
Create bar chart of feature importance.

---

##### `plot_dependence(X: pd.DataFrame, feature_name: str) -> plt.Figure`
Show how predictions depend on a feature.

---

##### `plot_beeswarm(X: pd.DataFrame, top_n: int = 12) -> plt.Figure`
Create beeswarm plot of feature impacts.

---

## Configuration

### AppConfig
```python
from config import AppConfig

# Paths
AppConfig.MODEL_CACHE_FILE      # Path to cached model
AppConfig.LOCAL_DATA_DIR         # Input data directory
AppConfig.TEMP_DATA_DIR          # Temporary files

# Training
AppConfig.CV_SPLITS              # Number of CV folds (default 5)
AppConfig.RANDOM_SEED            # Reproducibility seed (default 42)

# UI
AppConfig.PAGE_TITLE             # Streamlit page title
AppConfig.LANGUAGES              # Supported languages

# Data
AppConfig.MIN_SAMPLE_SIZE        # Minimum samples for analysis (default 30)
AppConfig.MISSING_TOKENS         # Strings treated as missing
```

### list_available_models()
```python
from config import list_available_models

models = list_available_models()
# ['LogisticRegression', 'RandomForest', 'XGBoost', ...]
```

### get_model_params(model_name)
```python
from config import get_model_params

params = get_model_params("RandomForest")
# {'n_estimators': 400, 'random_state': 42, ...}
```

---

## Data Classes

### PreparedDataset
```python
from dataclasses import dataclass
import pandas as pd

@dataclass
class PreparedDataset:
    data: pd.DataFrame          # Prepared data
    feature_columns: List[str]  # Feature names
    info: dict                  # Statistics
```

### TrainedArtifacts
```python
@dataclass
class TrainedArtifacts:
    model: Pipeline                        # Best model
    leaderboard: pd.DataFrame             # Model rankings
    oof_predictions: Dict[str, np.ndarray] # CV predictions
    feature_columns: List[str]            # Features used
    fitted_models: Dict[str, Pipeline]    # All trained models
    best_model_name: str                  # Name of best
```

---

### sts_calculator.py

**STS Score acquisition via WebSocket automation.**

The module queries the official STS Risk Calculator (`acsdriskcalc.research.sts.org`) via its WebSocket/Shiny interface. This is not a documented public API — it automates the same web calculator that clinicians use manually.

Key behaviours:
- Results are cached on disk, keyed by canonicalized patient payload. A cache hit skips the network query entirely.
- Each patient is retried up to 4 times with back-off before being marked as failed.
- If all retries fail but a prior cached value exists, it is returned and flagged as `stale_fallback` in the execution record.
- Severity: partial failures (some patients fail, usable results remain) → warning; `n_usable == 0` or `fail_ratio ≥ 0.5` → blocking error.

**Supported procedures:** Isolated CABG, Isolated AVR, Isolated MVR, AVR+CABG, MVR+CABG, MV Repair, MV Repair+CABG.

---

### ai_risk_inference.py

**Frozen-model inference core. No Streamlit dependencies.**

All three inference flows (individual, batch, temporal validation) route through this module. The frozen pipeline (`predict_proba`) is never retrained or recalibrated after the bundle is saved.

---

#### `_run_ai_risk_inference_row(*, model_pipeline, feature_columns, reference_df, row_dict, patient_id, numeric_cols, language) -> dict`
Unified per-row inference. Encapsulates input assembly, dtype alignment, feature cleaning, prediction, and completeness assessment.

**Returns dict with keys:**
- `probability` — float, or `None` on failure
- `completeness` — dict from `assess_input_completeness`, or `None` on failure
- `incident` — `None` on success; `{"patient_id", "stage", "reason"}` on failure
- `model_input` — cleaned DataFrame passed to `predict_proba`, or `None` on failure

---

#### `apply_frozen_model_to_temporal_cohort(*, model_pipeline, feature_columns, reference_df, temporal_data, language, progress_callback) -> dict`
Row-level inference over a prepared temporal cohort DataFrame.

**Parameters:**
- `temporal_data`: DataFrame already processed by `prepare_master_dataset`
- `progress_callback`: optional `callback(i, n)` for driving a Streamlit progress bar

**Returns dict with keys:**
- `probabilities` — list of floats, `NaN` for failed rows
- `completeness` — list of level strings
- `incidents` — list of `{patient_id, stage, reason}` dicts
- `n_total`, `n_failed` — integers

---

#### `_build_input_row(feature_columns, form) -> pd.DataFrame`
Assembles a single-row DataFrame from a raw input dict. Handles truncated column names (Excel export artefact), derives combined-surgery flags, and fills categorical defaults for boolean comorbidities.

---

#### `_align_input_to_training_schema(input_df, reference_df) -> pd.DataFrame`
Coerces column dtypes to match the training-time schema (numeric coercion with symbol stripping; categorical casting).

---

#### `_get_numeric_columns_from_pipeline(model_pipeline) -> set`
Extracts the set of numeric column names from a trained sklearn Pipeline's `prep` step.

---

#### `_patient_identifier_from_row(row_dict, fallback_index) -> str`
Returns the best-effort patient identifier for incident reporting. Checks `Name`, `Nome`, `_patient_key`; falls back to `row_{fallback_index + 1}`.

---

### export_helpers.py

**Pure statistical summary export helpers. No project-level imports.**

---

#### `build_statistical_summary(triple_ci, calib_df, formal_df, delong_df, reclass_df, threshold, threshold_metrics, n_triple, model_version, language) -> str`
Assembles a Markdown statistical report from input DataFrames. Returns a single Markdown string with section headers and tables.

---

#### `statistical_summary_to_dataframes(md_text) -> Dict[str, pd.DataFrame]`
Parses a Markdown summary into a `dict` of DataFrames (one per section/table).

---

#### `statistical_summary_to_xlsx(md_text) -> bytes`
Converts the Markdown summary to XLSX bytes. Each section becomes a separate worksheet.

---

#### `statistical_summary_to_csv(md_text) -> str`
Converts the Markdown summary to a CSV string. All sections are concatenated with section-header comments.

---

#### `statistical_summary_to_pdf(md_text) -> bytes`
Converts the Markdown summary to PDF bytes. Requires `fpdf2`. Returns `b""` if `fpdf2` is not installed.

---

### temporal_validation.py

**Temporal cohort helpers. No Streamlit dependencies.**

---

#### `extract_year_quarter_range(data: pd.DataFrame) -> tuple`
Returns `(start, end)` year-quarter strings derived from `surgery_year` / `surgery_quarter` columns. Format: `"2024-Q1"` or `"Unknown"` when columns are absent.

---

#### `check_temporal_overlap(training_start, training_end, validation_start, validation_end) -> dict`
Compares training and validation temporal ranges. Accepts `"2024-Q1"` or plain year strings.

**Returns dict with keys:** `overlap` (bool), `status`, `severity`, `message_en`, `message_pt`.

Status values:
- `no_overlap` — validation strictly after training (ideal); severity: `success`
- `overlap` — periods overlap; severity: `warning`
- `validation_before_training` — retrograde validation; severity: `error`
- `unknown` — could not parse one or more ranges; severity: `warning`

---

#### `format_locked_model_for_display(metadata, language) -> pd.DataFrame`
Formats locked-model metadata as a two-column (Property / Value) DataFrame for display in the Temporal Validation tab.

---

#### `build_temporal_validation_summary(cohort_summary, performance_df, pairwise_df, calibration_df, risk_category_df, metadata, threshold, language) -> str`
Builds a Markdown report for temporal validation results, including cohort summary, discrimination/calibration tables, pairwise comparison, and risk category distribution.

---

## Error Handling

All functions document exceptions in docstrings:

```python
try:
    artifacts = train_and_select_model(df, features)
except ValueError as e:
    print(f"Data preparation failed: {e}")
except FileNotFoundError as e:
    print(f"Model file not found: {e}")
```
