# AI Risk - Architecture & Design

## System Overview

AI Risk is a Streamlit-based clinical decision support system for predicting 30-day mortality risk in cardiac surgery patients. It combines machine learning models trained on local data with established clinical risk scores (EuroSCORE II, STS).

```
┌─────────────────────────────────────────┐
│         User Interface (Streamlit)      │
├─────────────────────────────────────────┤
│  app.py - Main application logic        │
│  - Data input & upload                  │
│  - Model selection & configuration      │
│  - Prediction interface                 │
│  - Analysis & comparison                │
├─────────────────────────────────────────┤
│  Core Processing Layer                  │
├─────────────────────────────────────────┤
│  risk_data.py      | Data preparation   │
│  euroscore.py      | EuroSCORE calc     │
│  modeling.py       | ML model training  │
│  stats_compare.py  | Statistical tests  │
│  explainability.py | SHAP explanations  │
├─────────────────────────────────────────┤
│  Configuration Layer                    │
├─────────────────────────────────────────┤
│  config/base_config.py     | App config │
│  config/model_config.py    | ML params  │
├─────────────────────────────────────────┤
│  Data Storage                           │
├─────────────────────────────────────────┤
│  Data/                  | User datasets  │
│  ia_risk_bundle.joblib  | Cached models │
│  .ia_risk_cache/        | Temp files    │
└─────────────────────────────────────────┘
```

## Module Responsibilities

### app.py
**Main application entry point**
- Streamlit UI setup and routing
- User authentication (optional)
- Data source management (upload, Google Sheets, local)
- Model selection and prediction interface
- Result visualization and export

### risk_data.py
**Data integration and preparation**
- Excel/CSV/Database reading
- Column mapping and normalization
- Data validation and eligibility criteria
- Patient matching across tables
- Feature engineering

### modeling.py
**ML model training and selection**
- Feature preprocessing (imputation, scaling, encoding)
- Candidate model instantiation
- Cross-validation training
- Model evaluation and ranking
- Out-of-fold predictions
- Feature importance extraction

### euroscore.py
**EuroSCORE II calculation**
- Formula implementation (logistic regression coefficients)
- Variable mapping from dataset
- Risk category assignment

### stats_compare.py
**Statistical analysis and comparison**
- Performance metrics (AUC, sensitivity, specificity)
- ROC curves and calibration
- Model comparison tests (DeLong, bootstrap)
- Decision curve analysis
- Confidence intervals

### explainability.py
**SHAP-based model interpretation**
- Feature importance (global)
- Local explanations per prediction
- Dependence plots
- Streamlit UI integration

## Data Flow

### Training Pipeline
```
1. Load Excel from local_data/
2. risk_data.prepare_master_dataset()
   ├─ Read Preoperative, Pre-Echocardiogram, Postoperative
   ├─ Normalize and map columns
   ├─ Match patients across tables
   ├─ Apply eligibility criteria
   └─ Return prepared DataFrame
3. modeling.train_and_select_model()
   ├─ Clean features
   ├─ Split preprocessing (numeric vs categorical)
   ├─ Create candidate pipelines
   ├─ Grouped cross-validation
   ├─ Select best model
   └─ Return TrainedArtifacts
4. Cache model to ia_risk_bundle.joblib
```

### Prediction Pipeline
```
1. User enters patient data
2. ui.parse_manual_input() → patient dict
3. modeling.prepare_features(patient)
   ├─ Apply same preprocessing as training
   └─ Return feature vector
4. model.predict_proba(features) → probability
5. explainer.local_explanation() → SHAP values
6. Display risk score + contributing factors
```

## Configuration Management

### base_config.py
Controls all application-level settings:
```python
from config import AppConfig

# Paths
model_path = AppConfig.MODEL_CACHE_FILE
cache_dir = AppConfig.TEMP_DATA_DIR

# Model training
cv_splits = AppConfig.CV_SPLITS
random_seed = AppConfig.RANDOM_SEED

# UI
page_title = AppConfig.PAGE_TITLE
```

### model_config.py
Controls ML hyperparameters:
```python
from config import get_model_params

# Get parameters for a model
xgb_params = get_model_params("XGBoost")

# List all available models
from config import list_available_models
models = list_available_models()
```

## Extension Points

### Adding a New Model
1. Add hyperparameters to `config/model_config.py`:
```python
MODEL_HYPERPARAMS["NewModel"] = {
    "param1": value1,
    "param2": value2,
}
```

2. Update `modeling.py`'s `_build_candidates()`:
```python
from sklearn_new import NewModelClassifier
candidates["NewModel"] = NewModelClassifier(**get_model_params("NewModel"))
```

### Adding New Features
1. Map column names in `risk_data.py`:
```python
FLAT_ALIAS_TO_APP_COLUMNS["new_feature"] = "Column Name"
```

2. Feature engineering in `risk_data.py` or `modeling.py`

### Adding Statistical Tests
1. Implement in `stats_compare.py`
2. Call from `app.py` in analysis section

## Performance Considerations

- **Model caching**: Trained models cached with file timestamp check
- **Data caching**: `@st.cache_data` for expensive operations
- **SHAP computation**: Uses TreeExplainer for speed (fast for tree models)
- **Grouped CV**: Prevents patient leakage between folds

## Security & Privacy

- Patient names/dates used only for internal matching
- No IDs or PHI stored in model cache
- Recommend: Run on local network (not internet-facing)
- Cache cleared on exit
