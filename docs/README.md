# AI Risk - Expanded README

## Quick Start

### Installation
```bash
# Clone repository
git clone <repo-url>
cd "AI Risk - Claude"

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run app.py
```

Browser should open automatically at `http://localhost:8501`

---

## What is AI Risk?

**AI Risk** is a research tool for **30-day mortality risk** stratification in cardiac surgery patients using:

1. **Machine Learning (AI Risk)**
   - Trained on your institutional data
   - 6 candidate models (LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost, StackingEnsemble)
   - Automatic model selection via grouped cross-validation with clinical-usability guardrails
   
2. **EuroSCORE II**
   - Published risk prediction model (Nashef et al., 2012)
   - 18 risk factors, 27 coefficients
   - Computed locally from the published formula

3. **STS Score (Society of Thoracic Surgeons)**
   - STS Adult Cardiac Surgery Predicted Risk of Mortality
   - Obtained via automated querying of the official STS web calculator (WebSocket/Shiny interface)
   - Cached on disk per patient; retried up to 4 times on failure; stale fallback if prior cache exists

All three predictions are **compared statistically** on your data.

---

## Key Features

### 🎯 Risk Prediction
- Single patient or batch predictions
- Risk categories: Low (<5%), Intermediate (5-15%), High (>15%)
- Probability outputs: 0-100%

### 📊 Statistical Analysis
- **Performance metrics**: AUC, sensitivity, specificity, NPV/PPV
- **Calibration**: Brier score, Hosmer-Lemeshow test, calibration curves
- **Model comparison**: Bootstrap CI, DeLong test, decision curves
- **Reclassification**: NRI, IDI metrics

### 🔍 Model Interpretability (SHAP)
- **Global features**: Which variables matter most?
- **Local explanations**: Why this specific prediction?
- **Dependence plots**: How does risk change with age, EF, etc.?
- **Contribution analysis**: Top 5 factors driving this patient's risk

### 🔄 Data Management
- Upload Excel data
- Google Sheets integration
- Local data storage
- Automatic caching

---

## Architecture

### System Layers

```
┌──────────────────────────────────────────────┐
│  Streamlit UI (app.py) — 10 tabs             │
├──────────────────────────────────────────────┤
│  Inference Layer                             │
│  • ai_risk_inference.py  ← frozen-model core │
├──────────────────────────────────────────────┤
│  Core Processing Layer                       │
│  • modeling.py        ← ML training          │
│  • risk_data.py       ← Loading & prep       │
│  • stats_compare.py   ← Statistical analysis │
│  • euroscore.py       ← EuroSCORE II formula │
│  • sts_calculator.py  ← STS Score (WebSocket)│
│  • explainability.py  ← SHAP                 │
├──────────────────────────────────────────────┤
│  Metadata / Export / Temporal Layer          │
│  • model_metadata.py      ← Versioning, audit│
│  • export_helpers.py      ← Report export    │
│  • temporal_validation.py ← Temporal helpers │
├──────────────────────────────────────────────┤
│  Configuration Layer                         │
│  config/ (Centralized settings)              │
└──────────────────────────────────────────────┘
```

### Data Flow

**Training:**
```
File upload → risk_data (normalize + match) → modeling (StratifiedGroupKFold CV)
  → per-fold calibration → auto-select with guardrails → ia_risk_bundle.joblib
```

**Frozen inference (individual / batch / temporal — all share the same path):**
```
Input dict → ai_risk_inference (_build_input_row → _align_schema → clean_features
  → predict_proba → assess_completeness) → {probability, completeness, incident}
```

**STS Score:**
```
Patient dict → sts_calculator (cache lookup → WebSocket query w/ retries → stale fallback)
  → probability + execution record (OK / warning / blocking error)
```

---

## Configuration

All settings centralized in `config/` folder:

### app level (`config/base_config.py`)
```python
from config import AppConfig

AppConfig.CV_SPLITS = 5          # Cross-validation folds
AppConfig.RANDOM_SEED = 42       # Reproducibility
AppConfig.MIN_SAMPLE_SIZE = 30   # Minimum for analysis
```

### model level (`config/model_config.py`)
```python
from config import get_model_params

params = get_model_params("XGBoost")
# Easily modify hyperparameters for all models
```

---

## Data Requirements

### Input Format: Excel (`.xlsx`)

**Required sheets:**
- `Preoperative` - Clinical data before surgery
- `Pre-Echocardiogram` - Cardiac imaging
- `Postoperative` - Outcomes (death/survival)

**Optional sheets:**
- `EuroSCORE II` - Manual scores
- `STS Score` - Operative mortality
- `EuroSCORE II Automático` - Calculated scores

**Matching:** Patient name + Procedure date

See [docs/DATA_FORMAT.md](./DATA_FORMAT.md) for detailed specifications.

---

## Model Selection

### Available Models

| Model | Notes |
|-------|-------|
| **Logistic Regression** | Baseline; uncalibrated |
| **Random Forest** | Sigmoid (Platt) calibration; current default |
| **XGBoost** | Isotonic calibration |
| **LightGBM** | Isotonic calibration |
| **CatBoost** | Isotonic calibration |
| **StackingEnsemble** | Combines the above; uncalibrated |

**Automatic selection:** All 6 candidates are trained under StratifiedGroupKFold cross-validation (same patient never in both folds). Per-model calibration is applied inside each fold. The best model is auto-selected using calibrated out-of-fold AUC as the primary criterion, subject to clinical-usability guardrails (AUC floor, Brier baseline, dynamic range, threshold coverage). Models that fail any guardrail remain visible in the leaderboard and can be force-selected manually.

---

## Performance Evaluation

### Metrics Reported

**Discrimination** (how well it separates risk groups)
- AUC-ROC: 0.5 (random) to 1.0 (perfect)
- AUPRC: Useful when outcome is rare
- Sensitivity/Specificity: At chosen threshold

**Calibration** (does prob match reality?)
- Brier score: MSE between predicted and actual (lower better)
- Calibration curve: Visual comparison
- Hosmer-Lemeshow test: Statistical fit

**Decision-making**
- Decision Curve Analysis: Clinical utility by threshold
- Net Reclassification Index (NRI): How many patients reclassified?
- Integrated Discrimination Index (IDI): Improved ranking?

---

## Explainability (SHAP)

Why did the model predict 25% risk for this patient?

```python
from explainability import ModelExplainer

explainer = ModelExplainer(trained_model, X_train)

# 1. Global: What matters overall?
importance = explainer.global_importance(X_test, top_n=10)

# 2. Local: What drove THIS prediction?
fig, prob = explainer.local_explanation(X_test, idx=0)

# 3. Dependence: How does AGE affect predictions?
fig = explainer.plot_dependence(X_test, 'Age (years)')
```

Output shows:
- ✅ **Increases risk**: Red bars (e.g., LVEF 25%)
- ❌ **Decreases risk**: Blue bars (e.g., Young age)
- 📊 **Magnitude**: Longer bar = Stronger effect

---

## Troubleshooting

### App won't start
```bash
# Check Python version
python --version  # Should be 3.8+

# Check dependencies
pip list | grep streamlit

# Clear cache
rm -rf .streamlit/cache
streamlit run app.py
```

### Data loading fails
- [ ] Check Excel has 3 required sheets
- [ ] Verify column names match (case-sensitive)
- [ ] Ensure Name & Procedure Date columns exist
- [ ] Check for duplicate patient-date combinations

See [docs/DATA_FORMAT.md](./DATA_FORMAT.md#troubleshooting) for details.

### Model training takes too long
- [ ] Reduce `CV_SPLITS` in config
- [ ] Use smaller dataset
- [ ] Disable slower models (CatBoost)
- [ ] Increase `SHAP_N_SAMPLES` threshold

### SHAP plots are slow
- [ ] Use `TreeExplainer` (automatic for tree models)
- [ ] Reduce number of samples for explanation
- [ ] Disable beeswarm plots for large n

---

## Important Notes

### 🔒 Privacy & Security
- **Patient names/dates**: Used only for internal matching, never stored
- **Model cache**: Contains no PHI or patient identifiers
- **Recommended**: Run on local network, not internet-facing
- **Data**: Recommend anonymization before upload

### ⚠️ Clinical Use
- **NOT FDA-approved** - Use for research only
- **Not a substitute** for clinical judgment
- **Validate on your data** before clinical use
- **Interpret with clinical context** - consider outliers

### 📊 Data Requirements
- **Minimum**: ~100 patients with outcomes for reliable models
- **Ideal**: 500-1000+ with good outcome event distribution
- **Missing data**: Handled via imputation, but >60% missing is problematic

---

## Frequently Asked Questions

**Q: Can I use data from multiple institutions?**
A: Yes! Combine Excel sheets first, then import. Consider testing institutional differences.

**Q: What if I have very few deaths?**
A: Models may have wider confidence intervals and reduced calibration. Consider collecting more data or using a simpler model (LogisticRegression). Class weighting is intentionally not applied, so that predicted probabilities reflect the true event rate.

**Q: Can I export the trained model?**
A: Yes, `ia_risk_bundle.joblib` is the pickled model bundle. Python only (use `joblib.load()`).

**Q: How do I update the model with new data?**
A: Delete `ia_risk_bundle.joblib` and re-run app.py with updated Excel file.

**Q: Can I use this for real-time predictions?**
A: Currently Streamlit-based. For production: export model, use FastAPI/Flask, containerize with Docker.

---

## Development

### Installation for Development
```bash
pip install -r requirements.txt
pip install pytest black flake8 mypy  # Dev tools
```

### Run Tests
```bash
pytest tests/  # All tests
pytest tests/test_modeling.py -v  # Specific test file, verbose
```

### Code Quality
```bash
black . --line-length=100     # Format
flake8 . --max-line-length=100  # Lint
mypy . --ignore-missing-imports  # Type checking
```

See [docs/CONTRIBUTING.md](./CONTRIBUTING.md) for detailed contribution guidelines.

---

## Performance Benchmarks

On typical dataset (n=500, 30 features):

| Operation | Time |
|-----------|------|
| Data preparation | 2-5 sec |
| Train 8 models (5-fold CV) | 30-60 sec |
| SHAP global importance | 10-20 sec |
| Single prediction + SHAP | 1-2 sec |
| Full app startup | 15-30 sec |

*Times vary by hardware, data complexity, and model selection*

---

## References

### Papers
- EuroSCORE II: https://www.eur.org/
- SHAP: Lundberg, S. M., & Lee, S. I. (2017). NIPS

### Further Reading
- Scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
- Streamlit: https://docs.streamlit.io/

---

## Support & Contributing

- **Issues**: Create GitHub issue
- **Questions**: Use Discussions
- **Contributions**: See [CONTRIBUTING.md](./CONTRIBUTING.md)
- **License**: [Specify your license here]

---

## Changelog

**Version 2.1.0** (2026-03-22)
- ✨ Added SHAP explainability module
- 🔧 Centralized configuration (config/ folder)
- 📚 Comprehensive documentation
- ✅ Type hints throughout codebase

**Version 2.0.0**
- Initial release with 8 ML models
- EuroSCORE II integration
- Statistical comparison framework

---

**Last Updated:** March 22, 2026  
**Maintainer:** [Your Name/Institution]
