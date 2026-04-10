# Contributing Guide

## Development Setup

### 1. Clone and Install
```bash
git clone <repository>
cd "IA Risk - Claude"

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-cov black flake8 mypy
```

### 2. Project Structure
```
├── app.py                    # Main Streamlit app
├── modeling.py               # ML training
├── risk_data.py              # Data loading/prep
├── euroscore.py              # EuroSCORE II
├── stats_compare.py          # Statistical tests
├── explainability.py         # SHAP integration
│
├── config/
│   ├── __init__.py
│   ├── base_config.py        # App configuration
│   └── model_config.py       # ML hyperparameters
│
├── docs/                     # Documentation
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── DATA_FORMAT.md
│   └── CONTRIBUTING.md
│
├── tests/                    # Unit tests
│
└── local_data/               # User datasets (gitignored)
```

---

## Code Style

### Format Code
```bash
# Format with black
black . --line-length=100

# Check with flake8
flake8 . --max-line-length=100

# Type checking
mypy . --ignore-missing-imports
```

### Naming Conventions
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private functions: `_leading_underscore`
- Private variables: `__double_underscore` (rarely used)

### Docstring Format
**All functions require docstrings:**

```python
def calculate_risk(df: pd.DataFrame, model) -> np.ndarray:
    """Calculate mortality risk for patient cohort.
    
    Uses the trained model to generate probability predictions.
    High risk: p > 0.15, Low risk: p < 0.05.
    
    Args:
        df: Patient data with features
        model: Trained sklearn model
        
    Returns:
        Array of probabilities [0-1]
    
    Raises:
        ValueError: If model is not fitted
        KeyError: If required columns missing from df
    
    Example:
        >>> risks = calculate_risk(test_df, trained_model)
        >>> high_risk = risks[risks > 0.15]
    """
    # implementation
```

---

## Adding Features

### Adding a New Model

1. **Update config** (`config/model_config.py`):
```python
MODEL_HYPERPARAMS["GradientBoosting"] = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "random_state": 42,
}
```

2. **Update imports** (top of `modeling.py`):
```python
from sklearn.ensemble import GradientBoostingClassifier
HAS_GB = True  # Add to availability check
```

3. **Update `_build_candidates()`** in `modeling.py`:
```python
if HAS_GB:
    candidates["GradientBoosting"] = GradientBoostingClassifier(
        **get_model_params("GradientBoosting")
    )
```

4. **Test the new model:**
```bash
pytest tests/test_modeling.py::test_candidate_models
```

### Adding a New Feature

1. **Add to column mapping** (`risk_data.py`):
```python
FLAT_ALIAS_TO_APP_COLUMNS["new_feature_name"] = "Column Header in Excel"
```

2. **Add preprocessing logic** (if needed) in `risk_data.py`:
```python
def _process_new_feature(value):
    """Custom processing for new feature."""
    # implementation
```

3. **Update documentation** (`docs/DATA_FORMAT.md`)

### Adding a Statistical Test

1. **Implement function** in `stats_compare.py`:
```python
def new_statistical_test(y: np.ndarray, p: np.ndarray) -> dict:
    """Description of the test.
    
    Args:
        y: Binary outcome
        p: Predictions
        
    Returns:
        Dictionary with test results
    """
    # implementation
    return {"statistic": value, "p_value": p_value}
```

2. **Call from** `app.py` analysis section

---

## Testing

### Run Tests
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_modeling.py::test_train_model

# With coverage
pytest --cov=. tests/

# Verbose output
pytest -v
```

### Test Structure
```
tests/
├── conftest.py              # Shared fixtures
├── test_risk_data.py        # Data loading tests
├── test_modeling.py         # Model training tests
├── test_euroscore.py        # EuroSCORE tests
├── test_stats_compare.py    # Statistical tests
└── test_explainability.py   # SHAP tests
```

### Example Test
```python
import pytest
import pandas as pd
from risk_data import prepare_master_dataset

def test_prepare_master_dataset():
    """Test data preparation with sample file."""
    df = prepare_master_dataset("tests/fixtures/sample_data.xlsx")
    
    assert len(df.data) > 0
    assert "morte_30d" in df.data.columns
    assert len(df.feature_columns) > 0

def test_missing_required_sheet():
    """Test error handling for missing sheets."""
    with pytest.raises(ValueError):
        prepare_master_dataset("tests/fixtures/incomplete.xlsx")
```

---

## Git Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/new-model
# or
git checkout -b fix/data-loading-bug
```

### 2. Make Changes
```bash
# Write code, tests
git add .
git commit -m "Add Random Forest model to candidates"
```

### 3. Format and Test
```bash
black .
flake8 .
mypy .
pytest
```

### 4. Push and Create PR
```bash
git push origin feature/new-model
# Create pull request on GitHub
```

### Commit Message Format
```
<type>: <short summary>

<detailed explanation if needed>

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat: add SHAP explainability module`
- `fix: handle missing EuroSCORE variables`
- `docs: update API reference`
- `test: add tests for bootstrapping function`

---

## Performance Optimization

### Profile Code
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
train_and_select_model(df, features)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Common Optimizations
- Use `@st.cache_data` for expensive operations in Streamlit
- Use `@functools.lru_cache` for pure functions
- Vectorize operations with NumPy instead of loops
- Use `n_jobs=-1` for parallel processing

---

## Documentation Standards

### Update These When Adding Features
- `docs/ARCHITECTURE.md` - System changes
- `docs/API.md` - New functions
- `docs/DATA_FORMAT.md` - Data changes
- Function docstrings - Always
- `README.md` - Major changes

### Generate Documentation
```bash
# If using Sphinx (optional setup)
cd docs
sphinx-build -b html . _build
```

---

## Debugging Tips

### Enable Debug Mode
```python
from config import AppConfig
AppConfig.DEBUG = True
```

### Print Debug Info
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Data shape: {df.shape}")
logger.warning("Feature has missing values")
```

### Use Streamlit Debugging
```python
import streamlit as st

st.write("Debug Info:")
st.write(df.head())
st.write(model.feature_names_in_)
```

---

## Releasing a New Version

1. Update version in `config/base_config.py`:
```python
MODEL_VERSION = "2026-04-01-v6-new-features"
```

2. Update `requirements.txt` with tested versions

3. Create GitHub release with changelog

4. Tag commit:
```bash
git tag v1.2.0
git push origin v1.2.0
```

---

## Support & Questions

- **Issues**: Create GitHub issue with:
  - Error message/traceback
  - Steps to reproduce
  - System info (Python version, OS)
  
- **Discussions**: Use GitHub Discussions for design questions

- **Documentation**: Check `docs/` folder and docstrings first
