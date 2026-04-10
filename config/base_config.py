"""Base configuration for AI Risk - Cardiac Surgery application."""

from pathlib import Path
from enum import Enum


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "dev"
    PRODUCTION = "prod"


class AppConfig:
    """Central configuration for AI Risk application.
    
    All hardcoded values are centralized here for easy modification.
    Override individual settings by creating a subclass.
    """
    
    # ==== Paths ====
    MODEL_CACHE_FILE = Path("ia_risk_bundle.joblib")
    APP_CACHE_DIR = Path(".ia_risk_cache")
    TEMP_DATA_DIR = APP_CACHE_DIR / "temp_data"
    LOCAL_DATA_DIR = Path("local_data")
    UPLOAD_CACHE_FILE = TEMP_DATA_DIR / "uploaded_source.xlsx"
    GSHEETS_CACHE_FILE = TEMP_DATA_DIR / "google_sheets_tables.xlsx"
    
    # ==== Model & Training ====
    MODEL_VERSION = "2026-03-29-v12-calibrated-oof"
    RANDOM_SEED = 42
    N_JOBS = -1
    
    # ==== Cross-Validation ====
    CV_STRATEGY = "StratifiedGroupKFold"  # Options: StratifiedKFold, StratifiedGroupKFold
    CV_SPLITS = 5
    GROUP_KEY_COLUMN = "_patient_key"
    TARGET_COLUMN = "morte_30d"
    
    # ==== Streamlit UI ====
    PAGE_TITLE = "AI Risk \u2014 Cardiac Surgery Risk Stratification"
    LAYOUT = "wide"
    THEME = "light"
    LANGUAGE_DEFAULT = "English"
    LANGUAGES = ["English", "Português"]
    
    # ==== Data Validation ====
    MIN_SAMPLE_SIZE = 30
    MIN_CLASSES = 2
    MIN_FEATURE_COMPLETION = 0.6  # 60% non-null for numeric conversion
    
    MISSING_TOKENS = {
        "",
        "-",
        "nan",
        "none",
        "not applicable",
        "unknown",
        "not informed",
    }
    
    # ==== Required Data Structure ====
    REQUIRED_SOURCE_TABLES = [
        "Preoperative",
        "Pre-Echocardiogram",
        "Postoperative",
    ]
    
    OPTIONAL_SOURCE_TABLES = [
        "EuroSCORE II",
        "EuroSCORE II Automático",
        "STS Score",
    ]
    
    # ==== Performance & Metrics ====
    N_BOOTSTRAP_SAMPLES = 2000
    BOOTSTRAP_SEED = 42
    CONFIDENCE_LEVEL = 0.95
    SIGNIFICANCE_LEVEL = 0.05
    
    # ==== Risk Classification ====
    RISK_THRESHOLDS = {
        "low": 0.05,
        "intermediate": 0.15,
        "high": 1.0,
    }
    
    # ==== Feature Importance ====
    SHAP_N_SAMPLES = 50  # Reduced for performance
    SHAP_TOP_N_FEATURES = 15
    
    # ==== Environment ====
    ENV = Environment.DEVELOPMENT
    DEBUG = False
    
    @classmethod
    def get_all(cls) -> dict:
        """Return all configuration as dictionary."""
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration for common issues."""
        if cls.CV_SPLITS < 2:
            raise ValueError("CV_SPLITS must be at least 2")
        if cls.MIN_SAMPLE_SIZE < 10:
            raise ValueError("MIN_SAMPLE_SIZE should be at least 10")
        if cls.MODEL_VERSION == "":
            raise ValueError("MODEL_VERSION cannot be empty")
