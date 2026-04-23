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
    MODEL_VERSION = "2026-04-23-v14-statistical-robustness"
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
        "--",
        "nan",
        "none",
        "na",
        "n/a",
        "null",
        "not applicable",
        "unknown",
        "not informed",
        "não informado",
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

    # ==== STS Score Cache (Phase 2: transparent caching + revalidation) ====
    # Persistent on-disk cache so STS Score calculations are not a black box.
    # The cache key is a hash of the clinically relevant STS Score input
    # fields (exactly what would be sent to the STS Score website).  A
    # cached entry is only returned if its STS_SCORE_INTEGRATION_VERSION
    # matches and it is within the TTL window below.  Bumping
    # STS_SCORE_INTEGRATION_VERSION invalidates every prior entry.
    STS_SCORE_CACHE_DIR = APP_CACHE_DIR / "sts_score_cache"
    STS_SCORE_CACHE_TTL_DAYS = 14
    STS_SCORE_INTEGRATION_VERSION = "sts-score-v1-2026-04"

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
