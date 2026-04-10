"""Machine learning model hyperparameters configuration."""

from typing import Dict, Any
from config.base_config import AppConfig


# ==== Base Model Hyperparameters ====

MODEL_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "LogisticRegression": {
        "C": 0.1,
        "l1_ratio": 0,
        "max_iter": 5000,
        "random_state": AppConfig.RANDOM_SEED,
    },

    "RandomForest": {
        "n_estimators": 400,
        "random_state": AppConfig.RANDOM_SEED,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "n_jobs": AppConfig.N_JOBS,
    },

    "MLP": {
        "hidden_layer_sizes": (32, 16),
        "activation": "relu",
        "alpha": 1e-2,
        "learning_rate_init": 1e-3,
        "max_iter": 600,
        "random_state": AppConfig.RANDOM_SEED,
    },

    "XGBoost": {
        "n_estimators": 350,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "eval_metric": "logloss",
        "random_state": AppConfig.RANDOM_SEED,
    },

    "LightGBM": {
        "n_estimators": 350,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": AppConfig.RANDOM_SEED,
        "verbose": -1,
    },

    "CatBoost": {
        "iterations": 350,
        "depth": 4,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": AppConfig.RANDOM_SEED,
        "verbose": False,
    },
}

# ==== Stacking Ensemble Base Models ====

STACKING_BASE_MODELS = [
    ("lr", "LogisticRegression"),
    ("rf", "RandomForest"),
    ("xgb_or_lgbm", "XGBoost"),  # Falls back to LightGBM if XGBoost unavailable
]

STACKING_META_LEARNER = {
    "type": "LogisticRegression",
    "params": {
        "C": 0.1,
        "max_iter": 3000,
        "random_state": AppConfig.RANDOM_SEED,
    }
}

STACKING_PARAMS = {
    "estimators": STACKING_BASE_MODELS,
    "final_estimator_type": STACKING_META_LEARNER["type"],
    "final_estimator_params": STACKING_META_LEARNER["params"],
    "stack_method": "predict_proba",
    "passthrough": False,
    "cv": AppConfig.CV_SPLITS,
    "n_jobs": AppConfig.N_JOBS,
}

# ==== Preprocessing Pipeline ====

PREPROCESSING_PARAMS = {
    "numeric": {
        "imputer_strategy": "median",
        "scaler_type": "StandardScaler",
    },
    "categorical": {
        "imputer_strategy": "most_frequent",
        "encoder_type": "TargetEncoder",
        "smooth": "auto",
        "post_imputer_strategy": "median",
    }
}


def get_model_params(model_name: str) -> Dict[str, Any]:
    """Get hyperparameters for a specific model.
    
    Args:
        model_name: Name of the model (e.g., "XGBoost", "RandomForest")
    
    Returns:
        Dictionary of model hyperparameters
    
    Raises:
        ValueError: If model_name not found in MODEL_HYPERPARAMS
    
    Example:
        >>> params = get_model_params("RandomForest")
        >>> rf = RandomForestClassifier(**params)
    """
    if model_name not in MODEL_HYPERPARAMS:
        available = ", ".join(MODEL_HYPERPARAMS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    return MODEL_HYPERPARAMS[model_name].copy()


def list_available_models() -> list:
    """Return list of configured model names."""
    return list(MODEL_HYPERPARAMS.keys())
