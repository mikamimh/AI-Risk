from dataclasses import dataclass
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, TargetEncoder
from sklearn.calibration import CalibratedClassifierCV

from config import AppConfig, get_model_params

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


# ---------------------------------------------------------------------------
# Probability clipping bounds
# ---------------------------------------------------------------------------

_PROB_EPS = 1e-6    # Numerical-stability epsilon only (log, logit operations).
                    # Consistent with stats_compare.py (calibration_intercept_slope,
                    # hosmer_lemeshow_test).  The calibrated probability IS the final
                    # clinical output — no aggressive clipping.

# Columns where "None" is a valid clinical value (means "no disease"), not missing data
_NONE_IS_VALID_COLUMNS = {
    "Aortic Stenosis", "Aortic Regurgitation",
    "Mitral Stenosis", "Mitral Regurgitation",
    "Tricuspid Regurgitation",
}

# Valve severity columns — ordinal encoding (clinically ordered)
_VALVE_SEVERITY_COLS = [
    "Aortic Stenosis", "Aortic Regurgitation",
    "Mitral Stenosis", "Mitral Regurgitation",
    "Tricuspid Regurgitation",
]
_VALVE_SEVERITY_ORDER = ["None", "Trivial", "Mild", "Moderate", "Severe"]


# ---------------------------------------------------------------------------
# Clipped Pipeline wrapper
# ---------------------------------------------------------------------------

class ClippedPipeline:
    """Wraps a fitted sklearn Pipeline (or CalibratedClassifierCV).

    Applies only a minimal numerical-stability epsilon (1e-6) to
    predict_proba — the calibrated probability IS the clinical output.
    Transparent wrapper: attribute access delegates to the inner pipeline,
    so SHAP / permutation importance / named_steps all work normally.

    When wrapping a CalibratedClassifierCV, ``named_steps`` is resolved from
    the underlying fitted estimator clone.
    """

    def __init__(self, pipeline):
        self._pipeline = pipeline

    @property
    def _inner_pipe(self) -> Pipeline:
        """Return a fitted sklearn Pipeline, even if wrapped by CalibratedClassifierCV.

        CalibratedClassifierCV stores fitted clones in ``calibrated_classifiers_``.
        We return the first one's estimator so that ``named_steps["model"]``
        gives access to fitted attributes (feature_importances_, coef_, etc.).
        """
        obj = self._pipeline
        if hasattr(obj, "calibrated_classifiers_") and obj.calibrated_classifiers_:
            inner = obj.calibrated_classifiers_[0].estimator
            if isinstance(inner, Pipeline):
                return inner
        if hasattr(obj, "estimator") and isinstance(obj.estimator, Pipeline):
            return obj.estimator
        return obj

    def _coerce_dtypes(self, X):
        """Force numeric columns to numeric dtype before the pipeline sees them."""
        if not isinstance(X, pd.DataFrame):
            return X
        inner = self._inner_pipe
        prep = inner.named_steps.get("prep") if hasattr(inner, "named_steps") else None
        if prep is None:
            return X
        X = X.copy()
        for tname, _, cols in prep.transformers_:
            if tname == "num":
                for c in cols:
                    if c in X.columns and X[c].dtype == object:
                        X[c] = pd.to_numeric(
                            X[c].astype(str).str.replace(',', '.', regex=False),
                            errors="coerce",
                        )
        return X

    @property
    def named_steps(self):
        """Expose named_steps from the inner Pipeline."""
        return self._inner_pipe.named_steps

    def predict_proba(self, X):
        raw = self._pipeline.predict_proba(self._coerce_dtypes(X))
        # Minimal epsilon clip for numerical stability only (log/logit).
        # The calibrated probability is the true clinical output.
        raw[:, 1] = np.clip(raw[:, 1], _PROB_EPS, 1 - _PROB_EPS)
        raw[:, 0] = 1 - raw[:, 1]
        return raw

    def predict(self, X):
        return self._pipeline.predict(self._coerce_dtypes(X))

    def fit(self, X, y):
        self._pipeline.fit(X, y)
        return self

    def __getattr__(self, name):
        # Try inner pipeline first (for named_steps, etc.), then outer
        inner = object.__getattribute__(self, "_inner_pipe")
        if hasattr(inner, name):
            return getattr(inner, name)
        return getattr(object.__getattribute__(self, "_pipeline"), name)

    def __getstate__(self):
        return {"_pipeline": self._pipeline}

    def __setstate__(self, state):
        self._pipeline = state["_pipeline"]


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def _clean_object_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _missing_common = {"", "-", "nan", "Unknown", "Not applicable"}
    for c in out.columns:
        if out[c].dtype == object:
            s = out[c].astype(str).str.strip()
            tokens = _missing_common if c in _NONE_IS_VALID_COLUMNS else _missing_common | {"None"}
            s = s.where(~s.isin(tokens), other=np.nan)
            out[c] = s
    return out


def _maybe_numeric(s: pd.Series) -> pd.Series:
    """Try to convert a Series to numeric.

    Handles common formatting issues in clinical data:
    - Comma as decimal separator: "1,08" → 1.08
    - Comma as thousands separator: "191,000" → 191000
    - Mixed formats in the same column

    For single-row DataFrames (individual prediction / batch row-by-row),
    the 30% threshold is effectively disabled — any parseable value converts.

    The threshold is computed among non-null values only, so columns that
    are mostly NaN (e.g. after "-" → NaN cleanup) still convert correctly
    when the non-null portion is numeric.
    """
    # First try direct conversion
    x = pd.to_numeric(s, errors="coerce")
    n_non_null = s.notna().sum()
    if n_non_null == 0:
        return x

    # Try fixing comma formatting: "191,000" → "191000", "1,08" → "1.08"
    # Always attempt this before the threshold check, because a column like
    # LVEF may have 86% values parseable directly but 14% with comma decimals
    # — returning early would silently lose those.
    import re

    def _fix_comma(v):
        if pd.isna(v) or str(v).strip().lower() in {"nan", "none", "", "-"}:
            return v
        txt = str(v).strip()
        return txt.replace(",", ".")
    fixed = s.map(_fix_comma)
    x2 = pd.to_numeric(fixed, errors="coerce")

    # Pick the version that recovers more values
    best = x2 if x2.notna().sum() >= x.notna().sum() else x

    if len(s) <= 5:
        if best.notna().any():
            return best
    else:
        pct_numeric = best.notna().sum() / max(n_non_null, 1)
        if pct_numeric >= 0.7:
            return best
    # For single rows: return coerced version
    if len(s) <= 5:
        return best
    return s


def clean_features(
    df: pd.DataFrame,
    numeric_columns: set | None = None,
) -> pd.DataFrame:
    """Clean and standardize feature data.

    Parameters
    ----------
    numeric_columns : set, optional
        Explicit set of column names that should be treated as numeric.
        When provided, only these columns are converted via _maybe_numeric;
        all other object columns are kept as categorical strings.
        This prevents single-row predictions from losing categorical values
        like "Emergent Salvage" → NaN.
    """
    out = _clean_object_missing(df)
    for c in out.columns:
        if out[c].dtype == object:
            if numeric_columns is not None:
                if c in numeric_columns:
                    out[c] = _maybe_numeric(out[c])
            else:
                out[c] = _maybe_numeric(out[c])

    # Clinically impossible zeros → NaN (e.g. BSA=0 when height/weight missing)
    _ZERO_IS_MISSING = {"BSA, m2"}
    for c in _ZERO_IS_MISSING:
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c]):
            out.loc[out[c] == 0, c] = np.nan

    return out


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline.

    Numeric: median imputation + StandardScaler.
    Valve severity (ordinal): OrdinalEncoder with clinically meaningful order
        (None=0, Trivial=1, Mild=2, Moderate=3, Severe=4) + median imputation.
        This avoids TargetEncoder's small-sample bias on these columns.
    Categorical: mode imputation + TargetEncoder (encodes each category as the
        smoothed mean of the target variable, producing 1 numeric feature per
        categorical column instead of N one-hot columns).
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    valve_cols = [c for c in _VALVE_SEVERITY_COLS if c in X.columns and c not in numeric_cols]
    categorical_cols = [c for c in X.columns if c not in numeric_cols and c not in valve_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    valve_pipe = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="constant", fill_value="None")),
            ("ordinal_enc", OrdinalEncoder(
                categories=[_VALVE_SEVERITY_ORDER] * len(valve_cols),
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
            )),
            ("post_imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target_enc", TargetEncoder(
                smooth="auto",
                random_state=AppConfig.RANDOM_SEED,
            )),
            ("post_imputer", SimpleImputer(strategy="median")),
        ]
    )

    transformers = [("num", num_pipe, numeric_cols)]
    if valve_cols:
        transformers.append(("valve", valve_pipe, valve_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

@dataclass
class TrainedArtifacts:
    model: object
    leaderboard: pd.DataFrame
    oof_predictions: Dict[str, np.ndarray]
    """Calibrated OOF probabilities (Platt-scaled for tree models, raw for
    others).  Used by the leaderboard, triple comparison, and all downstream
    metrics.  Legacy name kept for serialisation compatibility."""
    feature_columns: List[str]
    fitted_models: Dict[str, object]
    best_model_name: str
    calibration_method: str = "sigmoid"
    oof_raw: Dict[str, np.ndarray] | None = None
    """Uncalibrated OOF probabilities.  Stored for optional auditing /
    raw-vs-calibrated comparison; never used for ranking or reporting."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _optimal_threshold_youden(y: np.ndarray, proba: np.ndarray) -> float:
    """Find threshold maximizing Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y, proba)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx])


def _select_best_model(leaderboard: pd.DataFrame) -> str:
    """Select the best model based on leaderboard ranking.

    The leaderboard is sorted by AUC (desc) then AUPRC (desc).
    The top-ranked model wins — no artificial preference for any family.
    """
    return str(leaderboard.iloc[0]["Modelo"])


def _build_candidates() -> Dict[str, object]:
    """Build candidate models WITHOUT class weighting.

    Models learn from natural class proportions so that predict_proba
    returns probabilities that reflect the true event rate.  This avoids
    the inflated probabilities caused by class_weight='balanced' or
    scale_pos_weight >> 1.
    """
    candidates: Dict[str, object] = {
        "LogisticRegression": LogisticRegression(**get_model_params("LogisticRegression")),
        "RandomForest": RandomForestClassifier(**get_model_params("RandomForest")),
        "MLP": MLPClassifier(**get_model_params("MLP")),
    }

    if HAS_XGB:
        candidates["XGBoost"] = XGBClassifier(**get_model_params("XGBoost"))

    if HAS_LGBM:
        candidates["LightGBM"] = LGBMClassifier(**get_model_params("LightGBM"))

    if HAS_CATBOOST:
        candidates["CatBoost"] = CatBoostClassifier(**get_model_params("CatBoost"))

    # Stacking ensemble
    stack_base = [
        ("lr", LogisticRegression(**get_model_params("LogisticRegression"))),
        ("rf", RandomForestClassifier(**get_model_params("RandomForest"))),
    ]

    if HAS_XGB:
        stack_base.append(("xgb", XGBClassifier(**get_model_params("XGBoost"))))
    elif HAS_LGBM:
        stack_base.append(("lgbm", LGBMClassifier(**get_model_params("LightGBM"))))

    candidates["StackingEnsemble"] = StackingClassifier(
        estimators=stack_base,
        final_estimator=LogisticRegression(**get_model_params("LogisticRegression")),
        stack_method="predict_proba",
        passthrough=False,
        cv=AppConfig.CV_SPLITS,
        n_jobs=AppConfig.N_JOBS,
    )

    return candidates


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_and_select_model(
    df: pd.DataFrame,
    feature_columns: List[str],
    y_col: str = "morte_30d",
    group_col: str = "_patient_key",
    progress_callback=None,
) -> TrainedArtifacts:
    """Train candidate models and select the best performer.

    Strategy:
        1. Cross-validate each model to get out-of-fold (OOF) predictions.
           For tree-based models, calibration (Platt scaling) is applied
           *inside* each fold so that calibrated OOF predictions are honest.
        2. Compute leaderboard metrics on calibrated OOF probabilities.
        3. Fit final model on ALL data with Platt calibration for trees.
        4. Select best model by AUC (desc), AUPRC (desc) tiebreaker.

    The leaderboard and the final model use the same calibration strategy,
    so comparative metrics reflect what the deployed model actually outputs.
    """
    X = clean_features(df[feature_columns])
    non_empty_cols = [c for c in X.columns if not X[c].isna().all()]
    X = X[non_empty_cols].copy()
    y = df[y_col].astype(int).values
    groups = df[group_col].values if group_col in df.columns else None

    if len(X) == 0:
        raise ValueError("No eligible rows were found after applying inclusion and matching rules.")
    if len(np.unique(y)) < 2:
        raise ValueError("The outcome has fewer than 2 classes after preprocessing.")

    preprocessor = build_preprocessor(X)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    candidates = _build_candidates()

    # Cross-validation setup
    if groups is not None:
        unique_groups = pd.Series(groups).nunique()
        max_splits = min(AppConfig.CV_SPLITS, int(unique_groups), int(pos), int(neg))
        if max_splits < 2:
            raise ValueError("Not enough grouped samples to run cross-validation.")
        cv = StratifiedGroupKFold(n_splits=max_splits, shuffle=True, random_state=AppConfig.RANDOM_SEED)
    else:
        max_splits = min(AppConfig.CV_SPLITS, int(pos), int(neg), len(X))
        if max_splits < 2:
            raise ValueError("Not enough samples to run cross-validation.")
        cv = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=AppConfig.RANDOM_SEED)

    _TREE_MODELS = {"RandomForest", "XGBoost", "LightGBM", "CatBoost"}
    _CALIB_METHOD = "sigmoid"  # Platt scaling; change to "isotonic" if n_events >> 50

    rows = []
    oof_pred_raw: Dict[str, np.ndarray] = {}
    oof_pred_cal: Dict[str, np.ndarray] = {}
    fitted_pipes: Dict[str, object] = {}

    _n_candidates = len(candidates)
    for _idx, (name, estimator) in enumerate(candidates.items()):
        if progress_callback:
            progress_callback(
                phase="cross_validation",
                current=_idx,
                total=_n_candidates,
                model_name=name,
            )

        pipe = Pipeline(steps=[("prep", preprocessor), ("model", estimator)])

        # ------------------------------------------------------------------
        # Out-of-fold predictions (raw + calibrated)
        # ------------------------------------------------------------------
        # Raw OOF via sklearn shortcut
        proba_raw = cross_val_predict(
            pipe, X, y, cv=cv, method="predict_proba", groups=groups
        )[:, 1]

        # Calibrated OOF: for tree-based models, fit calibrator inside each
        # outer fold so that evaluation is honest (no data seen during
        # calibration leaks into evaluation).
        #
        # Limitation: CalibratedClassifierCV uses StratifiedKFold internally
        # and does not accept a ``groups`` parameter, so patient grouping is
        # not enforced in the *inner* calibration splits.  The risk is minor
        # because (a) calibration only fits 2 parameters (sigmoid), (b) the
        # inner CV is applied within the outer-fold training set, and (c) few
        # patients have multiple surgeries.  This is documented as a known
        # limitation for methodological transparency.
        if name in _TREE_MODELS:
            proba_cal = np.full(len(y), np.nan)
            for train_idx, test_idx in cv.split(X, y, groups):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr = y[train_idx]
                fold_pipe = Pipeline(
                    steps=[("prep", build_preprocessor(X_tr)), ("model", estimator)]
                )
                fold_pos = int(y_tr.sum())
                fold_neg = int(len(y_tr) - fold_pos)
                # Keep inner CV small (≤3) to limit grouping-leak surface
                fold_cal_cv = min(3, fold_pos, fold_neg)
                if fold_cal_cv >= 2:
                    cal_fold = CalibratedClassifierCV(
                        fold_pipe, method=_CALIB_METHOD, cv=fold_cal_cv
                    )
                    cal_fold.fit(X_tr, y_tr)
                    proba_cal[test_idx] = cal_fold.predict_proba(X_te)[:, 1]
                else:
                    fold_pipe.fit(X_tr, y_tr)
                    proba_cal[test_idx] = fold_pipe.predict_proba(X_te)[:, 1]
        else:
            proba_cal = proba_raw.copy()

        # Use calibrated OOF for leaderboard metrics (honest evaluation)
        proba = proba_cal

        # Metrics at optimal threshold (Youden's J)
        auc = roc_auc_score(y, proba)
        auprc = average_precision_score(y, proba)
        brier = brier_score_loss(y, proba)

        opt_thresh = _optimal_threshold_youden(y, proba)
        pred = (proba >= opt_thresh).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        sens = float(tp / (tp + fn)) if (tp + fn) else np.nan
        spec = float(tn / (tn + fp)) if (tn + fp) else np.nan

        rows.append(
            {
                "Modelo": name,
                "AUC": auc,
                "AUPRC": auprc,
                "Brier": brier,
                "Sensibilidade": sens,
                "Especificidade": spec,
            }
        )
        oof_pred_raw[name] = proba_raw
        oof_pred_cal[name] = proba_cal

        # ------------------------------------------------------------------
        # Fit final model on ALL data
        # ------------------------------------------------------------------
        if progress_callback:
            progress_callback(
                phase="final_fit",
                current=_idx,
                total=_n_candidates,
                model_name=name,
            )
        pipe.fit(X, y)
        if name in _TREE_MODELS:
            cal_cv = min(3, int(pos), int(neg))  # match inner-fold strategy
            if cal_cv >= 2:
                cal = CalibratedClassifierCV(pipe, method=_CALIB_METHOD, cv=cal_cv)
                cal.fit(X, y)
                fitted_pipes[name] = ClippedPipeline(cal)
            else:
                fitted_pipes[name] = ClippedPipeline(pipe)
        else:
            fitted_pipes[name] = ClippedPipeline(pipe)

    if progress_callback:
        progress_callback(phase="selecting_best", current=_n_candidates, total=_n_candidates, model_name="")

    leaderboard = (
        pd.DataFrame(rows)
        .sort_values(["AUC", "AUPRC"], ascending=[False, False])
        .reset_index(drop=True)
    )
    best_name = _select_best_model(leaderboard)
    best_model = fitted_pipes[best_name]

    return TrainedArtifacts(
        model=best_model,
        leaderboard=leaderboard,
        oof_predictions=oof_pred_cal,          # calibrated OOF (primary)
        feature_columns=non_empty_cols,
        fitted_models=fitted_pipes,
        best_model_name=best_name,
        calibration_method=_CALIB_METHOD,
        oof_raw=oof_pred_raw,                  # uncalibrated, for audit only
    )
