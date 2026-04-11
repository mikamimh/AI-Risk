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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, TargetEncoder
from sklearn.calibration import CalibratedClassifierCV

from config import AppConfig, get_model_params
from risk_data import (
    parse_number as _rd_parse_number,
    is_missing as _rd_is_missing,
    NONE_IS_VALID_COLUMNS,
)
from stats_compare import calibration_intercept_slope

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

# NONE_IS_VALID_COLUMNS imported from risk_data (single source of truth)

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
    for c in out.columns:
        if out[c].dtype == object:
            s = out[c].astype(str).str.strip()
            mask = s.apply(lambda v, col=c: _rd_is_missing(v, column=col))
            s = s.where(~mask, other=np.nan)
            out[c] = s
    return out


def _maybe_numeric(s: pd.Series) -> pd.Series:
    """Try to convert a Series to numeric using :func:`risk_data.parse_number`.

    Handles comma-decimal, BR/EN thousands, and percentage suffixes.
    Only converts the column if >= 60 % of non-null values parse as
    numeric (threshold disabled for <= 5-row DataFrames, e.g. individual
    predictions).
    """
    n_non_null = s.notna().sum()
    if n_non_null == 0:
        return pd.to_numeric(s, errors="coerce")

    converted = s.map(_rd_parse_number)

    if len(s) <= 5:
        if converted.notna().any():
            return converted
    else:
        pct_numeric = converted.notna().sum() / max(n_non_null, 1)
        if pct_numeric >= 0.6:
            return converted
    if len(s) <= 5:
        return converted
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
    youden_thresholds: Dict[str, float] | None = None
    """Per-model optimal threshold (Youden's J) from OOF predictions."""
    best_youden_threshold: float | None = None
    """Youden threshold for the best model.  Stored for optional use as
    an alternative decision threshold (the default remains 8 %)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _optimal_threshold_youden(y: np.ndarray, proba: np.ndarray) -> float:
    """Find threshold maximizing Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y, proba)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx])


def _select_best_model(
    leaderboard: pd.DataFrame,
    oof_predictions: "Dict[str, np.ndarray] | None" = None,
    y: "np.ndarray | None" = None,
    usability_floor: float = 0.08,
    min_auc: float = 0.60,
    min_dynamic_range: float = 0.15,
) -> str:
    """Select the best model based on leaderboard ranking + clinical guardrails.

    The leaderboard is sorted by AUC (desc) then AUPRC (desc). By default
    the top-ranked model wins — no artificial preference for any family.

    Clinical-usability filter (two auditable guardrails)
    ----------------------------------------------------
    When ``oof_predictions`` and ``y`` are both provided, each model is
    checked against two explicit guardrails. A model is eligible for
    automatic selection only if it passes **both**:

    Guardrail A — Coverage at the clinical threshold
        ``(oof_cal < usability_floor).mean() > 0``
        At least one patient must be classifiable as low-risk at the
        default 8% cutoff. Catches upward-compressed distributions
        (e.g. a stacked meta-learner whose output is bounded above 8%)
        and Platt-calibrated models whose lower asymptote is above the
        clinical threshold.

    Guardrail B — Discrimination + probabilistic sanity
        Three robust checks that work for both Platt- and isotonic-
        calibrated outputs (the old logit-slope/intercept check is
        incompatible with isotonic, which produces exact zeros that
        blow up logit-space regression):

        B1. ``AUC >= min_auc`` (default 0.60).  Catches near-random
            models (e.g. an under-trained MLP whose AUC collapses to
            ~0.52).

        B2. ``Brier < prevalence * (1 - prevalence)``.  Brier skill
            score > 0.  A model whose Brier is no better than always
            predicting the cohort prevalence is probabilistically
            worthless.  Catches bimodal-collapse pathologies (pre-fix
            MLP had Brier 0.165 vs baseline 0.127).

        B3. ``p99 - p01 >= min_dynamic_range`` (default 0.15).  Catches
            compression pathologies where the output range is too
            narrow to separate patients (original StackingEnsemble had
            p01..p99 spanning only 0.102).

    Excluded models still appear in the leaderboard and remain
    force-selectable from the sidebar; only the automatic default changes.

    If every model fails the guardrails, the plain AUC/AUPRC top is
    returned as a fallback so training never crashes.
    """
    if oof_predictions is None or y is None:
        return str(leaderboard.iloc[0]["Modelo"])

    y_arr = np.asarray(y).astype(int)
    prevalence = float(y_arr.mean())
    baseline_brier = prevalence * (1.0 - prevalence)

    def _is_usable(name: str) -> bool:
        p = oof_predictions.get(name)
        if p is None:
            return True
        p = np.asarray(p, dtype=float)
        mask = ~np.isnan(p)
        p_valid = p[mask]
        if len(p_valid) == 0:
            return True
        y_valid = y_arr[mask] if len(y_arr) == len(p) else y_arr
        if len(np.unique(y_valid)) < 2:
            return True  # cannot compute discrimination; do not penalise

        # Guardrail A — Coverage at the clinical threshold
        if float((p_valid < usability_floor).mean()) <= 0.0:
            return False

        # Guardrail B1 — Discrimination sanity
        try:
            auc = float(roc_auc_score(y_valid, p_valid))
        except Exception:
            return True
        if auc < min_auc:
            return False

        # Guardrail B2 — Brier skill score > 0 (better than predicting prevalence)
        try:
            brier = float(brier_score_loss(y_valid, p_valid))
        except Exception:
            return True
        if brier >= baseline_brier:
            return False

        # Guardrail B3 — Dynamic range (no compression)
        span = float(np.percentile(p_valid, 99) - np.percentile(p_valid, 1))
        if span < min_dynamic_range:
            return False

        return True

    usable_mask = leaderboard["Modelo"].map(_is_usable)
    usable_lb = leaderboard[usable_mask]
    if len(usable_lb) == 0:
        return str(leaderboard.iloc[0]["Modelo"])
    return str(usable_lb.iloc[0]["Modelo"])


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
        # MLP retired as a candidate for this cohort.  Two independent
        # failure modes were observed on this 454-row / 68-event dataset:
        #   (1) no early stopping -> Adam overtrains into bimodal sigmoid
        #       saturation (slope 0.111, Brier 0.165, Youden 0.007).
        #   (2) early stopping (default or escalated with
        #       n_iter_no_change=30, validation_fraction=0.2) -> Adam
        #       stops before the network learns anything discriminative
        #       (AUC collapses to ~0.52, near-constant 0.36 plateau).
        # The dataset is below the data-volume threshold where MLPs can
        # compete with tree models on medical tabular data.  Retiring
        # keeps training time down and removes a candidate that cannot
        # pass the clinical-usability guardrails under any configuration
        # tried.  The MLP hyperparameter block in config/model_config.py
        # is left in place as documentation of what was tried.
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

    # Meta-learner: use independent hyperparameters, NOT
    # get_model_params("LogisticRegression").  The base LR is tuned for the
    # ~50-feature preprocessed space with C=0.1 (strong L2).  The meta-
    # learner instead sees only 3 inputs — the base models' positive-class
    # probabilities, each in [0, 1] — and inheriting C=0.1 crushes its
    # coefficients so heavily that sigmoid(intercept + coef.p) is bounded
    # to roughly [0.13, 0.38], making every prediction exceed the 8%
    # clinical threshold.  C=1.0 (sklearn default) gives the meta-learner
    # enough dynamic range to emit probabilities that span the decision
    # boundary.
    #
    # passthrough=True: the meta-learner also sees the preprocessed raw
    # features in addition to the 3 base probabilities.  Without
    # passthrough the 3-way consensus of (raw LR, raw RF, raw XGB)
    # produced a legitimate ensemble floor around 0.087 — above the 8%
    # clinical threshold — because no patient had all three base models
    # simultaneously predicting very low risk.  Passthrough gives the
    # meta-LR enough feature signal to dip below 8% for patients whose
    # preoperative feature pattern is genuinely low-risk, restoring
    # coverage below the clinical threshold.
    candidates["StackingEnsemble"] = StackingClassifier(
        estimators=stack_base,
        final_estimator=LogisticRegression(
            C=1.0,
            max_iter=3000,
            random_state=AppConfig.RANDOM_SEED,
        ),
        stack_method="predict_proba",
        passthrough=True,
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

    # ── Calibration configuration ─────────────────────────────────────
    # Default (XGBoost / LightGBM / CatBoost): isotonic regression with
    # inner CV = min(3, fold_pos, fold_neg), ensemble=False.  Rationale:
    # linear Platt has an intrinsic lower asymptote of sigmoid(intercept)
    # which on this ~15%-prevalence cohort lands around 0.09 — above the
    # 8 % clinical threshold.  That floor crushed the three boosting
    # models' raw distributions (which spanned nearly to zero) onto a
    # calibrated minimum of ~0.089-0.112, eliminating coverage below
    # 8 %.  Isotonic regression is non-parametric and monotonic without
    # a lower asymptote, so low raw scores map to low calibrated scores.
    # Combined with ensemble=False (one isotonic on concatenated inner-CV
    # raw scores instead of averaging K per-fold fits), the calibrator
    # uses enough data to remain stable for those models.
    #
    # RandomForest override: Platt (sigmoid), inner CV = min(5, fold_pos,
    # fold_neg), ensemble=False.  Empirical OOF evaluation on this
    # 454-row / 68-event cohort (run under identical outer splits to
    # the training loop) showed isotonic produces a pathological
    # logit-scale regression for RF:
    #
    #     strategy             Brier     int       slope     <8%
    #     -----------------    ------    ------    ------    -----
    #     raw (no calibration) 0.1158    +1.017    +1.558    25.1 %
    #     isotonic cv=3        0.1192    -1.340    +0.155    28.0 %   <-- was default
    #     sigmoid  cv=3        0.1161    -0.079    +0.995    26.7 %
    #     sigmoid  cv=5        0.1151    +0.016    +1.015    31.7 %   <-- chosen
    #
    # Isotonic collapsed 25 % of the probability mass onto plateaus at
    # exactly 0 and exactly 1, destroying logit-scale linearity (slope
    # 0.155) and inflating Brier relative to the raw model.  Platt is
    # a 2-parameter fit that is much more stable on a 68-event cohort,
    # and for RF specifically it *improves* rather than destroys
    # calibration.  The feared lower-asymptote collapse does NOT happen
    # for RF because the raw minimum (0.014) is far enough below 8 %
    # that Platt's intercept translates to a calibrated minimum of
    # 0.033 — still well under the clinical threshold, with 31.7 % of
    # patients below 8 % (more than isotonic produced).
    #
    # The RF override is intentionally narrow (sigmoid, cv=5).  A later
    # audit of LightGBM and CatBoost under isotonic cv=3 showed the same
    # kind of plateau pathology that motivated the RF fix — LightGBM
    # pinned 16.7 % of patients to exactly 0.000 and CatBoost pinned
    # 4.8 %, both at slope < 0.20.  Bumping the inner CV cap from 3 to 5
    # (same knob, unchanged method) cuts LightGBM's plateau to ~6.6 %,
    # lowers its Brier from 0.1182 to 0.1167, and gives CatBoost a
    # consistent small improvement (Brier 0.1225 -> 0.1209).  Sigmoid was
    # rejected for both because Platt's lower asymptote lands above the
    # 8 % clinical threshold (Guardrail A would auto-exclude them).
    # XGBoost stays at the default (isotonic cv=3): no safe alternative
    # was found in the audit.  LogisticRegression and StackingEnsemble
    # remain uncalibrated (they are not in _TREE_MODELS).
    _CALIB_METHOD_DEFAULT = "isotonic"
    _CALIB_CV_CAP_DEFAULT = 3
    _CALIB_OVERRIDES = {
        "RandomForest": {"method": "sigmoid", "cv_cap": 5},
        "LightGBM": {"method": "isotonic", "cv_cap": 5},
        "CatBoost": {"method": "isotonic", "cv_cap": 5},
    }

    def _calib_config_for(name: str) -> tuple[str, int]:
        cfg = _CALIB_OVERRIDES.get(name, {})
        return (
            cfg.get("method", _CALIB_METHOD_DEFAULT),
            int(cfg.get("cv_cap", _CALIB_CV_CAP_DEFAULT)),
        )

    rows = []
    oof_pred_raw: Dict[str, np.ndarray] = {}
    oof_pred_cal: Dict[str, np.ndarray] = {}
    fitted_pipes: Dict[str, object] = {}
    youden_thresholds: Dict[str, float] = {}

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
            _cal_method, _cal_cv_cap = _calib_config_for(name)
            proba_cal = np.full(len(y), np.nan)
            for train_idx, test_idx in cv.split(X, y, groups):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr = y[train_idx]
                fold_pipe = Pipeline(
                    steps=[("prep", build_preprocessor(X_tr)), ("model", estimator)]
                )
                fold_pos = int(y_tr.sum())
                fold_neg = int(len(y_tr) - fold_pos)
                # Inner CV cap is per-model (see _CALIB_OVERRIDES):
                # RandomForest uses 5, other tree models use 3.
                fold_cal_cv = min(_cal_cv_cap, fold_pos, fold_neg)
                if fold_cal_cv >= 2:
                    # ensemble=False: fit ONE calibrator on concatenated
                    # inner-CV raw scores instead of averaging per-fold
                    # calibrators.  The averaged mode (default
                    # ensemble=True) produced lower-asymptote artefacts
                    # for the boosting models (~0.09-0.11), pushing
                    # their calibrated minimum above the 8% clinical
                    # threshold despite raw distributions spanning
                    # nearly to zero.  With ensemble=False, the final
                    # calibrated score is calibrator(f(x)) where f is a
                    # single base model fit on all training data and
                    # the calibrator parameters are fit against
                    # out-of-fold raw scores — preserving dynamic range.
                    cal_fold = CalibratedClassifierCV(
                        fold_pipe,
                        method=_cal_method,
                        cv=fold_cal_cv,
                        ensemble=False,
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
                "Limiar_Youden": float(opt_thresh),
            }
        )
        oof_pred_raw[name] = proba_raw
        oof_pred_cal[name] = proba_cal
        youden_thresholds[name] = float(opt_thresh)

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
            _cal_method, _cal_cv_cap = _calib_config_for(name)
            cal_cv = min(_cal_cv_cap, int(pos), int(neg))  # match inner-fold strategy
            if cal_cv >= 2:
                # ensemble=False matches the OOF calibration strategy
                # above.  See the longer comment on _CALIB_OVERRIDES
                # for the per-model method and cv_cap rationale.
                cal = CalibratedClassifierCV(
                    pipe,
                    method=_cal_method,
                    cv=cal_cv,
                    ensemble=False,
                )
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
    # Clinical-usability filter: reject models that either (A) have no
    # coverage below the 8% clinical threshold, or (B) show a grossly
    # pathological calibration slope/intercept on OOF.  See docstring of
    # _select_best_model for the exact bounds and rationale.
    best_name = _select_best_model(
        leaderboard, oof_predictions=oof_pred_cal, y=y
    )
    best_model = fitted_pipes[best_name]

    # Calibration method reported on the bundle = what the BEST model
    # actually uses.  Per-model overrides (see _CALIB_OVERRIDES) mean
    # this string is no longer a single global value.
    _best_cal_method, _ = _calib_config_for(best_name) if best_name in _TREE_MODELS else ("none", 0)

    return TrainedArtifacts(
        model=best_model,
        leaderboard=leaderboard,
        oof_predictions=oof_pred_cal,          # calibrated OOF (primary)
        feature_columns=non_empty_cols,
        fitted_models=fitted_pipes,
        best_model_name=best_name,
        calibration_method=_best_cal_method,
        oof_raw=oof_pred_raw,                  # uncalibrated, for audit only
        youden_thresholds=youden_thresholds,
        best_youden_threshold=youden_thresholds.get(best_name),
    )
