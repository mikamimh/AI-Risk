"""Model explainability using SHAP (SHapley Additive exPlanations).

This module provides tools for understanding and interpreting model predictions
using SHAP values, including global feature importance and local explanations.
"""

from typing import Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def _import_shap():
    """Import shap at call time so the module works even if shap is installed
    after the initial import. Raises ImportError with a clear message if absent."""
    try:
        import shap  # noqa: PLC0415
        return shap
    except ImportError:
        raise ImportError("shap is not installed. Install with: pip install shap")


class ModelExplainer:
    """SHAP-based model explainer for feature importance and local predictions.

    Uses KernelExplainer with the full sklearn Pipeline so that preprocessing
    is applied correctly and SHAP values are expressed in the original feature
    space (before one-hot encoding / scaling).

    Computation is lazy and cached for performance.

    Example:
        >>> explainer = ModelExplainer(model, X_train)
        >>> importance = explainer.global_importance(X_test)
        >>> fig = explainer.local_explanation(X_test, idx=0)
    """

    def __init__(self, model, X_train: pd.DataFrame, max_samples: int = 100):
        """Initialize SHAP explainer.

        Args:
            model: Trained sklearn Pipeline or estimator with predict_proba
            X_train: Training data for SHAP baseline
            max_samples: Maximum background samples for KernelExplainer

        Raises:
            ImportError: If shap is not installed
            ValueError: If model doesn't support probability predictions
        """
        _import_shap()  # fail early with a clear message if shap is absent

        self.model = model
        self.X_train = X_train.copy()
        self.max_samples = max_samples
        self._explainer = None

        try:
            if hasattr(model, "predict_proba"):
                model.predict_proba(X_train.iloc[:1])
            else:
                raise ValueError("Model must have predict_proba method")
        except Exception as e:
            raise ValueError(f"Model cannot generate probabilities: {e}")

    @property
    def explainer(self):
        """Lazy-load SHAP explainer (expensive computation)."""
        if self._explainer is None:
            self._explainer = self._build_explainer()
        return self._explainer

    def _build_explainer(self):
        """Build SHAP KernelExplainer using the full pipeline.

        Background and inputs are kept as numpy arrays for shap consistency.
        The model function reconstructs a named DataFrame so the sklearn
        Pipeline receives the same format it was fitted with.
        """
        _import_shap()  # ensure shap is available
        feature_names = list(self.X_train.columns)
        sample_size = min(len(self.X_train), self.max_samples)
        # Use pandas sampling + to_numpy() to avoid shap.sample() returning a
        # DataFrame whose column names pollute the DenseData background object,
        # causing "Specifying the columns using strings is only supported for
        # dataframes" when shap_values() later receives a plain numpy array.
        rng = np.random.default_rng(42)
        idx = rng.choice(len(self.X_train), size=sample_size, replace=False)
        background = self.X_train.iloc[idx].to_numpy()  # pure numpy, no column metadata

        def _predict(x: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(x, columns=feature_names)
            return self.model.predict_proba(df)[:, 1]

        import shap as _shap_mod  # noqa: PLC0415
        return _shap_mod.KernelExplainer(_predict, background)

    def _compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values — passes numpy to avoid pandas/shap version mismatches."""
        shap_values = self.explainer.shap_values(X.values)
        if isinstance(shap_values, list):
            return shap_values[1]
        return shap_values

    def global_importance(
        self,
        X: pd.DataFrame,
        top_n: int = 15,
        method: str = "mean_abs_shap",
    ) -> pd.DataFrame:
        """Compute global feature importance across dataset.

        Args:
            X: Evaluation dataset
            top_n: Number of top features to return
            method: 'mean_abs_shap' or 'mean_shap'

        Returns:
            DataFrame with columns: feature, mean_abs_shap, std_shap, mean_shap
        """
        shap_values = self._compute_shap_values(X)
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
            "std_shap": np.std(shap_values, axis=0),
            "mean_shap": np.mean(shap_values, axis=0),
        })
        return importance_df.sort_values("mean_abs_shap", ascending=False).head(top_n)

    def plot_importance(
        self,
        X: pd.DataFrame,
        top_n: int = 15,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Plot global feature importance as horizontal bar chart."""
        importance = self.global_importance(X, top_n=top_n)
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(importance["feature"][::-1], importance["mean_abs_shap"][::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top {top_n} Features - Global Importance")
        plt.tight_layout()
        return fig

    def local_explanation(self, X: pd.DataFrame, idx: int) -> Tuple[plt.Figure, float]:
        """Generate force plot explaining a single prediction.

        Returns:
            Tuple of (matplotlib figure, prediction probability)
        """
        shap = _import_shap()
        sample = X.iloc[[idx]]
        shap_values = self._compute_shap_values(sample)

        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1]

        try:
            fig = shap.force_plot(
                expected_value,
                shap_values[0],
                sample.iloc[0],
                matplotlib=True,
            )
            plt.tight_layout()
        except Exception as e:
            warnings.warn(f"Could not generate force plot: {e}. Using waterfall instead.")
            fig = self._waterfall_explanation(sample, idx)

        prob = self.model.predict_proba(sample)[0, 1]
        return fig, prob

    def _waterfall_explanation(self, X: pd.DataFrame, idx: int) -> plt.Figure:
        """Fallback: generate waterfall plot for a single prediction."""
        shap = _import_shap()
        shap_values = self._compute_shap_values(X)

        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1]

        fig = shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=expected_value,
                data=X.iloc[0].values,
                feature_names=X.columns.tolist(),
            ),
            show=False,
        )
        return fig

    def plot_dependence(
        self,
        X: pd.DataFrame,
        feature_name: str,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Plot how predictions depend on a specific feature."""
        shap = _import_shap()
        if feature_name not in X.columns:
            raise ValueError(f"Feature '{feature_name}' not found in dataset")

        shap_values = self._compute_shap_values(X)
        feature_idx = list(X.columns).index(feature_name)

        fig = plt.figure(figsize=figsize)
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            show=False,
            feature_names=X.columns.tolist(),
        )
        plt.tight_layout()
        return fig

    def plot_beeswarm(
        self,
        X: pd.DataFrame,
        top_n: int = 12,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """Create beeswarm plot showing impact of top features."""
        shap = _import_shap()
        shap_values = self._compute_shap_values(X)

        fig = plt.figure(figsize=figsize)
        shap.summary_plot(
            shap_values,
            X,
            plot_type="beeswarm",
            show=False,
            max_display=top_n,
        )
        plt.tight_layout()
        return fig

    def top_features_for_sample(
        self,
        X: pd.DataFrame,
        idx: int,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Get top SHAP features driving prediction for a single sample.

        Returns:
            DataFrame with columns: feature, feature_value, shap_value, impact
        """
        shap_values = self._compute_shap_values(X.iloc[[idx]])

        feature_contributions = pd.DataFrame({
            "feature": X.columns,
            "feature_value": X.iloc[idx].values,
            "shap_value": shap_values[0],
        })
        feature_contributions["abs_shap"] = np.abs(feature_contributions["shap_value"])
        feature_contributions["impact"] = feature_contributions["shap_value"].apply(
            lambda x: "increases" if x > 0 else "decreases"
        )

        return feature_contributions.nlargest(top_n, "abs_shap")[
            ["feature", "feature_value", "shap_value", "impact"]
        ]


def show_explainability_ui(explainer: ModelExplainer, X_test: pd.DataFrame, y_test: np.ndarray):
    """Streamlit UI for model explainability (SHAP visualizations)."""
    if not HAS_STREAMLIT:
        raise ImportError("streamlit required for show_explainability_ui")

    st.subheader("Model Explainability (SHAP)")

    explanation_type = st.selectbox(
        "Select visualization",
        ["Global Importance", "Feature Dependence", "Beeswarm Plot", "Single Prediction"],
    )

    if explanation_type == "Global Importance":
        top_n = st.slider("Number of features", 5, min(20, len(X_test.columns)), 15)
        fig = explainer.plot_importance(X_test, top_n=top_n)
        st.pyplot(fig)
        importance_df = explainer.global_importance(X_test, top_n=top_n)
        st.dataframe(importance_df, width="stretch")

    elif explanation_type == "Feature Dependence":
        feature = st.selectbox("Select feature", X_test.columns)
        fig = explainer.plot_dependence(X_test, feature)
        st.pyplot(fig)

    elif explanation_type == "Beeswarm Plot":
        top_n = st.slider("Number of features", 5, min(20, len(X_test.columns)), 12)
        fig = explainer.plot_beeswarm(X_test, top_n=top_n)
        st.pyplot(fig)

    elif explanation_type == "Single Prediction":
        idx = st.number_input(
            "Select patient (row index)",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0,
        )
        fig, prob = explainer.local_explanation(X_test, idx)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.pyplot(fig)
        with col2:
            st.metric("Prediction Probability", f"{prob:.1%}")
            top_features = explainer.top_features_for_sample(X_test, idx, top_n=7)
            st.write("**Top Contributing Features:**")
            st.dataframe(top_features, width="stretch")
