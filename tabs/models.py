"""Models tab — extracted from app.py (tab index 5).

Pure extraction: all logic, text, i18n, and UI elements are identical to the
original inline code.  The only structural change is that shared state is
accessed through ``ctx`` (:class:`tabs.TabContext`) instead of bare local
variables in ``app.py``.

The three ``@st.cache_data`` SHAP functions are moved here as module-level
cached functions (they reference only standard imports and ``_feat_display_name``).
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance

from euroscore import COEF as EURO_COEF, EURO_CONST
from modeling import clean_features
from ai_risk_inference import _safe_select_features
from risk_data import MISSINGNESS_INDICATOR_COLUMNS

if TYPE_CHECKING:
    from tabs import TabContext


# ---------------------------------------------------------------------------
# Feature display-name helpers (mirrors app.py)
# ---------------------------------------------------------------------------

_FEAT_PREFIXES = (
    "cat__onehot__",
    "cat__target_enc__",
    "cat__ordinal__",
    "cat__",
    "num__",
    "valve__",
    "ord__",
)


def _feat_display_name(name: str) -> str:
    s = str(name)
    for prefix in _FEAT_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def _resolve_base_feature(encoded_feature: str, feature_columns: list) -> str:
    if encoded_feature in feature_columns:
        return encoded_feature
    for feat in sorted(feature_columns, key=len, reverse=True):
        if encoded_feature.startswith(feat + "_"):
            return feat
    return encoded_feature


def _feature_group(base_feature: str, tr) -> str:
    clinical = {
        "Age (years)", "Sex", "Preoperative NYHA", "CCS4", "Diabetes", "PVD", "Previous surgery",
        "Dialysis", "IE", "HF", "Hypertension", "Dyslipidemia", "CVA", "Cancer ≤ 5 yrs",
        "Arrhythmia Remote", "Arrhythmia Recent", "Family Hx of CAD", "Smoking (Pack-year)",
        "Alcohol", "Pneumonia", "Chronic Lung Disease", "Poor mobility",
        "Critical preoperative state", "Coronary Symptom", "Left Main Stenosis ≥ 50%",
        "Proximal LAD Stenosis ≥ 70%", "No. of Diseased Vessels",
    }
    lab = {
        "Weight (kg)", "Height (cm)", "Cr clearance, ml/min *", "Creatinine (mg/dL)", "Hematocrit (%)",
        "WBC Count (10³/μL)", "Platelet Count (cells/μL)", "INR", "PTT", "KDIGO †",
        *MISSINGNESS_INDICATOR_COLUMNS,
    }
    echo = {
        "Pré-LVEF, %", "PSAP", "TAPSE", "Aortic Stenosis", "Aortic Regurgitation",
        "Mitral Stenosis", "Mitral Regurgitation", "Tricuspid Regurgitation", "Aortic Root Abscess",
        "AVA (cm²)", "MVA (cm²)", "Aortic Mean gradient (mmHg)", "Mitral Mean gradient (mmHg)",
        "PHT Aortic", "PHT Mitral", "Vena contracta", "Vena contracta (mm)",
    }
    procedure = {
        "Surgery", "Surgical Priority", "cirurgia_combinada", "peso_procedimento", "thoracic_aorta_flag",
        "Anticoagulation/ Antiaggregation", "Suspension of Anticoagulation (day)", "Preoperative Medications",
    }
    if base_feature in clinical:
        return tr("Clinical", "Clínico")
    if base_feature in lab:
        return tr("Laboratory", "Laboratorial")
    if base_feature in echo:
        return tr("Echocardiographic", "Ecocardiográfico")
    if base_feature in procedure:
        return tr("Procedure", "Procedimento")
    return tr("Other", "Outro")


# ---------------------------------------------------------------------------
# SHAP cached functions (moved from app.py)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_shap_global(
    xlsx_path: str,
    model_name: str,
    top_n: int,
    _pipe,
    _X: pd.DataFrame,
) -> pd.DataFrame:
    """Compute global SHAP importance using TreeExplainer on preprocessed features."""
    try:
        import shap as _shap
    except ImportError:
        return pd.DataFrame()

    estimator = _pipe.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame()

    prep = _pipe.named_steps["prep"]
    X_proc = prep.transform(_X)
    feat_names = [_feat_display_name(n) for n in prep.get_feature_names_out()]

    explainer = _shap.TreeExplainer(estimator)
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", category=UserWarning, module="shap")
        shap_values = explainer.shap_values(X_proc)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_dir = shap_values.mean(axis=0)
    df_imp = pd.DataFrame({
        "Feature": feat_names,
        "Mean |SHAP|": mean_abs,
        "Mean SHAP": mean_dir,
    }).sort_values("Mean |SHAP|", ascending=False).head(top_n).reset_index(drop=True)
    return df_imp


@st.cache_data(show_spinner=False)
def _cached_shap_beeswarm(
    xlsx_path: str,
    model_name: str,
    top_n: int,
    _pipe,
    _X: pd.DataFrame,
):
    """Generate SHAP beeswarm plot (summary_plot) for tree-based models."""
    try:
        import shap as _shap
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    estimator = _pipe.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return None

    prep = _pipe.named_steps["prep"]
    X_proc = prep.transform(_X)
    feat_names = [_feat_display_name(n) for n in prep.get_feature_names_out()]

    explainer = _shap.TreeExplainer(estimator)
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", category=UserWarning, module="shap")
        shap_values = explainer.shap_values(X_proc)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    fig = plt.figure(figsize=(10, 8))
    _shap.summary_plot(
        shap_values,
        X_proc,
        feature_names=feat_names,
        plot_type="dot",
        show=False,
        max_display=top_n,
    )
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def _cached_shap_dependence(
    xlsx_path: str,
    model_name: str,
    feature_name: str,
    _pipe,
    _X: pd.DataFrame,
):
    """Generate SHAP dependence plot for a specific feature."""
    try:
        import shap as _shap
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    estimator = _pipe.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return None

    prep = _pipe.named_steps["prep"]
    X_proc = prep.transform(_X)
    feat_names = [_feat_display_name(n) for n in prep.get_feature_names_out()]

    explainer = _shap.TreeExplainer(estimator)
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", category=UserWarning, module="shap")
        shap_values = explainer.shap_values(X_proc)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    if feature_name not in feat_names:
        return None
    feature_idx = feat_names.index(feature_name)

    plt.close("all")
    plt.figure(figsize=(10, 6))
    _shap.dependence_plot(
        feature_idx,
        shap_values,
        X_proc,
        feature_names=feat_names,
        show=False,
    )
    plt.tight_layout()
    return plt.gcf()


# ---------------------------------------------------------------------------
# Model helper functions (moved from app.py, used only in models tab)
# ---------------------------------------------------------------------------

def model_weight_table(artifacts, prepared, model_name: str, tr, top_n: int = 20) -> tuple:
    pipe = artifacts.fitted_models[model_name]
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    if model.__class__.__name__ == "StackingClassifier" and hasattr(model, "final_estimator_"):
        final = model.final_estimator_
        if hasattr(final, "coef_") and hasattr(model, "estimators"):
            names = [n for n, _ in model.estimators]
            vals = final.coef_.ravel()
            w = pd.DataFrame({"Model": names[: len(vals)], "Weight": vals[: len(names)]})
            w["Absolute impact"] = w["Weight"].abs()
            return w.sort_values("Absolute impact", ascending=False).head(top_n), "stacking"
        return pd.DataFrame(), "opaque"

    feature_names = [_feat_display_name(n) for n in prep.get_feature_names_out()]

    if hasattr(model, "coef_"):
        vals = np.asarray(model.coef_).ravel()
        n = min(len(vals), len(feature_names))
        w = pd.DataFrame({"Variable": feature_names[:n], "Coefficient": vals[:n]})
        w["Absolute impact"] = w["Coefficient"].abs()
        return w.sort_values("Absolute impact", ascending=False).head(top_n), "coefficient"

    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_).ravel()
        n = min(len(vals), len(feature_names))
        w = pd.DataFrame({"Variable": feature_names[:n], "Importance": vals[:n]})
        w["Absolute impact"] = w["Importance"].abs()
        return w.sort_values("Absolute impact", ascending=False).head(top_n), "importance"

    return pd.DataFrame(), "opaque"


@st.cache_data(show_spinner=False)
def _cached_permutation_importance_table(
    xlsx_path: str,
    model_name: str,
    top_n: int,
    show_all: bool,
    _model,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_columns: list,
    _tr_lang: str,
) -> pd.DataFrame:
    # _tr_lang is passed as a cache key for language; create a local tr
    def _tr(en: str, pt: str) -> str:
        return en if _tr_lang == "English" else pt

    result = permutation_importance(_model, X, y, n_repeats=10, random_state=42, scoring="roc_auc")
    imp = pd.DataFrame(
        {
            _tr("Variable", "Variável"): feature_columns,
            _tr("Importance", "Importância"): result.importances_mean,
        }
    )

    def _fg(base_feature: str) -> str:
        return _feature_group(base_feature, _tr)

    imp[_tr("Group", "Grupo")] = imp[_tr("Variable", "Variável")].map(_fg)
    imp[_tr("Ranking", "Ranking")] = imp[_tr("Importance", "Importância")].rank(ascending=False, method="dense").astype(int)
    imp = imp.sort_values(_tr("Importance", "Importância"), ascending=False)
    return imp if show_all else imp.head(top_n)


def logistic_clinical_coefficients_table(artifacts, prepared, tr, top_n: int = 20, show_all: bool = False) -> pd.DataFrame:
    if "LogisticRegression" not in artifacts.fitted_models:
        return pd.DataFrame()
    pipe = artifacts.fitted_models["LogisticRegression"]
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame()

    raw_features = artifacts.feature_columns
    feature_names = [_feat_display_name(n) for n in prep.get_feature_names_out()]
    vals = np.asarray(model.coef_).ravel()
    rows = []
    for feat, coef in zip(feature_names, vals):
        base = _resolve_base_feature(feat, raw_features)
        rows.append(
            {
                "encoded_feature": feat,
                tr("Variable", "Variável"): base,
                tr("Coefficient", "Coeficiente"): float(coef),
                tr("Absolute impact", "Impacto absoluto"): abs(float(coef)),
            }
        )
    df_coef = pd.DataFrame(rows)
    if df_coef.empty:
        return df_coef
    idx = df_coef.groupby(tr("Variable", "Variável"))[tr("Absolute impact", "Impacto absoluto")].idxmax()
    out = df_coef.loc[idx].copy()
    out[tr("Direction", "Direção")] = np.where(
        out[tr("Coefficient", "Coeficiente")] >= 0,
        tr("Higher risk tendency", "Tendência a maior risco"),
        tr("Lower risk tendency", "Tendência a menor risco"),
    )
    out[tr("Group", "Grupo")] = out[tr("Variable", "Variável")].map(lambda x: _feature_group(x, tr))
    out[tr("Ranking", "Ranking")] = out[tr("Absolute impact", "Impacto absoluto")].rank(ascending=False, method="dense").astype(int)
    out = out.sort_values(tr("Absolute impact", "Impacto absoluto"), ascending=False)
    cols = [
        tr("Variable", "Variável"),
        tr("Group", "Grupo"),
        tr("Coefficient", "Coeficiente"),
        tr("Direction", "Direção"),
        tr("Absolute impact", "Impacto absoluto"),
        tr("Ranking", "Ranking"),
    ]
    out = out[cols]
    return out if show_all else out.head(top_n)


def model_table_column_config(kind: str, tr) -> dict:
    if kind == "coefficient":
        return {
            "Variable": st.column_config.TextColumn(
                tr("Variable", "Variável"),
                help=tr(
                    "Clinical or encoded feature used by the model.",
                    "Variável clínica ou atributo codificado usado pelo modelo.",
                ),
            ),
            "Coefficient": st.column_config.NumberColumn(
                tr("Coefficient", "Coeficiente"),
                help=tr(
                    "Logistic coefficient after preprocessing. Positive values suggest higher predicted risk, negative values suggest lower predicted risk.",
                    "Coeficiente logístico após o pré-processamento. Valores positivos sugerem maior risco previsto e valores negativos sugerem menor risco previsto.",
                ),
                format="%.4f",
            ),
            "Absolute impact": st.column_config.NumberColumn(
                tr("Absolute impact", "Impacto absoluto"),
                help=tr(
                    "Absolute magnitude of the coefficient, useful for ranking the strongest effects.",
                    "Magnitude absoluta do coeficiente, útil para classificar os efeitos mais fortes.",
                ),
                format="%.4f",
            ),
        }
    if kind == "importance":
        return {
            tr("Variable", "Variável"): st.column_config.TextColumn(
                tr("Variable", "Variável"),
                help=tr(
                    "Clinical feature evaluated in the final selected model.",
                    "Variável clínica avaliada no modelo final selecionado.",
                ),
            ),
            tr("Group", "Grupo"): st.column_config.TextColumn(
                tr("Group", "Grupo"),
                help=tr(
                    "Clinical domain of the variable: clinical, laboratory, echocardiographic, or procedure-related.",
                    "Domínio clínico da variável: clínico, laboratorial, ecocardiográfico ou relacionado ao procedimento.",
                ),
            ),
            tr("Importance", "Importância"): st.column_config.NumberColumn(
                tr("Importance", "Importância"),
                help=tr(
                    "Permutation importance: estimated drop in model performance when the variable is randomly shuffled. Higher values mean greater relevance to the final model.",
                    "Importância por permutação: queda estimada no desempenho do modelo quando a variável é embaralhada aleatoriamente. Valores maiores significam maior relevância no modelo final.",
                ),
                format="%.5f",
            ),
            tr("Ranking", "Ranking"): st.column_config.NumberColumn(
                tr("Ranking", "Ranking"),
                help=tr(
                    "Position of the variable after ordering from most to least important.",
                    "Posição da variável após ordenar da mais importante para a menos importante.",
                ),
                format="%d",
            ),
        }
    if kind == "logistic_clinical":
        return {
            tr("Variable", "Variável"): st.column_config.TextColumn(
                tr("Variable", "Variável"),
                help=tr(
                    "Clinical variable represented in the logistic regression model.",
                    "Variável clínica representada no modelo de regressão logística.",
                ),
            ),
            tr("Group", "Grupo"): st.column_config.TextColumn(
                tr("Group", "Grupo"),
                help=tr(
                    "Clinical domain of the variable.",
                    "Domínio clínico da variável.",
                ),
            ),
            tr("Coefficient", "Coeficiente"): st.column_config.NumberColumn(
                tr("Coefficient", "Coeficiente"),
                help=tr(
                    "Representative logistic coefficient. Positive coefficients suggest higher predicted risk and negative coefficients suggest lower predicted risk.",
                    "Coeficiente logístico representativo. Valores positivos sugerem maior risco previsto e negativos sugerem menor risco previsto.",
                ),
                format="%.4f",
            ),
            tr("Direction", "Direção"): st.column_config.TextColumn(
                tr("Direction", "Direção"),
                help=tr(
                    "Clinical reading of the coefficient sign.",
                    "Leitura clínica do sinal do coeficiente.",
                ),
            ),
            tr("Absolute impact", "Impacto absoluto"): st.column_config.NumberColumn(
                tr("Absolute impact", "Impacto absoluto"),
                help=tr(
                    "Absolute magnitude of the coefficient, used to rank stronger effects regardless of sign.",
                    "Magnitude absoluta do coeficiente, usada para ranquear efeitos mais fortes independentemente do sinal.",
                ),
                format="%.4f",
            ),
            tr("Ranking", "Ranking"): st.column_config.NumberColumn(
                tr("Ranking", "Ranking"),
                help=tr(
                    "Order from strongest to weakest absolute effect.",
                    "Ordem do efeito absoluto mais forte para o mais fraco.",
                ),
                format="%d",
            ),
        }
    if kind == "stacking":
        return {
            "Model": st.column_config.TextColumn(
                tr("Model", "Modelo"),
                help=tr(
                    "Base model used inside the stacking ensemble.",
                    "Modelo base usado dentro do ensemble por stacking.",
                ),
            ),
            "Weight": st.column_config.NumberColumn(
                tr("Weight", "Peso"),
                help=tr(
                    "Weight assigned by the stacking meta-model to the prediction of each base model. This is not a direct clinical variable weight.",
                    "Peso atribuído pelo meta-modelo do stacking à predição de cada modelo base. Não é um peso direto de variável clínica.",
                ),
                format="%.4f",
            ),
            "Absolute impact": st.column_config.NumberColumn(
                tr("Absolute impact", "Impacto absoluto"),
                help=tr(
                    "Absolute magnitude of the base-model weight, useful for ranking which base model contributes most to the ensemble.",
                    "Magnitude absoluta do peso do modelo base, útil para ranquear qual modelo contribui mais para o ensemble.",
                ),
                format="%.4f",
            ),
        }
    if kind == "euroscore":
        return {
            tr("Factor", "Fator"): st.column_config.TextColumn(
                tr("Factor", "Fator"),
                help=tr(
                    "Variable included in the official EuroSCORE II formula.",
                    "Variável incluída na fórmula oficial do EuroSCORE II.",
                ),
            ),
            tr("Coefficient", "Coeficiente"): st.column_config.NumberColumn(
                tr("Coefficient", "Coeficiente"),
                help=tr(
                    "Published EuroSCORE II logistic coefficient. Higher positive coefficients generally indicate stronger contribution to risk.",
                    "Coeficiente logístico publicado do EuroSCORE II. Coeficientes positivos maiores geralmente indicam contribuição mais forte para o risco.",
                ),
                format="%.6f",
            ),
        }
    return {}


def render(ctx: "TabContext") -> None:
    tr = ctx.tr
    artifacts = ctx.artifacts
    prepared = ctx.prepared
    forced_model = ctx.forced_model
    xlsx_path = ctx.xlsx_path
    language = ctx.language
    _fig_to_png_bytes = ctx.fig_to_png_bytes
    _chart_download_buttons = ctx.chart_download_buttons

    st.subheader(tr("Understand the selected model", "Entenda o modelo selecionado"))

    model_docs_en = {
        "LogisticRegression": {
            "como_funciona": (
                "Assigns a weight to each clinical variable, sums them, and converts the result into a 30-day mortality probability."
            ),
            "comportamento": (
                "Stable and transparent. Each factor changes risk in a predictable way. "
                "Works well when risk increases gradually across variables."
            ),
            "forcas": "High interpretability, good calibration, and easy clinical communication.",
            "limitacoes": "May miss complex nonlinear interactions.",
            "quando_erra": (
                "Can underperform when risk depends on very specific variable combinations."
            ),
        },
        "RandomForest": {
            "como_funciona": (
                "Builds many decision trees on different samples and averages their predictions."
            ),
            "comportamento": (
                "Captures nonlinear relationships and interactions automatically. Robust to noise, "
                "but can smooth extreme probabilities."
            ),
            "forcas": "Strong tabular performance with limited feature engineering.",
            "limitacoes": "Less explainable than logistic regression; probabilities may need calibration.",
            "quando_erra": (
                "May underperform for rare patient profiles and very extreme risk estimates."
            ),
        },
        "XGBoost": {
            "como_funciona": (
                "Trains trees sequentially; each new tree corrects previous errors."
            ),
            "comportamento": (
                "Very strong for complex tabular patterns and imbalanced outcomes. "
                "Can achieve high discrimination but needs careful overfitting control."
            ),
            "forcas": "High predictive performance in many clinical datasets.",
            "limitacoes": "Harder to explain; often requires tuning and calibration monitoring.",
            "quando_erra": (
                "May learn spurious patterns in small subgroups and become miscalibrated over time."
            ),
        },
        "LightGBM": {
            "como_funciona": (
                "A gradient-boosted tree model optimized for speed and scalability."
            ),
            "comportamento": (
                "Finds nonlinear interactions efficiently; on smaller datasets it may be parameter-sensitive."
            ),
            "forcas": "Fast training and strong tabular performance.",
            "limitacoes": "Indirect interpretability; performance depends on proper configuration.",
            "quando_erra": (
                "Can be unstable with small samples and many categories, especially in underrepresented subgroups."
            ),
        },
        "CatBoost": {
            "como_funciona": (
                "Boosted trees with native handling of categorical variables and built-in overfitting control."
            ),
            "comportamento": (
                "Often stable when there are many categorical and missing values, with less manual preprocessing."
            ),
            "forcas": "Excellent for heterogeneous clinical tabular data.",
            "limitacoes": "Still complex for individual-case explanation.",
            "quando_erra": (
                "Can underperform when key variables are missing or when case-mix shifts over time."
            ),
        },
        "MLP": {
            "como_funciona": (
                "A multilayer neural network that learns complex combinations of variables."
            ),
            "comportamento": (
                "Captures sophisticated patterns, but can be unstable and overfit in small datasets."
            ),
            "forcas": "Highly flexible for complex relationships.",
            "limitacoes": "Lower transparency and sensitive to dataset size/quality.",
            "quando_erra": (
                "Can overfit in small/noisy datasets, especially for rare profiles."
            ),
        },
        "StackingEnsemble": {
            "como_funciona": (
                "Combines predictions from multiple base models and uses a meta-model for final probability."
            ),
            "comportamento": (
                "Can improve robustness when base models make different errors, at the cost of complexity."
            ),
            "forcas": "Integrates strengths of multiple algorithms.",
            "limitacoes": "Harder to explain, validate, and maintain in routine care.",
            "quando_erra": (
                "Can fail when base models share the same bias or when epidemiologic profile shifts."
            ),
        },
    }

    model_docs_pt = {
        "LogisticRegression": {
            "como_funciona": "Atribui pesos às variáveis e estima a probabilidade de óbito em 30 dias.",
            "comportamento": "É estável e transparente, com efeito previsível de cada variável.",
            "forcas": "Alta interpretabilidade e boa comunicação clínica.",
            "limitacoes": "Pode perder interações não lineares complexas.",
            "quando_erra": "Pode falhar quando o risco depende de combinações muito específicas de fatores.",
        },
        "RandomForest": {
            "como_funciona": "Combina muitas árvores de decisão e faz média das previsões.",
            "comportamento": "Captura não linearidades, mas tende a suavizar probabilidades extremas.",
            "forcas": "Bom desempenho tabular sem muita engenharia manual.",
            "limitacoes": "Menos explicável que regressão logística.",
            "quando_erra": "Pode piorar em perfis raros e riscos muito extremos.",
        },
        "XGBoost": {
            "como_funciona": "Treina árvores em sequência corrigindo erros anteriores.",
            "comportamento": "Alto poder discriminativo, com risco de sobreajuste sem controle.",
            "forcas": "Alta performance preditiva.",
            "limitacoes": "Exige ajuste fino e monitoramento de calibração.",
            "quando_erra": "Pode aprender padrões espúrios em subgrupos pequenos.",
        },
        "LightGBM": {
            "como_funciona": "Boosting de árvores otimizado para velocidade e escala.",
            "comportamento": "Eficiente em interações; pode ser sensível em amostras pequenas.",
            "forcas": "Rápido e eficaz em dados tabulares.",
            "limitacoes": "Interpretação indireta.",
            "quando_erra": "Pode ser instável em subgrupos pouco representados.",
        },
        "CatBoost": {
            "como_funciona": "Boosting com tratamento nativo de variáveis categóricas.",
            "comportamento": "Tende a ser estável com muitos dados categóricos e faltantes.",
            "forcas": "Muito bom para dados clínicos heterogêneos.",
            "limitacoes": "Ainda é complexo para explicar caso a caso.",
            "quando_erra": "Perde desempenho quando faltam variáveis-chave.",
        },
        "MLP": {
            "como_funciona": "Rede neural em camadas para padrões complexos.",
            "comportamento": "Flexível, porém sensível a base pequena e ruído.",
            "forcas": "Capta relações complexas.",
            "limitacoes": "Menor transparência clínica.",
            "quando_erra": "Pode sobreajustar em bases pequenas.",
        },
        "StackingEnsemble": {
            "como_funciona": "Combina previsões de vários modelos em um meta-modelo final.",
            "comportamento": "Pode aumentar robustez ao combinar erros diferentes.",
            "forcas": "Integra forças de múltiplos algoritmos.",
            "limitacoes": "Mais difícil de validar e explicar.",
            "quando_erra": "Falha quando os modelos base compartilham o mesmo viés.",
        },
    }

    model_docs = model_docs_en if language == "English" else model_docs_pt

    d = model_docs.get(
        forced_model,
        {
            "como_funciona": tr("Model not documented.", "Modelo não documentado."),
            "comportamento": "",
            "forcas": "",
            "limitacoes": "",
            "quando_erra": "",
        },
    )

    st.markdown(tr(f"### Selected model: `{forced_model}`", f"### Modelo selecionado: `{forced_model}`"))
    st.markdown(tr("**How it works**", "**Como funciona**"))
    st.write(d["como_funciona"])
    st.markdown(tr("**How it behaves in practice**", "**Como ele se comporta na prática**"))
    st.write(d["comportamento"])
    st.markdown(tr("**Strengths**", "**Pontos fortes**"))
    st.write(d["forcas"])
    st.markdown(tr("**Limitations**", "**Limitações**"))
    st.write(d["limitacoes"])
    st.markdown(tr("**When this model usually fails**", "**Quando este modelo costuma errar**"))
    st.write(d["quando_erra"])

    st.markdown(tr("**How to interpret in practice**", "**Como interpretar na prática**"))
    st.markdown(
        tr(
            "- Risk is reported as a probability (0% to 100%).\n"
            "- The app classifies risk as Low (<5%), Intermediate (5-15%), and High (>15%).\n"
            "- Changes in patient profile may shift probability even without class change.\n"
            "- Changes in case-mix over time may change the best-performing model.",
            "- O risco mostrado é uma probabilidade (0% a 100%).\n"
            "- O app classifica em Baixo (<5%), Intermediário (5-15%) e Alto (>15%).\n"
            "- Mudança no perfil do paciente pode deslocar a probabilidade mesmo sem mudar a classe.\n"
            "- Mudança de base (novos pacientes) pode alterar qual modelo fica melhor.",
        )
    )

    # ── Top predictors summary ──────────────────────────────────────────
    w, w_kind = model_weight_table(artifacts, prepared, forced_model, tr, top_n=20)
    if not w.empty and w_kind in ("importance", "coefficient"):
        _name_col = w.columns[0]
        _top3_names = w.head(3)[_name_col].tolist()
        _top3_str = ", ".join(f"**{n}**" for n in _top3_names)
        if w_kind == "importance":
            _top_narrative_en = (
                f"The three variables with the greatest relative importance for this model are {_top3_str}. "
                "These rankings reflect contribution to model performance, not direct clinical effect size."
            )
            _top_narrative_pt = (
                f"As três variáveis com maior importância relativa neste modelo são {_top3_str}. "
                "Esses rankings refletem contribuição ao desempenho do modelo, não efeito clínico direto."
            )
        else:
            _top_narrative_en = (
                f"The three variables with the largest absolute coefficients in this model are {_top3_str}. "
                "Positive coefficients suggest higher risk; negative suggest lower risk."
            )
            _top_narrative_pt = (
                f"As três variáveis com maiores coeficientes absolutos neste modelo são {_top3_str}. "
                "Coeficientes positivos sugerem maior risco; negativos sugerem menor risco."
            )
        st.markdown(tr("**Top predictors**", "**Principais preditores**"))
        st.caption(tr(_top_narrative_en, _top_narrative_pt))
        st.dataframe(w.head(5), width="stretch", column_config=model_table_column_config(w_kind, tr))
        st.divider()

    st.markdown(tr("**Model explanation table**", "**Tabela explicativa do modelo**"))
    if w_kind == "coefficient":
        st.caption(
            tr(
                "These are model coefficients after preprocessing. For categorical variables, the values refer to encoded categories rather than the raw clinical field as a whole.",
                "Estes são coeficientes do modelo após o pré-processamento. Para variáveis categóricas, os valores se referem às categorias codificadas, e não ao campo clínico bruto como um todo.",
            )
        )
    elif w_kind == "importance":
        st.caption(
            tr(
                "These values represent variable importance, not direct clinical effect size. They indicate how much the variable helps the model, but not whether it increases or decreases risk.",
                "Esses valores representam importância de variável, e não efeito clínico direto. Eles indicam o quanto a variável ajuda o modelo, mas não se aumenta ou reduz o risco.",
            )
        )
    elif w_kind == "stacking":
        st.caption(
            tr(
                "These are weights of the base models inside the stacking ensemble, not direct weights of the clinical variables.",
                "Esses são pesos dos modelos base dentro do ensemble por stacking, e não pesos diretos das variáveis clínicas.",
            )
        )

    if w.empty:
        st.info(
            tr(
                "This model does not expose clinically interpretable per-variable weights in a simple way. Use global importance, calibration, discrimination, and sensitivity analysis.",
                "Este modelo não expõe pesos por variável clinicamente interpretáveis de forma simples. Nesses casos, a interpretação deve se apoiar em importância global, calibração, discriminação e análise de sensibilidade.",
            )
        )
    else:
        st.dataframe(w, width="stretch", column_config=model_table_column_config(w_kind, tr))

    st.markdown(tr("**Clinical variables and importance**", "**Variáveis clínicas e importância**"))
    show_all_final = st.checkbox(
        tr("Show all variables for final model", "Ver todas as variáveis do modelo final"),
        value=False,
        key="show_all_final_importance",
    )
    X_perm = clean_features(_safe_select_features(prepared.data, artifacts.feature_columns))
    y_perm = prepared.data["morte_30d"].astype(int).values
    perm_table = _cached_permutation_importance_table(
        xlsx_path,
        forced_model,
        20,
        show_all_final,
        artifacts.fitted_models[forced_model],
        X_perm,
        y_perm,
        artifacts.feature_columns,
        language,
    )
    st.caption(
        tr(
            "Permutation importance estimates how much model performance worsens when a variable is randomly shuffled. This is the recommended summary for the selected final model, but it should be interpreted as exploratory because it is computed on the fitted dataset.",
            "A importância por permutação estima o quanto o desempenho do modelo piora quando uma variável é embaralhada aleatoriamente. Este é o resumo mais indicado para o modelo final selecionado, mas deve ser interpretado como exploratório porque é calculado na base ajustada.",
        )
    )
    st.dataframe(perm_table, width="stretch", column_config=model_table_column_config("importance", tr))

    st.markdown(tr("**SHAP global importance (selected model)**", "**Importância global SHAP (modelo selecionado)**"))
    _shap_model_estimator = artifacts.fitted_models[forced_model].named_steps["model"]
    if not hasattr(_shap_model_estimator, "feature_importances_"):
        st.info(
            tr(
                "SHAP global visualization is available only for tree-based models (RandomForest, XGBoost, LightGBM, CatBoost). The currently selected model does not expose tree structure.",
                "A visualização global SHAP está disponível apenas para modelos baseados em árvore (RandomForest, XGBoost, LightGBM, CatBoost). O modelo selecionado não expõe estrutura de árvore.",
            )
        )
    else:
        st.caption(
            tr(
                "Unlike permutation importance (which only shows magnitude), SHAP values also show direction: positive mean SHAP pushes predicted risk up, negative pushes it down. Computed on the full fitted dataset using TreeExplainer.",
                "Ao contrário da importância por permutação (que mostra apenas magnitude), os valores SHAP também mostram direção: SHAP médio positivo aumenta o risco previsto, negativo reduz. Calculado na base ajustada completa com TreeExplainer.",
            )
        )
        with st.spinner(tr("Computing SHAP global importance…", "Calculando importância global SHAP…")):
            shap_global_df = _cached_shap_global(
                xlsx_path,
                forced_model,
                20,
                artifacts.fitted_models[forced_model],
                X_perm,
            )
        if shap_global_df.empty:
            st.info(tr("SHAP global importance could not be computed.", "Importância global SHAP não pôde ser calculada."))
        else:
            shap_global_df.columns = [
                tr("Feature", "Variável"),
                tr("Mean |SHAP|", "SHAP médio |absoluto|"),
                tr("Mean SHAP (direction)", "SHAP médio (direção)"),
            ]
            st.dataframe(shap_global_df, width="stretch")

        # --- SHAP Beeswarm Plot ---
        st.markdown(tr("**SHAP Beeswarm plot**", "**Gráfico Beeswarm SHAP**"))
        st.caption(
            tr(
                "Each dot is one patient-feature pair. Position on x-axis shows the SHAP value (impact on prediction). Color shows the feature value (red = high, blue = low). This reveals both importance and direction of each variable's effect.",
                "Cada ponto é um par paciente-variável. A posição no eixo x mostra o valor SHAP (impacto na predição). A cor indica o valor da variável (vermelho = alto, azul = baixo). Isso revela importância e direção do efeito de cada variável.",
            )
        )
        with st.spinner(tr("Generating beeswarm plot…", "Gerando gráfico beeswarm…")):
            beeswarm_fig = _cached_shap_beeswarm(
                xlsx_path,
                forced_model,
                15,
                artifacts.fitted_models[forced_model],
                X_perm,
            )
        if beeswarm_fig is not None:
            st.pyplot(beeswarm_fig)
            _chart_download_buttons(shap_global_df, _fig_to_png_bytes(beeswarm_fig), "shap_beeswarm")
        else:
            st.info(tr("Beeswarm plot could not be generated.", "Gráfico beeswarm não pôde ser gerado."))

        # --- SHAP Dependence Plot ---
        st.markdown(tr("**SHAP Feature dependence**", "**Dependência de variável SHAP**"))
        st.caption(
            tr(
                "Shows how a single variable's value affects the model prediction. Each dot is a patient; the y-axis shows that variable's SHAP contribution to the predicted risk.",
                "Mostra como o valor de uma variável individual afeta a predição do modelo. Cada ponto é um paciente; o eixo y mostra a contribuição SHAP daquela variável ao risco previsto.",
            )
        )
        _shap_dep_features = shap_global_df[tr("Feature", "Variável")].tolist() if not shap_global_df.empty else []
        if _shap_dep_features:
            selected_feature = st.selectbox(
                tr("Select feature for dependence plot", "Selecione variável para gráfico de dependência"),
                _shap_dep_features,
                key="shap_dep_feature",
            )
            with st.spinner(tr("Generating dependence plot…", "Gerando gráfico de dependência…")):
                dep_fig = _cached_shap_dependence(
                    xlsx_path,
                    forced_model,
                    selected_feature,
                    artifacts.fitted_models[forced_model],
                    X_perm,
                )
            if dep_fig is not None:
                st.pyplot(dep_fig)
                _dep_data = pd.DataFrame({"Feature": [selected_feature]})
                _chart_download_buttons(_dep_data, _fig_to_png_bytes(dep_fig), f"shap_dependence_{selected_feature}")
            else:
                st.info(tr("Dependence plot could not be generated for this feature.", "Gráfico de dependência não pôde ser gerado para esta variável."))

    st.markdown(tr("**Logistic regression clinical coefficients**", "**Coeficientes clínicos da regressão logística**"))
    show_all_lr = st.checkbox(
        tr("Show all logistic regression variables", "Ver todas as variáveis da regressão logística"),
        value=False,
        key="show_all_lr_coeffs",
    )
    lr_table = logistic_clinical_coefficients_table(artifacts, prepared, tr, top_n=20, show_all=show_all_lr)
    st.caption(
        tr(
            "These coefficients come from the fitted logistic regression model. Positive coefficients suggest a tendency toward higher risk, whereas negative coefficients suggest a tendency toward lower risk. For categorical variables, the table keeps the category with the strongest absolute coefficient as the representative effect.",
            "Esses coeficientes vêm do modelo de regressão logística ajustado. Coeficientes positivos sugerem tendência a maior risco, enquanto coeficientes negativos sugerem tendência a menor risco. Para variáveis categóricas, a tabela mantém a categoria com maior coeficiente absoluto como efeito representativo.",
        )
    )
    st.dataframe(lr_table, width="stretch", column_config=model_table_column_config("logistic_clinical", tr))

    st.markdown(tr("**EuroSCORE II coefficients (official)**", "**Coeficientes do EuroSCORE II (oficial)**"))
    euro_coef_df = pd.DataFrame(
        {
            tr("Factor", "Fator"): [tr("Constant", "Constante")] + list(EURO_COEF.keys()),
            tr("Coefficient", "Coeficiente"): [EURO_CONST] + [EURO_COEF[k] for k in EURO_COEF.keys()],
        }
    )
    st.dataframe(euro_coef_df, width="stretch", column_config=model_table_column_config("euroscore", tr))

    # ── Feature Importance Export ──────────────────────────────────────────
    st.divider()
    st.markdown(tr("### Export", "### Exportar"))

    def _build_feature_importance_xlsx() -> bytes:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            try:
                shap_df = _cached_shap_global(xlsx_path, forced_model, 20, artifacts.fitted_models[forced_model], clean_features(_safe_select_features(prepared.data, artifacts.feature_columns)))
                if shap_df is not None and not shap_df.empty:
                    shap_export = shap_df.copy()
                    shap_export.insert(0, "rank", range(1, len(shap_export) + 1))
                    shap_export.to_excel(writer, sheet_name="SHAP_Importance", index=False)
            except Exception:
                pass
            try:
                from config.model_config import get_model_params
                params = get_model_params(forced_model)
                pd.DataFrame([{"Parameter": k, "Value": str(v)} for k, v in params.items()]).to_excel(
                    writer, sheet_name="Model_Parameters", index=False
                )
            except Exception:
                pass
            pd.DataFrame({"Feature": artifacts.feature_columns}).to_excel(
                writer, sheet_name="Feature_List", index=False
            )
        return buf.getvalue()

    st.download_button(
        label=tr("Download Feature Importance (XLSX)", "Baixar Importância de Features (XLSX)"),
        data=_build_feature_importance_xlsx(),
        file_name="ai_risk_feature_importance.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
