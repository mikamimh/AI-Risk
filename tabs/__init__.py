"""Tab modules for AI Risk.

Each sub-module exposes a single ``render(ctx)`` function that receives a
:class:`TabContext` carrying every shared dependency the tab needs.  This
avoids passing 15+ positional arguments and keeps the contract explicit.

The ``TabContext`` is assembled once in ``app.py`` after bootstrap and
passed verbatim — tab modules never import from ``app.py``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd


@dataclasses.dataclass
class TabContext:
    """Shared state passed from app.py to every tab renderer.

    Fields are grouped logically:

    1. **i18n** — translation helpers and active language.
    2. **Data** — prepared dataset, artifacts, working dataframe.
    3. **Model** — selected model name, best model name, bundle metadata.
    4. **Config** — paths, thresholds, constants.
    5. **UI helpers** — download-button wrappers and display utilities that
       live in ``app.py`` and capture Streamlit / language state via closure.
    """

    # ── 1. i18n ──────────────────────────────────────────────────────────
    tr: Callable[[str, str], str]
    hp: Callable[[str, str], str]
    language: str

    # ── 2. Data ──────────────────────────────────────────────────────────
    prepared: Any                     # PreparedData (has .data, .info, .feature_columns)
    artifacts: Any                    # TrainingArtifacts (has .fitted_models, .oof_predictions, …)
    df: pd.DataFrame                  # Working dataframe with ia_risk_oof, euroscore_calc, …

    # ── 3. Model ─────────────────────────────────────────────────────────
    forced_model: str                 # User-selected model name from sidebar
    best_model_name: str              # Best model from leaderboard
    bundle_info: Dict[str, Any]       # Bundle metadata dict (saved_at, training_source, …)

    # ── 4. Config ────────────────────────────────────────────────────────
    xlsx_path: str                    # Path to the active data source
    default_threshold: float          # Clinical threshold (0.08)
    model_version: str                # MODEL_VERSION constant
    has_sts: bool                     # Whether STS WebSocket library is available

    # ── 5. UI helpers (closures from app.py) ─────────────────────────────
    csv_download_btn: Callable        # _csv_download_btn(df, filename, label)
    txt_download_btn: Callable        # _txt_download_btn(text, filename, label)
    bytes_download_btn: Callable      # _bytes_download_btn(data, filename, label, mime, key=)
    update_phase: Callable            # _update_phase(slot, phase_num, phase_total, label)
    sts_score_patient_ids: Callable   # _sts_score_patient_ids(rows) -> list
    general_table_column_config: Callable  # general_table_column_config(kind) -> dict
    stats_table_column_config: Callable    # stats_table_column_config(kind) -> dict
    format_ppv_npv: Callable          # _format_ppv_npv(df) -> df
    to_csv_bytes: Callable            # _to_csv_bytes(df) -> bytes
    safe_prob: Callable               # _safe_prob(x) -> float

    # ── 5b. Plot helpers (closures from app.py) ──────────────────────────
    plot_roc: Callable                # _plot_roc(scores, y)
    plot_calibration: Callable        # _plot_calibration(scores, y)
    plot_boxplots: Callable           # _plot_boxplots(df_plot)
    plot_ia_model_boxplots: Callable  # _plot_ia_model_boxplots(y_true, oof_predictions)
    plot_dca: Callable                # _plot_dca(curve_df)

    # ── 5c. Text builders (closures from app.py) ─────────────────────────
    build_methods_text: Callable      # build_methods_text(mode) -> str
    build_results_text: Callable      # build_results_text(mode, context) -> str
