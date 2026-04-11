"""Bundle I/O helpers for AI Risk.

Extracted from ``app.py`` during the Phase 4 conservative refactor.
These are pure data-transform helpers with **no** Streamlit dependency:

* :func:`bundle_signature` - build the cache-key signature for a training
  source based on its file stat and the current model version.
* :func:`serialize_bundle` - convert a training bundle (containing
  ``PreparedData`` and ``TrainedArtifacts`` dataclasses) into plain dicts
  suitable for ``joblib.dump``.  Streamlit reloads module objects between
  reruns, so pickling dataclass instances directly would tie the cache to
  specific class references; plain dicts sidestep that.
* :func:`deserialize_bundle` - reverse :func:`serialize_bundle`, rebuilding
  the dataclass instances via an ``importlib.reload`` of ``modeling`` and
  ``observability`` so the freshly loaded classes are used.

Both serialize/deserialize also handle the Phase 3 ``run_report`` field
(a :class:`observability.RunReport`) via its ``to_dict`` / ``from_dict``
methods.

No methodology is changed.  The cache file path itself
(``MODEL_CACHE_FILE``) is deliberately left in ``app.py`` because it is
owned by the cache policy layer (``load_train_bundle`` and
``load_cached_bundle_only``) which remain in the app for this phase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from config import AppConfig
from risk_data import PreparedData

MODEL_VERSION = AppConfig.MODEL_VERSION


def bundle_signature(xlsx_path: str) -> Dict[str, object]:
    """Signature used as the cache key for a training source."""
    p = Path(xlsx_path)
    stt = p.stat()
    return {
        "xlsx_path": str(p.resolve()),
        "xlsx_mtime_ns": int(stt.st_mtime_ns),
        "xlsx_size": int(stt.st_size),
        "model_version": MODEL_VERSION,
    }


def serialize_bundle(bundle: Dict[str, object]) -> Dict[str, object]:
    """Convert dataclasses to plain dicts for pickle compatibility.

    Streamlit may reload modules between runs, creating new class objects.
    Pickle requires the exact same class reference, so we store plain dicts.
    """
    out = dict(bundle)
    prepared = out["prepared"]
    out["prepared"] = {
        "data": prepared.data,
        "feature_columns": prepared.feature_columns,
        "info": prepared.info,
    }
    artifacts = out["artifacts"]
    out["artifacts"] = {
        "model": artifacts.model,
        "leaderboard": artifacts.leaderboard,
        "oof_predictions": artifacts.oof_predictions,       # calibrated OOF (primary)
        "oof_raw": getattr(artifacts, "oof_raw", None),     # uncalibrated, audit only
        "feature_columns": artifacts.feature_columns,
        "fitted_models": artifacts.fitted_models,
        "best_model_name": artifacts.best_model_name,
        "calibration_method": getattr(artifacts, "calibration_method", "sigmoid"),
        "youden_thresholds": getattr(artifacts, "youden_thresholds", None),
        "best_youden_threshold": getattr(artifacts, "best_youden_threshold", None),
    }
    # Phase 3: persist the run report as a plain dict so module reloads
    # don't break unpickling.
    run_report = out.get("run_report")
    if run_report is not None and hasattr(run_report, "to_dict"):
        out["run_report"] = run_report.to_dict()
    return out


def deserialize_bundle(bundle: Dict[str, object]) -> Dict[str, object]:
    """Reconstruct dataclasses from plain dicts."""
    import importlib
    import modeling as _mod
    importlib.reload(_mod)
    TrainedArtifacts = _mod.TrainedArtifacts
    out = dict(bundle)
    p = out["prepared"]
    if isinstance(p, dict):
        out["prepared"] = PreparedData(
            data=p["data"],
            feature_columns=p["feature_columns"],
            info=p["info"],
        )
    a = out["artifacts"]
    if isinstance(a, dict):
        out["artifacts"] = TrainedArtifacts(
            model=a["model"],
            leaderboard=a["leaderboard"],
            oof_predictions=a["oof_predictions"],           # calibrated OOF
            feature_columns=a["feature_columns"],
            fitted_models=a["fitted_models"],
            best_model_name=a["best_model_name"],
            calibration_method=a.get("calibration_method", "sigmoid"),
            oof_raw=a.get("oof_raw"),                       # may be None in old caches
            youden_thresholds=a.get("youden_thresholds"),
            best_youden_threshold=a.get("best_youden_threshold"),
        )
    # Phase 3: rebuild RunReport from persisted dict (legacy bundles without
    # the key simply carry no report).
    rr = out.get("run_report")
    if isinstance(rr, dict):
        import observability as _obs
        importlib.reload(_obs)
        out["run_report"] = _obs.RunReport.from_dict(rr)
    return out
