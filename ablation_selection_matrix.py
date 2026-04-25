"""Ablation matrix: feature-block variants x model-selection policies.

Standalone script. It does not modify production training code.

Why this exists
---------------
The app already has targeted ablations for specific pipeline decisions. This
script adds a broader sensitivity matrix:

* feature-block variants are retrained per seed;
* selection policies are evaluated post-hoc on the same calibrated OOF
  predictions from each training run.

That keeps the experiment reasonably cheap while separating two questions:

1. Which feature representation is stable?
2. Which model-selection rule would have been chosen from the same candidates?

Usage
-----
    python ablation_selection_matrix.py
    python ablation_selection_matrix.py --n-runs 50
    python ablation_selection_matrix.py --n-runs 20 --n-jobs -1
    python ablation_selection_matrix.py --dataset local_data/Dataset_2025.xlsx
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from config.base_config import AppConfig
import config.model_config as model_config
import modeling
from risk_data import MISSINGNESS_INDICATOR_COLUMNS, prepare_master_dataset
from stats_compare import calibration_intercept_slope, classification_metrics_at_threshold


DEFAULT_DATASET = "local_data/Dataset_2025.xlsx"
DEFAULT_OUTPUT = "ablation_selection_matrix_results.csv"
DEFAULT_SUMMARY = "ablation_selection_matrix_summary.csv"
DEFAULT_N_RUNS = 20
DEFAULT_N_JOBS = 1
BASE_SEED = 42
CLINICAL_THRESHOLD = 0.08

SURGICAL_DERIVED_COLS = frozenset(
    {"cirurgia_combinada", "peso_procedimento", "thoracic_aorta_flag"}
)

SPARSE_ECHO_PATTERNS = (
    "Vena contracta",
    "MVA",
    "Mitral Mean gradient",
    "Mitral Stenosis",
    "PHT Aortic",
    "KDIGO",
)


def _match_any_pattern(columns: list[str], patterns: tuple[str, ...]) -> set[str]:
    matched: set[str] = set()
    for col in columns:
        lower = col.lower()
        if any(pattern.lower() in lower for pattern in patterns):
            matched.add(col)
    return matched


def _feature_variants(df: pd.DataFrame, base_features: list[str]) -> dict[str, list[str]]:
    """Return pre-specified feature-block variants.

    These are sensitivity analyses, not free subset search. The variants are
    intentionally coarse and clinically interpretable.
    """
    base = [c for c in base_features if c in df.columns]
    missingness = set(MISSINGNESS_INDICATOR_COLUMNS)
    sparse_echo_kdigo = _match_any_pattern(base, SPARSE_ECHO_PATTERNS)

    variants: dict[str, list[str]] = {
        "baseline": base,
        "no_missingness_indicators": [c for c in base if c not in missingness],
        "no_surgery_derived": [c for c in base if c not in SURGICAL_DERIVED_COLS],
        "no_sparse_echo_kdigo": [c for c in base if c not in sparse_echo_kdigo],
    }

    if "procedure_group" in df.columns:
        with_pg = list(base)
        if "procedure_group" not in with_pg:
            with_pg.append("procedure_group")
        variants["add_procedure_group"] = with_pg

    # Compact, publication-friendly sensitivity variant: removes the blocks
    # most likely to overfit small samples while leaving core clinical data.
    variants["lean_no_missingness_no_sparse"] = [
        c for c in base if c not in missingness and c not in sparse_echo_kdigo
    ]

    # Preserve insertion order and remove duplicates inside each variant.
    return {name: list(dict.fromkeys(cols)) for name, cols in variants.items()}


def _selection_policies(
    artifacts: modeling.TrainedArtifacts,
    y: np.ndarray,
) -> dict[str, str]:
    """Select a model under several auditable policies from one training run."""
    lb = artifacts.leaderboard.copy()
    policies: dict[str, str] = {}

    policies["current_guardrails"] = str(artifacts.best_model_name)
    policies["auc_only"] = str(lb.iloc[0]["Modelo"])
    policies["auprc_only"] = str(lb.sort_values(["AUPRC", "AUC"], ascending=[False, False]).iloc[0]["Modelo"])
    policies["brier_only"] = str(lb.sort_values(["Brier", "AUC"], ascending=[True, False]).iloc[0]["Modelo"])

    cal_rows = []
    for name, proba in artifacts.oof_predictions.items():
        try:
            cal = calibration_intercept_slope(y, np.asarray(proba, dtype=float))
            slope = float(cal.get("Calibration slope", np.nan))
        except Exception:
            slope = np.nan
        auc = float(lb.loc[lb["Modelo"] == name, "AUC"].iloc[0])
        cal_rows.append({"Modelo": name, "AUC": auc, "slope_dist": abs(slope - 1.0)})
    cal_df = pd.DataFrame(cal_rows)
    if not cal_df.empty:
        top_auc = float(cal_df["AUC"].max())
        near_top = cal_df[cal_df["AUC"] >= top_auc - 0.01]
        policies["auc_with_calibration_tiebreak"] = str(
            near_top.sort_values(["slope_dist", "AUC"], ascending=[True, False]).iloc[0]["Modelo"]
        )

    if "RandomForest" in artifacts.oof_predictions:
        policies["fixed_randomforest"] = "RandomForest"
    if "LogisticRegression" in artifacts.oof_predictions:
        policies["fixed_logistic"] = "LogisticRegression"

    return policies


def _metrics_for_selection(
    *,
    variant: str,
    seed: int,
    selection_policy: str,
    selected_model: str,
    n_features: int,
    y: np.ndarray,
    proba: np.ndarray,
    leaderboard: pd.DataFrame,
) -> dict[str, object]:
    proba = np.asarray(proba, dtype=float)
    cal = calibration_intercept_slope(y, proba)
    at8 = classification_metrics_at_threshold(y, proba, CLINICAL_THRESHOLD)
    lb_row = leaderboard.loc[leaderboard["Modelo"] == selected_model]

    if lb_row.empty:
        auc = roc_auc_score(y, proba)
        auprc = average_precision_score(y, proba)
        brier = brier_score_loss(y, proba)
    else:
        auc = float(lb_row["AUC"].iloc[0])
        auprc = float(lb_row["AUPRC"].iloc[0])
        brier = float(lb_row["Brier"].iloc[0])

    return {
        "variant": variant,
        "seed": seed,
        "selection_policy": selection_policy,
        "selected_model": selected_model,
        "n_features": int(n_features),
        "AUC": float(auc),
        "AUPRC": float(auprc),
        "Brier": float(brier),
        "Calibration_intercept": float(cal.get("Calibration intercept", np.nan)),
        "Calibration_slope": float(cal.get("Calibration slope", np.nan)),
        "pct_oof_below_8pct": float((proba < CLINICAL_THRESHOLD).mean()),
        "Sensitivity_8pct": float(at8.get("Sensitivity", np.nan)),
        "Specificity_8pct": float(at8.get("Specificity", np.nan)),
        "PPV_8pct": float(at8.get("PPV", np.nan)),
        "NPV_8pct": float(at8.get("NPV", np.nan)),
    }


def _run_one_variant(
    *,
    df: pd.DataFrame,
    feature_columns: list[str],
    variant: str,
    seed: int,
    n_jobs: int,
) -> list[dict[str, object]]:
    original_seed = AppConfig.RANDOM_SEED
    original_n_jobs = AppConfig.N_JOBS
    original_rf_n_jobs = model_config.MODEL_HYPERPARAMS.get("RandomForest", {}).get("n_jobs")
    try:
        AppConfig.RANDOM_SEED = seed
        # Ablation scripts are often run from constrained Windows shells where
        # joblib thread/process pools can fail with WinError 5. The CLI default
        # is serial mode, but --n-jobs can be raised on unrestricted machines.
        AppConfig.N_JOBS = int(n_jobs)
        if "RandomForest" in model_config.MODEL_HYPERPARAMS:
            model_config.MODEL_HYPERPARAMS["RandomForest"]["n_jobs"] = int(n_jobs)
        artifacts = modeling.train_and_select_model(df, feature_columns)
        y = df["morte_30d"].astype(int).values
        policies = _selection_policies(artifacts, y)
        rows = []
        for policy, model_name in policies.items():
            proba = artifacts.oof_predictions.get(model_name)
            if proba is None:
                continue
            rows.append(
                _metrics_for_selection(
                    variant=variant,
                    seed=seed,
                    selection_policy=policy,
                    selected_model=model_name,
                    n_features=len(artifacts.feature_columns),
                    y=y,
                    proba=proba,
                    leaderboard=artifacts.leaderboard,
                )
            )
        return rows
    finally:
        AppConfig.RANDOM_SEED = original_seed
        AppConfig.N_JOBS = original_n_jobs
        if original_rf_n_jobs is not None and "RandomForest" in model_config.MODEL_HYPERPARAMS:
            model_config.MODEL_HYPERPARAMS["RandomForest"]["n_jobs"] = original_rf_n_jobs


def _summarise(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["variant", "selection_policy"]
    rows = []
    for (variant, policy), grp in results.groupby(group_cols, sort=False):
        winners = grp["selected_model"].value_counts()
        rows.append(
            {
                "variant": variant,
                "selection_policy": policy,
                "n_runs": int(len(grp)),
                "n_features": int(grp["n_features"].median()),
                "AUC_mean": float(grp["AUC"].mean()),
                "AUC_std": float(grp["AUC"].std(ddof=1)),
                "AUPRC_mean": float(grp["AUPRC"].mean()),
                "AUPRC_std": float(grp["AUPRC"].std(ddof=1)),
                "Brier_mean": float(grp["Brier"].mean()),
                "Brier_std": float(grp["Brier"].std(ddof=1)),
                "Slope_mean": float(grp["Calibration_slope"].mean()),
                "Slope_std": float(grp["Calibration_slope"].std(ddof=1)),
                "Sensitivity_8pct_mean": float(grp["Sensitivity_8pct"].mean()),
                "Specificity_8pct_mean": float(grp["Specificity_8pct"].mean()),
                "pct_below_8_mean": float(grp["pct_oof_below_8pct"].mean()),
                "top_model": str(winners.index[0]) if len(winners) else "",
                "top_model_pct": float(winners.iloc[0] / len(grp)) if len(winners) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)

    baseline = summary[
        (summary["variant"] == "baseline")
        & (summary["selection_policy"] == "current_guardrails")
    ]
    if not baseline.empty:
        base = baseline.iloc[0]
        summary["delta_AUC_vs_baseline"] = summary["AUC_mean"] - float(base["AUC_mean"])
        summary["delta_AUPRC_vs_baseline"] = summary["AUPRC_mean"] - float(base["AUPRC_mean"])
        summary["delta_Brier_vs_baseline"] = summary["Brier_mean"] - float(base["Brier_mean"])
        summary["delta_SlopeAbsErr_vs_baseline"] = (
            (summary["Slope_mean"] - 1.0).abs() - abs(float(base["Slope_mean"]) - 1.0)
        )
    return summary


def run_ablation_matrix(
    dataset_path: str,
    n_runs: int,
    n_jobs: int,
    output_path: str,
    summary_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading dataset: {dataset_path}")
    prepared = prepare_master_dataset(dataset_path)
    df = prepared.data
    base_features = prepared.feature_columns
    y = df["morte_30d"].astype(int).values
    print(
        f"  rows={len(df)}, events={int(y.sum())}, "
        f"prevalence={y.mean():.1%}, base_features={len(base_features)}"
    )

    variants = _feature_variants(df, base_features)
    seeds = [BASE_SEED + i for i in range(n_runs)]
    rows: list[dict[str, object]] = []

    for variant, features in variants.items():
        print(f"\n--- {variant} ({len(features)} requested features) ---")
        for i, seed in enumerate(seeds, start=1):
            print(f"  run {i}/{n_runs}, seed={seed} ... ", end="", flush=True)
            try:
                new_rows = _run_one_variant(
                    df=df,
                    feature_columns=features,
                    variant=variant,
                    seed=seed,
                    n_jobs=n_jobs,
                )
                rows.extend(new_rows)
                current = [r for r in new_rows if r["selection_policy"] == "current_guardrails"]
                winner = current[0]["selected_model"] if current else "?"
                auc = current[0]["AUC"] if current else np.nan
                print(f"winner={winner}, AUC={auc:.4f}")
            except Exception as exc:
                print(f"FAILED: {exc}", file=sys.stderr)
                traceback.print_exc()

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No ablation rows were produced.")

    summary = _summarise(results)
    results.to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"\nResults saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    print("\nTop summary sorted by AUC then Brier:")
    cols = [
        "variant",
        "selection_policy",
        "n_runs",
        "n_features",
        "AUC_mean",
        "AUPRC_mean",
        "Brier_mean",
        "Slope_mean",
        "top_model",
        "top_model_pct",
        "delta_AUC_vs_baseline",
        "delta_Brier_vs_baseline",
    ]
    show_cols = [c for c in cols if c in summary.columns]
    shown = summary.sort_values(["AUC_mean", "Brier_mean"], ascending=[False, True])
    print(shown[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run feature-block x model-selection ablation matrix."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=(
            "Parallel jobs for estimators that support it. Default 1 is safest "
            "on Windows/sandboxed shells; use -1 on unrestricted local runs."
        ),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    if args.n_runs < 1:
        raise SystemExit("--n-runs must be >= 1")
    if not Path(args.dataset).exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    run_ablation_matrix(args.dataset, args.n_runs, args.n_jobs, args.output, args.summary)


if __name__ == "__main__":
    main()
