"""Standalone script to audit model selection stability across random seeds.

Runs train_and_select_model N times with sequential seeds and records the
key metrics for each run. Outputs a CSV and a console summary.

Usage:
    python audit_model_stability.py
    python audit_model_stability.py --n-runs 20 --dataset local_data/Dataset_2025.xlsx
    python audit_model_stability.py --output my_stability_results.csv
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ── Must not import Streamlit ──────────────────────────────────────────────
from config.base_config import AppConfig
from modeling import train_and_select_model
from risk_data import prepare_master_dataset
from stats_compare import calibration_intercept_slope


_DEFAULT_DATASET = "local_data/Dataset_2025.xlsx"
_DEFAULT_N_RUNS = 50
_DEFAULT_OUTPUT = "audit_model_stability_results.csv"
_BASE_SEED = 42


def _run_once(
    df: pd.DataFrame,
    feature_columns: list,
    seed: int,
    run_idx: int,
    total: int,
) -> dict:
    """Train once with a given seed and return the metrics dict."""
    original_seed = AppConfig.RANDOM_SEED
    try:
        AppConfig.RANDOM_SEED = seed
        artifacts = train_and_select_model(
            df,
            feature_columns,
            y_col="morte_30d",
            group_col="_patient_key",
        )
        y = df["morte_30d"].astype(int).values
        best = artifacts.best_model_name
        oof = artifacts.oof_predictions.get(best, np.full(len(y), np.nan))

        auc = float(artifacts.leaderboard.loc[
            artifacts.leaderboard["Modelo"] == best, "AUC"
        ].iloc[0]) if not artifacts.leaderboard.empty else np.nan

        auprc = float(artifacts.leaderboard.loc[
            artifacts.leaderboard["Modelo"] == best, "AUPRC"
        ].iloc[0]) if not artifacts.leaderboard.empty else np.nan

        brier = float(artifacts.leaderboard.loc[
            artifacts.leaderboard["Modelo"] == best, "Brier"
        ].iloc[0]) if not artifacts.leaderboard.empty else np.nan

        cal = calibration_intercept_slope(y, oof)
        pct_below_8pct = float((oof < 0.08).mean()) if len(oof) > 0 else np.nan

        youden_thresh = (
            artifacts.best_youden_threshold
            if artifacts.best_youden_threshold is not None
            else np.nan
        )

        row = {
            "run": run_idx + 1,
            "seed": seed,
            "best_model": best,
            "AUC": auc,
            "AUPRC": auprc,
            "Brier": brier,
            "Calibration_intercept": cal["Calibration intercept"],
            "Calibration_slope": cal["Calibration slope"],
            "pct_oof_below_8pct": pct_below_8pct,
            "Youden_threshold": youden_thresh,
        }
    finally:
        AppConfig.RANDOM_SEED = original_seed

    return row


def run_stability_audit(
    dataset_path: str,
    n_runs: int,
    output_path: str,
) -> pd.DataFrame:
    """Run the stability audit and return the results DataFrame."""
    print(f"Loading dataset: {dataset_path}")
    prepared = prepare_master_dataset(dataset_path)
    df = prepared.data
    feature_columns = prepared.feature_columns
    print(
        f"  {len(df)} rows, {len(feature_columns)} features, "
        f"prevalence={df['morte_30d'].mean():.1%}"
    )

    seeds = [_BASE_SEED + i for i in range(n_runs)]
    rows = []

    _have_tqdm = False
    try:
        from tqdm import tqdm
        _have_tqdm = True
    except ImportError:
        pass

    iterable = range(n_runs)
    if _have_tqdm:
        iterable = tqdm(iterable, desc="Stability runs", unit="run")

    for i in iterable:
        seed = seeds[i]
        if not _have_tqdm:
            print(f"  Run {i + 1}/{n_runs} (seed={seed}) ...", end=" ", flush=True)
        try:
            row = _run_once(df, feature_columns, seed, i, n_runs)
            rows.append(row)
            if not _have_tqdm:
                print(f"winner={row['best_model']}, AUC={row['AUC']:.4f}")
        except Exception as exc:
            print(f"\n  [WARN] Run {i + 1} failed: {exc}", file=sys.stderr)
            traceback.print_exc()

    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    if results.empty:
        print("No successful runs.")
        return results

    print("\n=== Model selection frequency ===")
    freq = results["best_model"].value_counts()
    for model, count in freq.items():
        print(f"  {model}: {count}/{len(results)} runs ({count / len(results):.0%})")

    print("\n=== Metric summary (mean ± std, range) ===")
    for metric in ["AUC", "AUPRC", "Brier", "Calibration_intercept", "Calibration_slope"]:
        if metric in results.columns:
            s = results[metric].dropna()
            if not s.empty:
                print(
                    f"  {metric}: "
                    f"{s.mean():.4f} ± {s.std():.4f}  "
                    f"[{s.min():.4f}, {s.max():.4f}]"
                )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Audit model selection stability across random seeds."
    )
    parser.add_argument(
        "--dataset",
        default=_DEFAULT_DATASET,
        help=f"Path to the dataset XLSX (default: {_DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=_DEFAULT_N_RUNS,
        help=f"Number of runs (default: {_DEFAULT_N_RUNS})",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"Error: dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    run_stability_audit(args.dataset, args.n_runs, args.output)


if __name__ == "__main__":
    main()
