"""Ablation: Previous surgery free-text vs structured binary/count features.

Compares two representations of Prior Surgery across multiple seeds:

  Baseline:    Previous surgery (free-text TargetEncoder, current default)
  Structured:  previous_surgery_any + previous_surgery_count_est +
               previous_surgery_has_combined (binary/numeric, no TargetEncoder)

Runs each variant with N seeds (default 20) and reports mean ± std of AUC,
AUPRC, and Brier. A lower std in the structured variant would indicate more
stable model selection.

Usage:
    python ablation_previous_surgery_structured.py
    python ablation_previous_surgery_structured.py --n-runs 10 --dataset local_data/Dataset_2025.xlsx
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from config.base_config import AppConfig
from modeling import train_and_select_model
from risk_data import prepare_master_dataset, NEVER_FEATURE_COLUMNS

_DEFAULT_DATASET = "local_data/Dataset_2025.xlsx"
_DEFAULT_N_RUNS = 20
_BASE_SEED = 42

_STRUCTURED_COLS = [
    "previous_surgery_any",
    "previous_surgery_count_est",
    "previous_surgery_has_combined",
]
_BASELINE_COL = "Previous surgery"


def _run_variant(
    df: pd.DataFrame,
    feature_columns: list,
    seed: int,
    variant_name: str,
) -> dict:
    original_seed = AppConfig.RANDOM_SEED
    try:
        AppConfig.RANDOM_SEED = seed
        artifacts = train_and_select_model(df, feature_columns)
        best = artifacts.best_model_name
        lb_row = artifacts.leaderboard.loc[artifacts.leaderboard["Modelo"] == best]
        auc = float(lb_row["AUC"].iloc[0]) if not lb_row.empty else np.nan
        auprc = float(lb_row["AUPRC"].iloc[0]) if not lb_row.empty else np.nan
        brier = float(lb_row["Brier"].iloc[0]) if not lb_row.empty else np.nan
        return {
            "variant": variant_name,
            "seed": seed,
            "best_model": best,
            "AUC": auc,
            "AUPRC": auprc,
            "Brier": brier,
        }
    finally:
        AppConfig.RANDOM_SEED = original_seed


def run_ablation(dataset_path: str, n_runs: int, output_path: str) -> pd.DataFrame:
    print(f"Loading dataset: {dataset_path}")
    prepared = prepare_master_dataset(dataset_path)
    df = prepared.data
    base_feature_columns = prepared.feature_columns

    # Verify structured columns exist (they should — computed by pipeline)
    missing_structured = [c for c in _STRUCTURED_COLS if c not in df.columns]
    if missing_structured:
        print(f"Warning: structured columns missing from data: {missing_structured}")
        print("Cannot run structured variant.")
        sys.exit(1)

    # Build structured feature set: remove Previous surgery, add structured cols
    structured_features = [c for c in base_feature_columns if c != _BASELINE_COL]
    for col in _STRUCTURED_COLS:
        if col not in structured_features and col in df.columns:
            structured_features.append(col)
    # Belt-and-suspenders: never include NEVER_FEATURE_COLUMNS
    structured_features = [c for c in structured_features if c not in NEVER_FEATURE_COLUMNS]

    seeds = [_BASE_SEED + i for i in range(n_runs)]
    rows = []

    _have_tqdm = False
    try:
        from tqdm import tqdm
        _have_tqdm = True
    except ImportError:
        pass

    for variant, feat_cols in [
        ("baseline", base_feature_columns),
        ("structured", structured_features),
    ]:
        print(f"\n--- Variant: {variant} ({len(feat_cols)} features) ---")
        iterable = range(n_runs)
        if _have_tqdm:
            iterable = tqdm(iterable, desc=f"{variant}", unit="run")

        for i in iterable:
            seed = seeds[i]
            if not _have_tqdm:
                print(f"  Run {i + 1}/{n_runs} (seed={seed}) ...", end=" ", flush=True)
            try:
                row = _run_variant(df, feat_cols, seed, variant)
                rows.append(row)
                if not _have_tqdm:
                    print(f"winner={row['best_model']}, AUC={row['AUC']:.4f}")
            except Exception as exc:
                print(f"\n  [WARN] Run {i + 1} failed: {exc}", file=sys.stderr)
                traceback.print_exc()

    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print("\n=== Summary by variant ===")
    for variant, grp in results.groupby("variant"):
        print(f"\nVariant: {variant}")
        for metric in ["AUC", "AUPRC", "Brier"]:
            s = grp[metric].dropna()
            if not s.empty:
                print(
                    f"  {metric}: {s.mean():.4f} ± {s.std():.4f}  "
                    f"[{s.min():.4f}, {s.max():.4f}]"
                )
        print(f"  Model wins: {grp['best_model'].value_counts().to_dict()}")

    # Compare variance
    if len(results["variant"].unique()) == 2:
        print("\n=== Variance comparison (structured vs baseline) ===")
        for metric in ["AUC", "AUPRC", "Brier"]:
            std_base = results.loc[results["variant"] == "baseline", metric].std()
            std_struct = results.loc[results["variant"] == "structured", metric].std()
            if std_base > 0:
                rel = (std_struct - std_base) / std_base * 100
                sign = "LOWER" if rel < 0 else "HIGHER"
                print(
                    f"  {metric}: structured std={std_struct:.4f}, baseline std={std_base:.4f}  "
                    f"→ {sign} by {abs(rel):.1f}%"
                )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: Previous surgery structured vs free-text."
    )
    parser.add_argument("--dataset", default=_DEFAULT_DATASET)
    parser.add_argument("--n-runs", type=int, default=_DEFAULT_N_RUNS)
    parser.add_argument(
        "--output",
        default="ablation_previous_surgery_structured_results.csv",
    )
    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"Error: dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    run_ablation(args.dataset, args.n_runs, args.output)


if __name__ == "__main__":
    main()
