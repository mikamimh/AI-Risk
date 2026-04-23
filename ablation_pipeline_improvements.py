"""Ablation: 5 pipeline improvement hypotheses tested with N seeds.

Compares baseline pipeline against feature representation changes:
  V1: Surgery text → procedure_group (9 categories)
  V2: Binary Yes/No → direct 0/1 numeric encoding
  V3: V1 + V2 combined
  V4: Echo missingness indicator added
  V5: V1 + V2 + V4 (all improvements)

Each variant trained with N seeds (default 20). Outputs CSV + console summary.

Usage:
    python ablation_pipeline_improvements.py
    python ablation_pipeline_improvements.py --n-runs 10
    python ablation_pipeline_improvements.py --dataset local_data/Dataset_2025.xlsx --output my_results.csv
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from config.base_config import AppConfig
from modeling import train_and_select_model
from risk_data import (
    NEVER_FEATURE_COLUMNS,
    is_missing,
    prepare_master_dataset,
)
from stats_compare import calibration_intercept_slope

_DEFAULT_DATASET = "local_data/Dataset_2025.xlsx"
_DEFAULT_N_RUNS = 20
_DEFAULT_OUTPUT = "ablation_pipeline_improvements_results.csv"
_BASE_SEED = 42

# Binary columns: variables that have exactly 2 non-null unique values after
# cleaning (typically Yes/No). These go through TargetEncoder in the current
# pipeline but could be encoded as 0/1 numeric.
_KNOWN_BINARY_COLS = {
    "IE",
    "Left Main Stenosis ≥ 50%",
    "Proximal LAD Stenosis ≥ 70%",
    "CCS4",
    "Hypertension",
    "Diabetes",
    "Dyslipidemia",
    "CVA",
    "PVD",
    "Alcohol",
    "Cancer ≤ 5 yrs",
    "Family Hx of CAD",
    "Anticoagulation/ Antiaggregation",
    "Pneumonia",
    "Dialysis",
    "Chronic Lung Disease",
    "Critical preoperative state",
    "Poor mobility",
}


def _convert_binary_to_numeric(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Convert known binary categorical columns to 0/1 numeric.

    Positive tokens (Yes, Sim, True, 1) → 1.0
    Negative tokens (No, Não, False, 0) → 0.0
    Everything else (NaN, unknown) → NaN (will be median-imputed later)
    """
    out = df.copy()
    _positive = {"yes", "sim", "true", "1", "1.0", "treated", "active", "possible"}
    _negative = {"no", "não", "nao", "false", "0", "0.0"}

    for col in feature_columns:
        if col not in _KNOWN_BINARY_COLS or col not in out.columns:
            continue
        s = out[col].astype(str).str.strip().str.lower()
        numeric = pd.Series(np.nan, index=out.index)
        numeric[s.isin(_positive)] = 1.0
        numeric[s.isin(_negative)] = 0.0
        out[col] = numeric
    return out


def _add_echo_missingness_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing_echo_key: 1 if ALL three key echo variables are missing."""
    out = df.copy()
    _echo_key_cols = ["Pré-LVEF, %", "Aortic Stenosis", "Mitral Regurgitation"]
    all_missing = pd.Series(True, index=out.index)
    for col in _echo_key_cols:
        if col in out.columns:
            col_missing = out[col].apply(lambda v, c=col: is_missing(v, column=c))
            all_missing = all_missing & col_missing
        # If column doesn't exist, treat as missing (True) — already handled by init True
    out["missing_echo_key"] = all_missing.astype(int)
    return out


def _build_variant(
    df: pd.DataFrame,
    base_features: list,
    variant: str,
) -> tuple:
    """Return (modified_df, modified_feature_columns) for a given variant."""
    features = list(base_features)
    out = df.copy()

    if variant == "baseline":
        return out, features

    apply_v1 = variant in ("V1_procedure_group", "V3_v1_plus_v2", "V5_full")
    apply_v2 = variant in ("V2_binary_direct", "V3_v1_plus_v2", "V5_full")
    apply_v4 = variant in ("V4_echo_missing", "V5_full")

    # V1: Replace Surgery text with procedure_group (9-category taxonomy)
    if apply_v1:
        if "Surgery" in features:
            features.remove("Surgery")
        if "procedure_group" in out.columns and "procedure_group" not in features:
            features.append("procedure_group")

    # V2: Convert known binary columns to 0/1 numeric
    if apply_v2:
        out = _convert_binary_to_numeric(out, features)

    # V4: Add echo missingness composite indicator
    if apply_v4:
        out = _add_echo_missingness_indicator(out)
        if "missing_echo_key" not in features:
            features.append("missing_echo_key")

    # Belt-and-suspenders: remove never-feature cols and non-existent cols
    features = [c for c in features if c not in NEVER_FEATURE_COLUMNS and c in out.columns]

    return out, features


def _run_variant(
    df: pd.DataFrame,
    feature_columns: list,
    seed: int,
    variant_name: str,
) -> dict:
    """Train one variant with one seed and return metrics dict."""
    original_seed = AppConfig.RANDOM_SEED
    try:
        AppConfig.RANDOM_SEED = seed
        artifacts = train_and_select_model(df, feature_columns)
        y = df["morte_30d"].astype(int).values
        best = artifacts.best_model_name
        oof = artifacts.oof_predictions.get(best, np.full(len(y), np.nan))

        lb_row = artifacts.leaderboard.loc[artifacts.leaderboard["Modelo"] == best]
        auc = float(lb_row["AUC"].iloc[0]) if not lb_row.empty else np.nan
        auprc = float(lb_row["AUPRC"].iloc[0]) if not lb_row.empty else np.nan
        brier = float(lb_row["Brier"].iloc[0]) if not lb_row.empty else np.nan

        cal = calibration_intercept_slope(y, oof)
        pct_below_8 = float((oof < 0.08).mean()) if len(oof) > 0 else np.nan

        return {
            "variant": variant_name,
            "seed": seed,
            "best_model": best,
            "n_features": len(feature_columns),
            "AUC": auc,
            "AUPRC": auprc,
            "Brier": brier,
            "Calibration_intercept": cal["Calibration intercept"],
            "Calibration_slope": cal["Calibration slope"],
            "pct_oof_below_8pct": pct_below_8,
        }
    finally:
        AppConfig.RANDOM_SEED = original_seed


def run_ablation(dataset_path: str, n_runs: int, output_path: str) -> pd.DataFrame:
    """Run the full ablation and return the results DataFrame."""
    print(f"Loading dataset: {dataset_path}")
    prepared = prepare_master_dataset(dataset_path)
    df = prepared.data
    base_features = prepared.feature_columns
    print(
        f"  {len(df)} rows, {len(base_features)} features, "
        f"prevalence={df['morte_30d'].mean():.1%}"
    )

    variants = [
        "baseline",
        "V1_procedure_group",
        "V2_binary_direct",
        "V3_v1_plus_v2",
        "V4_echo_missing",
        "V5_full",
    ]

    seeds = [_BASE_SEED + i for i in range(n_runs)]
    rows = []

    _have_tqdm = False
    try:
        from tqdm import tqdm
        _have_tqdm = True
    except ImportError:
        pass

    for variant in variants:
        df_v, features_v = _build_variant(df, base_features, variant)
        print(f"\n--- {variant} ({len(features_v)} features) ---")

        iterable = range(n_runs)
        if _have_tqdm:
            iterable = tqdm(iterable, desc=variant, unit="run")

        for i in iterable:
            seed = seeds[i]
            if not _have_tqdm:
                print(f"  Run {i + 1}/{n_runs} (seed={seed}) ...", end=" ", flush=True)
            try:
                row = _run_variant(df_v, features_v, seed, variant)
                rows.append(row)
                if not _have_tqdm:
                    print(
                        f"winner={row['best_model']}, "
                        f"AUC={row['AUC']:.4f}, "
                        f"slope={row['Calibration_slope']:.3f}"
                    )
            except Exception as exc:
                print(f"\n  [WARN] Run {i + 1} failed: {exc}", file=sys.stderr)
                traceback.print_exc()

    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)

    summary_rows = []
    for variant in variants:
        grp = results[results["variant"] == variant]
        if grp.empty:
            continue
        top_model_counts = grp["best_model"].value_counts()
        summary_rows.append({
            "Variant": variant,
            "N_features": int(grp["n_features"].iloc[0]),
            "AUC_mean": grp["AUC"].mean(),
            "AUC_std": grp["AUC"].std(),
            "AUPRC_mean": grp["AUPRC"].mean(),
            "AUPRC_std": grp["AUPRC"].std(),
            "Brier_mean": grp["Brier"].mean(),
            "Brier_std": grp["Brier"].std(),
            "Slope_mean": grp["Calibration_slope"].mean(),
            "Slope_std": grp["Calibration_slope"].std(),
            "Intercept_mean": grp["Calibration_intercept"].mean(),
            "pct_below_8_mean": grp["pct_oof_below_8pct"].mean(),
            "Top_model": top_model_counts.index[0],
            "Top_model_pct": top_model_counts.iloc[0] / len(grp),
        })

    summary = pd.DataFrame(summary_rows)

    # Performance table
    print(
        f"\n{'Variant':<25s} {'N_feat':>6s} "
        f"{'AUC':>12s} {'Brier':>12s} {'Slope':>12s} {'Top model':>18s}"
    )
    print("-" * 90)
    for _, r in summary.iterrows():
        print(
            f"{r['Variant']:<25s} "
            f"{r['N_features']:>6.0f} "
            f"{r['AUC_mean']:.4f}±{r['AUC_std']:.4f} "
            f"{r['Brier_mean']:.4f}±{r['Brier_std']:.4f} "
            f"{r['Slope_mean']:.3f}±{r['Slope_std']:.3f} "
            f"{r['Top_model']:>14s} ({r['Top_model_pct']:.0%})"
        )

    # Delta table vs baseline
    if "baseline" in summary["Variant"].values:
        base = summary[summary["Variant"] == "baseline"].iloc[0]
        print(
            f"\n{'Variant':<25s} {'ΔAUC':>9s} {'ΔAUPRC':>9s} "
            f"{'ΔBrier':>8s} {'ΔSlope':>8s} {'ΔStd(AUC)':>10s}"
        )
        print("-" * 73)
        for _, r in summary.iterrows():
            if r["Variant"] == "baseline":
                continue
            d_auc = r["AUC_mean"] - base["AUC_mean"]
            d_auprc = r["AUPRC_mean"] - base["AUPRC_mean"]
            d_brier = r["Brier_mean"] - base["Brier_mean"]
            d_slope = r["Slope_mean"] - base["Slope_mean"]
            d_std = r["AUC_std"] - base["AUC_std"]
            print(
                f"{r['Variant']:<25s} "
                f"{d_auc:>+.4f} "
                f"{d_auprc:>+.4f} "
                f"{d_brier:>+.4f} "
                f"{d_slope:>+.4f} "
                f"{d_std:>+.4f}"
            )

        print("\nLegenda dos deltas (vs baseline):")
        print("  ΔAUC > 0     → melhor discriminação")
        print("  ΔAUPRC > 0   → melhor discriminação nos eventos")
        print("  ΔBrier < 0   → melhor calibração probabilística")
        print("  ΔSlope → 0.0 que aproxima slope de 1.0 = melhor calibração logit")
        print("  ΔStd(AUC) < 0 → mais estável entre seeds")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: 5 pipeline improvement hypotheses."
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
        help=f"Seeds to run per variant (default: {_DEFAULT_N_RUNS})",
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

    run_ablation(args.dataset, args.n_runs, args.output)


if __name__ == "__main__":
    main()
