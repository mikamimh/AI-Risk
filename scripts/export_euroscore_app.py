"""Export, per patient, the EuroSCORE II exactly as the app computes it today.

Reuses the app's own data path without reimplementing anything:
  * loads the cached bundle (``AppConfig.MODEL_CACHE_FILE``) and checks that its
    signature matches ``local_data/Dataset_2025.xlsx`` (same check as the app);
  * takes ``bundle["data"]`` (output of ``risk_data.prepare_master_dataset`` +
    STS Score from the web calculator, as stored by the app);
  * recomputes ``euroscore_calc`` with ``euroscore.euroscore_from_row`` and
    fills ``euroscore_sheet_clean`` exactly as ``app.py`` does after loading.

Writes ``local_data/export_euroscore_app.csv`` (patient data — never commit it)
and prints the AUC of ``euroscore_calc`` vs ``morte_30d`` in the triple cohort
(``sts_score`` not null), the cohort used by the comparison tab.

Usage (from the repo root):
    python scripts/export_euroscore_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from bundle_io import bundle_signature, deserialize_bundle, normalize_payload  # noqa: E402
from config import AppConfig  # noqa: E402
from euroscore import euroscore_from_row  # noqa: E402

XLSX_PATH = ROOT / "local_data" / "Dataset_2025.xlsx"
OUT_PATH = ROOT / "local_data" / "export_euroscore_app.csv"
BUNDLE_PATH = ROOT / AppConfig.MODEL_CACHE_FILE


def load_app_dataframe() -> pd.DataFrame:
    payload = joblib.load(BUNDLE_PATH)
    expected = bundle_signature(str(XLSX_PATH))
    if payload.get("signature") != expected:
        raise SystemExit(
            "Cached bundle signature does not match the current dataset/model version.\n"
            f"  bundle:   {payload.get('signature')}\n  expected: {expected}\n"
            "Retrain in the app first so the export reflects what the app shows."
        )
    bundle = deserialize_bundle(normalize_payload(payload)["bundle"])
    df = bundle["data"].copy()

    # Same post-load steps as app.py (base_df preparation).
    df["euroscore_calc"] = df.apply(euroscore_from_row, axis=1)
    if "euroscore_sheet_clean" not in df.columns:
        df["euroscore_sheet_clean"] = pd.to_numeric(df.get("euroscore_sheet"), errors="coerce")
    if "sts_score" not in df.columns:
        df["sts_score"] = np.nan
    return df


def main() -> None:
    df = load_app_dataframe()

    out = pd.DataFrame(
        {
            "patient_id": df["Name"],
            "morte_30d": df["morte_30d"],
            "euroscore_calc": df["euroscore_calc"],
            "euroscore_sheet_clean": df["euroscore_sheet_clean"],
            "sts_score": df["sts_score"],
        }
    )
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved {len(out)} rows -> {OUT_PATH}")

    triple = out[out["sts_score"].notna()]
    valid = triple[["morte_30d", "euroscore_calc"]].dropna()
    auc = roc_auc_score(valid["morte_30d"].astype(int), valid["euroscore_calc"])
    print(
        f"Triple cohort (sts_score not null): n={len(triple)}, "
        f"n with euroscore_calc={len(valid)}, deaths={int(valid['morte_30d'].sum())}"
    )
    print(f"AUC euroscore_calc vs morte_30d: {auc:.4f} (rounded {auc:.3f})")


if __name__ == "__main__":
    main()
