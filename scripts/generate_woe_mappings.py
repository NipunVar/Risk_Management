# scripts/generate_woe_mappings.py
"""
Generate WoE mappings for the features used by the model.
Input: training CSV that contains aggregated features (or raw transactions + agg step).
You must have a target column named 'is_high_risk' (0/1) in the training aggregated data.
Output: saves mappings to models/woe_mappings.joblib
"""
import joblib
import pandas as pd
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Edit this path to point to your training aggregated CSV (or a file you will create by aggregating)
TRAIN_AGG_CSV = PROJECT_ROOT / "notebooks" / "train_aggregated.csv"  # produce this by running your training aggregation pipeline

# Features you want WoE for (the model expects these)
FEATURES_TO_BIN = [
    "Recency", "Frequency", "MeanAmount", "StdAmount", "AvgTransactionHour",
    "TotalDebits", "DebitCount", "CreditCount", "TransactionVolatility",
    "MonetaryAmount", "NetCashFlow", "DebitCreditRatio"
]

NUM_BINS = 5
OUTPUT_PATH = MODELS_DIR / "woe_mappings.joblib"

def create_binned_column(series, q=NUM_BINS):
    # use qcut but allow duplicates removal
    try:
        return pd.qcut(series, q=q, labels=False, duplicates="drop")
    except Exception:
        # fallback to rank-based bins
        return pd.cut(series.rank(method="dense"), bins=q, labels=False)

def compute_woe(df, binned_col, target_col):
    # returns dict mapping bin -> WoE
    total_good = df[target_col].sum()
    total_bad = df[target_col].count() - total_good
    woe_map = {}
    for bin_val in sorted(df[binned_col].dropna().unique()):
        bin_df = df[df[binned_col] == bin_val]
        good = bin_df[target_col].sum()
        bad = bin_df[target_col].count() - good
        # Avoid zero division
        if good == 0 or bad == 0:
            woe_map[int(bin_val)] = 0.0
        else:
            woe_map[int(bin_val)] = float(np.log((good / total_good) / (bad / total_bad)))
    return woe_map

def main():
    if not TRAIN_AGG_CSV.exists():
        raise FileNotFoundError(f"Training aggregated CSV not found at {TRAIN_AGG_CSV}. Create it using your aggregation pipeline.")

    df = pd.read_csv(TRAIN_AGG_CSV)
    if "is_high_risk" not in df.columns:
        raise ValueError("Training aggregated CSV must have 'is_high_risk' target column (0/1).")

    mappings = {}
    for feat in FEATURES_TO_BIN:
        if feat not in df.columns:
            print(f"[WARN] {feat} not in training data columns; skipping.")
            continue
        binned_name = f"{feat}_binned"
        df[binned_name] = create_binned_column(df[feat], q=NUM_BINS)
        woe_map = compute_woe(df, binned_name, "is_high_risk")
        mappings[feat] = {
            "woe_map": woe_map,
            "bins_used": NUM_BINS
        }
        print(f"Computed WoE for {feat}: bins {sorted(df[binned_name].dropna().unique())}")

    joblib.dump(mappings, OUTPUT_PATH)
    print("Saved WoE mappings to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
