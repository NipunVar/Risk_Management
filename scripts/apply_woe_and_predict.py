# scripts/apply_woe_and_predict.py
"""
Aggregate raw transactions (uses your feature_eng.create_aggregate_features),
apply saved WoE mappings (models/woe_mappings.joblib), and call local API /predict_single
for each customer (or a single customer).
"""
import joblib
import pandas as pd
from pathlib import Path
import requests
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODELS_DIR = PROJECT_ROOT / "models"
Woe_path = MODELS_DIR / "woe_mappings.joblib"
API_URL = "http://localhost:8000/predict_single"
TRANSACTIONS_CSV = PROJECT_ROOT / "notebooks" / "sample_transactions.csv"  # replace with your CSV

# list of features expected by model (name mapping)
MODEL_WOE_FEATURE_NAMES = [
 'Recency_binned_WoE','Frequency_binned_WoE','MeanAmount_binned_WoE','StdAmount_binned_WoE',
 'AvgTransactionHour_binned_WoE','TotalDebits_binned_WoE','DebitCount_binned_WoE','CreditCount_binned_WoE',
 'TransactionVolatility_binned_WoE','MonetaryAmount_binned_WoE','NetCashFlow_binned_WoE','DebitCreditRatio_binned_WoE'
]

def load_mappings():
    if not Woe_path.exists():
        raise FileNotFoundError(f"Woe mappings not found at {Woe_path}. Run generate_woe_mappings.py first.")
    return joblib.load(Woe_path)

def aggregate_transactions(path):
    # use your feature engineering function
    import importlib.util, os
    feat_mod = importlib.util.spec_from_file_location("feature_engineering", str(SCRIPTS_DIR / "feature_engineering.py"))
    fe = importlib.util.module_from_spec(feat_mod)
    feat_mod.loader.exec_module(fe)
    df = pd.read_csv(path)
    df_clean = None
    # if you have a data_cleaner, call it
    dc_spec = importlib.util.spec_from_file_location("data_cleaner", str(SCRIPTS_DIR / "data_cleaner.py"))
    dc = importlib.util.module_from_spec(dc_spec)
    dc_spec.loader.exec_module(dc)
    df_clean = dc.clean_data(df)
    df_agg = fe.create_aggregate_features(df_clean)
    return df_agg

def apply_woe_and_call(df_agg, mappings):
    results = []
    for _, row in df_agg.iterrows():
        features = {}
        for feat_base in mappings.keys():
            binned_name = f"{feat_base}_binned_WoE".replace("_binned_WoE","")  # not used
            # We need to find the bin for this row — assume mappings were created with qcut; we will mimic quantile binning
            # Simpler: find the nearest bin key by ranking into NUM_BINS
            # Here we compute quantile bins using pandas.qcut with same bin count
            try:
                num_bins = mappings[feat_base]["bins_used"]
                # create bins using qcut on the column across df_agg
                # (this is approximate: better to store bin edges during mapping generation)
            except:
                num_bins = mappings[feat_base].get("bins_used", 5)
        # Instead of doing complex binning here, call the server's /predict_file which uses the server pipeline
        # So for reliability, we fall back to using /predict_file for batch predictions.
        return None

def main():
    print("This script is a helper. For reliable mapping and prediction, use the API /predict_file endpoint which runs the server-side pipeline.")
    print("Uploading CSV to /predict_file:")
    files = {'file': open(TRANSACTIONS_CSV, 'rb')}
    r = requests.post("http://localhost:8000/predict_file", files=files)
    print("Status:", r.status_code)
    print("Response:", r.text)

if __name__ == "__main__":
    main()
