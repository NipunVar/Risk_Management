# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
import importlib.util
import time
from io import BytesIO
import requests

# -------------------------
# Configuration / paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODELS_DIR = PROJECT_ROOT / "models"

CLASSIFIER_FILE = MODELS_DIR / "credit_risk_prob_model.pkl"
SCORE_FILE = MODELS_DIR / "credit_score_model.pkl"

SAMPLE_CSV = PROJECT_ROOT / "notebooks" / "sample_transactions.csv"

# -------------------------
# Dynamic import helper
# -------------------------
def import_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

data_cleaner = None
feature_eng = None
credit_score_mod = None
try:
    if (SCRIPTS_DIR / "data_cleaner.py").exists():
        data_cleaner = import_from_path("data_cleaner", SCRIPTS_DIR / "data_cleaner.py")
    if (SCRIPTS_DIR / "feature_engineering.py").exists():
        feature_eng = import_from_path("feature_engineering", SCRIPTS_DIR / "feature_engineering.py")
    if (SCRIPTS_DIR / "credit_score.py").exists():
        credit_score_mod = import_from_path("credit_score", SCRIPTS_DIR / "credit_score.py")
except Exception as e:
    st.warning(f"Could not import scripts: {e}")

# -------------------------
# Load models
# -------------------------
@st.cache_resource
def load_models():
    clf = None
    scorer = None
    clf_loaded = False
    scorer_loaded = False
    if CLASSIFIER_FILE.exists():
        try:
            clf = joblib.load(CLASSIFIER_FILE)
            clf_loaded = True
        except Exception as e:
            st.error(f"Failed to load classifier: {e}")
    if SCORE_FILE.exists():
        try:
            scorer = joblib.load(SCORE_FILE)
            scorer_loaded = True
        except Exception as e:
            st.error(f"Failed to load score model: {e}")
    return clf, scorer, clf_loaded, scorer_loaded

clf, scorer, clf_loaded, scorer_loaded = load_models()

# -------------------------
# Model WOE features
# -------------------------
MODEL_WOE_FEATURES = [
 'Recency_binned_WoE','Frequency_binned_WoE','MeanAmount_binned_WoE','StdAmount_binned_WoE',
 'AvgTransactionHour_binned_WoE','TotalDebits_binned_WoE','DebitCount_binned_WoE','CreditCount_binned_WoE',
 'TransactionVolatility_binned_WoE','MonetaryAmount_binned_WoE','NetCashFlow_binned_WoE','DebitCreditRatio_binned_WoE'
]

# -------------------------
# Utility functions
# -------------------------
def model_metadata():
    files = []
    if MODELS_DIR.exists():
        for p in MODELS_DIR.glob("*"):
            if p.is_file():
                stat = p.stat()
                files.append({
                    "filename": p.name,
                    "size_kb": round(stat.st_size/1024, 2),
                    "modified": time.ctime(stat.st_mtime),
                    "path": str(p)
                })
    return files

def predict_from_features(features: dict):
    df = pd.DataFrame([features])
    if clf is not None:
        X = df[MODEL_WOE_FEATURES].fillna(0)
        try:
            df["risk_probability"] = clf.predict_proba(X)[:,1]
        except Exception:
            df["risk_probability"] = clf.predict(X)
    else:
        df["risk_probability"] = 0.5

    if scorer is not None:
        try:
            Xs = df[MODEL_WOE_FEATURES].fillna(0)
            df["credit_score"] = scorer.predict(Xs).round().astype(int)
        except Exception:
            if credit_score_mod is not None:
                df = credit_score_mod.assign_credit_score(df)
    else:
        if credit_score_mod is not None:
            df = credit_score_mod.assign_credit_score(df)

    return df

def predict_from_transactions_csv(df_transactions: pd.DataFrame):
    api_url = "http://localhost:8000/predict_file"
    csv_bytes = df_transactions.to_csv(index=False).encode("utf-8")
    files = {"file": ("upload.csv", BytesIO(csv_bytes), "text/csv")}
    try:
        r = requests.post(api_url, files=files, timeout=30)
        r.raise_for_status()
    except Exception as e:
        st.error(f"Failed to call API {api_url}: {e}")
        return None
    result = r.json()
    df_out = pd.DataFrame(result.get("predictions", []))
    return df_out

# -------------------------
# Sanitizer (avoid pyarrow conversion errors)
# -------------------------
def sanitize_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    df2 = df.copy()
    for col in df2.columns:
        if df2[col].dtype == object:
            df2[col] = df2[col].fillna("").astype(str)
            continue
        try:
            converted = pd.to_numeric(df2[col], errors="coerce")
            if converted.notna().sum() / max(1, len(converted)) >= 0.95:
                df2[col] = converted
            else:
                df2[col] = df2[col].astype(str)
        except Exception:
            df2[col] = df2[col].astype(str)
    df2 = df2.reset_index(drop=True)
    return df2

# -------------------------
# IMAGE: find all PNGs in project root
# -------------------------
def find_pngs_in_root(root: Path):
    pngs = []
    for p in sorted(root.glob("*.png")):
        pngs.append(p)
    return pngs

PROJECT_PNGS = find_pngs_in_root(PROJECT_ROOT)  # list of Path objects

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Credit Risk - Streamlit UI", layout="wide")
st.title("Credit Risk Scoring")

# Sidebar header & mode
st.sidebar.header("Options")
mode = st.sidebar.radio("Mode", ["Single (WoE features)", "Batch from CSV", "Models info", "Quick demo sample CSV"])

# Sidebar: list model files
st.sidebar.markdown("**Model files**")
md = model_metadata()
if md:
    for m in md:
        st.sidebar.markdown(f"- **{m['filename']}** · {m['size_kb']} KB · modified {m['modified']}")
else:
    st.sidebar.markdown("- No models found in /models")

# -------------------------
# Sidebar: one expander per PNG (name -> image)
# -------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Project images")
if PROJECT_PNGS:
    for img_path in PROJECT_PNGS:
        name = img_path.stem  # filename without extension
        with st.sidebar.expander(name, expanded=False):
            try:
                st.image(str(img_path), use_container_width=True)
            except Exception as e:
                st.write(f"Unable to display `{img_path.name}`: {e}")
else:
    st.sidebar.markdown("_No .png files found in project root_")

# Also offer a toggle to show images inline (main panel)
show_images_main = st.sidebar.checkbox("Show images in main panel", value=False)

if show_images_main and PROJECT_PNGS:
    st.header("Project images")
    for img_path in PROJECT_PNGS:
        st.subheader(img_path.stem)
        try:
            st.image(str(img_path), use_container_width=True)
        except Exception as e:
            st.write(f"Unable to display `{img_path.name}`: {e}")

# -------------------------
# MODE: Models info
# -------------------------
if mode == "Models info":
    st.header("Model metadata")
    df_md = pd.DataFrame(md) if md else pd.DataFrame(columns=["filename", "size_kb", "modified", "path"])
    st.dataframe(sanitize_df_for_display(df_md))
    st.write("Classifier loaded:", bool(clf_loaded))
    st.write("Score model loaded:", bool(scorer_loaded))

# -------------------------
# MODE: Single (WoE features)
# -------------------------
elif mode == "Single (WoE features)":
    st.header("Single prediction — provide WoE features")
    st.markdown("Provide the 12 WoE-transformed features exactly as used in training.")
    cols = st.columns(3)
    default_vals = {
        'Recency_binned_WoE': -0.12, 'Frequency_binned_WoE': 0.45, 'MeanAmount_binned_WoE': 0.10,
        'StdAmount_binned_WoE': -0.05, 'AvgTransactionHour_binned_WoE': 0.02, 'TotalDebits_binned_WoE': 0.30,
        'DebitCount_binned_WoE': -0.10, 'CreditCount_binned_WoE': 0.05, 'TransactionVolatility_binned_WoE': 0.12,
        'MonetaryAmount_binned_WoE': 0.20, 'NetCashFlow_binned_WoE': -0.08, 'DebitCreditRatio_binned_WoE': 0.07
    }
    inputs = {}
    for i, feat in enumerate(MODEL_WOE_FEATURES):
        with cols[i % 3]:
            inputs[feat] = st.number_input(feat, value=float(default_vals.get(feat, 0.0)), format="%.6f")

    if st.button("Predict single"):
        with st.spinner("Running prediction..."):
            out = predict_from_features(inputs)
            st.success("Prediction complete")
            out_display = sanitize_df_for_display(out)
            st.dataframe(out_display.T)

# -------------------------
# MODE: Batch from CSV
# -------------------------
elif mode == "Batch from CSV":
    st.header("Batch prediction from transactions CSV")
    st.markdown("Upload a transactions CSV. The pipeline will clean, aggregate, compute features and predict per CustomerId.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("Use sample CSV file from project (notebooks/sample_transactions.csv)", value=False)
    if use_sample:
        if SAMPLE_CSV.exists():
            uploaded_file = open(SAMPLE_CSV, "rb")
        else:
            st.error(f"Sample CSV not found at {SAMPLE_CSV}")

    if uploaded_file is not None:
        if isinstance(uploaded_file, (str, bytes, BytesIO)):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_csv(uploaded_file)
        st.write("Raw data preview")
        st.dataframe(sanitize_df_for_display(df_raw.head(10)))

        if st.button("Run batch prediction"):
            with st.spinner("Running cleaning, feature engineering and predictions..."):
                df_out = predict_from_transactions_csv(df_raw)
                if df_out is None:
                    st.error("Prediction pipeline failed (feature_engineering or data_cleaner not found).")
                else:
                    st.success("Batch prediction completed")
                    df_out_display = sanitize_df_for_display(df_out)
                    st.dataframe(df_out_display.head(50))
                    csv = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")

# -------------------------
# MODE: Quick demo sample CSV
# -------------------------
elif mode == "Quick demo sample CSV":
    st.header("Quick demo using sample_transactions.csv")
    st.markdown(f"Sample path used: `{SAMPLE_CSV}`")
    if SAMPLE_CSV.exists():
        if st.button("Run demo using sample file"):
            df_demo = pd.read_csv(SAMPLE_CSV)
            st.write("Sample transactions")
            st.dataframe(sanitize_df_for_display(df_demo))
            with st.spinner("Sending to pipeline..."):
                df_out = predict_from_transactions_csv(df_demo)
                if df_out is not None:
                    st.success("Demo prediction complete")
                    df_out_display = sanitize_df_for_display(df_out)
                    st.dataframe(df_out_display)
                    csv = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download demo predictions", data=csv, file_name="demo_predictions.csv", mime="text/csv")
                else:
                    st.error("Pipeline failed. Check that scripts/data_cleaner.py and scripts/feature_engineering.py exist and are importable.")

st.markdown("---")
st.caption("This Streamlit app uses your project's models and scripts (scripts/* and models/*). Run `streamlit run streamlit_app.py` in the project root to start.")
