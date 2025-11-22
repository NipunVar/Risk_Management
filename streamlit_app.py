# streamlit_app.py
"""
Full updated Streamlit app (complete).
- Dedicated "Images" mode with one section per PNG (each in its own expander).
- Sidebar shows only the "Project images" header and the Mode radio.
- All DataFrame outputs are sanitized to avoid PyArrow ArrowInvalid errors.
- Models loading, single prediction, and batch CSV prediction remain intact.
- Uses environment variable API_URL for the /predict_file endpoint (fallback to localhost).
"""

import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
import importlib.util
import time
from io import BytesIO
import requests
import json

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
    # Do not crash app if imports fail; show a gentle warning later
    st.warning(f"Could not import scripts: {e}")

# -------------------------
# Load models (if present)
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
    # Use environment variable (works on Streamlit Cloud when set) and fallback to localhost for local dev
    API_URL = os.getenv("API_URL", "http://localhost:8000/predict_file")
    api_url = API_URL

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
def _to_json_safe(x):
    try:
        return json.dumps(x, default=str)
    except Exception:
        return str(x)

def sanitize_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stricter sanitizer that guarantees Arrow-compatible dtypes for display.

    Rules:
    - Numeric columns: only if *all* non-null values parse as numeric (no stray strings).
    - Datetime/timedelta preserved when possible.
    - Categorical -> object converted safely.
    - Complex objects (list/dict/set) are JSON-serialized to strings.
    - Any remaining object columns are converted to plain Python str with NaNs -> "".
    """
    if df is None:
        return df

    df2 = df.copy()

    for col in df2.columns:
        series = df2[col]

        # 1) Convert categorical to object safely (new recommended check)
        try:
            if isinstance(series.dtype, pd.CategoricalDtype):
                try:
                    series = series.astype(object)
                except Exception:
                    series = series.astype(str)
                df2[col] = series
                series = df2[col]
        except Exception:
            pass  # defensive

        # 2) If dtype is numeric already, coerce to numeric and continue
        if pd.api.types.is_numeric_dtype(series):
            df2[col] = pd.to_numeric(series, errors="coerce")
            continue

        # 3) Preserve datetime/timedelta when possible
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
            try:
                df2[col] = pd.to_datetime(series, errors="coerce")
            except Exception:
                pass
            continue

        # 4) If the column contains complex Python objects (list/dict/set), JSON-serialize them
        sample_has_complex = False
        try:
            non_null_sample = series.dropna().head(20).tolist()
            for v in non_null_sample:
                if isinstance(v, (list, dict, set)):
                    sample_has_complex = True
                    break
        except Exception:
            sample_has_complex = False

        if sample_has_complex:
            df2[col] = series.fillna("").apply(_to_json_safe)
            continue

        # 5) Numeric check: only accept numeric if ALL non-null values parse as numeric.
        coerced = pd.to_numeric(series, errors="coerce")
        non_null_mask = series.notna()
        if non_null_mask.any():
            all_non_null_parsable = coerced[non_null_mask].notna().all()
        else:
            all_non_null_parsable = False

        if all_non_null_parsable:
            df2[col] = coerced
            continue

        # 6) Otherwise convert to plain strings (no pd.NA, no objects)
        try:
            s = series.fillna("")
            def _safe_str(x):
                if pd.isna(x) or x == "":
                    return ""
                if isinstance(x, (list, dict, set)):
                    return _to_json_safe(x)
                return str(x)
            df2[col] = s.apply(_safe_str)
        except Exception:
            df2[col] = series.apply(lambda x: "" if pd.isna(x) else str(x))

    # Final pass: ensure object columns are safe strings and no pd.NA remain
    for c in df2.select_dtypes(include=["object"]).columns:
        try:
            df2[c] = df2[c].fillna("").apply(lambda x: "" if pd.isna(x) else str(x))
        except Exception:
            df2[c] = df2[c].astype(str).fillna("")

    # Ensure index is a simple RangeIndex (avoid strange index objects)
    try:
        df2 = df2.reset_index(drop=True)
    except Exception:
        pass

    return df2

def display_df_for_streamlit(df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    """
    Prepare a DataFrame for display in Streamlit safely:
    - Runs sanitize_df_for_display
    - Forces every non-datetime column to plain Python strings (so Arrow won't try to convert to numeric)
    - Optionally truncates rows for display
    Returns the prepared DataFrame (which is safe to pass directly to st.dataframe)
    """
    if df is None:
        return df
    df_safe = sanitize_df_for_display(df)

    # Optionally truncate (but leave original indexing intact) for speed
    if max_rows is not None and len(df_safe) > max_rows:
        df_safe = df_safe.head(max_rows)

    # Convert non-datetime columns to strings (explicitly), avoiding 'nan' text
    for c in df_safe.columns:
        # Preserve datetime/timedelta columns as-is
        if pd.api.types.is_datetime64_any_dtype(df_safe[c]) or pd.api.types.is_timedelta64_dtype(df_safe[c]):
            continue
        # Force to string and replace "nan" with ""
        df_safe[c] = df_safe[c].apply(lambda x: "" if pd.isna(x) else str(x))

    # As a last precaution ensure dtype is object for pyarrow
    for c in df_safe.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_safe[c]) and not pd.api.types.is_timedelta64_dtype(df_safe[c]):
            df_safe[c] = df_safe[c].astype(object)

    return df_safe

# -------------------------
# Image discovery & descriptions
# -------------------------
def find_pngs_in_root(root: Path):
    return sorted([p for p in root.glob("*.png") if p.is_file()])

PROJECT_PNGS = find_pngs_in_root(PROJECT_ROOT)
PNG_MAP = {p.stem: p for p in PROJECT_PNGS}

# Descriptions: edit as needed. Keys are file stems (filename without .png)
IMAGE_DESCRIPTIONS = {
    "actual_prediction": "Actual vs Predicted — scatter plot with fitted line showing how predicted credit scores compare to actuals.",
    "classfication": "Sample classification table showing predicted labels and 'is_high_risk' flag.",
    "confusion_mat": "Confusion matrix for the classifier — visualises true vs predicted counts.",
    "correlation": "Correlation heatmap showing relationships across all features (including binned WoE features).",
    "creditScore": "Example output table showing computed credit scores and ratings.",
    "FICO_Score": "FICO scoring buckets and helpful reference chart for mapping scores to categories.",
    "gb": "Gradient Boosting classifier training snapshot (model properties).",
    "LR": "Linear Regression snapshot used for credit score modeling.",
    "pred": "Sample predictions output snapshot.",
    "rfms_space": "RFMS 3D scatter showing Recency/Frequency/Monetary segmentation.",
    "ROC-Curve": "ROC curve and AUC indicating classifier performance.",
    "selected_features": "List of selected features used for building the model."
}

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Credit Risk - Streamlit UI", layout="wide")
st.title("Credit Risk Scoring")

# Sidebar: ONLY the Project images heading (per request)
st.sidebar.header("Project images")
st.sidebar.markdown("This app includes a dedicated Images page. Use the main page to browse image sections.")

# Sidebar: Mode selection (modes unchanged)
st.sidebar.markdown("---")
mode = st.sidebar.radio("Mode", ["Single (WoE features)", "Batch from CSV", "Models info", "Images"])

# ---------------------------------------------------------------------
#  IMAGES MODE: separate page that shows only images (one section per PNG)
# ---------------------------------------------------------------------
if mode == "Images":
    st.header("Project images — sections")
    st.markdown("This page contains a separate **section** for each image in the repository. Expand a section to view the full image and its description.")
    if not PROJECT_PNGS:
        st.info("No PNG images found in the project root. Place PNG files next to this script.")
    else:
        # Option to expand all
        expand_all = st.checkbox("Show all expanded", value=False)
        # Show each image as its own section (expander)
        for p in PROJECT_PNGS:
            title = p.stem
            desc = IMAGE_DESCRIPTIONS.get(title, "No description available for this image.")
            # set expanded state according to checkbox
            with st.expander(title, expanded=expand_all):
                st.subheader(title)
                st.markdown(desc)
                try:
                    st.image(str(p), use_container_width=True)
                except Exception as e:
                    st.error(f"Unable to display image {p.name}: {e}")
                st.markdown("---")

    # end of Images mode: return early so nothing else renders
    st.stop()

# -------------------------
# MODE: Models info
# -------------------------
if mode == "Models info":
    st.header("Model metadata")
    md = model_metadata()
    df_md = pd.DataFrame(md) if md else pd.DataFrame(columns=["filename", "size_kb", "modified", "path"])
    st.dataframe(display_df_for_streamlit(df_md))
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
            out_display = display_df_for_streamlit(out)  # safe display wrapper
            # show transposed display as before, but ensure it's safe (we convert everything to strings)
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
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Unable to read uploaded CSV: {e}")
            df_raw = None

        if df_raw is not None:
            st.write("Raw data preview")
            st.dataframe(display_df_for_streamlit(df_raw.head(10), max_rows=10))

            if st.button("Run batch prediction"):
                with st.spinner("Running cleaning, feature engineering and predictions..."):
                    df_out = predict_from_transactions_csv(df_raw)
                    if df_out is None:
                        st.error("Prediction pipeline failed (feature_engineering or data_cleaner not found).")
                    else:
                        st.success("Batch prediction completed")
                        df_out_display = display_df_for_streamlit(df_out, max_rows=50)
                        st.dataframe(df_out_display)
                        csv = df_out.to_csv(index=False).encode("utf-8")
                        st.download_button("Download predictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")

# Footer
st.markdown("---")
st.caption("This Streamlit app uses your project's models and scripts (scripts/* and models/*). Run `streamlit run streamlit_app.py` in the project root to start.")
