# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import os
import joblib
from pathlib import Path
import importlib.util
from io import BytesIO
import time

# -------------------------
# Helper: dynamic import for your script modules (scripts/)
# -------------------------
def import_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Determine project root and script/model paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ALT_MNT = Path("/mnt/data")

if (SCRIPTS_DIR / "data_loader.py").exists():
    SCRIPTS_PATH = SCRIPTS_DIR
elif (ALT_MNT / "data_loader.py").exists():
    SCRIPTS_PATH = ALT_MNT
else:
    SCRIPTS_PATH = Path().resolve() / "scripts"

DATA_LOADER_PATH = str(SCRIPTS_PATH / "data_loader.py")
DATA_CLEANER_PATH = str(SCRIPTS_PATH / "data_cleaner.py")
FEATURE_ENG_PATH = str(SCRIPTS_PATH / "feature_engineering.py")
CREDIT_SCORE_PATH = str(SCRIPTS_PATH / "credit_score.py")

# Import user's modules dynamically (will raise if not found)
try:
    data_loader = import_from_path("data_loader", DATA_LOADER_PATH)
    data_cleaner = import_from_path("data_cleaner", DATA_CLEANER_PATH)
    feature_eng = import_from_path("feature_engineering", FEATURE_ENG_PATH)
    credit_score_mod = import_from_path("credit_score", CREDIT_SCORE_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to import one of the script modules: {e}")

# Models dir inside repo root
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Names of model files we expect (you showed these in the repo)
DEFAULT_CLASSIFIER_FILENAME = "credit_risk_prob_model.pkl"
DEFAULT_SCORE_FILENAME = "credit_score_model.pkl"

CLASSIFIER_PATH = MODEL_DIR / DEFAULT_CLASSIFIER_FILENAME
SCORE_PATH = MODEL_DIR / DEFAULT_SCORE_FILENAME

# In-memory loaded models (None if not available)
credit_risk_model = None
credit_score_model = None

def try_load_models():
    global credit_risk_model, credit_score_model
    # Load classifier if exists
    if CLASSIFIER_PATH.exists():
        try:
            credit_risk_model = joblib.load(CLASSIFIER_PATH)
        except Exception as e:
            # keep None on failure but log
            credit_risk_model = None
            print(f"[WARN] Failed to load classifier at {CLASSIFIER_PATH}: {e}")

    # Load score/regressor if exists
    if SCORE_PATH.exists():
        try:
            credit_score_model = joblib.load(SCORE_PATH)
        except Exception as e:
            credit_score_model = None
            print(f"[WARN] Failed to load score model at {SCORE_PATH}: {e}")

# Try loading at startup
try_load_models()

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Credit Risk FastAPI", version="1.1")

class PredictRequest(BaseModel):
    features: Optional[Dict[str, Any]] = None
    CustomerId: Optional[str] = None

@app.get("/")
def root():
    return {
        "message": "Credit Risk API - FastAPI",
        "docs": "/docs",
        "predict_single": "/predict_single (POST)",
        "predict_file": "/predict_file (POST)",
        "models": "/models (GET)",
        "upload_model": "/upload_model (POST)"
    }

# -------------------------
# Model management endpoints
# -------------------------
@app.get("/models")
def list_models() -> Dict[str, Any]:
    """
    List model files present in models/ with metadata (size, mtime).
    """
    files = []
    for p in MODEL_DIR.glob("*"):
        if p.is_file():
            stat = p.stat()
            files.append({
                "filename": p.name,
                "path": str(p),
                "size_bytes": stat.st_size,
                "modified_ts": int(stat.st_mtime),
                "modified_iso": time.ctime(stat.st_mtime)
            })
    return {"models": files, "loaded_classifier": bool(credit_risk_model), "loaded_score_model": bool(credit_score_model)}

@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    """
    Upload a serialized model (.joblib, .pkl) and save to models/.
    Use this to push a trained model into models/.
    """
    contents = await file.read()
    fname = file.filename
    if not fname.endswith((".joblib", ".pkl", ".sav")):
        raise HTTPException(status_code=400, detail="Model file must be .joblib or .pkl or .sav")
    save_path = MODEL_DIR / fname
    with open(save_path, "wb") as f:
        f.write(contents)

    # If uploaded filenames match expected defaults, attempt to load into memory
    if save_path.name == DEFAULT_CLASSIFIER_FILENAME:
        try:
            global credit_risk_model
            credit_risk_model = joblib.load(save_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Saved but failed to load classifier: {e}")
    if save_path.name == DEFAULT_SCORE_FILENAME:
        try:
            global credit_score_model
            credit_score_model = joblib.load(save_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Saved but failed to load score model: {e}")

    return {"status": "ok", "saved_path": str(save_path)}

# -------------------------
# Prediction endpoints
# -------------------------
@app.post("/predict_single")
def predict_single(payload: PredictRequest):
    """
    Predict for a single feature dict.
    If a loaded classifier is available it will be used to compute risk_probability.
    If a loaded credit_score_model (regressor) is available it will be used to predict a numeric credit score.
    Otherwise we fall back to credit_score_mod.assign_credit_score using risk_probability.
    """
    if payload.features is None:
        raise HTTPException(status_code=400, detail="Send a JSON body with 'features' dict")

    df = pd.DataFrame([payload.features])

    # If classifier exists, use it to compute risk_probability
    if credit_risk_model is not None:
        try:
            # Ensure correct columns: take numeric columns from df; real app should use the exact training columns
            X = df.select_dtypes(include=["number"]).fillna(0)
            if hasattr(credit_risk_model, "predict_proba"):
                proba = credit_risk_model.predict_proba(X)[:, 1]
            else:
                # classifier might return direct predictions or scores
                proba = credit_risk_model.predict(X)
            df["risk_probability"] = proba
        except Exception as e:
            # fallback to any provided risk_probability
            if "risk_probability" not in df.columns:
                df["risk_probability"] = 0.5
    else:
        if "risk_probability" not in df.columns:
            df["risk_probability"] = 0.5

    # If a credit score regressor exists, try to predict score
    if credit_score_model is not None:
        try:
            X_score = df.select_dtypes(include=["number"]).fillna(0)
            pred_score = credit_score_model.predict(X_score)
            # clamp and round to int
            df["credit_score_model_pred"] = pd.Series(pred_score).round().astype(int)
            out_score = int(df["credit_score_model_pred"].iloc[0])
            rating_df = credit_score_mod.assign_credit_score(pd.DataFrame([{"risk_probability": df["risk_probability"].iloc[0]}]))
            out_rating = str(rating_df["Rating"].iloc[0])
            return {
                "risk_probability": float(df["risk_probability"].iloc[0]),
                "credit_score_pred_by_model": out_score,
                "rating_based_on_risk_probability": out_rating
            }
        except Exception as e:
            # fall through to assign_credit_score
            pass

    # fallback: use credit_score.assign_credit_score which maps risk_probability -> credit_score
    df = credit_score_mod.assign_credit_score(df)
    out = {
        "risk_probability": float(df["risk_probability"].iloc[0]),
        "credit_score": int(df["credit_score"].iloc[0]),
        "rating": str(df["Rating"].iloc[0])
    }
    return out

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    """
    Upload a CSV file of transactions or precomputed features.
    Pipeline:
      - Load CSV
      - Clean with scripts.data_cleaner.clean_data
      - Aggregate using scripts.feature_engineering.create_aggregate_features
      - Use loaded classifier to predict probabilities (if available)
      - Use loaded score model to predict numeric score (if available)
      - Otherwise generate credit score using credit_score.assign_credit_score
    """
    contents = await file.read()
    try:
        df = pd.read_csv(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    # 1) Clean the raw transactions (your data_cleaner.clean_data expects certain columns)
    df_clean = data_cleaner.clean_data(df)
    if df_clean is None:
        raise HTTPException(status_code=500, detail="Data cleaning failed")

    # 2) Feature engineering: aggregate per CustomerId
    df_agg = feature_eng.create_aggregate_features(df_clean)
    if df_agg is None or df_agg.empty:
        raise HTTPException(status_code=500, detail="Feature engineering failed or no aggregated rows")

    # 3) Predict risk probability
    if credit_risk_model is not None:
        try:
            X = df_agg.select_dtypes(include=["number"]).fillna(0)
            if hasattr(credit_risk_model, "predict_proba"):
                proba = credit_risk_model.predict_proba(X)[:, 1]
            else:
                proba = credit_risk_model.predict(X)
            df_agg["risk_probability"] = proba
        except Exception as e:
            # fallback heuristic
            df_agg["risk_probability"] = 1 - (df_agg["Monetary"] / (df_agg["Monetary"].max() + 1))
            df_agg["risk_probability"] = df_agg["risk_probability"].clip(0, 1)
    else:
        # simple heuristic if no classifier
        df_agg["risk_probability"] = 1 - (df_agg["Monetary"] / (df_agg["Monetary"].max() + 1))
        df_agg["risk_probability"] = df_agg["risk_probability"].clip(0, 1)

    # 4) Predict credit score using score model if exists, else map from risk_probability
    if credit_score_model is not None:
        try:
            X_score = df_agg.select_dtypes(include=["number"]).fillna(0)
            score_pred = credit_score_model.predict(X_score)
            df_agg["credit_score_model_pred"] = pd.Series(score_pred).round().astype(int)
            # Also compute rating for consistency using credit_score.assign_credit_score from risk_probability
            rating_df = credit_score_mod.assign_credit_score(df_agg[["risk_probability"]].copy())
            df_agg["Rating_from_risk_map"] = rating_df["Rating"].values
        except Exception as e:
            # fallback mapping
            df_agg = credit_score_mod.assign_credit_score(df_agg)
    else:
        df_agg = credit_score_mod.assign_credit_score(df_agg)

    # Prepare response subset to avoid huge payloads
    cols_to_return = ["CustomerId", "risk_probability"]
    if "credit_score" in df_agg.columns:
        cols_to_return.append("credit_score")
        cols_to_return.append("Rating")
    if "credit_score_model_pred" in df_agg.columns:
        cols_to_return.append("credit_score_model_pred")
        cols_to_return.append("Rating_from_risk_map")

    # dedupe and convert to records
    df_out = df_agg.reset_index(drop=True)
    result = df_out[cols_to_return].to_dict(orient="records")
    return {"n_rows": len(result), "predictions": result}

# -------------------------
# Utility endpoint to reload models (admin)
# -------------------------
@app.post("/reload_models")
def reload_models():
    """
    Force reloading models from disk (useful after uploading new models via /upload_model).
    """
    try:
        try_load_models()
        return {"status": "ok", "loaded_classifier": bool(credit_risk_model), "loaded_score_model": bool(credit_score_model)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
