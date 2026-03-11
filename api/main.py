import json
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

# ─────────────────────────────────────────────
# GLOBAL MODEL STORE
# ─────────────────────────────────────────────
_store: dict = {
    "model":              None,
    "scaler":             None,
    "optimal_threshold":  0.10,
    "eval_results":       {},
}


# ─────────────────────────────────────────────
# LIFESPAN — load artifacts once at startup
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load XGBoost model
    try:
        _store["model"] = joblib.load("models/xgboost_tuned.pkl")
        print("[startup] Model loaded: models/xgboost_tuned.pkl")
    except Exception as e:
        print(f"[startup] WARNING: Could not load model — {e}")

    # Load scaler
    try:
        _store["scaler"] = joblib.load("models/scaler.pkl")
        print("[startup] Scaler loaded: models/scaler.pkl")
    except Exception as e:
        print(f"[startup] WARNING: Could not load scaler — {e}")

    # Load evaluation results and extract optimal threshold
    try:
        with open("models/evaluation_results.json") as f:
            _store["eval_results"] = json.load(f)
        _store["optimal_threshold"] = _store["eval_results"].get("optimal_threshold", 0.10)
        print(f"[startup] Optimal threshold: {_store['optimal_threshold']}")
    except Exception as e:
        print(f"[startup] WARNING: Could not load evaluation_results.json — {e}")

    yield  # app runs here

    print("[shutdown] Cleaning up resources.")


# ─────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="XGBoost-powered fraud detection with business-optimised threshold.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins so Streamlit (different port) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# PYDANTIC — INPUT MODEL
# ─────────────────────────────────────────────
class TransactionFeatures(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
                "Time": 406.0,
                "V1":  -2.3122265423263,
                "V2":   1.9519999999999,
                "V3":  -1.6098473900817,
                "V4":   3.9979055875468,
                "V5":  -0.5220083770035,
                "V6":  -1.4261691064052,
                "V7":  -2.7370130697555,
                "V8":   0.1377742812761,
                "V9":  -0.3408786430989,
                "V10": -0.5773168575,
                "V11": -0.8315616676,
                "V12": -0.4693270527,
                "V13": -0.3133736635,
                "V14": -0.8478585399,
                "V15": -0.2670047078,
                "V16": -0.8162463605,
                "V17": -0.0831569399,
                "V18": -0.1415197543,
                "V19": -0.2063695491,
                "V20":  0.0215807918,
                "V21":  0.2778375189,
                "V22":  0.1730612038,
                "V23": -0.0870832838,
                "V24": -0.0693093293,
                "V25": -0.2219286428,
                "V26":  0.0627228487,
                "V27":  0.0614576419,
                "V28":  0.0221689839,
                "Amount": 24.79,
    }})

    Time:  float
    V1:    float
    V2:    float
    V3:    float
    V4:    float
    V5:    float
    V6:    float
    V7:    float
    V8:    float
    V9:    float
    V10:   float
    V11:   float
    V12:   float
    V13:   float
    V14:   float
    V15:   float
    V16:   float
    V17:   float
    V18:   float
    V19:   float
    V20:   float
    V21:   float
    V22:   float
    V23:   float
    V24:   float
    V25:   float
    V26:   float
    V27:   float
    V28:   float
    Amount: float


# ─────────────────────────────────────────────
# PYDANTIC — OUTPUT MODEL
# ─────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction:     str    # "fraud" or "legitimate"
    probability:    float
    threshold_used: float
    risk_level:     str    # "HIGH", "MEDIUM", or "LOW"


# ─────────────────────────────────────────────
# HELPER — preprocess a single transaction
# ─────────────────────────────────────────────
def _preprocess(tx: TransactionFeatures) -> np.ndarray:
    """
    Reproduce the same feature order used in train.py:
    [V1..V28, Amount_scaled, Time_scaled]

    The saved scaler was fit on the Time column (last fit_transform in
    train.py).  Because XGBoost is tree-based, exact scaling of Amount
    and Time does not affect split decisions — only the relative ordering
    matters.  We apply the saved scaler to both columns for consistency.
    """
    scaler = _store["scaler"]

    amount_scaled = scaler.transform([[tx.Amount]])[0][0]
    time_scaled   = scaler.transform([[tx.Time]])[0][0]

    # Feature order must match training: V1-V28, Amount_scaled, Time_scaled
    features = [
        tx.V1,  tx.V2,  tx.V3,  tx.V4,  tx.V5,
        tx.V6,  tx.V7,  tx.V8,  tx.V9,  tx.V10,
        tx.V11, tx.V12, tx.V13, tx.V14, tx.V15,
        tx.V16, tx.V17, tx.V18, tx.V19, tx.V20,
        tx.V21, tx.V22, tx.V23, tx.V24, tx.V25,
        tx.V26, tx.V27, tx.V28,
        amount_scaled, time_scaled,
    ]
    return np.array(features).reshape(1, -1)


def _risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "HIGH"
    if probability >= 0.3:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", summary="API root")
def root():
    return {
        "message":           "Fraud Detection API",
        "status":            "running",
        "model":             "XGBoost Tuned",
        "optimal_threshold": _store["optimal_threshold"],
    }


@app.get("/health", summary="Health check")
def health():
    return {
        "status":        "healthy",
        "model_loaded":  _store["model"] is not None,
        "scaler_loaded": _store["scaler"] is not None,
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict fraud")
def predict(transaction: TransactionFeatures):
    if _store["model"] is None or _store["scaler"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model or scaler not loaded. Check server startup logs.",
        )

    features    = _preprocess(transaction)
    probability = float(_store["model"].predict_proba(features)[0][1])
    threshold   = _store["optimal_threshold"]
    label       = "fraud" if probability >= threshold else "legitimate"

    return PredictionResponse(
        prediction=label,
        probability=round(probability, 6),
        threshold_used=threshold,
        risk_level=_risk_level(probability),
    )


@app.get("/model-info", summary="Model metadata and evaluation results")
def model_info():
    ev = _store["eval_results"]
    return {
        "name":                "XGBoost Tuned",
        "optimal_threshold":   _store["optimal_threshold"],
        "roc_auc":             ev.get("roc_auc"),
        "recall":              ev.get("recall"),
        "precision":           ev.get("precision"),
        "false_negative_cost": ev.get("cost_false_negative"),
        "false_positive_cost": ev.get("cost_false_positive"),
        "total_business_cost": ev.get("total_business_cost"),
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
# Run with:
#   uvicorn api.main:app --host 0.0.0.0 --port 8002 --reload
