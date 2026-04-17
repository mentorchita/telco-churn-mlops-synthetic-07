"""
Telco Churn ML Inference Service
FastAPI app that loads a sklearn model and serves predictions.
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telco Churn Predictor",
    description="Sklearn model serving churn probability predictions",
    version=os.getenv("MODEL_VERSION", "1.0.0"),
)

# ── Model loading ──────────────────────────────────────────────────────────
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/churn_model.pkl"))
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
model = None


@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {MODEL_PATH}")
    else:
        logger.warning(f"Model not found at {MODEL_PATH} — using dummy model")
        model = None


# ── Schemas ────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    tenure: int = Field(..., ge=0, le=72, description="Months as customer (0–72)")
    monthly_charges: float = Field(..., ge=0, description="Monthly bill in USD")
    total_charges: Optional[float] = Field(None, description="Inferred if not provided")
    contract_type: str = Field("month-to-month", description="month-to-month | one_year | two_year")
    internet_service: str = Field("Fiber optic", description="DSL | Fiber optic | No")
    payment_method: str = Field("Electronic check", description="Payment method")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "monthly_charges": 65.50,
                "contract_type": "month-to-month",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check",
            }
        }


class PredictResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    prediction: str                    # "churn" | "no_churn"
    model_version: str
    threshold_used: float


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/")
def root():
    return {"service": "churn-ml-service", "version": os.getenv("MODEL_VERSION", "1.0.0")}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    total = req.total_charges if req.total_charges else req.tenure * req.monthly_charges

    features = {
        "tenure": req.tenure,
        "MonthlyCharges": req.monthly_charges,
        "TotalCharges": total,
        "Contract_Month-to-month": int(req.contract_type == "month-to-month"),
        "Contract_One year": int(req.contract_type == "one_year"),
        "Contract_Two year": int(req.contract_type == "two_year"),
        "InternetService_DSL": int(req.internet_service == "DSL"),
        "InternetService_Fiber optic": int(req.internet_service == "Fiber optic"),
        "InternetService_No": int(req.internet_service == "No"),
        "PaymentMethod_Electronic check": int(req.payment_method == "Electronic check"),
    }

    df = pd.DataFrame([features])

    if model is not None:
        prob = float(model.predict_proba(df)[0][1])
    else:
        # Dummy heuristic for demo without real model
        prob = float(np.clip(
            (1 / (1 + req.tenure / 12)) * (req.monthly_charges / 100) * 0.8, 0, 1
        ))
        logger.warning("Using dummy heuristic — no model loaded")

    prediction = "churn" if prob >= THRESHOLD else "no_churn"
    logger.info(f"Prediction: {prediction} (prob={prob:.3f}, tenure={req.tenure})")

    return PredictResponse(
        churn_probability=round(prob, 4),
        prediction=prediction,
        model_version=os.getenv("MODEL_VERSION", "1.0.0"),
        threshold_used=THRESHOLD,
    )


@app.get("/metrics")
def metrics():
    """Lightweight metrics endpoint — replace with Prometheus in Topic 8."""
    return {
        "model_loaded": model is not None,
        "threshold": THRESHOLD,
        "model_version": os.getenv("MODEL_VERSION", "1.0.0"),
    }
