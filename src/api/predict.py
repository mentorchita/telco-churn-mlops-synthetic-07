import joblib
import pandas as pd
import os
import sys
from typing import Dict

# Import mlflow only if needed, but don't fail startup
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: mlflow not available, will use local model only", file=sys.stderr)

# Local model path fallback
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.pkl")

# MLflow model settings
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', '').strip()
MLFLOW_MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME', 'ChurnModel')
MLFLOW_MODEL_STAGE = os.getenv('MLFLOW_MODEL_STAGE', 'Production')

model = None
model_source = None

# Try MLflow registry only if tracking URI is explicitly provided and mlflow is available
if MLFLOW_AVAILABLE and MLFLOW_TRACKING_URI:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
        model = mlflow.sklearn.load_model(model_uri)
        model_source = f"MLflow registry ({model_uri})"
        print(f"✓ Model loaded from {model_source}")
    except Exception as e:
        print(f"Warning: Could not load from MLflow: {e}", file=sys.stderr)
        # Fall through to local model attempt

# Fallback to local model file
if model is None:
    try:
        model = joblib.load(MODEL_PATH)
        model_source = f"local file ({MODEL_PATH})"
        print(f"✓ Model loaded from {model_source}")
    except Exception as e:
        print(f"Warning: Could not load local model: {e}", file=sys.stderr)
        model = None
        model_source = None

if model is None:
    print("Warning: No model loaded - predictions will fail until model is available", file=sys.stderr)


def preprocess_features(features: Dict) -> pd.DataFrame:
    """Return a DataFrame suitable for the saved sklearn Pipeline.

    Do minimal numeric coercion and otherwise leave categorical values as-is
    so the pipeline's ColumnTransformer / OneHotEncoder can handle them.
    """
    df = pd.DataFrame([features])

    # Ensure numeric columns are numeric (common potential issue)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if 'MonthlyCharges' in df.columns:
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    if 'tenure' in df.columns:
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

    # Fill NA with reasonable defaults (pipeline may still raise if unexpected)
    df = df.fillna({
        'TotalCharges': 0,
        'MonthlyCharges': 0,
        'tenure': 0
    })

    return df


def predict_churn(features: Dict) -> Dict:
    if model is None:
        return {"error": "Model not loaded"}

    try:
        X = preprocess_features(features)

        # Some MLflow-loaded models may be pyfunc wrappers; prefer predict_proba when available
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X)[0][1]
        else:
            # fallback to predict (binary 0/1) and map to probability-like value
            pred = model.predict(X)[0]
            prob = float(pred)

        pred = 1 if prob >= 0.5 else 0

        return {
            "churn_probability": round(float(prob), 4),
            "churn_prediction": int(pred),
            "features_used": list(X.columns)
        }
    except Exception as e:
        return {"error": str(e)}