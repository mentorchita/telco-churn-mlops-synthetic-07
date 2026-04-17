#!/usr/bin/env python
"""
Test that API can be imported and started without errors.
This helps diagnose startup issues.
"""

import sys
import os

# Set a dummy model path for testing
os.environ['MODEL_PATH'] = 'models/churn_model.pkl'

print("Testing API imports...")

try:
    print("1. Importing FastAPI and utils...")
    from fastapi import FastAPI
    print("   ✓ FastAPI OK")
    
    print("2. Importing API models...")
    from src.api.models import CustomerFeatures, PredictionResponse
    print("   ✓ Models OK")
    
    print("3. Importing predict module...")
    from src.api import predict as predict_module
    print(f"   ✓ Predict OK (model: {predict_module.model_source or 'None (will fail at runtime)'})")
    
    print("4. Importing main app...")
    from src.api.main import app
    print("   ✓ Main app OK")
    
    print("\n✓ All imports successful!")
    print(f"\nModel status: {predict_module.model_source or 'NOT LOADED'}")
    
    if predict_module.model is None:
        print("\n⚠ Warning: Model not loaded! You need to either:")
        print("  - Place a trained model at: models/churn_model.pkl")
        print("  - Run: python pipelines/train.py")
        print("  - Or set MLFLOW_TRACKING_URI and register a model")
    
except Exception as e:
    print(f"\n✗ Import failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
