#!/usr/bin/env python
"""
Simple test script for /predict endpoint.
Requires the API to be running at http://localhost:8000

Usage:
  python test_api_predict.py
"""

import requests
import json

API_URL = "http://localhost:8000"

# Sample customer features - adjust values to match your data
sample_customer = {
    "tenure": 12,
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
}

def test_health():
    """Test the /health endpoint"""
    print("Testing /health endpoint...")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        print(f"Status: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict():
    """Test the /predict endpoint with sample data"""
    print("\nTesting /predict endpoint...")
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json=sample_customer,
            timeout=5
        )
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            result = resp.json()
            print(f"Prediction result:")
            print(f"  Churn probability: {result['churn_probability']}")
            print(f"  Churn prediction: {result['churn_prediction']}")
            print(f"  Features used: {len(result['features_used'])} columns")
        else:
            print(f"Error response: {resp.text}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print(f"Testing API at {API_URL}\n")
    
    health_ok = test_health()
    if not health_ok:
        print("\nHealth check failed! Make sure the API is running:")
        print("  uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
        exit(1)
    
    predict_ok = test_predict()
    
    if health_ok and predict_ok:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
        exit(1)
