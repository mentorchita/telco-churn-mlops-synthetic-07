from pydantic import BaseModel
from typing import List, Optional

class CustomerFeatures(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    churn_prediction: int  # 0 або 1
    features_used: List[str]