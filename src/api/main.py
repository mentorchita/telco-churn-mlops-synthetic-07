from fastapi import FastAPI
app = FastAPI()

@app.get("/predict")
def predict():
    return {"churn": 0.5}  # Placeholder API для інференсу
