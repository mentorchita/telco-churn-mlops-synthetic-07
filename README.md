# Telco Customer Churn - Synthetic Dataset with MLOps Pipeline

## Overview
This repository provides a synthetic dataset generator for Telco Customer Churn prediction, along with a full MLOps pipeline. It includes tools for data generation with built-in data drift, model training, experiment tracking using MLflow, API serving with FastAPI, monitoring, and deployment. The dataset is entirely synthetic (no real customer data) and is inspired by the public Telco Customer Churn dataset on Kaggle, licensed under CC BY-NC-SA 4.0.
Key features:

Generate 100,000+ records spanning 2023-01-01 to 2024-12-31.
Simulate gradual concept drift (e.g., growth in fiber optic adoption, decline in electronic checks, reducing churn rates).

Realistic feature dependencies and a RecordDate column for time-based analysis.
MLOps integration: Data Version Control (DVC), Airflow for orchestration, MLflow for experiment tracking and model registry, Kubernetes for deployment, and monitoring for drift detection.

## Repository Structure

- .dvc/: DVC configuration for data and pipeline tracking.
- airflow/dags/: Airflow DAGs for ML workflows.
- conf/ and config/: Configuration files for experiments and pipelines.
- data/: Generated synthetic data (e.g., telco_customers.csv).
- deployment/: Kubernetes manifests for production deployment.
- docs/: Additional documentation and diagrams.
- mlflow/: MLflow configurations, registration scripts, and setup guide.
- mlflow_db/: Persistent storage for MLflow database.
- models/: Trained model artifacts.
- monitoring/: Scripts for data/concept drift detection, A/B testing, and shadow datasets.
- notebooks/: Jupyter notebooks for data exploration and analysis.
- pipelines/: Training and prediction pipelines (e.g., train.py, predict.py).
- src/: Source code for data generation (e.g., generate_dataset_ext.py).
- tests/: Unit tests (e.g., test_api_predict.py).

Dockerfile and Dockerfile.api: Docker images for the project and API.

- docker-compose.yml: Composes services like data generator, Jupyter, API, and MLflow.

## Installation

Clone the repository:textgit clone https://github.com/mentorchita/telco-churn-mlops-synthetic-05.git

cd telco-churn-mlops-synthetic-05

Create a virtual environment and install dependencies:textpython -m venv venv
```sh
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
```sh
pip install -r requirements.txt
```
```sh
pip install -r requirements-ml.txt  # For MLflow and training dependencies
```
```sh
pip install -r requirements-api.txt  # For FastAPI
```
```sh
pip install -r requirements-dev.txt  # Optional: For linting, Jupyter, etc.
```
## Usage

### Data Generation

Generate synthetic data using the provided scripts.

Standard generation:
```sh
python src/generate_dataset.py
```
Custom generation:
```sh
python src/generate_dataset.py --samples 100000 --output-dir data/ --start-date 2022-01-01 --end-date 2024-12-31
```

Enhanced generation (using config.yaml):
```sh
python src/generate_dataset_ext.py --samples 20000 --conv-samples 3000
```

Output files will be placed in data/ (e.g., telco_customers.csv, support_conversations.csv).

### Makefile Commands

Use make for streamlined workflows:

- make help: List all commands.
- make install: Install base dependencies.
- make install-dev: Install development tools (e.g., Ruff, Black, Jupyter).
- make generate-ext: Generate extended dataset.
- make explore: Launch Jupyter.
- make lint: Check code style.
- make format: Fix code style.
- make clean-data: Clean generated data.
- make train: Train the churn model (logs to MLflow).
- make docker-up: Start all services via Docker Compose.
- make jupyter-up: Launch Jupyter container.
- make jupyter-down: Stop Jupyter.
- make jupyter-logs: View Jupyter logs (includes access token).

## ML Training
Запустіть `make train` для тренування моделі churn prediction.

## Testing the Predict API

The `/predict` endpoint accepts customer features as JSON and returns churn prediction.

### Quick test with curl:

```bash
docker t \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Python script test:

```bash
# Install requests if not already installed
pip install requests

# Run the test script
python test_api_predict.py
```

The test script will:
1. Check `/health` endpoint (confirms API is running and model is loaded)

2. Send sample customer data to `/predict`

3. Display the churn prediction result (probability and binary classification)

### Via Docker Compose:

```bash
# Start all services (generator, jupyter, api, mlflow)

docker-compose up --build

# In another terminal, test the API

curl http://localhost:8000/health


```

## Deployment
Use deployment/ for Kubernetes manifests to deploy the API and MLflow in production.

## Monitoring
Scripts in monitoring/ handle data/concept drift detection, A/B testing, and shadow datasets. Integrate with MLflow for comparing model versions.

## License
MIT License. See LICENSE for details.

dvc.yaml: DVC pipeline definitions.

Makefile: Convenience commands for setup, generation, training, and more.

requirements-*.txt: Python dependencies for base, API, dev, and ML.

## ML Training
Запустіть `make train` для тренування моделі churn prediction.

## Deployment
Використовуйте Kubernetes manifests в deployment/ для production.

## Monitoring
Скрипти для дріфту в monitoring/.
