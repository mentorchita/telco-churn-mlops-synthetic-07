# Telco Customer Churn - Synthetic Dataset with MLOps Pipeline
# Kubernetes Deployment — Telco Churn MLOps Stack

Topic 7 materials for the **Modern MLOps · LLMOps · AgenticOps** course.  
Deploys three microservices (ML inference, RAG retrieval, LLM Agent) to a
**k3s** Kubernetes cluster on Ubuntu.

> ⚠️ This guide targets **k3s on Ubuntu/Debian**.  
> For minikube (macOS/Windows) see `k8s/scripts/deploy-minikube.sh`.


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
```
telco-churn-mlops-synthetic-07/
├── k8s/
│   ├── base/                          ← Core K8s manifests
│   │   ├── namespace.yaml             ← namespace: mlops
│   │   ├── configmap.yaml             ← ml-config, rag-config, agent-config
│   │   ├── secrets.yaml               ← llm-secrets (placeholders only)
│   │   ├── rag-pvc.yaml               ← PersistentVolumeClaim for ChromaDB
│   │   ├── ml-deployment.yaml         ← churn-predictor FastAPI (port 8000)
│   │   ├── ml-service.yaml
│   │   ├── ml-hpa.yaml                ← HorizontalPodAutoscaler 2→10 pods
│   │   ├── rag-deployment.yaml        ← ChromaDB + FastAPI (port 8001)
│   │   ├── rag-service.yaml
│   │   ├── agent-deployment.yaml      ← LLM Agent FastAPI (port 8002)
│   │   ├── agent-service.yaml
│   │   ├── ingress.yaml               ← Traefik ingress: /ml /rag /agent
│   │   └── kustomization.yaml
│   ├── overlays/
│   │   ├── dev/kustomization.yaml     ← 1 replica, :dev tags, DEBUG logs
│   │   └── prod/kustomization.yaml    ← 3 replicas, pinned tags, WARNING logs
│   └── scripts/
│       ├── deploy-k3s.sh              ← ✅ One-shot deploy on k3s (use this)
│       ├── deploy-minikube.sh         ← minikube variant (macOS/Windows)
│       ├── demo.sh                    ← Interactive 6-step demo
│       ├── ingest-knowledge.sh        ← Load knowledge_base.json into RAG
│       ├── rotate-secret.sh           ← Rotate OpenAI key + rolling restart
│       └── teardown.sh                ← Remove all resources
│
├── services/
│   ├── ml/app.py                      ← FastAPI sklearn inference (port 8000)
│   ├── rag/app.py                     ← FastAPI ChromaDB service (port 8001)
│   └── agent/app.py                   ← FastAPI LLM agent (port 8002)
│
├── models/churn_model.pkl             ← sklearn Pipeline (joblib) — retrain first!
├── data/
│   ├── telco_customers.csv            ← training data
│   └── knowledge_base.json            ← RAG knowledge base (8 docs)
└── pipelines/train.py                 ← Train and save churn_model.pkl
```

---

## Prerequisites

| Tool | Min version | Install |
|------|-------------|---------|
| k3s | v1.28+ | `curl -sfL https://get.k3s.io \| sh -` |
| kubectl | 1.28+ | included with k3s |
| Docker | 24+ | `sudo apt-get install -y docker.io` |
| Python | 3.11+ | `sudo apt-get install -y python3.11` |

### k3s vs minikube — what changed

| | minikube | k3s (this guide) |
|---|---|---|
| Ingress controller | nginx addon | **Traefik built-in** — no addon needed |
| Image loading | `eval $(minikube docker-env)` | `docker save \| sudo k3s ctr images import -` |
| StorageClass | `standard` | **`local-path`** |
| Cluster IP | `minikube ip` | `kubectl get nodes -o jsonpath=...` |
| Stop cluster | `minikube stop` | `sudo systemctl stop k3s` |

---

## Quick Start on k3s (15 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/mentorchita/telco-churn-mlops-synthetic-07
cd telco-churn-mlops-synthetic-07

# 2. Set up kubectl without sudo (one-time setup)
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER ~/.kube/config

# 3. Set your OpenAI API key (optional — fallback runs without it)
export OPENAI_API_KEY="sk-proj-..."

# 4. One-shot deploy (trains model + builds images + deploys everything)
chmod +x k8s/scripts/*.sh
./k8s/scripts/deploy-k3s.sh

# 5. Run the interactive demo
./k8s/scripts/demo.sh
```

---

## Manual Step-by-Step Deploy

### 1. Verify k3s

```bash
sudo k3s kubectl get nodes
# NAME      STATUS   ROLES                  AGE
# vagrant   Ready    control-plane,master   ...

# One-time kubectl setup:
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER ~/.kube/config
kubectl get nodes   # works without sudo now
```


## Manual Installation

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
Run `make train` 
### 2. Train the model

> The `models/churn_model.pkl` in the repo may be a `numpy.ndarray` (predictions array),
> not a sklearn Pipeline. **Always retrain before deploying.**

```bash
python3 pipelines/train.py
# Expected: "Accuracy (no mlflow): 0.79xx"  +  "Model saved to models/churn_model.pkl"

# Verify:
python3 -c "
import joblib
m = joblib.load('models/churn_model.pkl')
print(type(m).__name__)   # must print: Pipeline
"
```

 

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
### 3. Build images and import into k3s

```bash
# Build with host Docker daemon (do NOT use eval $(minikube docker-env))
docker build -t churn-predictor:latest services/ml/
docker build -t churn-rag:latest       services/rag/
docker build -t churn-agent:latest     services/agent/

# Import into k3s containerd (required after every build)
docker save churn-predictor:latest | sudo k3s ctr images import -
docker save churn-rag:latest       | sudo k3s ctr images import -
docker save churn-agent:latest     | sudo k3s ctr images import -

# Verify
sudo k3s ctr images ls | grep churn
```

### 4. Namespace and secrets

```bash
kubectl apply -f k8s/base/namespace.yaml

kubectl create secret generic llm-secrets \
  --from-literal=openai_api_key="${OPENAI_API_KEY:-placeholder}" \
  --from-literal=mlflow_tracking_uri="http://mlflow:5000" \
  --from-literal=mlflow_tracking_token="demo-token" \
  -n mlops \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/base/configmap.yaml
```

### 5. PVC — must set storageClass: local-path for k3s

```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rag-pvc
  namespace: mlops
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: local-path
  resources:
    requests:
      storage: 5Gi
EOF
```

### 6. Deploy all services

```bash
kubectl apply -f k8s/base/ml-deployment.yaml
kubectl apply -f k8s/base/ml-service.yaml
kubectl apply -f k8s/base/ml-hpa.yaml
kubectl apply -f k8s/base/rag-deployment.yaml
kubectl apply -f k8s/base/rag-service.yaml
kubectl apply -f k8s/base/agent-deployment.yaml
kubectl apply -f k8s/base/agent-service.yaml
```

### 7. Ingress — patch to Traefik

```bash
kubectl apply -f k8s/base/ingress.yaml

# k3s has Traefik built-in; patch ingressClassName
kubectl patch ingress mlops-ingress -n mlops \
  --type='json' \
  -p='[{"op":"replace","path":"/spec/ingressClassName","value":"traefik"}]'
```

### 8. Mount model via hostPath

```bash
MODELS_DIR=$(realpath models/)
kubectl patch deployment churn-ml-service -n mlops --patch "
spec:
  template:
    spec:
      containers:
      - name: churn-ml
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
      volumes:
      - name: model-storage
        hostPath:
          path: ${MODELS_DIR}
          type: Directory
"
kubectl rollout status deployment/churn-ml-service -n mlops
```

### 9. Configure /etc/hosts

```bash
NODE_IP=$(kubectl get nodes \
  -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
echo "$NODE_IP  mlops.local" | sudo tee -a /etc/hosts
```

### 10. Load knowledge base into RAG

```bash
./k8s/scripts/ingest-knowledge.sh
```

---

## API Reference

### ML Service — port 8000 / Ingress path `/ml`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | `{"status":"ok","model_loaded":true}` |
| POST | `/predict` | Predict churn probability |
| GET | `/metrics` | Model metadata |

> ⚠️ **Field names are case-sensitive and match `telco_customers.csv` exactly.**  
> Use `MonthlyCharges` not `monthly_charges`. Use `Contract` not `contract_type`.

```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 8, "MonthlyCharges": 79.90, "TotalCharges": 639.20,
    "Contract": "Month-to-month", "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "PhoneService": "Yes", "MultipleLines": "No", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "PaperlessBilling": "Yes"
  }' | python3 -m json.tool
```

### RAG Service — port 8001 / Ingress path `/rag`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | `{"chroma_ready":true,"doc_count":8}` |
| POST | `/ingest` | Index documents |
| POST | `/query` | Semantic search |
| DELETE | `/collection` | Clear all docs (lab reset) |

### Agent Service — port 8002 / Ingress path `/agent`

Agent uses **snake_case** field names (its own schema):

```bash
curl -s -X POST http://localhost:8003/run \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST-7890",
    "tenure": 8,
    "monthly_charges": 79.90,
    "contract_type": "month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check"
  }' | python3 -m json.tool
```

---

## Lab Exercises

### Scale deployments
```bash
kubectl scale deployment churn-ml-service --replicas=4 -n mlops
kubectl get pods -n mlops -w
kubectl scale deployment churn-ml-service --replicas=2 -n mlops
```

### Rolling update — deploy new model
```bash
# Retrain model (simulates v1.1)
python3 pipelines/train.py

# Pod reads new pkl from hostPath automatically on restart
kubectl rollout restart deployment/churn-ml-service -n mlops
kubectl rollout status  deployment/churn-ml-service -n mlops
kubectl rollout history deployment/churn-ml-service -n mlops

# Rollback if needed
kubectl rollout undo deployment/churn-ml-service -n mlops
```

### HPA in action
```bash
kubectl get hpa -n mlops
kubectl top pods -n mlops   # requires metrics-server
# Install: kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### Kustomize overlays
```bash
kubectl apply -k k8s/overlays/dev/
kubectl apply -k k8s/overlays/prod/
```

### Rotate the OpenAI API key
```bash
export OPENAI_API_KEY="sk-proj-new-key"
./k8s/scripts/rotate-secret.sh
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImagePullBackOff` | Image not in k3s containerd | `docker save <img> \| sudo k3s ctr images import -` |
| `CrashLoopBackOff` | App crash on startup | `kubectl logs <pod> -n mlops` |
| `chroma_ready: false` | chromadb/httpx conflict | `chromadb==0.4.24 numpy==1.26.4 httpx==0.27.2` in `services/rag/requirements.txt` |
| `model_loaded: false` | Wrong pkl type or no hostPath | Retrain: `python3 pipelines/train.py`, add hostPath patch |
| `columns are missing: RecordDate` | Old pkl trained with RecordDate | Retrain — `train.py` now drops it automatically |
| Ingress ADDRESS empty | Wrong ingressClassName | Patch to `traefik` (see Step 7 above) |
| PVC `Pending` | StorageClass missing | Set `storageClassName: local-path` |
| `Expecting value` on curl | Port-forward not running | `kubectl port-forward svc/churn-ml-svc 8001:80 -n mlops &` |

### Universal debug sequence
```bash
kubectl get pods -n mlops                     # overall status
kubectl describe pod <pod-name> -n mlops      # events + scheduling errors
kubectl logs <pod-name> -n mlops              # application output
kubectl exec -it <pod-name> -n mlops -- sh    # shell inside pod
```

---

## Teardown
```bash
./k8s/scripts/teardown.sh           # remove namespace + all resources
./k8s/scripts/teardown.sh --all     # also stop k3s service
```

---

## Notes

- `imagePullPolicy: IfNotPresent` must be set on all deployments so k3s uses
  locally imported images without a registry.
- The agent and RAG services work without an OpenAI key — rule-based fallback
  and default ChromaDB embeddings are used respectively.
- Secrets in `k8s/base/secrets.yaml` contain placeholder values only.
  Always use `kubectl create secret` or a secrets manager in production.
- After every `docker build`, re-run `docker save | sudo k3s ctr images import -`.
  k3s **cannot** see Docker images automatically.

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
