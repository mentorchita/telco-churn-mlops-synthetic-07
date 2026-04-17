# Kubernetes Deployment — Telco Churn MLOps Stack

Topic 7 materials for the **Modern MLOps · LLMOps · AgenticOps** course.  
Deploys three microservices (ML inference, RAG retrieval, LLM Agent) to a local
Kubernetes cluster (minikube or k3s).

---

## Repository Structure

```
telco-churn-mlops-synthetic-05/
├── k8s/
│   ├── base/                       ← Core K8s manifests
│   │   ├── namespace.yaml
│   │   ├── configmap.yaml          ← ml-config, rag-config, agent-config
│   │   ├── secrets.yaml            ← llm-secrets (placeholder — see below)
│   │   ├── rag-pvc.yaml            ← PersistentVolumeClaim for ChromaDB
│   │   ├── ml-deployment.yaml      ← churn-predictor FastAPI service
│   │   ├── ml-service.yaml
│   │   ├── ml-hpa.yaml             ← HorizontalPodAutoscaler 2→10 pods
│   │   ├── rag-deployment.yaml     ← LangChain + ChromaDB service
│   │   ├── rag-service.yaml
│   │   ├── agent-deployment.yaml   ← LLM agent orchestrator
│   │   ├── agent-service.yaml
│   │   ├── ingress.yaml            ← NGINX ingress: /ml /rag /agent
│   │   └── kustomization.yaml
│   └── overlays/
│       ├── dev/kustomization.yaml  ← 1 replica, :dev tags, DEBUG logs
│       └── prod/kustomization.yaml ← 3 replicas, pinned tags, WARNING logs
│
├── services/
│   ├── ml/
│   │   ├── app.py                  ← FastAPI sklearn inference (port 8000)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── rag/
│   │   ├── app.py                  ← FastAPI ChromaDB service (port 8001)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── agent/
│       ├── app.py                  ← FastAPI LLM agent (port 8002)
│       ├── Dockerfile
│       └── requirements.txt
│
└── scripts/
    ├── deploy-minikube.sh          ← One-shot build + deploy
    ├── demo.sh                     ← Interactive 6-step demo
    ├── rotate-secret.sh            ← Rotate OpenAI key + rolling restart
    └── teardown.sh                 ← Remove all resources
```

---

## Prerequisites

| Tool       | Min version | Install |
|------------|-------------|---------|
| minikube   | 1.32+       | `brew install minikube` |
| kubectl    | 1.28+       | `brew install kubectl` |
| Docker     | 24+         | [docker.com](https://docker.com) |
| Python     | 3.11+       | For local testing |

---

## Quick Start (10 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/mentorchita/telco-churn-mlops-synthetic-05
cd telco-churn-mlops-synthetic-05

# 2. Set your OpenAI API key (optional — fallback runs without it)
export OPENAI_API_KEY="sk-proj-..."

# 3. One-shot deploy
chmod +x scripts/*.sh
./scripts/deploy-minikube.sh

# 4. Run the interactive demo
./scripts/demo.sh
```

---

## Manual Step-by-Step Deploy

### 1. Start minikube
```bash
minikube start --cpus=4 --memory=8192 --driver=docker
minikube addons enable ingress
minikube addons enable metrics-server
eval $(minikube docker-env)
```

### 2. Build images
```bash
docker build -t churn-predictor:latest services/ml/
docker build -t churn-rag:latest       services/rag/
docker build -t churn-agent:latest     services/agent/
```

### 3. Create namespace + secrets
```bash
kubectl apply -f k8s/base/namespace.yaml

kubectl create secret generic llm-secrets \
  --from-literal=openai_api_key="${OPENAI_API_KEY:-placeholder}" \
  -n mlops
```

### 4. Apply all manifests
```bash
kubectl apply -f k8s/base/configmap.yaml
kubectl apply -f k8s/base/rag-pvc.yaml
kubectl apply -f k8s/base/ -n mlops
```

### 5. Verify
```bash
kubectl get pods,svc,ingress -n mlops
kubectl rollout status deployment/churn-ml-service -n mlops
```

---

## API Reference

### ML Service (port 8000 / path `/ml`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/predict` | Predict churn probability |
| GET | `/metrics` | Model metadata |

**Example `/predict` request:**
```json
{
  "tenure": 8,
  "monthly_charges": 79.90,
  "contract_type": "month-to-month",
  "internet_service": "Fiber optic",
  "payment_method": "Electronic check"
}
```

**Response:**
```json
{
  "churn_probability": 0.7421,
  "prediction": "churn",
  "model_version": "1.0.0",
  "threshold_used": 0.5
}
```

---

### RAG Service (port 8001 / path `/rag`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + doc count |
| POST | `/ingest` | Index documents into ChromaDB |
| POST | `/query` | Semantic search |
| DELETE | `/collection` | Clear all documents |

---

### Agent Service (port 8002 / path `/agent`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + dependency status |
| POST | `/run` | Full pipeline: ML + RAG + LLM → report |

---

## Lab Exercises

### Scale deployments
```bash
kubectl scale deployment churn-ml-service --replicas=4 -n mlops
kubectl get pods -n mlops -w
kubectl scale deployment churn-ml-service --replicas=2 -n mlops
```

### Rolling update (simulate new model version)
```bash
# Tag image as v1.1
docker tag churn-predictor:latest churn-predictor:1.1

# Trigger rolling update
kubectl set image deployment/churn-ml-service \
  churn-ml=churn-predictor:1.1 -n mlops

kubectl rollout status deployment/churn-ml-service -n mlops
kubectl rollout history deployment/churn-ml-service -n mlops

# Rollback
kubectl rollout undo deployment/churn-ml-service -n mlops
```

### HPA in action
```bash
kubectl get hpa -n mlops
kubectl top pods -n mlops         # requires metrics-server addon
```

### Use Kustomize overlays
```bash
# Deploy dev overlay
kubectl apply -k k8s/overlays/dev/

# Deploy prod overlay
kubectl apply -k k8s/overlays/prod/
```

### Rotate the OpenAI API key
```bash
export OPENAI_API_KEY="sk-proj-new-key-here"
./scripts/rotate-secret.sh
```

---

## Teardown
```bash
./scripts/teardown.sh           # Remove namespace + all resources
./scripts/teardown.sh --all     # Also stop minikube
```

---

## Notes

- `imagePullPolicy: IfNotPresent` is set on all deployments so minikube uses
  locally built images without a registry.
- The agent and RAG services work without an OpenAI key — they fall back to
  rule-based logic and default ChromaDB embeddings respectively.
- Secrets in `k8s/base/secrets.yaml` contain placeholder values only.
  Always use `kubectl create secret` or a secrets manager in production.
