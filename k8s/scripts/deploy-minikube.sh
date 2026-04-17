#!/usr/bin/env bash
# =============================================================================
# deploy-minikube.sh
# One-shot script: build images + deploy full Telco Churn MLOps stack to minikube
# Usage:
#   export OPENAI_API_KEY="sk-proj-..."
#   chmod +x scripts/deploy-minikube.sh
#   ./scripts/deploy-minikube.sh
# =============================================================================
set -euo pipefail

NAMESPACE="mlops"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
info "Checking required tools..."
for cmd in minikube kubectl docker; do
  command -v "$cmd" &>/dev/null || error "$cmd is not installed."
done

# ── Start minikube ────────────────────────────────────────────────────────────
info "Starting minikube..."
if ! minikube status | grep -q "Running"; then
  minikube start --cpus=4 --memory=8192 --driver=docker
else
  info "minikube already running."
fi

info "Enabling addons..."
minikube addons enable ingress       2>/dev/null || true
minikube addons enable metrics-server 2>/dev/null || true

# ── Point Docker to minikube ──────────────────────────────────────────────────
info "Configuring Docker to use minikube registry..."
eval "$(minikube docker-env)"

# ── Build images ──────────────────────────────────────────────────────────────
info "Building Docker images..."
docker build -t churn-predictor:latest "$REPO_ROOT/services/ml/"
docker build -t churn-rag:latest       "$REPO_ROOT/services/rag/"
docker build -t churn-agent:latest     "$REPO_ROOT/services/agent/"
info "All images built successfully."

# ── Namespace ─────────────────────────────────────────────────────────────────
info "Creating namespace..."
kubectl apply -f "$REPO_ROOT/k8s/base/namespace.yaml"

# ── Secrets ───────────────────────────────────────────────────────────────────
info "Creating secrets..."
if [ -z "${OPENAI_API_KEY:-}" ]; then
  warning "OPENAI_API_KEY not set — agent will use rule-based fallback"
  OPENAI_API_KEY="placeholder-not-set"
fi

kubectl create secret generic llm-secrets \
  --from-literal=openai_api_key="$OPENAI_API_KEY" \
  --from-literal=mlflow_tracking_uri="http://mlflow:5000" \
  --from-literal=mlflow_tracking_token="demo-token" \
  -n "$NAMESPACE" \
  --dry-run=client -o yaml | kubectl apply -f -

# ── ConfigMaps ────────────────────────────────────────────────────────────────
info "Applying ConfigMaps..."
kubectl apply -f "$REPO_ROOT/k8s/base/configmap.yaml"

# ── Storage ───────────────────────────────────────────────────────────────────
info "Creating PersistentVolumeClaim for RAG storage..."
kubectl apply -f "$REPO_ROOT/k8s/base/rag-pvc.yaml"

# ── Deployments ───────────────────────────────────────────────────────────────
info "Deploying services..."
kubectl apply -f "$REPO_ROOT/k8s/base/ml-deployment.yaml"
kubectl apply -f "$REPO_ROOT/k8s/base/ml-service.yaml"
kubectl apply -f "$REPO_ROOT/k8s/base/ml-hpa.yaml"
kubectl apply -f "$REPO_ROOT/k8s/base/rag-deployment.yaml"
kubectl apply -f "$REPO_ROOT/k8s/base/rag-service.yaml"
kubectl apply -f "$REPO_ROOT/k8s/base/agent-deployment.yaml"
kubectl apply -f "$REPO_ROOT/k8s/base/agent-service.yaml"
kubectl apply -f "$REPO_ROOT/k8s/base/ingress.yaml"

# ── Wait for rollout ──────────────────────────────────────────────────────────
info "Waiting for deployments to be ready..."
kubectl rollout status deployment/churn-ml-service    -n "$NAMESPACE" --timeout=120s
kubectl rollout status deployment/churn-rag-service   -n "$NAMESPACE" --timeout=120s
kubectl rollout status deployment/churn-agent-service -n "$NAMESPACE" --timeout=120s

# ── /etc/hosts hint ───────────────────────────────────────────────────────────
MINIKUBE_IP=$(minikube ip)
info "Add to /etc/hosts if not present:"
echo -e "  ${YELLOW}$MINIKUBE_IP  mlops.local${NC}"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
info "Deployment complete!"
echo ""
echo -e "${GREEN}Resources in namespace '$NAMESPACE':${NC}"
kubectl get pods,svc,ingress -n "$NAMESPACE"
echo ""
echo -e "${GREEN}Quick test (port-forward):${NC}"
echo "  kubectl port-forward svc/churn-ml-svc 8001:80 -n $NAMESPACE &"
echo "  curl http://localhost:8001/health"
echo "  curl -X POST http://localhost:8001/predict \\"
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"tenure":8,"monthly_charges":79.9,"contract_type":"month-to-month","internet_service":"Fiber optic","payment_method":"Electronic check"}'"'"
echo ""
echo -e "${GREEN}Or run the full demo:${NC}"
echo "  ./scripts/demo.sh"
