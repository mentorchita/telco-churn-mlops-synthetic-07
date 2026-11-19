#!/usr/bin/env bash
# =============================================================================
# deploy-k3s.sh
# One-shot script: build images + deploy full Telco Churn MLOps stack to k3s
# Usage:
#   export OPENAI_API_KEY="sk-proj-..."
#   chmod +x scripts/deploy-k3s.sh
#   ./scripts/deploy-k3s.sh
# =============================================================================
set -euo pipefail

NAMESPACE="mlops"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
info "Checking required tools..."
command -v docker &>/dev/null || error "docker is not installed (needed to build images)."
command -v curl   &>/dev/null || error "curl is not installed."

# ── Install / start k3s ───────────────────────────────────────────────────────
if ! command -v k3s &>/dev/null; then
  info "k3s not found — installing (Traefik ingress, metrics-server, local-path storage included)..."
  curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644
else
  info "k3s already installed."
  if ! systemctl is-active --quiet k3s; then
    info "Starting k3s service..."
    sudo systemctl start k3s
  fi
fi

# kubectl: use the k3s bundled one if no standalone kubectl is present
if ! command -v kubectl &>/dev/null; then
  kubectl() { k3s kubectl "$@"; }
  export -f kubectl 2>/dev/null || true
fi

if [ ! -r "$KUBECONFIG" ]; then
  warning "$KUBECONFIG is not readable — fixing permissions with sudo."
  sudo chmod 644 "$KUBECONFIG"
fi

info "Waiting for the k3s node to be Ready..."
kubectl wait --for=condition=Ready node --all --timeout=120s

# ── Build images ──────────────────────────────────────────────────────────────
# k3s uses containerd, not Docker, so images built by Docker must be imported.
info "Building Docker images..."
docker build -t churn-predictor:latest "$REPO_ROOT/services/ml/"
docker build -t churn-rag:latest       "$REPO_ROOT/services/rag/"
docker build -t churn-agent:latest     "$REPO_ROOT/services/agent/"
info "All images built successfully."

info "Importing images into k3s containerd..."
for img in churn-predictor:latest churn-rag:latest churn-agent:latest; do
  docker save "$img" | sudo k3s ctr images import -
done

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
# k3s ships the default "local-path" StorageClass, so the PVC works as is.
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

# Images with the :latest tag default to imagePullPolicy=Always, which would try
# to pull from a registry. Force local (imported) images instead.
info "Setting imagePullPolicy=IfNotPresent for local images..."
for d in churn-ml-service churn-rag-service churn-agent-service; do
  kubectl patch deployment "$d" -n "$NAMESPACE" --type=json \
    -p='[{"op":"add","path":"/spec/template/spec/containers/0/imagePullPolicy","value":"IfNotPresent"}]'
done

# ── Wait for rollout ──────────────────────────────────────────────────────────
info "Waiting for deployments to be ready..."
kubectl rollout status deployment/churn-ml-service    -n "$NAMESPACE" --timeout=120s
kubectl rollout status deployment/churn-rag-service   -n "$NAMESPACE" --timeout=120s
kubectl rollout status deployment/churn-agent-service -n "$NAMESPACE" --timeout=120s

# ── /etc/hosts hint ───────────────────────────────────────────────────────────
NODE_IP=$(kubectl get node -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
info "Add to /etc/hosts if not present (Traefik listens on ports 80/443 of the node):"
echo -e "  ${YELLOW}$NODE_IP  mlops.local${NC}"
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
