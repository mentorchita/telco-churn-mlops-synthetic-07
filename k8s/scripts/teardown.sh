#!/usr/bin/env bash
# =============================================================================
# teardown.sh
# Remove all Telco Churn MLOps resources from the cluster.
# Usage:
#   ./scripts/teardown.sh              # removes namespace + resources
#   ./scripts/teardown.sh --all        # also stops minikube
# =============================================================================
set -euo pipefail

NAMESPACE="mlops"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC}  $*"; }

info "Deleting all resources in namespace '$NAMESPACE'..."
kubectl delete namespace "$NAMESPACE" --ignore-not-found=true

info "Resources removed."

if [[ "${1:-}" == "--all" ]]; then
  warning "Stopping minikube..."
  minikube stop
  info "minikube stopped."
fi

echo ""
info "Teardown complete. To redeploy: ./scripts/deploy-minikube.sh"
