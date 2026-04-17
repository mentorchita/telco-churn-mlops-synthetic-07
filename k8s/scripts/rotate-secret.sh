#!/usr/bin/env bash
# =============================================================================
# rotate-secret.sh
# Rotate the OpenAI API key in K8s Secret and trigger rolling restarts.
# Usage:
#   export OPENAI_API_KEY="sk-proj-new-key"
#   ./scripts/rotate-secret.sh
# =============================================================================
set -euo pipefail

NAMESPACE="mlops"
GREEN='\033[0;32m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC}  $*"; }

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY env variable must be set."
  exit 1
fi

info "Patching llm-secrets with new key..."
kubectl create secret generic llm-secrets \
  --from-literal=openai_api_key="$OPENAI_API_KEY" \
  -n "$NAMESPACE" \
  --dry-run=client -o yaml | kubectl apply -f -

info "Triggering rolling restart of services that consume the secret..."
kubectl rollout restart deployment/churn-rag-service   -n "$NAMESPACE"
kubectl rollout restart deployment/churn-agent-service -n "$NAMESPACE"

kubectl rollout status deployment/churn-rag-service   -n "$NAMESPACE" --timeout=90s
kubectl rollout status deployment/churn-agent-service -n "$NAMESPACE" --timeout=90s

info "Secret rotation complete. New pods are using the updated key."
