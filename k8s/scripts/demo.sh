#!/usr/bin/env bash
# =============================================================================
# demo.sh
# Interactive demonstration of the Telco Churn MLOps stack running on K8s.
# Opens port-forwards, injects demo data, and fires test requests.
#
# Usage:
#   ./scripts/demo.sh
# =============================================================================
set -euo pipefail

NAMESPACE="mlops"
ML_PORT=8001
RAG_PORT=8002
AGENT_PORT=8003

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${GREEN}▶${NC} $*"; }
section() { echo -e "\n${CYAN}${BOLD}══ $* ══${NC}"; }
pause()   { echo -e "${YELLOW}[Press Enter to continue]${NC}"; read -r; }

# ── Port-forward all services in background ───────────────────────────────────
section "Starting port-forwards"
info "Forwarding ML service    → localhost:$ML_PORT"
kubectl port-forward svc/churn-ml-svc    "$ML_PORT:80"    -n "$NAMESPACE" &>/tmp/pf-ml.log    &
PF_ML=$!
info "Forwarding RAG service   → localhost:$RAG_PORT"
kubectl port-forward svc/churn-rag-svc   "$RAG_PORT:80"   -n "$NAMESPACE" &>/tmp/pf-rag.log   &
PF_RAG=$!
info "Forwarding Agent service → localhost:$AGENT_PORT"
kubectl port-forward svc/churn-agent-svc "$AGENT_PORT:80" -n "$NAMESPACE" &>/tmp/pf-agent.log &
PF_AGENT=$!

# Cleanup on exit
cleanup() {
  info "Cleaning up port-forwards..."
  kill $PF_ML $PF_RAG $PF_AGENT 2>/dev/null || true
}
trap cleanup EXIT

sleep 3  # wait for port-forwards to establish

# ── Health checks ─────────────────────────────────────────────────────────────
section "Step 1: Health Checks"
echo ""
info "ML Service health:"
curl -s "http://localhost:$ML_PORT/health" | python3 -m json.tool
echo ""
info "RAG Service health:"
curl -s "http://localhost:$RAG_PORT/health" | python3 -m json.tool
echo ""
info "Agent Service health:"
curl -s "http://localhost:$AGENT_PORT/health" | python3 -m json.tool
pause

# ── ML Service predictions ────────────────────────────────────────────────────
section "Step 2: ML Service — Churn Predictions"

echo ""
info "High-risk customer (short tenure, month-to-month, fiber optic):"
curl -s -X POST "http://localhost:$ML_PORT/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure":3,"monthly_charges":89.50,"contract_type":"month-to-month","internet_service":"Fiber optic","payment_method":"Electronic check"}' \
  | python3 -m json.tool

echo ""
info "Low-risk customer (long tenure, two-year contract):"
curl -s -X POST "http://localhost:$ML_PORT/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure":48,"monthly_charges":45.00,"contract_type":"two_year","internet_service":"DSL","payment_method":"Bank transfer"}' \
  | python3 -m json.tool
pause

# ── RAG Service: ingest + query ───────────────────────────────────────────────
section "Step 3: RAG Service — Ingest Knowledge & Query"

echo ""
info "Ingesting churn knowledge base documents..."
curl -s -X POST "http://localhost:$RAG_PORT/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "Customers with month-to-month contracts churn at 43% vs 11% for two-year contracts.",
      "Fiber optic internet customers show 30% higher churn rates due to intense market competition and price sensitivity.",
      "Electronic check payment users have a 45% churn rate compared to 15-18% for automatic payment methods.",
      "Customers in their first 12 months are 3x more likely to churn than customers with 2+ years tenure.",
      "Offering a discounted annual contract to month-to-month customers reduces churn by 22% on average.",
      "Customers with monthly charges over $80 are the highest churn segment and respond well to bundle discounts."
    ],
    "metadatas": [
      {"source": "churn_analysis_2024", "topic": "contract_type"},
      {"source": "churn_analysis_2024", "topic": "internet_service"},
      {"source": "churn_analysis_2024", "topic": "payment_method"},
      {"source": "churn_analysis_2024", "topic": "tenure"},
      {"source": "retention_playbook", "topic": "contract_upgrade"},
      {"source": "retention_playbook", "topic": "pricing"}
    ]
  }' | python3 -m json.tool

echo ""
info "Querying the knowledge base:"
curl -s -X POST "http://localhost:$RAG_PORT/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Why do fiber optic customers on month-to-month contracts churn more?","top_k":3}' \
  | python3 -m json.tool
pause

# ── Agent Service: full pipeline ──────────────────────────────────────────────
section "Step 4: Agent Service — Full Pipeline (ML + RAG + LLM)"

echo ""
info "Running agent analysis on a high-risk customer..."
curl -s -X POST "http://localhost:$AGENT_PORT/run" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST-7890",
    "tenure": 8,
    "monthly_charges": 79.90,
    "contract_type": "month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check"
  }' | python3 -m json.tool

echo ""
info "Running agent analysis on a low-risk customer..."
curl -s -X POST "http://localhost:$AGENT_PORT/run" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST-1234",
    "tenure": 36,
    "monthly_charges": 55.00,
    "contract_type": "one_year",
    "internet_service": "DSL",
    "payment_method": "Credit card"
  }' | python3 -m json.tool
pause

# ── Scaling demo ──────────────────────────────────────────────────────────────
section "Step 5: Scale the ML Service"
info "Current pod count:"
kubectl get pods -n "$NAMESPACE" -l app=churn-ml

echo ""
info "Scaling ML service from 2 → 4 replicas..."
kubectl scale deployment churn-ml-service --replicas=4 -n "$NAMESPACE"
kubectl rollout status deployment/churn-ml-service -n "$NAMESPACE" --timeout=60s

echo ""
info "New pod count:"
kubectl get pods -n "$NAMESPACE" -l app=churn-ml

echo ""
info "Scaling back to 2..."
kubectl scale deployment churn-ml-service --replicas=2 -n "$NAMESPACE"
pause

# ── Rolling update demo ───────────────────────────────────────────────────────
section "Step 6: Rolling Update — Simulate Model v1.1 Deploy"
info "Triggering rolling update via annotation (simulates new image tag)..."
kubectl annotate deployment churn-ml-service \
  kubernetes.io/change-cause="demo-rolling-update-v1.1" \
  -n "$NAMESPACE" --overwrite

kubectl rollout status deployment/churn-ml-service -n "$NAMESPACE" --timeout=60s

echo ""
info "Rollout history:"
kubectl rollout history deployment/churn-ml-service -n "$NAMESPACE"

echo ""
info "Demo complete! All K8s resources:"
kubectl get all -n "$NAMESPACE"
