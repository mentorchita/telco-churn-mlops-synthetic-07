#!/usr/bin/env bash
# =============================================================================
# ingest-knowledge.sh
# Load data/knowledge_base.json into the RAG service (ChromaDB) via /ingest.
#
# Usage:
#   ./k8s/scripts/ingest-knowledge.sh              # port-forward + ingest
#   ./k8s/scripts/ingest-knowledge.sh --clear       # wipe collection first
#   RAG_PORT=8010 ./k8s/scripts/ingest-knowledge.sh # use a different local port
#   RAG_URL=http://mlops.local/rag ./k8s/scripts/ingest-knowledge.sh
#                                                    # skip port-forward, use Ingress instead
#
# Requires: kubectl, curl, python3 (no external K8s deps for JSON handling).
# =============================================================================
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
NAMESPACE="${NAMESPACE:-mlops}"
SERVICE="${SERVICE:-churn-rag-svc}"
RAG_PORT="${RAG_PORT:-8002}"          # local port, matches ingress.yaml / demo.sh convention
RAG_URL="${RAG_URL:-}"                # set to skip port-forward (e.g. Ingress URL)
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-60}" # seconds to wait for /health to be ready
CLEAR_FIRST=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KB_FILE="${KB_FILE:-$REPO_ROOT/data/knowledge_base.json}"

# ── Pretty output ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Args ─────────────────────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --clear) CLEAR_FIRST=true ;;
    -h|--help)
      grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) warning "Unknown argument: $arg (ignored)" ;;
  esac
done

# ── Pre-flight checks ────────────────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || { error "python3 is required."; exit 1; }
command -v curl    >/dev/null 2>&1 || { error "curl is required.";    exit 1; }

if [[ ! -f "$KB_FILE" ]]; then
  error "Knowledge base file not found: $KB_FILE"
  exit 1
fi
info "Knowledge base file: $KB_FILE"

# ── Port-forward (unless RAG_URL was supplied) ──────────────────────────────
PF_PID=""
cleanup() {
  if [[ -n "$PF_PID" ]]; then
    info "Stopping port-forward (pid $PF_PID)..."
    kill "$PF_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -z "$RAG_URL" ]]; then
  command -v kubectl >/dev/null 2>&1 || { error "kubectl is required (or set RAG_URL to skip port-forward)."; exit 1; }

  info "Port-forwarding svc/$SERVICE -> localhost:$RAG_PORT (namespace: $NAMESPACE)..."
  kubectl port-forward "svc/$SERVICE" "$RAG_PORT:80" -n "$NAMESPACE" &>/tmp/pf-rag-ingest.log &
  PF_PID=$!
  RAG_URL="http://localhost:$RAG_PORT"
else
  info "Using external RAG_URL: $RAG_URL"
fi

# ── Wait for the service to be ready ────────────────────────────────────────
info "Waiting for RAG service health (timeout: ${HEALTH_TIMEOUT}s)..."
elapsed=0
until curl -s -f "$RAG_URL/health" 2>/dev/null | grep -q '"chroma_ready":true'; do
  sleep 2
  elapsed=$((elapsed + 2))
  if (( elapsed >= HEALTH_TIMEOUT )); then
    error "RAG service not ready after ${HEALTH_TIMEOUT}s. Check: kubectl logs -l app=churn-rag -n $NAMESPACE"
    exit 1
  fi
done
info "RAG service is ready."

# ── Optionally clear the collection first (idempotent re-ingestion) ─────────
if [[ "$CLEAR_FIRST" == true ]]; then
  warning "Clearing existing collection (--clear)..."
  curl -s -X DELETE "$RAG_URL/collection" | python3 -m json.tool
fi

# ── Build the /ingest payload from knowledge_base.json ──────────────────────
info "Building ingest payload from $(basename "$KB_FILE")..."
PAYLOAD="$(python3 - "$KB_FILE" <<'PYEOF'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    entries = json.load(f)

documents, ids, metadatas = [], [], []
for entry in entries:
    title = entry.get("title", "")
    content = entry.get("content", "")
    documents.append(f"{title}: {content}" if title else content)
    ids.append(f"kb_{entry.get('id')}")
    metadatas.append({"source": "knowledge_base.json", "title": title})

print(json.dumps({"documents": documents, "ids": ids, "metadatas": metadatas}))
PYEOF
)"

DOC_COUNT="$(python3 -c 'import json,sys; print(len(json.loads(sys.argv[1])["documents"]))' "$PAYLOAD")"
info "Prepared $DOC_COUNT document(s) for ingestion."

# ── Ingest ───────────────────────────────────────────────────────────────────
info "POST $RAG_URL/ingest ..."
RESPONSE="$(curl -s -f -X POST "$RAG_URL/ingest" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")" || { error "Ingest request failed."; exit 1; }

echo "$RESPONSE" | python3 -m json.tool

TOTAL="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["total_in_collection"])' "$RESPONSE")"
info "Ingestion complete — $TOTAL document(s) now in the collection."

echo ""
info "Try a query:"
echo "  curl -s -X POST \"$RAG_URL/query\" -H \"Content-Type: application/json\" \\"
echo "    -d '{\"question\":\"How do I reset my modem?\",\"top_k\":3}' | python3 -m json.tool"
