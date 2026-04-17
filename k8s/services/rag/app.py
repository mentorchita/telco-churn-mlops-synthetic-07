"""
Telco Churn RAG Service
FastAPI app: ingest documents + answer questions via ChromaDB + OpenAI embeddings.
"""

import os
import logging
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telco Churn RAG Service",
    description="Document ingestion and retrieval-augmented query service",
    version="1.0.0",
)

# ── ChromaDB setup ─────────────────────────────────────────────────────────
CHROMA_PATH = os.getenv("CHROMA_PATH", "/data/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "telco_churn_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

chroma_client = None
collection = None


@app.on_event("startup")
def init_chroma():
    global chroma_client, collection
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBEDDING_MODEL,
        ) if OPENAI_API_KEY else embedding_functions.DefaultEmbeddingFunction()

        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB ready at {CHROMA_PATH}, collection '{COLLECTION_NAME}', {collection.count()} docs")
    except Exception as e:
        logger.error(f"ChromaDB init failed: {e}")


# ── Schemas ────────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    documents: list[str] = Field(..., description="List of text chunks to index")
    ids: Optional[list[str]] = Field(None, description="Optional IDs; auto-generated if omitted")
    metadatas: Optional[list[dict]] = Field(None, description="Optional metadata per document")

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    "Customers with month-to-month contracts churn at 43% vs 11% for two-year.",
                    "Fiber optic users have higher churn rates due to competitive market pricing.",
                ],
                "metadatas": [
                    {"source": "analysis_report", "topic": "contract"},
                    {"source": "analysis_report", "topic": "internet_service"},
                ],
            }
        }


class IngestResponse(BaseModel):
    indexed: int
    total_in_collection: int


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question to answer")
    top_k: Optional[int] = Field(None, description="Number of results (overrides env default)")
    filter_metadata: Optional[dict] = Field(None, description="Optional ChromaDB metadata filter")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Why do fiber optic customers churn more often?",
                "top_k": 3,
            }
        }


class QueryResult(BaseModel):
    document: str
    distance: float
    metadata: dict


class QueryResponse(BaseModel):
    question: str
    results: list[QueryResult]
    total_in_collection: int


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    count = collection.count() if collection else -1
    return {"status": "ok", "chroma_ready": collection is not None, "doc_count": count}


@app.get("/")
def root():
    return {"service": "churn-rag-service", "version": "1.0.0"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    if collection is None:
        raise HTTPException(status_code=503, detail="ChromaDB not ready")

    ids = req.ids or [f"doc_{i}_{hash(d) & 0xFFFFFF}" for i, d in enumerate(req.documents)]
    metadatas = req.metadatas or [{} for _ in req.documents]

    collection.upsert(documents=req.documents, ids=ids, metadatas=metadatas)
    total = collection.count()
    logger.info(f"Ingested {len(req.documents)} docs — total: {total}")
    return IngestResponse(indexed=len(req.documents), total_in_collection=total)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if collection is None:
        raise HTTPException(status_code=503, detail="ChromaDB not ready")
    if collection.count() == 0:
        raise HTTPException(status_code=404, detail="No documents indexed yet. POST to /ingest first.")

    k = req.top_k or TOP_K
    kwargs = {"query_texts": [req.question], "n_results": min(k, collection.count())}
    if req.filter_metadata:
        kwargs["where"] = req.filter_metadata

    results = collection.query(**kwargs)

    items = []
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        items.append(QueryResult(document=doc, distance=round(dist, 4), metadata=meta or {}))

    logger.info(f"Query '{req.question[:60]}' → {len(items)} results")
    return QueryResponse(question=req.question, results=items, total_in_collection=collection.count())


@app.delete("/collection")
def clear_collection():
    """Clear all documents — useful for lab resets."""
    if collection is None:
        raise HTTPException(status_code=503, detail="ChromaDB not ready")
    count_before = collection.count()
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
    logger.warning(f"Cleared {count_before} documents from collection")
    return {"deleted": count_before, "remaining": collection.count()}
