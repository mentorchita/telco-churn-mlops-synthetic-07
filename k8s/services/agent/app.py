"""
Telco Churn Agent Service
LLM agent that orchestrates ML + RAG services to analyse a customer
and produce a churn-risk report with recommended retention actions.
"""

import os
import logging
import json
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telco Churn Agent Service",
    description="LLM agent: combines ML prediction + RAG context → retention report",
    version="1.0.0",
)

# ── Config ─────────────────────────────────────────────────────────────────
ML_URL = os.getenv("ML_SERVICE_URL", "http://churn-ml-svc:80")
RAG_URL = os.getenv("RAG_SERVICE_URL", "http://churn-rag-svc:80")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))

openai_client = None


@app.on_event("startup")
def init_openai():
    global openai_client
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info(f"OpenAI client ready, model={LLM_MODEL}")
    else:
        logger.warning("OPENAI_API_KEY not set — agent will use rule-based fallback")


# ── Schemas ────────────────────────────────────────────────────────────────
class CustomerProfile(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    tenure: int = Field(..., ge=0, le=72)
    monthly_charges: float = Field(..., ge=0)
    total_charges: Optional[float] = None
    contract_type: str = Field("month-to-month")
    internet_service: str = Field("Fiber optic")
    payment_method: str = Field("Electronic check")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST-7890",
                "tenure": 8,
                "monthly_charges": 79.90,
                "contract_type": "month-to-month",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check",
            }
        }


class AgentResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str                  # "low" | "medium" | "high"
    prediction: str
    rag_context: list[str]
    recommendation: str
    model_version: str


# ── Internal helpers ───────────────────────────────────────────────────────
async def call_ml(profile: CustomerProfile) -> dict:
    payload = {
        "tenure": profile.tenure,
        "monthly_charges": profile.monthly_charges,
        "total_charges": profile.total_charges,
        "contract_type": profile.contract_type,
        "internet_service": profile.internet_service,
        "payment_method": profile.payment_method,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(f"{ML_URL}/predict", json=payload)
        r.raise_for_status()
        return r.json()


async def call_rag(question: str, top_k: int = 3) -> list[str]:
    payload = {"question": question, "top_k": top_k}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(f"{RAG_URL}/query", json=payload)
            r.raise_for_status()
            data = r.json()
            return [item["document"] for item in data.get("results", [])]
        except Exception as e:
            logger.warning(f"RAG query failed: {e} — proceeding without context")
            return []


def risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "high"
    if prob >= 0.4:
        return "medium"
    return "low"


def rule_based_recommendation(prob: float, profile: CustomerProfile) -> str:
    """Fallback when no OpenAI key is configured."""
    if prob < 0.4:
        return (
            f"Customer {profile.customer_id} has LOW churn risk ({prob:.0%}). "
            "No immediate action needed. Continue standard engagement."
        )
    if prob < 0.7:
        actions = []
        if profile.contract_type == "month-to-month":
            actions.append("offer a discounted one-year contract")
        if profile.monthly_charges > 70:
            actions.append("review their plan for a cost-saving bundle")
        if profile.tenure < 12:
            actions.append("assign a dedicated onboarding specialist")
        return (
            f"Customer {profile.customer_id} has MEDIUM churn risk ({prob:.0%}). "
            f"Recommended actions: {'; '.join(actions) or 'schedule a satisfaction call'}."
        )
    return (
        f"Customer {profile.customer_id} has HIGH churn risk ({prob:.0%}). "
        "URGENT: Escalate to retention team immediately. "
        "Offer significant discount or contract upgrade within 48 hours."
    )


def llm_recommendation(profile: CustomerProfile, ml_result: dict, rag_context: list[str]) -> str:
    if openai_client is None:
        return rule_based_recommendation(ml_result["churn_probability"], profile)

    context_block = "\n".join(f"- {c}" for c in rag_context) if rag_context else "No additional context available."

    prompt = f"""You are a senior customer retention analyst at a telecom company.

CUSTOMER PROFILE:
- ID: {profile.customer_id}
- Tenure: {profile.tenure} months
- Monthly charges: ${profile.monthly_charges:.2f}
- Contract: {profile.contract_type}
- Internet service: {profile.internet_service}
- Payment method: {profile.payment_method}

ML MODEL OUTPUT:
- Churn probability: {ml_result['churn_probability']:.1%}
- Prediction: {ml_result['prediction']}
- Risk level: {risk_level(ml_result['churn_probability'])}

RELEVANT KNOWLEDGE BASE CONTEXT:
{context_block}

Write a concise (3-4 sentences) retention recommendation for this customer.
Be specific and actionable. Address the key risk factors directly."""

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_ready": openai_client is not None,
        "ml_url": ML_URL,
        "rag_url": RAG_URL,
    }


@app.get("/")
def root():
    return {"service": "churn-agent-service", "version": "1.0.0"}


@app.post("/run", response_model=AgentResponse)
async def run(profile: CustomerProfile):
    logger.info(f"Agent run for customer {profile.customer_id}")

    # Step 1 — ML prediction
    try:
        ml_result = await call_ml(profile)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ML service error: {e}")

    prob = ml_result["churn_probability"]
    level = risk_level(prob)

    # Step 2 — RAG context retrieval
    rag_question = (
        f"What are the main reasons for churn in customers with "
        f"{profile.contract_type} contract and {profile.internet_service} internet service?"
    )
    rag_docs = await call_rag(rag_question, top_k=3)

    # Step 3 — LLM recommendation
    recommendation = llm_recommendation(profile, ml_result, rag_docs)

    logger.info(f"Customer {profile.customer_id}: prob={prob:.3f}, risk={level}")

    return AgentResponse(
        customer_id=profile.customer_id,
        churn_probability=prob,
        risk_level=level,
        prediction=ml_result["prediction"],
        rag_context=rag_docs,
        recommendation=recommendation,
        model_version=ml_result.get("model_version", "unknown"),
    )
