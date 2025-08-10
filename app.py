import os
import time
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI
from pinecone import Pinecone

# -----------------------------------------------------------------------------
# Config & clients
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "rob-brain")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY")

EMBED_MODEL = "text-embedding-3-small"  # 1536 dims (matches your index)

# OpenAI
oai = OpenAI(api_key=OPENAI_API_KEY)

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Namespaces we use
ALL_NAMESPACES = ["fiction", "nonfiction", "trading", "personal", "reference"]

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="Rob Forever Brain API", version="2.1.0")

# Allow calls from your GPT/action UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class SearchBody(BaseModel):
    query: str
    top_k: int = 8
    namespace: Optional[str] = None   # None or "all" -> search every namespace
    symbols: Optional[List[str]] = None
    tags: Optional[List[str]] = None

class AskBody(BaseModel):
    question: str
    top_k: int = 12
    namespace: Optional[str] = None       # None or "all" => search all
    min_score: Optional[float] = None     # default applied in code
    symbols: Optional[List[str]] = None
    tags: Optional[List[str]] = None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def embed_query(text: str) -> List[float]:
    """Return an embedding vector for a query."""
    t0 = time.time()
    resp = oai.embeddings.create(model=EMBED_MODEL, input=text)
    vec = resp.data[0].embedding
    logging.info(f"🧠 Embed time: {time.time() - t0:.3f}s (dim={len(vec)})")
    return vec

def list_namespaces() -> List[str]:
    """Return namespaces present in the index."""
    try:
        stats = index.describe_index_stats()
        ns = sorted(list(stats.get("namespaces", {}).keys()))
        return ns if ns else ALL_NAMESPACES
    except Exception as e:
        logging.error(f"describe_index_stats failed: {e}")
        return ALL_NAMESPACES

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    """Basic health + what namespaces exist right now."""
    return {
        "ok": True,
        "index": PINECONE_INDEX,
        "namespaces": list_namespaces(),
    }

@app.post("/search")
def search(body: SearchBody, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """
    Legacy search (kept for compatibility).
    If body.namespace is None or 'all', search ALL namespaces and combine results.
    """
    if body.namespace and body.namespace != "all":
        namespaces_to_use = [body.namespace]
    else:
        namespaces_to_use = ALL_NAMESPACES

    logging.info("🔍 /search request")
    logging.info(f"Query: {body.query}")
    logging.info(f"Namespaces searched: {namespaces_to_use}")
    logging.info(f"Top K per namespace: {body.top_k}")

    vec = embed_query(body.query)

    combined_results: List[Dict[str, Any]] = []
    for ns in namespaces_to_use:
        try:
            t0 = time.time()
            res = index.query(
                vector=vec,
                top_k=body.top_k,
                namespace=ns,
                include_metadata=True,
            )
            took = time.time() - t0
            matches = getattr(res, "matches", []) or []
            logging.info(f"📦 Namespace '{ns}' returned {len(matches)} results in {took:.3f}s")

            for m in matches:
                md = getattr(m, "metadata", {}) or {}
                combined_results.append({
                    "namespace": ns,
                    "id": getattr(m, "id", None),
                    "score": getattr(m, "score", None),
                    "text": md.get("text"),
                    "metadata": md,
                })

        except Exception as e:
            logging.error(f"❌ Error searching namespace '{ns}': {e}")

    for i, hit in enumerate(combined_results[:3]):
        snippet = (hit.get("text") or "")[:160].replace("\n", " ")
        logging.info(f"▶ Hit {i+1} [{hit.get('namespace')}:{hit.get('score')}] {snippet!r}")

    logging.info(f"✅ Total combined results: {len(combined_results)}")

    return {
        "namespaces_queried": namespaces_to_use,
        "hits": combined_results,
        "count": len(combined_results),
    }

@app.post("/ask")
def ask(body: AskBody, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """
    Orchestrated endpoint for your GPT:
      1) Always search vectors first (all namespaces by default)
      2) If strong hits exist (>= min_score), return them as context
      3) Otherwise return use_web=true so GPT falls back to web search
    """
    min_score = body.min_score if body.min_score is not None else 0.34
    if body.namespace and body.namespace != "all":
        namespaces_to_use = [body.namespace]
    else:
        namespaces_to_use = ALL_NAMESPACES

    logging.info("🧭 /ask request")
    logging.info(f"Q: {body.question}")
    logging.info(f"Namespaces: {namespaces_to_use} | top_k={body.top_k} | min_score={min_score}")

    vec = embed_query(body.question)

    combined: List[Dict[str, Any]] = []
    for ns in namespaces_to_use:
        try:
            t0 = time.time()
            res = index.query(
                vector=vec,
                top_k=body.top_k,
                namespace=ns,
                include_metadata=True,
            )
            took = time.time() - t0
            matches = getattr(res, "matches", []) or []
            logging.info(f"📚 ns='{ns}' -> {len(matches)} in {took:.3f}s")

            for m in matches:
                md = getattr(m, "metadata", {}) or {}
                combined.append({
                    "namespace": ns,
                    "id": getattr(m, "id", None),
                    "score": float(getattr(m, "score", 0.0) or 0.0),
                    "text": md.get("text"),
                    "title": md.get("title"),
                    "source": md.get("source"),
                    "type": md.get("type"),
                    "tags": md.get("tags"),
                })
        except Exception as e:
            logging.error(f"❌ Error in namespace '{ns}': {e}")

    # Filter by score
    strong = [h for h in combined if h.get("score", 0.0) >= min_score]
    strong_sorted = sorted(strong, key=lambda x: x["score"], reverse=True)

    # Log a peek
    for i, hit in enumerate(strong_sorted[:3]):
        snippet = (hit.get("text") or "")[:160].replace("\n", " ")
        logging.info(f"⭐ strong[{i}] {hit['namespace']} {hit['score']:.3f} {hit.get('title')!r} :: {snippet!r}")

    if not strong_sorted:
        logging.info("➡️ No strong matches. Instructing client to use web.")
        return {
            "answered_from": "none",
            "use_web": True,
            "reason": "no_hits_above_threshold",
            "min_score_used": min_score,
            "namespaces_queried": namespaces_to_use,
        }

    # Build compact context for the GPT to answer from
    context = []
    for h in strong_sorted[: body.top_k]:
        context.append({
            "namespace": h["namespace"],
            "score": h["score"],
            "title": h.get("title"),
            "source": h.get("source"),
            "text": h.get("text"),
        })

    return {
        "answered_from": "vector",
        "use_web": False,
        "min_score_used": min_score,
        "namespaces_queried": namespaces_to_use,
        "hits_count": len(strong_sorted),
        "context": context,
        "suggested_instruction": (
            "Use the context excerpts to answer the user's question. "
            "Cite titles/sources from context. If something is unclear or missing, say so."
        ),
    }
