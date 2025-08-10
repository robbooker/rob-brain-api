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
app = FastAPI(title="Rob Forever Brain API", version="2.0.0")

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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def embed_query(text: str) -> List[float]:
    """Return an embedding vector for a query."""
    t0 = time.time()
    resp = oai.embeddings.create(
        model="text-embedding-3-large",  # 3072-dim to match your Pinecone index
        input=text
    )
    vec = resp.data[0].embedding
    logging.info(f"🧠 Embed time: {time.time() - t0:.3f}s (dim={len(vec)})")
    return vec


def list_namespaces() -> List[str]:
    """Return namespaces present in the index."""
    try:
        stats = index.describe_index_stats()
        # 'namespaces' may be a dict of namespace->stats; fall back to ALL_NAMESPACES
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
    Search Pinecone.
    - If body.namespace is None or 'all', search ALL namespaces and combine results.
    - Otherwise search just the provided namespace.
    """
    # Decide namespaces
    if body.namespace and body.namespace != "all":
        namespaces_to_use = [body.namespace]
    else:
        namespaces_to_use = ALL_NAMESPACES

    # Log incoming request
    logging.info("🔍 Search request received")
    logging.info(f"Query: {body.query}")
    logging.info(f"Namespaces searched: {namespaces_to_use}")
    logging.info(f"Top K per namespace: {body.top_k}")

    # Embed once
    vec = embed_query(body.query)

    # Query each namespace
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

            # Normalize match objects to plain dicts the GPT can read easily
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

    # Optional: quick peek at the first 3 snippets in logs
    for i, hit in enumerate(combined_results[:3]):
        snippet = (hit.get("text") or "")[:160].replace("\n", " ")
        logging.info(f"▶ Hit {i+1} [{hit.get('namespace')}:{hit.get('score')}] {snippet!r}")

    logging.info(f"✅ Total combined results: {len(combined_results)}")

    return {
        "namespaces_queried": namespaces_to_use,
        "hits": combined_results,
        "count": len(combined_results),
    }
