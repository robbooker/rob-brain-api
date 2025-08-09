# app.py — Rob’s Forever Brain API (fiction/nonfiction/trading/personal/reference)
# Unified search across ALL namespaces by default (+ tiny reranker)
#
# Env vars required:
#   OPENAI_API_KEY
#   PINECONE_API_KEY
#   PINECONE_INDEX      (e.g., "rob-brain")
#   FOREVER_BRAIN_BEARER  (token used for Bearer auth from your GPT)

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os

from openai import OpenAI
from pinecone import Pinecone

# ---------- Config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "rob-brain")
BEARER_TOKEN    = os.environ.get("FOREVER_BRAIN_BEARER")
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY and PINECONE_API_KEY.")

# Your permanent namespace set
NAMESPACES = ["fiction", "nonfiction", "trading", "personal", "reference"]

# ---------- Clients ----------
oai = OpenAI(api_key=OPENAI_API_KEY)
pc  = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# ---------- App ----------
app = FastAPI(title="Rob Forever Brain API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Models ----------
class DateRange(BaseModel):
    start: Optional[str] = None  # "YYYY-MM-DD"
    end: Optional[str] = None

class SearchBody(BaseModel):
    query: str
    top_k: int = 12
    namespace: Optional[str] = None   # omit or "all" -> search everything
    symbols: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    date_range: Optional[DateRange] = None

# ---------- Auth ----------
def require_bearer(authorization: Optional[str] = Header(default=None, alias="Authorization")):
    if not BEARER_TOKEN:
        raise HTTPException(status_code=500, detail="Server missing FOREVER_BRAIN_BEARER")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid bearer token")
    return True

# ---------- Namespace-aware keyword nudges for rerank ----------
KW_BY_NS = {
    "fiction": [
        "character", "narrative", "plot", "chapter", "novel", "story", "dialogue", "scene"
    ],
    "nonfiction": [
        "munger", "charlie munger", "paul graham", "marcus aurelius", "meditations",
        "stoic", "stoicism", "bias", "checklist", "opportunity cost", "decision",
        "virtue", "prudence", "discipline"
    ],
    "trading": [
        "short", "borrow", "fee", "ssr", "halt", "float", "dilution", "offering",
        "pnl", "profit", "loss", "gap", "fade", "resistance", "support", "liquidity",
        "ticker", "10q", "8k", "shelf", "at-the-market"
    ],
    "personal": [
        "journal", "reflection", "lesson", "goal", "plan", "review", "gratitude",
        "habit", "routine", "mood", "relationship"
    ],
    "reference": [
        "definition", "glossary", "table", "appendix", "dataset", "reference",
        "api", "specification", "manual", "guide"
    ],
}
def _keywords_for(ns: Optional[str]) -> List[str]:
    return KW_BY_NS.get((ns or "").lower(), [])

def _kw_hits(text: str, keywords: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for k in keywords if k in t)

# ---------- Routes ----------
@app.get("/healthz")
def healthz():
    return {"ok": True, "index": PINECONE_INDEX, "namespaces": NAMESPACES}

@app.post("/search")
def search(body: SearchBody, _: bool = Depends(require_bearer)):
    """
    If body.namespace is None or "all": search ALL namespaces and merge.
    If body.namespace is set (e.g., "trading"), search only that one.
    """
    # 1) Embed the query
    emb = oai.embeddings.create(model="text-embedding-3-large", input=body.query).data[0].embedding

    # 2) Optional metadata filters
    flt: Dict[str, Any] = {}
    if body.symbols:
        flt["symbol"] = {"$in": body.symbols}
    if body.tags:
        flt["tags"] = {"$in": body.tags}
    if body.date_range and (body.date_range.start or body.date_range.end):
        rng: Dict[str, Any] = {}
        if body.date_range.start: rng["$gte"] = body.date_range.start
        if body.date_range.end:   rng["$lte"] = body.date_range.end
        flt["date"] = rng
    flt = flt or None

    # 3) Which namespaces to hit?
    ns_param = (body.namespace or "").strip().lower()
    namespaces = NAMESPACES if (not ns_param or ns_param == "all") else [ns_param]

    # 4) Query each namespace, slightly over-fetch, then merge + rerank
    per_ns_k = max(min(body.top_k, 12), 6)  # 6..12 per namespace
    all_matches: List[Dict[str, Any]] = []

    for ns in namespaces:
        res = index.query(
            vector=emb,
            top_k=per_ns_k,
            include_metadata=True,
            namespace=ns,
            filter=flt,
        )
        matches = res.get("matches") or []
        for m in matches:
            md = m.get("metadata", {}) or {}
            txt = (md.get("text") or "") + " " + (md.get("title") or "")
            base = float(m.get("score") or 0.0)
            add  = 0.05 * _kw_hits(txt, _keywords_for(ns))  # gentle nudge
            all_matches.append({
                "namespace": ns,
                "text": md.get("text", ""),
                "metadata": md,
                "score": base,
                "combined": base + add
            })

    # 5) Merge, sort, trim
    all_matches.sort(key=lambda x: x["combined"], reverse=True)
    hits = all_matches[: body.top_k]

    return {
        "namespaces_queried": namespaces,
        "hits": hits,
        "count": len(hits)
    }
