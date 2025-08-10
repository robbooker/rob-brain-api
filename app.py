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

EMBED_MODEL = "text-embedding-3-large"

# ---- Bearer auth for Actions ----
FOREVER_BRAIN_BEARER = os.environ.get("FOREVER_BRAIN_BEARER")

def _check_bearer(authorization_header: str | None):
    """Raise 401 if the Authorization header is missing/invalid."""
    if not FOREVER_BRAIN_BEARER:
        # If you forgot to set it on Render, fail closed.
        raise HTTPException(status_code=500, detail="Server missing bearer key")
    if not authorization_header or not authorization_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization_header.split(" ", 1)[1].strip()
    if token != FOREVER_BRAIN_BEARER:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

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
    resp = oai.embeddings.create(model="text-embedding-3-large", input=text)
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
    """Basic health + what namespaces exist right now + index dimension."""
    try:
        desc = pc.describe_index(PINECONE_INDEX)   # <- control-plane call
        index_dim = getattr(desc, "dimension", None)
    except Exception as e:
        index_dim = f"unknown ({e})"

    return {
        "ok": True,
        "index": PINECONE_INDEX,
        "index_dimension": index_dim,              # <-- add this
        "embed_model_dim": 1536,                   # text-embedding-3-small
        "namespaces": list_namespaces(),
    }

@app.post("/search")
def search(body: SearchBody, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _check_bearer(authorization)   # ← add this line
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
    _check_bearer(authorization)   # ← add this line
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
# =========================
# /fees endpoint (trading)
# =========================
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class FeesBody(BaseModel):
    # Optional natural-language query to retrieve rows; default grabs broadly
    query: Optional[str] = "*"
    # How many rows to retrieve from Pinecone (use a big number to approximate "all")
    top_k: int = 5000
    # Namespace to search; your trading rows live here
    namespace: str = "trading"
    # Optional filters
    start: Optional[str] = None   # "YYYY-MM-DD"
    end: Optional[str] = None     # "YYYY-MM-DD"
    symbols: Optional[List[str]] = None  # e.g. ["XPON","XBP"]

def _parse_ymd(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None

def _num(x) -> float:
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        # strip commas etc.
        return float(str(x).replace(",", "").strip())
    except Exception:
        return 0.0

@app.post("/fees")
def fees(body: FeesBody, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    # Enforce bearer if you added _check_bearer previously; otherwise remove this line:
    if "_check_bearer" in globals():
        _check_bearer(authorization)

    t0 = time.time()
    # 1) Embed the query (broad "*" by default) and pull a big batch from Pinecone
    vec = embed_query(body.query or "*")

    try:
        res = index.query(
            vector=vec,
            top_k=body.top_k,
            namespace=body.namespace,
            include_metadata=True,
        )
    except Exception as e:
        logging.error(f"/fees Pinecone query failed: {e}")
        raise HTTPException(status_code=500, detail="Vector query failed")

    matches = getattr(res, "matches", []) or []

    # 2) Optional filters
    start_dt = _parse_ymd(body.start)
    end_dt   = _parse_ymd(body.end)
    want_symbols = set([s.upper() for s in (body.symbols or [])])

    def _keep(md: Dict[str, Any]) -> bool:
        # date filter
        dstr = (md.get("date") or "").strip()
        if start_dt or end_dt:
            try:
                ddt = datetime.strptime(dstr, "%Y-%m-%d")
            except Exception:
                return False
            if start_dt and ddt < start_dt:
                return False
            if end_dt and ddt > end_dt:
                return False
        # symbol filter
        if want_symbols:
            sym = (md.get("symbol") or "").upper()
            if sym not in want_symbols:
                return False
        return True

    # 3) Aggregate
    fee_cols = ["TransFee", "ECNTaker", "ECNMaker", "ORFFee", "TAFFee", "SECFee", "CATFee", "Commissions"]
    totals: Dict[str, float] = {k: 0.0 for k in fee_cols}
    borrow_total = 0.0
    overnight_total = 0.0

    rows_scanned = 0
    for m in matches:
        md = getattr(m, "metadata", {}) or {}
        if not _keep(md):
            continue
        rows_scanned += 1

        # Sum explicit fee columns if present
        for k in fee_cols:
            totals[k] += _num(md.get(k))

        # Classify borrow vs overnight from description + amount
        desc = (md.get("description") or md.get("text") or "").upper()
        amt  = _num(md.get("amount"))
        if "BORROW FEE" in desc:
            # Your convention:
            #  - contains " C "  -> borrow fee
            #  - otherwise       -> overnight fee
            if " C " in desc:
                borrow_total += amt
            else:
                overnight_total += amt

    fee_cols_total = sum(totals.values())
    grand_total = fee_cols_total + borrow_total + overnight_total

    took = time.time() - t0
    logging.info(f"🧾 /fees scanned {rows_scanned} rows in {took:.3f}s "
                 f"(fees={fee_cols_total:.2f}, borrow={borrow_total:.2f}, overnight={overnight_total:.2f})")

    # 4) Return a structured JSON + pretty strings the GPT can render verbatim
    pretty_lines = []
    pretty_lines.append(f"Rows scanned: {rows_scanned}\n")
    pretty_lines.append("--- Fee columns ---")
    for k in fee_cols:
        pretty_lines.append(f"{k:<10s}: ${totals[k]:.2f}")
    pretty_lines.append(f"\nBorrow fees (from Amount, ' C STOCK BORROW FEE '): ${borrow_total:.2f}")
    pretty_lines.append(f"Overnight fees (from Amount, 'STOCK BORROW FEE' no ' C '): ${overnight_total:.2f}")
    pretty_lines.append("\n=== ALL-IN FEES & COMMISSIONS ===")
    pretty_lines.append(f"${grand_total:.2f}")

    return {
        "ok": True,
        "filters": {
            "namespace": body.namespace,
            "query": body.query,
            "top_k": body.top_k,
            "start": body.start,
            "end": body.end,
            "symbols": list(want_symbols) if want_symbols else None,
        },
        "rows_scanned": rows_scanned,
        "totals": {
            "by_column": {k: round(v, 2) for k, v in totals.items()},
            "borrow_fees": round(borrow_total, 2),
            "overnight_fees": round(overnight_total, 2),
            "grand_total": round(grand_total, 2),
        },
        "pretty": "\n".join(pretty_lines),
    }
