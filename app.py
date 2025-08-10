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
from typing import Optional, List, Dict, Any
from fastapi import Body

# ---------- FEES HELPERS ----------
def _f(x) -> float:
    try:
        if x is None: return 0.0
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip().replace(",", "")
        return float(s) if s not in ("", "null", "None") else 0.0
    except Exception:
        return 0.0

def _pass_filters(md: Dict[str, Any],
                  file: Optional[str],
                  symbols: Optional[List[str]],
                  start: Optional[str],
                  end: Optional[str]) -> bool:
    # file filter
    if file and str(md.get("file", "")).strip() != file.strip():
        return False
    # date filter (YYYY-MM-DD)
    d = str(md.get("date", ""))[:10]
    if start and d and d < start:
        return False
    if end and d and d > end:
        return False
    # symbols filter
    if symbols:
        sy = str(md.get("symbol", "")).upper()
        want = {s.upper() for s in symbols}
        if sy not in want:
            return False
    return True

class FeesBody(BaseModel):
    file: Optional[str] = None
    namespace: str = "trading"
    query: str = "*"              # semantic “grab bag”; we’ll use a neutral embedding
    top_k: int = 5000
    start: Optional[str] = None   # "YYYY-MM-DD"
    end: Optional[str] = None
    symbols: Optional[List[str]] = None

@app.post("/fees")
def fees(body: FeesBody):
    _check_bearer(None)  # if you’re enforcing bearer here; otherwise remove

    # 1) Get a neutral embedding so we can pull lots of rows
    vec = embed_query("broker fee commission borrow overnight summary query")

    # 2) Query Pinecone
    res = index.query(
        vector=vec,
        top_k=body.top_k,
        namespace=body.namespace,
        include_metadata=True,
    )
    matches = getattr(res, "matches", []) or []

    # 3) Accumulators
    fee_cols = ["TransFee", "ECNTaker", "ECNMaker", "ORFFee", "TAFFee", "SECFee", "CATFee", "Commissions"]
    by_col = {k: 0.0 for k in fee_cols}
    borrow_total = 0.0
    overnight_total = 0.0
    trades_fees_total = 0.0  # sums metadata.fees_total for trade rows
    rows_scanned = 0

    # 4) Walk results & sum
    for m in matches:
        md = getattr(m, "metadata", {}) or {}
        if not _pass_filters(md, body.file, body.symbols, body.start, body.end):
            continue
        rows_scanned += 1

        # per-column fees if present (many rows won’t have these, that’s fine)
        for k in fee_cols:
            by_col[k] += _f(md.get(k))

        # trades_fees_total: sum the roll‑up fee you stored on trade rows
        trades_fees_total += _f(md.get("fees_total"))

        # borrow/overnight totals (you tagged these on fee rows)
        ft = (md.get("fee_type") or "").lower()
        amt = _f(md.get("amount"))
        if ft == "borrow_fee":
            borrow_total += amt
        elif ft == "overnight_fee":
            overnight_total += amt

    # 5) Build totals
    fee_cols_sum = sum(by_col.values())
    # Grand total matches local script logic: fee columns + borrow + overnight.
    # NOTE: If fee columns are zero (not embedded per-row), trades_fees_total covers trade fees.
    grand_total = fee_cols_sum + borrow_total + overnight_total
    # Provide both, so you can compare:
    combined_including_trades_rollup = grand_total + trades_fees_total if fee_cols_sum == 0.0 else grand_total

    pretty = []
    pretty.append(f"Rows scanned: {rows_scanned}\n")
    pretty.append("--- Fee columns ---")
    for k in fee_cols:
        pretty.append(f"{k:10s}: ${by_col[k]:.2f}")
    pretty.append(f"\nBorrow fees: ${borrow_total:.2f}")
    pretty.append(f"Overnight fees: ${overnight_total:.2f}")
    if fee_cols_sum == 0.0:
        pretty.append(f"\nfees_total (roll‑up on trade rows): ${trades_fees_total:.2f}")
        pretty.append("\n=== ALL-IN FEES (fee columns + borrow + overnight + trades fees_total) ===")
        pretty.append(f"${combined_including_trades_rollup:.2f}")
    else:
        pretty.append("\n=== ALL-IN FEES (fee columns + borrow + overnight) ===")
        pretty.append(f"${grand_total:.2f}")

    return {
        "ok": True,
        "filters": {
            "namespace": body.namespace,
            "query": body.query,
            "top_k": body.top_k,
            "start": body.start,
            "end": body.end,
            "symbols": body.symbols,
            "file": body.file,
        },
        "rows_scanned": rows_scanned,
        "totals": {
            "by_column": by_col,
            "borrow_fees": borrow_total,
            "overnight_fees": overnight_total,
            "trades_fees_total": trades_fees_total,
            "grand_total_fee_cols_borrow_overnight": grand_total,
            "grand_total_including_trades_rollup_if_needed": combined_including_trades_rollup,
        },
        "pretty": "\n".join(pretty),
    }
from datetime import datetime

@app.post("/fees_summary")
def fees_summary(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    top_k: int = 200,
    authorization: Optional[str] = Header(default=None)
):
    _check_bearer(authorization)

    vec = embed_query(symbol + " borrow fee")
    res = index.query(
        vector=vec,
        top_k=top_k,
        namespace="trading",
        include_metadata=True,
    )

    matches = getattr(res, "matches", []) or []
    rows = []

    total = 0.0
    for m in matches:
        md = m.metadata or {}
        sym = (md.get("symbol") or "").upper()
        amt = md.get("amount") or 0.0
        date_str = md.get("date")

        # Skip if not the same symbol
        if sym != symbol.upper():
            continue

        # Filter by date range if provided
        if start_date or end_date:
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
                if start_date and d < datetime.strptime(start_date, "%Y-%m-%d").date():
                    continue
                if end_date and d > datetime.strptime(end_date, "%Y-%m-%d").date():
                    continue
            except:
                pass

        total += amt
        rows.append({
            "date": date_str,
            "amount": amt,
            "description": md.get("description", ""),
            "fee_type": md.get("fee_type", ""),
            "score": m.score
        })

    return {
        "symbol": symbol.upper(),
        "total": total,
        "count": len(rows),
        "rows": rows
    }
