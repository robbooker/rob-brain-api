import os

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "120000"))  # ~120 KB
MAX_EXCERPT_CHARS = int(os.getenv("MAX_EXCERPT_CHARS", "1500"))    # per hit

import time
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from pydantic import BaseModel

# add near your other imports
from typing import Optional, List
from pydantic import BaseModel

# === Request body model for /fees ===
class FeesBody(BaseModel):
    file: Optional[str] = None
    namespace: str = "trading"
    query: str = "*"
    top_k: int = 5000
    start: Optional[str] = None
    end: Optional[str] = None
    symbols: Optional[List[str]] = None

from openai import OpenAI
from pinecone import Pinecone
from datetime import datetime

# -----------------------------------------------------------------------------
# Config & clients
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "rob-brain")

# Embedding model MUST match index dimension (3072)
EMBED_MODEL = "text-embedding-3-large"

# ---- Bearer auth for Actions ----
FOREVER_BRAIN_BEARER = os.environ.get("FOREVER_BRAIN_BEARER")

def _check_bearer(authorization_header: Optional[str]):
    """Raise 401 if the Authorization header is missing/invalid."""
    if not FOREVER_BRAIN_BEARER:
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
app = FastAPI(title="Rob Forever Brain API", version="2.2.0")

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
def _norm_date(s: str) -> str:
    """Return YYYY-MM-DD for mixed date strings like '7/9/2025' or '2025-07-09'."""
    if not s:
        return s
    # Try common formats
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%-m/%-d/%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    # macOS strptime may not support %-m; try zero-padded as last resort
    for fmt in ("%m/%d/%Y",):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    return s

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
    """Basic health + what namespaces exist right now + index dimension."""
    try:
        desc = pc.describe_index(PINECONE_INDEX)   # control-plane call
        index_dim = getattr(desc, "dimension", None)
    except Exception as e:
        index_dim = f"unknown ({e})"

    return {
        "ok": True,
        "index": PINECONE_INDEX,
        "index_dimension": index_dim,
        "embed_model": EMBED_MODEL,
        "embed_model_dim": 3072,
        "namespaces": list_namespaces(),
    }

@app.post("/search")
def search(body: SearchBody, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _check_bearer(authorization)
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

# =========================
# /ask endpoint
# =========================

 # (make sure you have: import os)

@app.post("/ask")
def ask(
    body: AskBody,
    authorization: Optional[str] = Header(default=None),
):
    """
    Orchestrated endpoint for your GPT:
      1) Always search vectors first (all namespaces by default)
      2) If strong hits exist (>= min_score), return them as context
      3) Otherwise return use_web=true so GPT falls back to web search
    """
    _check_bearer(authorization)

    ns = body.namespace or "all"
    min_score = body.min_score or 0.34
    top_k = body.top_k or 12

    # Vector search
    res = index.query(
        vector=embed_query(body.question),
        top_k=top_k,
        namespace=None if ns == "all" else ns,
        include_metadata=True,
    )
    matches = getattr(res, "matches", []) or []

    # Strong hits, then sort best-first
    strong_hits_unsorted = [m for m in matches if getattr(m, "score", 0.0) >= min_score]
    strong_hits = sorted(
        strong_hits_unsorted,
        key=lambda m: float(getattr(m, "score", 0.0) or 0.0),
        reverse=True,
    )

    # Caps
    MAX_CONTEXT_CHARS  = int(os.getenv("MAX_CONTEXT_CHARS", "120000"))  # total chars
    MAX_EXCERPT_CHARS  = int(os.getenv("MAX_EXCERPT_CHARS", "1500"))    # per hit
    MAX_CONTEXT_ITEMS  = int(os.getenv("MAX_CONTEXT_ITEMS", "200"))     # max items

    context: List[Dict[str, Any]] = []
    total_chars = 0
    seen_keys = set()  # dedupe by (title, source)

    for m in strong_hits:
        if len(context) >= MAX_CONTEXT_ITEMS:
            break

        md = getattr(m, "metadata", {}) or {}
        k = (md.get("title") or "", md.get("source") or "")
        if k in seen_keys:
            continue
        seen_keys.add(k)

        txt = (md.get("text") or "")
        if txt and len(txt) > MAX_EXCERPT_CHARS:
            txt = txt[:MAX_EXCERPT_CHARS] + "…"

        add_len = len(txt)
        if total_chars + add_len > MAX_CONTEXT_CHARS:
            break

        context.append({
            "namespace": md.get("namespace") or md.get("ns") or "",
            "score": float(getattr(m, "score", 0.0) or 0.0),
            "title": md.get("title") or "",
            "source": md.get("source") or "",
            "text": txt,
        })
        total_chars += add_len

    if not context:
        return {
            "answered_from": "none",
            "use_web": True,
            "min_score_used": min_score,
            "namespaces_queried": [ns],
            "hits_count": 0,
            "context": [],
            "suggested_instruction": "No strong vector hits found — fall back to web search."
        }

    return {
        "answered_from": "vector",
        "use_web": False,
        "min_score_used": min_score,
        "namespaces_queried": [ns],
        "hits_count": len(strong_hits),
        "context": context,
        "suggested_instruction": "Use the context excerpts to answer the user's question. Cite titles/sources from context. If something is unclear or missing, say so."
    }
# =========================
# /fees endpoint (auto‑chunking)
# =========================
from datetime import date, timedelta
import calendar

def _parse_iso(d: str) -> date | None:
    try:
        y, m, d = d.split("-")
        return date(int(y), int(m), int(d))
    except Exception:
        return None

def _month_end(dt: date) -> date:
    return date(dt.year, dt.month, calendar.monthrange(dt.year, dt.month)[1])

def _daterange_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Return small chunks to keep payloads tiny."""
    days = (end - start).days + 1
    # one month: split 1–15, 16–end
    if start.year == end.year and start.month == end.month and days > 15:
        mid = date(start.year, start.month, 15)
        return [(start, mid), (mid + timedelta(days=1), end)]
    # multi‑month or long windows: 10‑day chunks
    chunks = []
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=9), end)
        chunks.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return chunks

@app.post("/fees")
def fees(
    body: FeesBody,
    authorization: Optional[str] = Header(default=None),
):
    """
    Aggregate fee columns + borrow/overnight + trades rollup.
    If the requested window is big (or dates omitted), we auto‑chunk
    into smaller date ranges and combine totals to avoid size limits.
    """
    _check_bearer(authorization)

    # Resolve dates
    start_str = body.start
    end_str = body.end

    # If missing dates, default to the last full calendar month
    if not (start_str and end_str):
        today = date.today()
        first_this_month = today.replace(day=1)
        last_prev_month = first_this_month - timedelta(days=1)
        start_dt = last_prev_month.replace(day=1)
        end_dt = last_prev_month
    else:
        start_dt = _parse_iso(start_str)
        end_dt = _parse_iso(end_str)
        if not (start_dt and end_dt) or start_dt > end_dt:
            raise HTTPException(status_code=400, detail="Invalid start/end; use YYYY-MM-DD and start<=end")

    # Build chunks
    chunks = _daterange_chunks(start_dt, end_dt)

    # Accumulators
    by_col_totals = {
        "TransFee": 0.0, "ECNTaker": 0.0, "ECNMaker": 0.0,
        "ORFFee": 0.0, "TAFFee": 0.0, "SECFee": 0.0,
        "CATFee": 0.0, "Commissions": 0.0
    }
    borrow_fees_total = 0.0
    overnight_fees_total = 0.0
    trades_fees_total = 0.0
    rows_scanned_sum = 0

    # Reuse your existing inner scan logic, but run it per chunk
    for (c_start, c_end) in chunks:
        # call your existing vector scan exactly as before, but with
        # start=str(c_start), end=str(c_end)
        res = _scan_fees_once(
            file=body.file,
            namespace=body.namespace,
            query=body.query,
            top_k=body.top_k,
            start=str(c_start),
            end=str(c_end),
            symbols=body.symbols,
        )
        # res should be the same dict you already compute inside /fees today
        # Accumulate
        for k in by_col_totals:
            by_col_totals[k] += float(res["totals"]["by_column"].get(k, 0.0))
        borrow_fees_total      += float(res["totals"]["borrow_fees"])
        overnight_fees_total   += float(res["totals"]["overnight_fees"])
        trades_fees_total      += float(res["totals"]["trades_fees_total"])
        rows_scanned_sum       += int(res.get("rows_scanned", 0))

    grand_fee_cols = sum(by_col_totals.values()) + borrow_fees_total + overnight_fees_total
    grand_all_in   = grand_fee_cols + trades_fees_total

    return {
        "ok": True,
        "filters": {
            "namespace": body.namespace,
            "query": body.query,
            "top_k": body.top_k,
            "start": str(start_dt),
            "end": str(end_dt),
            "symbols": body.symbols,
            "file": body.file,
            "auto_chunked": len(chunks) > 1,
            "chunks": [{"start": str(a), "end": str(b)} for (a,b) in chunks],
        },
        "rows_scanned": rows_scanned_sum,
        "totals": {
            "by_column": by_col_totals,
            "borrow_fees": borrow_fees_total,
            "overnight_fees": overnight_fees_total,
            "trades_fees_total": trades_fees_total,
            "grand_total_fee_cols_borrow_overnight": grand_fee_cols,
            "grand_total_including_trades_rollup_if_needed": grand_all_in,
        },
        # keep your existing 'pretty' if you like, but it’s optional now
    }

# =========================
# /fees_summary endpoint
# =========================
from fastapi import Query

@app.post("/fees_summary")
def fees_summary(
    symbol: str = Query(..., description="Stock symbol to summarize"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    top_k: int = Query(400, description="Number of results to consider"),
    brief: bool = Query(False, description="If true, omit raw rows"),
    authorization: Optional[str] = Header(default=None)
):
    """
    Summarize borrow-related fees for a symbol in a date range.
    - Returns split totals (borrow vs overnight), grand total
    - Daily subtotals (date -> total)
    - Monthly rollup (days_counted, total, avg_per_day, max_day)
    """
    _check_bearer(authorization)

    ns = "trading"
    q = f"{symbol} borrow fee"  # simple query key; index is already pre-tagged

    # Build a vector from the query for semantic search
    vec = embed_query(q)

    # Query Pinecone
    res = index.query(
        vector=vec,
        top_k=top_k,
        namespace=ns,
        include_metadata=True,
    )
    matches = getattr(res, "matches", []) or []

    # Filter to rows for this symbol + in date range + with a borrow-fee type
    def in_range(dstr: str) -> bool:
        if not (start_date or end_date):
            return True
        nd = _norm_date(dstr)
        if start_date and nd < start_date:
            return False
        if end_date and nd > end_date:
            return False
        return True

    rows: List[Dict[str, Any]] = []
    for m in matches:
        md = (getattr(m, "metadata", None) or {})
        if (md.get("symbol") or "").upper() != symbol.upper():
            continue
        ft = (md.get("fee_type") or "").lower()
        if ft not in ("borrow_fee", "overnight_borrow_fee"):
            continue
        dstr = md.get("date") or ""
        if not in_range(dstr):
            continue
        rows.append({
            "date": _norm_date(dstr),
            "amount": float(md.get("amount", 0.0)),
            "description": md.get("description") or "",
            "fee_type": ft,
            "score": float(getattr(m, "score", 0.0) or 0.0),
        })

    # Totals split by fee type
    borrow_total = sum(r["amount"] for r in rows if r["fee_type"] == "borrow_fee")
    overnight_total = sum(r["amount"] for r in rows if r["fee_type"] == "overnight_borrow_fee")
    grand_total = borrow_total + overnight_total

    # Daily subtotals
    daily: Dict[str, float] = {}
    for r in rows:
        daily[r["date"]] = daily.get(r["date"], 0.0) + r["amount"]

    # Monthly rollup
    days_counted = len(daily)
    avg_per_day = (grand_total / days_counted) if days_counted else 0.0
    if daily:
        max_day_date = max(daily, key=lambda d: daily[d])
        max_day = {"date": max_day_date, "total": daily[max_day_date]}
    else:
        max_day = {"date": None, "total": 0.0}

    # Build compact response
    resp = {
        "symbol": symbol.upper(),
        "count": len(rows),
        "totals": {
            "borrow_fee_total": borrow_total,
            "overnight_borrow_fee_total": overnight_total,
            "grand_total": grand_total,
        },
        "daily_subtotals": daily,
        "monthly_summary": {
            "days_counted": days_counted,
            "total": grand_total,
            "avg_per_day": avg_per_day,
            "max_day": max_day,
        },
    }

    # Only include raw rows if NOT brief
    if not brief:
        resp["rows"] = rows

    return resp

# =========================
# /fees_rollup endpoint
# =========================
from fastapi import Query

# =========================
# /fees_rollup endpoint
# =========================
from fastapi import Query

@app.post("/fees_rollup")
def fees_rollup(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str]   = Query(None, description="End date YYYY-MM-DD"),
    top_k: int                = Query(5000, description="How many rows to scan"),
    file: Optional[str]       = Query(None, description="Optional: restrict to a specific uploaded CSV filename"),
    authorization: Optional[str] = Header(default=None),
):
    """
    Roll up borrow-related fees by symbol for a date window.
    Includes fee rows tagged as `borrow_fee` or `overnight_borrow_fee`.
    Returns: { by_symbol: [{symbol, total}], grand_total }
    """
    _check_bearer(authorization)

    # Inline, safe date parsing (ISO YYYY-MM-DD, then US M/D/YYYY)
    from datetime import datetime as _dt

    def _parse_any_date_local(val: str):
        if not val:
            return None
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%-m/%-d/%Y"):
            try:
                return _dt.strptime(val, fmt).date()
            except Exception:
                continue
        # macOS may not support %-m; try zero‑padded as last resort
        for fmt in ("%m/%d/%Y",):
            try:
                return _dt.strptime(val, fmt).date()
            except Exception:
                continue
        return None

    start_dt = _parse_any_date_local(start_date) if start_date else None
    end_dt   = _parse_any_date_local(end_date)   if end_date   else None

    def in_range(dstr: str) -> bool:
        if not (start_dt or end_dt):
            return True
        d = _parse_any_date_local(dstr)
        if not d:
            return False
        if start_dt and d < start_dt:
            return False
        if end_dt and d > end_dt:
            return False
        return True

    ns = "trading"
    # Neutral embedding to pull a wide net of fee rows
    vec = embed_query("stock borrow fee overnight borrow fee trading summary")

    res = index.query(
        vector=vec,
        top_k=top_k,
        namespace=ns,
        include_metadata=True,
    )
    matches = getattr(res, "matches", []) or []

    per_symbol: Dict[str, float] = {}
    grand_total = 0.0
    rows_scanned = 0

    for m in matches:
        md = getattr(m, "metadata", {}) or {}

        # Optional file filter
        if file and str(md.get("file", "")).strip() != file.strip():
            continue

        ft = (md.get("fee_type") or "").lower()
        if ft not in ("borrow_fee", "overnight_borrow_fee"):
            continue

        dstr = md.get("date") or ""
        if not in_range(dstr):
            continue

        sym = str(md.get("symbol", "")).upper().strip()
        if not sym:
            continue  # skip rows that don't have a symbol

        try:
            amt = float(md.get("amount") or 0.0)
        except Exception:
            continue

        per_symbol[sym] = per_symbol.get(sym, 0.0) + amt
        grand_total += amt
        rows_scanned += 1

    # Sort by total ascending (most negative first)
    by_symbol = [{"symbol": s, "total": t} for s, t in per_symbol.items()]
    by_symbol.sort(key=lambda x: x["total"])

    return {
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "top_k": top_k,
            "file": file,
            "namespace": ns,
        },
        "rows_scanned": rows_scanned,
        "by_symbol": by_symbol,
        "grand_total": grand_total,
    }
