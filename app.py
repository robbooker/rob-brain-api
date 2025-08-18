# app.py
# FastAPI app for Rob Brain API
# Endpoints: /healthz, /search, /ask, /fees, /fees_summary, /fees_rollup, /short_pnl

import os
import math
from datetime import datetime
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, Header, Query
from fastapi.middleware.cors import CORSMiddleware

# ==== OpenAI embeddings ====
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
# text-embedding-3-large has 3072 dims; -small has 1536
EMBED_DIM = 3072 if "large" in EMBED_MODEL else 1536

def embed_query(text: str) -> List[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

# ==== INDEX SETUP (swap for your existing one if different) ====
# We expect a vector index client with .query(vector, top_k, namespace, include_metadata=True)
# and optionally .describe_index_stats() for healthz. Replace with your known-good code if needed.
from pinecone import Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX", "rob-brain")
index = pc.Index(INDEX_NAME)

# ==== FastAPI ====
app = FastAPI(title="Rob Forever Brain API", version="2.2.6")

# --- Minimal OpenAPI for the GPT (nonfiction-only tools) ---
from fastapi.responses import JSONResponse

@app.get("/openapi_nonfiction.json")
def openapi_nonfiction():
    # Start from the full schema that FastAPI already generates
    full = app.openapi()

    # Keep only these paths
    allowed = {"/healthz", "/search", "/answer"}
    minimal_paths = {p: spec for p, spec in full.get("paths", {}).items() if p in allowed}

    # Build a trimmed schema
    minimal = dict(full)  # shallow copy is fine
    minimal["paths"] = minimal_paths

    # (Optional) make it clear in the title/version this is the slim spec used by GPT
    info = dict(minimal.get("info", {}))
    info["title"] = (info.get("title") or "API") + " — Nonfiction GPT Spec"
    minimal["info"] = info

    return JSONResponse(minimal)

# --- END Minimal OpenAPI for the GPT (nonfiction-only tools) ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ADD THIS BLOCK RIGHT HERE
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["servers"] = [
        {"url": "https://rob-brain-api-1.onrender.com", "description": "prod"}
    ]
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi
# END OF BLOCK


# ==== Helpers ====
def _check_bearer(authorization: Optional[str]) -> None:
    want = os.getenv("API_BEARER", "H@173y8004er")
    if not authorization:
        raise ValueError("missing Authorization header")
    if not authorization.lower().startswith("bearer "):
        raise ValueError("expected Bearer token")
    token = authorization.split(" ", 1)[1]
    if token != want:
        raise ValueError("invalid token")

def _parse_any_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).strip()
    # Try ISO first: YYYY-MM-DD
    try:
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return datetime.strptime(s, "%Y-%m-%d")
    except:
        pass
    # Try M/D/YYYY or MM/DD/YYYY
    for fmt in ("%m/%d/%Y", "%-m/%-d/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except:
            pass
    # Try D/M/YYYY (rare)
    for fmt in ("%d/%m/%Y",):
        try:
            return datetime.strptime(s, fmt)
        except:
            pass
    return None

def _safe_float(x, default=0.0) -> float:
    try:
        return float(str(x).replace(",", ""))
    except:
        return default

def _md_get(md: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in md and md[k] not in (None, ""):
            return md[k]
    return default

def _namespaces_to_query(ns_in: Optional[str]) -> list[str]:
    if ns_in is None:
        return ["nonfiction", "trading", "short-selling"]
    ns = (ns_in or "").strip().lower()
    if ns in ("", "all", "*"):
        return ["nonfiction", "trading", "short-selling"]
    return [ns_in]

# ==== Models (pydantic lite via typing) ====
from pydantic import BaseModel
class SearchBody(BaseModel):
    query: str
    top_k: int = 8
    namespace: Optional[str] = None
    symbols: Optional[List[str]] = None
    tags: Optional[List[str]] = None

class AskBody(BaseModel):
    question: str
    top_k: int = 12
    namespace: Optional[str] = None
    min_score: Optional[float] = None
    symbols: Optional[List[str]] = None
    tags: Optional[List[str]] = None

class FeesBody(BaseModel):
    file: Optional[str] = None
    namespace: str = "trading"
    query: str = "*"
    top_k: int = 5000
    start: Optional[str] = None
    end: Optional[str] = None
    symbols: Optional[List[str]] = None

# ==== /healthz ====
@app.get("/healthz")
def healthz():
    try:
        stats = index.describe_index_stats()
        idx_dim = EMBED_DIM
    except Exception:
        stats = {}
        idx_dim = EMBED_DIM
    return {
        "ok": True,
        "version": app.version,  # Add this line
        "index": INDEX_NAME,
        "index_dimension": idx_dim,
        "embed_model": EMBED_MODEL,
        "embed_model_dim": EMBED_DIM,
        "namespaces": list((stats.get("namespaces") or {}).keys()) or ["nonfiction", "trading"]
    }
    
# ==== /search (legacy) ====
@app.post("/search")
def search(body: SearchBody, authorization: Optional[str] = Header(default=None)):
    _check_bearer(authorization)

    namespaces = []
    if body.namespace in (None, "", "all"):
        # probe common namespaces
        namespaces = ["trading", "nonfiction"]
    else:
        namespaces = [body.namespace]

    vec = embed_query(body.query)
    all_hits = []
    for ns in namespaces:
        try:
            res = index.query(vector=vec, top_k=body.top_k, namespace=ns, include_metadata=True)
            matches = getattr(res, "matches", []) or []
            for m in matches:
                all_hits.append({
                    "namespace": ns,
                    "score": getattr(m, "score", None),
                    "metadata": getattr(m, "metadata", {}) or {},
                    "id": getattr(m, "id", None)
                })
        except Exception as e:
            # non-fatal per-namespace
            all_hits.append({"namespace": ns, "error": str(e)})

    return {"ok": True, "hits": all_hits}

# ==== /ask ====
@app.post("/ask")
def ask(body: AskBody, authorization: Optional[str] = Header(default=None)):
    _check_bearer(authorization)

    # choose namespaces
    if body.namespace in (None, "", "all"):
        namespaces = ["nonfiction", "trading"]
    else:
        namespaces = [body.namespace]

    min_score = 0.34 if body.min_score is None else float(body.min_score)
    vec = embed_query(body.question)

    hits = []
    for ns in namespaces:
        try:
            res = index.query(vector=vec, top_k=body.top_k, namespace=ns, include_metadata=True)
            matches = getattr(res, "matches", []) or []
            for m in matches:
                sc = getattr(m, "score", 0.0) or 0.0
                md = getattr(m, "metadata", {}) or {}
                text = md.get("text") or md.get("content") or ""
                title = md.get("title") or md.get("symbol") or md.get("file") or ""
                source = md.get("source") or md.get("file") or md.get("symbol") or ""
                hits.append({
                    "namespace": ns,
                    "score": sc,
                    "title": title,
                    "source": source,
                    "text": text
                })
        except Exception as e:
            hits.append({"namespace": ns, "error": str(e), "score": 0.0, "text": ""})

    strong = [h for h in hits if isinstance(h.get("score"), (int, float)) and h["score"] >= min_score]
    use_web = len(strong) == 0

    # pack a small context to keep responses lean
    # (You can send more if you like, but smaller is safer for GPT input)
    strong_sorted = sorted(strong, key=lambda x: x["score"], reverse=True)
    context = strong_sorted[: body.top_k]

    return {
        "answered_from": "vector" if not use_web else "none",
        "use_web": use_web,
        "min_score_used": min_score,
        "namespaces_queried": namespaces,
        "hits_count": len(context),
        "context": context,
        "suggested_instruction": "Use the context excerpts to answer the user's question. Cite titles/sources from context. If something is unclear or missing, say so."
    }

# ==== /fees (UPDATED) ====

from typing import Optional, List, Dict, Any
from fastapi import Header
from pydantic import BaseModel

# If you already have FeesBody defined elsewhere, you can remove this class.
class FeesBody(BaseModel):
    file: Optional[str] = None
    namespace: str = "trading"
    query: str = "*"          # kept for compatibility; not strictly used here
    top_k: int = 5000
    start: Optional[str] = None  # YYYY-MM-DD or other parseable format handled by _parse_any_date
    end:   Optional[str] = None
    symbols: Optional[List[str]] = None  # Optional filter list of symbols (uppercased)

def _fnum(v, default=0.0) -> float:
    """Robust float parser for strings like '1,234.56', blanks, None, etc."""
    try:
        if v is None:
            return float(default)
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).replace(",", "").strip()
        if s == "":
            return float(default)
        return float(s)
    except Exception:
        return float(default)

def _str(v) -> str:
    return "" if v is None else str(v).strip()

def _is_between(date_str: str, start_dt, end_dt) -> bool:
    if not (start_dt or end_dt):
        return True
    d = _parse_any_date(date_str)
    if not d:
        return False
    if start_dt and d < start_dt:
        return False
    if end_dt and d > end_dt:
        return False
    return True

@app.post("/fees")
def fees_post(
    body: FeesBody,
    authorization: Optional[str] = Header(default=None),
):
    """
    Fee rollup (date-window inclusive):
      - Borrow/Overnight fee rows are recognized via Type ('Borrow Fee' or 'Overnight Fee')
        and use the **Amount** column (signed).
      - Per-trade transaction fee columns are summed as-is (signed):
          TransFee, ECNTaker, ECNMaker, ORFFee, TAFFee, SECFee, CATFee, Commissions
      - Optional filters: file, symbols[], start, end
    Response:
      {
        filters: {...},
        rows_scanned: int,
        totals: {
          by_column: {...},               # sum of each transaction-fee column
          by_column_sum: float,           # sum of the 8 transaction-fee columns
          borrow_fees: float,             # sum of Borrow Fee amounts
          overnight_fees: float,          # sum of Overnight Fee amounts
          grand_total_all_in: float       # by_column_sum + borrow_fees + overnight_fees
        },
        by_symbol: {
          "SYMBOL": {
            transaction_fees: float,
            borrow_fees: float,
            overnight_fees: float,
            symbol_total: float
          }, ...
        },
        sample_rows: [ up to 25 filtered rows with light fields ]
      }
    """
    _check_bearer(authorization)

    ns = body.namespace or "trading"
    start_dt = _parse_any_date(body.start) if body.start else None
    end_dt   = _parse_any_date(body.end)   if body.end   else None
    sym_filter = set([s.strip().upper() for s in (body.symbols or [])]) if body.symbols else None
    file_in = _str(body.file)

    # Query a broad slice of rows likely to contain fee/trade information
    probe = "short trades transaction fees ECN maker taker ORF TAF SEC CAT commissions borrow fee overnight fee"
    vec = embed_query(probe)
    res = index.query(
        vector=vec,
        top_k=body.top_k,
        namespace=ns,
        include_metadata=True
    )
    matches = getattr(res, "matches", []) or []

    # Accumulators
    fee_columns = ["TransFee", "ECNTaker", "ECNMaker", "ORFFee", "TAFFee", "SECFee", "CATFee", "Commissions"]

    by_column_totals: Dict[str, float] = {k: 0.0 for k in fee_columns}
    transaction_by_symbol: Dict[str, float] = {}
    borrow_by_symbol: Dict[str, float] = {}
    overnight_by_symbol: Dict[str, float] = {}

    total_borrow = 0.0
    total_overnight = 0.0

    rows_scanned = 0
    filtered_rows: List[Dict[str, Any]] = []

    for m in matches:
        md = getattr(m, "metadata", {}) or {}

        # File filter (optional)
        if file_in and _str(md.get("file")) != file_in:
            continue

        date_str = _str(md.get("date"))
        if not _is_between(date_str, start_dt, end_dt):
            continue

        symbol = _str(md.get("symbol") or md.get("Symbol")).upper()
        if sym_filter and (not symbol or symbol not in sym_filter):
            continue

        typ = _str(md.get("type") or md.get("Type")).lower()

        # Collect per-row data for sample
        row_info = {
            "date": date_str,
            "symbol": symbol,
            "type": _str(md.get("type") or md.get("Type")),
            "file": _str(md.get("file")),
        }

        # Branch 1: Borrow/Overnight fee rows — use Amount column
        if typ in ("borrow fee", "overnight fee"):
            amount = _fnum(md.get("Amount") or md.get("amount"), 0.0)
            if typ == "borrow fee":
                total_borrow += amount
                if symbol:
                    borrow_by_symbol[symbol] = borrow_by_symbol.get(symbol, 0.0) + amount
            else:
                total_overnight += amount
                if symbol:
                    overnight_by_symbol[symbol] = overnight_by_symbol.get(symbol, 0.0) + amount

            row_info["amount"] = amount
            filtered_rows.append(row_info)
            rows_scanned += 1
            continue

        # Branch 2: Regular trade rows — sum transaction fee columns (signed)
        # Pull and sum the 8 columns
        row_tx_sum = 0.0
        for col in fee_columns:
            val = _fnum(md.get(col), 0.0)
            by_column_totals[col] += val
            row_tx_sum += val

        if abs(row_tx_sum) > 0:
            # Track per-symbol transaction fees
            if symbol:
                transaction_by_symbol[symbol] = transaction_by_symbol.get(symbol, 0.0) + row_tx_sum

            row_info.update({
                "bs": _str(md.get("bs") or md.get("B/S")),
                "tx_fees_sum": row_tx_sum,
            })
            filtered_rows.append(row_info)

        rows_scanned += 1

    # Compose totals
    by_column_sum = sum(by_column_totals.values())
    grand_total_all_in = by_column_sum + total_borrow + total_overnight

    # Per-symbol combined view
    by_symbol: Dict[str, Dict[str, float]] = {}
    all_symbols = set().union(transaction_by_symbol.keys(),
                              borrow_by_symbol.keys(),
                              overnight_by_symbol.keys())
    for s in sorted(all_symbols):
        tx  = transaction_by_symbol.get(s, 0.0)
        br  = borrow_by_symbol.get(s, 0.0)
        on  = overnight_by_symbol.get(s, 0.0)
        by_symbol[s] = {
            "transaction_fees": round(tx, 2),
            "borrow_fees": round(br, 2),
            "overnight_fees": round(on, 2),
            "symbol_total": round(tx + br + on, 2),
        }

    # Keep sample modest
    sample_rows = filtered_rows[:25]

    return {
        "filters": {
            "file": file_in,
            "start": body.start,
            "end": body.end,
            "top_k": body.top_k,
            "namespace": ns,
            "symbols": sorted(list(sym_filter)) if sym_filter else None,
        },
        "rows_scanned": rows_scanned,
        "totals": {
            "by_column": {k: round(v, 2) for k, v in by_column_totals.items()},
            "by_column_sum": round(by_column_sum, 2),
            "borrow_fees": round(total_borrow, 2),
            "overnight_fees": round(total_overnight, 2),
            "grand_total_all_in": round(grand_total_all_in, 2),
        },
        "by_symbol": by_symbol,
        "sample_rows": sample_rows,
    }
    
# ==== /fees_summary (symbol-level daily & monthly) ====
@app.post("/fees_summary")
def fees_summary(
    symbol: str = Query(..., description="Stock symbol to summarize"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str]   = Query(None, description="End date YYYY-MM-DD"),
    top_k: int                = Query(400, description="Number of results to consider"),
    brief: bool               = Query(False, description="If true, omit raw rows"),
    authorization: Optional[str] = Header(default=None),
):
    _check_bearer(authorization)

    start_dt = _parse_any_date(start_date) if start_date else None
    end_dt   = _parse_any_date(end_date)   if end_date   else None
    sym_up = symbol.upper()

    def in_range(dstr: str) -> bool:
        if not (start_dt or end_dt):
            return True
        d = _parse_any_date(dstr)
        if not d:
            return False
        if start_dt and d < start_dt: return False
        if end_dt   and d > end_dt:   return False
        return True

    ns = "trading"
    vec = embed_query(f"{sym_up} STOCK BORROW FEE overnight borrow fee trading activity")
    res = index.query(vector=vec, top_k=top_k, namespace=ns, include_metadata=True)
    matches = getattr(res, "matches", []) or []

    rows = []
    totals = {
        "borrow_fee_total": 0.0,
        "overnight_borrow_fee_total": 0.0,
        "grand_total": 0.0,
    }
    daily: Dict[str, float] = {}

    for m in matches:
        md = getattr(m, "metadata", {}) or {}
        if str(md.get("symbol", "")).upper() != sym_up:
            continue

        dstr = str(md.get("date", ""))
        if not in_range(dstr):
            continue

        ft = str(md.get("fee_type", "")).lower()
        amt = _safe_float(md.get("amount", 0.0), 0.0)
        if ft not in ("borrow_fee", "overnight_borrow_fee"):
            continue

        rows.append({
            "date": dstr,
            "amount": amt,
            "description": md.get("description", ""),
            "fee_type": ft,
            "score": getattr(m, "score", 0.0) or 0.0
        })

        if ft == "borrow_fee":
            totals["borrow_fee_total"] += amt
        else:
            totals["overnight_borrow_fee_total"] += amt

        daily[dstr] = daily.get(dstr, 0.0) + amt

    totals["grand_total"] = totals["borrow_fee_total"] + totals["overnight_borrow_fee_total"]

    # monthly rollup
    days = sorted(daily.items())
    days_counted = len(days)
    total_amt = totals["grand_total"]
    avg_per_day = (total_amt / days_counted) if days_counted else 0.0
    max_day = {"date": days[-1][0], "total": days[-1][1]} if days_counted else {"date": None, "total": 0.0}

    out = {
        "symbol": sym_up,
        "count": len(rows),
        "totals": totals,
        "daily_subtotals": {d: v for d, v in days},
        "monthly_summary": {
            "days_counted": days_counted,
            "total": total_amt,
            "avg_per_day": avg_per_day,
            "max_day": max_day,
        },
    }
    if not brief:
        out["rows"] = rows
    return out

# ==== /fees_rollup (by symbol for a window) ====
@app.post("/fees_rollup")
def fees_rollup(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str]   = Query(None, description="End date YYYY-MM-DD"),
    top_k: int                = Query(5000, description="How many rows to scan"),
    file: Optional[str]       = Query(None, description="Optional: restrict to a specific uploaded CSV filename"),
    authorization: Optional[str] = Header(default=None),
):
    _check_bearer(authorization)

    start_dt = _parse_any_date(start_date) if start_date else None
    end_dt   = _parse_any_date(end_date)   if end_date   else None

    def in_range(dstr: str) -> bool:
        if not (start_dt or end_dt):
            return True
        d = _parse_any_date(dstr)
        if not d:
            return False
        if start_dt and d < start_dt: return False
        if end_dt   and d > end_dt:   return False
        return True

    ns = "trading"
    vec = embed_query("stock borrow fee overnight borrow fee trading summary")
    res = index.query(vector=vec, top_k=top_k, namespace=ns, include_metadata=True)
    matches = getattr(res, "matches", []) or []

    per_symbol: Dict[str, float] = {}
    grand_total = 0.0
    rows_scanned = 0

    for m in matches:
        md = getattr(m, "metadata", {}) or {}
        if file and str(md.get("file", "")).strip() != file.strip():
            continue

        ft = (md.get("fee_type") or "").lower()
        if ft not in ("borrow_fee", "overnight_borrow_fee"):
            continue

        dstr = md.get("date") or ""
        if not in_range(dstr):
            continue

        sym = str(md.get("symbol", "")).upper()
        if not sym:
            continue

        amt = _safe_float(md.get("amount", 0.0), 0.0)

        per_symbol[sym] = per_symbol.get(sym, 0.0) + amt
        grand_total += amt
        rows_scanned += 1

    by_symbol = [{"symbol": s, "total": v} for s, v in per_symbol.items()]
    by_symbol.sort(key=lambda x: x["total"], reverse=False)

    pretty = ["Rows scanned: {}".format(rows_scanned),
              "--- Borrow-related fees by symbol ---"]
    for row in sorted(by_symbol, key=lambda x: abs(x["total"]), reverse=True):
        pretty.append(f'{row["symbol"]} : ${abs(row["total"]):,.2f}')
    pretty.append("")
    pretty.append("=== GRAND TOTAL ===")
    pretty.append(f"${abs(grand_total):,.2f}")

    return {
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "top_k": top_k,
            "file": file,
            "namespace": ns
        },
        "rows_scanned": rows_scanned,
        "by_symbol": by_symbol,
        "grand_total": round(grand_total, 2),
        "pretty": "\n".join(pretty)
    }

# ==== /short_pnl (aligned with breakdown matching) ====

from pydantic import BaseModel
from typing import Optional, Dict, List, Any, Tuple

def _short_key(date_str: str) -> Tuple:
    """
    Sort key: (parsed_date, SELL-first flag)
    We put SELL before BUY when the timestamp is identical so that opens precede covers.
    """
    d = _parse_any_date(date_str)
    # SELL should come before BUY on same timestamp -> SELL gets 0, BUY gets 1
    def bs_rank(bs: str) -> int:
        bsu = (bs or "").upper()
        return 0 if bsu == "SELL" else (1 if bsu == "BUY" else 2)
    # We'll return a callable; actual bs passed later. We handle bs in the sort by composing keys.
    return (d or _parse_any_date("1900-01-01")),  # base date part, we’ll extend per-row

def _short_pnl_core(
    start_date: Optional[str],
    end_date: Optional[str],
    top_k: int,
    file: Optional[str],
    authorization: Optional[str],
):
    """
    Realized P&L for SHORT trades (FIFO) with *no carry-in*:
      - Only rows where type == "Short"
      - Entry lot: bs == "SELL" (qty < 0)
      - Cover lot: bs == "BUY"  (qty > 0)
      - Symbol must be present and not MARKET
      - Only consider trades whose DATE is within [start_date, end_date] inclusive
      - On the same date, process SELL rows before BUY rows
      - PnL per match = (open_price - cover_price) * matched_shares
    """
    _check_bearer(authorization)

    # Parse date window
    start_dt = _parse_any_date(start_date) if start_date else None
    end_dt   = _parse_any_date(end_date)   if end_date   else None

    def in_range(dstr: str) -> bool:
        if not (start_dt or end_dt):
            return True
        d = _parse_any_date(dstr)
        if not d:
            return False
        if start_dt and d < start_dt: return False
        if end_dt   and d > end_dt:   return False
        return True

    ns = "trading"

    # Broad semantic probe to retrieve short trade rows
    vec = embed_query("short trades SELL BUY cover quantity price FIFO realized pnl")
    res = index.query(vector=vec, top_k=top_k, namespace=ns, include_metadata=True)
    matches = getattr(res, "matches", []) or []

    # Helper
    def fnum(v, default=0.0):
        if v is None:
            return float(default)
        try:
            if isinstance(v, str):
                v = v.replace(",", "").strip()
            return float(v)
        except Exception:
            return float(default)

    # Collect only rows that are:
    # - file matches (if provided)
    # - type == Short
    # - bs in SELL/BUY
    # - symbol present and not MARKET
    # - date within the requested range
    rows: List[Dict[str, Any]] = []
    for m in matches:
        md = getattr(m, "metadata", {}) or {}

        if file and str(md.get("file", "")).strip() != file.strip():
            continue

        t = str(md.get("type") or md.get("Type") or "").strip().upper()
        if t != "SHORT":
            continue

        bs = str(md.get("bs") or md.get("B/S") or "").strip().upper()
        if bs not in ("SELL", "BUY"):
            continue

        sym = str(md.get("symbol") or md.get("Symbol") or "").strip().upper()
        if not sym or sym == "MARKET":
            continue

        dstr = str(md.get("date", "")).strip()
        if not dstr or not in_range(dstr):
            continue

        qty = fnum(md.get("qty") or md.get("Quantity") or md.get("Qty"), 0.0)
        price = fnum(md.get("price") or md.get("Price"), 0.0)
        if qty == 0.0 or price == 0.0:
            continue

        d = _parse_any_date(dstr)
        if not d:
            continue

        rows.append({
            "date_str": dstr,
            "date": d,
            "bs": bs,
            "symbol": sym,
            "qty": qty,
            "price": price,
        })

    # Deterministic ordering:
    #  - by date ascending
    #  - SELL before BUY on the same date
    #  - stable on insertion (Python sort is stable)
    rows.sort(key=lambda r: (r["date"], 0 if r["bs"] == "SELL" else 1))

    # FIFO state scoped to the window ONLY (no pre-window preloading)
    open_lots: Dict[str, List[Dict[str, Any]]] = {}   # symbol -> [{shares, price, date}]
    realized_by_symbol: Dict[str, float] = {}

    rows_scanned = 0
    opens_used = 0
    covers_in_window = 0

    for r in rows:
        rows_scanned += 1

        sym   = r["symbol"]
        bs    = r["bs"]
        qty   = int(abs(r["qty"]))
        price = float(r["price"])
        dstr  = r["date_str"]

        if sym not in open_lots:
            open_lots[sym] = []
        if sym not in realized_by_symbol:
            realized_by_symbol[sym] = 0.0

        if bs == "SELL":
            # Entry (store positive shares)
            open_lots[sym].append({
                "shares": qty,
                "price": price,
                "date": dstr,
            })
            opens_used += 1
        else:
            # BUY (cover) — match only against window opens
            covers_in_window += 1
            cover = qty
            lots = open_lots[sym]
            i = 0
            while cover > 0 and i < len(lots):
                lot = lots[i]
                match_shares = min(cover, lot["shares"])
                pnl = (lot["price"] - price) * match_shares
                realized_by_symbol[sym] += pnl

                lot["shares"] -= match_shares
                cover -= match_shares
                if lot["shares"] == 0:
                    lots.pop(i)
                else:
                    i += 1
            # If unmatched cover remains (cover > 0), ignore the excess — no carry-in allowed.

    # Build outputs
    realized_list = [
        {"symbol": s, "realized_pnl": round(v, 2)}
        for s, v in realized_by_symbol.items()
        if abs(v) > 1e-9
    ]
    realized_list.sort(key=lambda x: x["realized_pnl"], reverse=True)

    total_realized = round(sum(x["realized_pnl"] for x in realized_list), 2)

    return {
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "top_k": top_k,
            "file": file,
            "namespace": ns
        },
        "rows_scanned": rows_scanned,
        "trades_used": len(rows),
        "debug_counts": {
            "opens_used": opens_used,
            "covers_in_window": covers_in_window,
            "opens_seen_pre_match": 0  # explicitly zero: no pre-window carry-in
        },
        "total_realized_pnl": total_realized,
        "sample": realized_list[:5],
        "realized_by_symbol": realized_list
    }
class ShortPnlBody(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    top_k: int = 5000
    file: Optional[str] = None

@app.post("/short_pnl")
def short_pnl_post(
    body: ShortPnlBody,
    authorization: Optional[str] = Header(default=None),
):
    return _short_pnl_core(
        start_date=body.start_date,
        end_date=body.end_date,
        top_k=body.top_k,
        file=body.file,
        authorization=authorization,
    )

@app.get("/short_pnl")
def short_pnl_get(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date:   Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    top_k:      int = Query(5000, description="How many rows to scan"),
    file:       Optional[str] = Query(None, description="Optional: restrict to a specific uploaded CSV filename"),
    authorization: Optional[str] = Header(default=None),
):
    return _short_pnl_core(start_date, end_date, top_k, file, authorization)
    
# ========= Short PnL BREAKDOWN (FIFO, robust, with pre-window burn) =========

from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

def _dt(s: str) -> Optional[datetime]:
    """Parse dates like '6/10/2025', '2025-06-10', etc. Returns None if bad."""
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%-m/%-d/%Y", "%m/%-d/%Y", "%-m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # last-resort tolerant parse
    try:
        m, d, y = s.split("/")
        return datetime(int(y), int(m), int(d))
    except Exception:
        return None

def _num(x, default=0.0) -> float:
    if x is None:
        return float(default)
    try:
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return float(x)
    except Exception:
        return float(default)

def _short_pnl_breakdown_core(
    start_date: Optional[str],
    end_date: Optional[str],
    top_k: int,
    file: Optional[str],
    symbol: Optional[str],
    limit: int,
    allow_carry: bool,      # False = both legs must be inside window (no carry-in)
    authorization: Optional[str]
):
    _check_bearer(authorization)

    ns = "trading"
    start_dt = _dt(start_date) if start_date else None
    end_dt   = _dt(end_date)   if end_date   else None

    def in_window(d: datetime) -> bool:
        if start_dt and d < start_dt: return False
        if end_dt   and d > end_dt:   return False
        return True

    # --- 1) Pull candidate rows from Pinecone
    vec = embed_query("short trades FIFO realized pnl SELL BUY cover shorting")
    res = index.query(vector=vec, top_k=top_k, namespace=ns, include_metadata=True)
    matches = getattr(res, "matches", []) or []

    # --- 2) Normalize / filter to atomic trade rows
    rows: List[Dict[str, Any]] = []
    rows_scanned = 0

    want_symbol = (symbol or "").strip().upper() or None
    want_file   = (file or "").strip() or None

    for m in matches:
        md = getattr(m, "metadata", {}) or {}

        # file filter
        if want_file and str(md.get("file", "")).strip() != want_file:
            continue

        # only Short type
        t = str(md.get("type") or md.get("Type") or "").strip().lower()
        if t != "short":
            continue

        # symbol
        sym = str(md.get("symbol") or md.get("Symbol") or "").strip().upper()
        if not sym or sym == "MARKET":
            continue
        if want_symbol and sym != want_symbol:
            continue

        # B/S must be SELL or BUY
        bs = str(md.get("bs") or md.get("B/S") or "").strip().upper()
        if bs not in ("SELL", "BUY"):
            continue

        # date
        dstr = str(md.get("date") or "").strip()
        d = _dt(dstr)
        if not d:
            continue

        # numbers
        qty   = _num(md.get("qty") or md.get("Quantity") or md.get("Qty"), 0.0)
        price = _num(md.get("price") or md.get("Price"), 0.0)
        if qty == 0.0 or price == 0.0:
            continue

        rows.append({
            "sym":   sym,
            "dt":    d,
            "date":  dstr,
            "bs":    bs,
            "qty":   qty,
            "price": price,
            "file":  md.get("file"),
        })
        rows_scanned += 1

    if not rows:
        return {
            "ok": True,
            "filters": {
                "start_date": start_date, "end_date": end_date,
                "top_k": top_k, "file": file, "symbol": symbol,
                "namespace": ns, "allow_carry": allow_carry
            },
            "rows_scanned": rows_scanned,
            "trades_used": 0,
            "legs_count": 0,
            "total_realized_pnl": 0.0,
            "legs": []
        }

    # --- 3) Sort chronologically; ensure SELL before BUY *within same day*
    rows.sort(key=lambda r: (r["dt"], 0 if r["bs"] == "SELL" else 1, -abs(r["qty"])))

    # --- 4) Pre-window burn: consume ALL activity before start_dt so inventory is clean
    opens: Dict[str, List[Dict[str, Any]]] = {}
    trades_used = 0

    if start_dt:
        for r in rows:
            if r["dt"] >= start_dt:
                break
            if r["sym"] not in opens: opens[r["sym"]] = []
            if r["bs"] == "SELL":
                # store positive lot size for easier math
                opens[r["sym"]].append({"shares": int(abs(r["qty"])), "price": r["price"], "dt": r["dt"]})
                trades_used += 1
            else:  # BUY
                cover = int(abs(r["qty"]))
                lots = opens[r["sym"]]
                i = 0
                while cover > 0 and i < len(lots):
                    take = min(cover, lots[i]["shares"])
                    lots[i]["shares"] -= take
                    cover -= take
                    if lots[i]["shares"] == 0:
                        lots.pop(i)
                    else:
                        i += 1
                trades_used += 1
        # If carry-in is NOT allowed, drop any remaining open lots from before window:
        if not allow_carry:
            opens = {}

    # --- 5) In-window matching and legs collection
    legs: List[Dict[str, Any]] = []
    for r in rows:
        if not in_window(r["dt"]):        # only iterate window rows
            continue
        sym = r["sym"]
        if sym not in opens: opens[sym] = []

        if r["bs"] == "SELL":
            opens[sym].append({"shares": int(abs(r["qty"])), "price": r["price"], "dt": r["dt"]})
            trades_used += 1
        else:  # BUY
            cover = int(abs(r["qty"]))
            lots = opens[sym]
            i = 0
            while cover > 0 and i < len(lots):
                lot = lots[i]
                take = min(cover, lot["shares"])
                pnl  = (lot["price"] - r["price"]) * take

                # record leg (cover is inside the window by construction here)
                # If carry is disallowed, the only possible lots are window lots (we reset opens above).
                legs.append({
                    "symbol": sym,
                    "shares": take,
                    "open_date":  lot["dt"].strftime("%-m/%-d/%Y"),
                    "open_price": round(lot["price"], 5),
                    "cover_date": r["dt"].strftime("%-m/%-d/%Y"),
                    "cover_price": round(r["price"], 5),
                    "pnl": round(pnl, 2),
                    "file": r["file"]
                })

                lot["shares"] -= take
                cover -= take
                if lot["shares"] == 0:
                    lots.pop(i)
                else:
                    i += 1
            trades_used += 1

    # apply symbol filter at the end if requested (keeps totals coherent per symbol)
    if want_symbol:
        legs = [x for x in legs if x["symbol"] == want_symbol]

    # limit output legs if requested
    if limit and limit > 0:
        legs = legs[:limit]

    total_pnl = round(sum(l["pnl"] for l in legs), 2)

    return {
        "ok": True,
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "top_k": top_k,
            "file": file,
            "symbol": symbol,
            "namespace": ns,
            "allow_carry": allow_carry
        },
        "rows_scanned": rows_scanned,
        "trades_used": trades_used,
        "legs_count": len(legs),
        "total_realized_pnl": total_pnl,
        "legs": legs
    }

# -------- FastAPI surfaces (GET + POST) --------

class ShortPnlBreakdownBody(BaseModel):
    start_date: Optional[str] = None
    end_date:   Optional[str] = None
    top_k:      int = 5000
    file:       Optional[str] = None
    symbol:     Optional[str] = None
    limit:      int = 100000
    allow_carry: bool = False  # default to NO carry-in

@app.get("/short_pnl_breakdown")
def short_pnl_breakdown_get(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date:   Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    top_k:      int = Query(5000, description="Vector rows to scan"),
    file:       Optional[str] = Query(None, description="Specific CSV filename"),
    symbol:     Optional[str] = Query(None, description="Restrict to one symbol"),
    limit:      int = Query(100000, description="Max legs to return"),
    allow_carry: bool = Query(False, description="Allow opens before start to close inside window"),
    authorization: Optional[str] = Header(default=None),
):
    return _short_pnl_breakdown_core(
        start_date, end_date, top_k, file, symbol, limit, allow_carry, authorization
    )

@app.post("/short_pnl_breakdown")
def short_pnl_breakdown_post(
    body: ShortPnlBreakdownBody,
    authorization: Optional[str] = Header(default=None),
):
    return _short_pnl_breakdown_core(
        body.start_date, body.end_date, body.top_k, body.file,
        body.symbol, body.limit, body.allow_carry, authorization
    )
    
# ==== /shorts_over_price (filter SHORT entries by entry/price/date) ====

@app.get("/shorts_over_price")
def shorts_over_price(
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD start (inclusive)"),
    end_date:   Optional[str] = Query(None, description="YYYY-MM-DD end (inclusive)"),
    min_price:  float         = Query(20.0, description="Minimum entry price to include"),
    file:       Optional[str] = Query(None, description="Restrict to this CSV filename"),
    entry_only: bool          = Query(True, description="Only include entry rows (qty<0 or side=Sell)"),
    top_k:      int           = Query(5000, description="Vector rows to scan"),
    namespace:  str           = Query("trading", description="Vector namespace"),
    authorization: Optional[str] = Header(default=None),
):
    """
    Returns SHORT trade rows whose PRICE > min_price within a date window.
    Defaults to *entry* rows only (qty<0 or side=Sell).
    Output: [{date, symbol, price, qty, side, description, file}]
    """
    _check_bearer(authorization)

    # date helpers
    start_dt = _parse_any_date(start_date) if start_date else None
    end_dt   = _parse_any_date(end_date)   if end_date   else None

    def in_range(dstr: str) -> bool:
        if not (start_dt or end_dt):
            return True
        d = _parse_any_date(dstr)
        if not d:
            return False
        if start_dt and d < start_dt: return False
        if end_dt   and d > end_dt:   return False
        return True

    # target the right rows in vector search, then filter tightly
    vec = embed_query("Type SHORT trades with price quantity side date symbol")
    res = index.query(vector=vec, top_k=top_k, namespace=namespace, include_metadata=True)
    matches = getattr(res, "matches", []) or []

    def num(*vals, default=0.0):
        for v in vals:
            if v is None: 
                continue
            try:
                if isinstance(v, str):
                    v = v.replace(",", "").strip()
                return float(v)
            except Exception:
                pass
        return float(default)

    out = []
    rows_scanned = 0

    for m in matches:
        md = getattr(m, "metadata", {}) or {}

        # optional file restriction
        if file and str(md.get("file", "")).strip() != file.strip():
            continue

        dstr = str(md.get("date", "")).strip()
        if not dstr or not in_range(dstr):
            continue

        # must be SHORT type
        tval = (md.get("Type") or md.get("type") or "").strip().upper()
        if tval != "SHORT":
            continue

        sym = (md.get("symbol") or md.get("Symbol") or "").strip().upper()
        if not sym or sym == "MARKET":
            continue

        price = num(md.get("Price"), md.get("price"))
        if not price or price <= float(min_price):
            continue

        qty = num(md.get("Quantity"), md.get("Qty"), md.get("quantity"), md.get("qty"))
        side = (md.get("B/S") or md.get("BS") or md.get("side") or md.get("Side") or "").strip().upper()

        if entry_only:
            is_entry = (qty < 0) or (side in ("SELL", "S"))
            if not is_entry:
                continue

        rows_scanned += 1
        out.append({
            "date": dstr,
            "symbol": sym,
            "price": round(price, 5),
            "qty": int(qty) if qty == int(qty) else qty,
            "side": side or ("SELL" if qty < 0 else "BUY"),
            "description": md.get("description") or md.get("Description") or "",
            "file": md.get("file", "")
        })

    # sort: date then symbol
    out.sort(key=lambda r: (r["date"], r["symbol"]))

    return {
        "ok": True,
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "min_price": min_price,
            "file": file,
            "entry_only": entry_only,
            "top_k": top_k,
            "namespace": namespace
        },
        "rows_scanned": rows_scanned,
        "count": len(out),
        "rows": out
    }

# ==== /answer (RAG answer with citations; multi-namespace support) ====

from pydantic import BaseModel
from fastapi import HTTPException, Header

class AnswerBody(BaseModel):
    query: str
    namespace: str = "nonfiction"   # "all" will fan out to multiple namespaces
    top_k: int = 12
    min_score: Optional[float] = None

def _vector_search_simple(query: str, namespace: str, top_k: int, min_score: Optional[float]) -> List[Dict[str, Any]]:
    """
    Runs an embedding search against a single namespace and returns
    a list of hits with normalized fields: score, metadata, id.
    """
    vec = embed_query(query)
    try:
        res = index.query(vector=vec, top_k=top_k, namespace=namespace, include_metadata=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector query failed: {e}")

    matches = getattr(res, "matches", []) or []
    hits = []
    for m in matches:
        sc = getattr(m, "score", 0.0) or 0.0
        if (min_score is not None) and sc < float(min_score):
            continue
        hits.append({
            "score": sc,
            "id": getattr(m, "id", None),
            "metadata": getattr(m, "metadata", {}) or {},
            "namespace": namespace,
        })
    return hits

def _vector_search_multi(query: str, namespaces: List[str], top_k: int, min_score: Optional[float]) -> List[Dict[str, Any]]:
    """
    Run vector search across multiple namespaces and merge results.
    Returns a list of hits with: namespace, score, id, metadata.
    """
    vec = embed_query(query)
    merged: List[Dict[str, Any]] = []
    for ns in namespaces:
        try:
            res = index.query(vector=vec, top_k=top_k, namespace=ns, include_metadata=True)
            matches = getattr(res, "matches", []) or []
            for m in matches:
                sc = float(getattr(m, "score", 0.0) or 0.0)
                if (min_score is not None) and sc < float(min_score):
                    continue
                merged.append({
                    "namespace": ns,
                    "score": sc,
                    "id": getattr(m, "id", None),
                    "metadata": getattr(m, "metadata", {}) or {},
                })
        except Exception as e:
            # non-fatal; record the error as a stub so we can see which ns failed
            merged.append({"namespace": ns, "score": 0.0, "id": None, "metadata": {"error": str(e)}})
    return merged

@app.post("/answer", summary="Answer")
def answer(body: AnswerBody, authorization: Optional[str] = Header(default=None)):
    """
    Synthesizes a short answer from vector search results in the requested namespace.
    - If namespace == 'all' (or '*' or ''), fans out across ['nonfiction','trading','short-selling'].
    - If OPENAI_API_KEY is missing or model call fails, returns a debug payload with snippets & citations.
    - Citations appear as [1], [2], ... based on snippet order.
    """
    _check_bearer(authorization)

    # 1) Vector search (single ns or multi-ns)
    ns_raw = (body.namespace or "").lower()
    if ns_raw in ("all", "*", ""):
        namespaces = ["nonfiction", "trading", "short-selling"]
        hits = _vector_search_multi(
            query=body.query,
            namespaces=namespaces,
            top_k=body.top_k,
            min_score=body.min_score
        )
        effective_namespace = "all"
    else:
        hits = _vector_search_simple(
            query=body.query,
            namespace=body.namespace,
            top_k=body.top_k,
            min_score=body.min_score
        )
        effective_namespace = body.namespace

    # 2) Build snippets and numbered context
    snippets: List[Dict[str, Any]] = []
    for h in hits:
        md = h.get("metadata", {}) or {}
        ns = h.get("namespace") or md.get("namespace") or effective_namespace
        snippets.append({
            "hash": md.get("uid") or md.get("hash") or h.get("id"),
            "source": md.get("source") or md.get("file") or md.get("symbol") or md.get("title") or "",
            "namespace": ns,
            "title": md.get("title") or md.get("symbol") or md.get("file") or "",
            "text": md.get("text") or md.get("description") or md.get("content") or "",
            "score": float(h.get("score", 0.0) or 0.0),
        })

    if not snippets:
        return {
            "answer": "I couldn’t find anything relevant in the vector index for that question.",
            "citations": [],
            "snippets": [],
            "used_top_k": 0,
            "namespace": effective_namespace
        }

    blocks = []
    for i, s in enumerate(snippets, start=1):
        txt = (s["text"] or "").strip()
        if not txt:
            continue
        # include namespace for clarity in merged results
        blocks.append(f"[{i}] {txt}\n(source={s['source']}, ns={s['namespace']}, hash={s['hash']})")
    context = "\n\n".join(blocks)

    # 3) If no OpenAI key, return debug (no model call)
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "answer": "(debug) No OPENAI_API_KEY set on the backend — returning snippets only.",
            "citations": [{"n": i+1, "hash": s["hash"], "source": s["source"], "namespace": s["namespace"]} for i, s in enumerate(snippets)],
            "snippets": snippets,
            "used_top_k": len(snippets),
            "namespace": effective_namespace
        }

    # 4) Call OpenAI to synthesize an answer with inline citations
    try:
        model = os.getenv("ANSWER_MODEL", "gpt-4o")
        prompt = f"""Answer the question using ONLY the context. If something is not in the context, say you can't find it.
Cite sources inline using square-bracket numbers like [1], [2] that refer to the context blocks below.

Question: {body.query}

Context:
{context}
"""
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "Be concise, factual, and cite sources with [1], [2], etc."},
                {"role": "user", "content": prompt},
            ],
        )
        answer_text = resp.choices[0].message.content
    except Exception as e:
        # Graceful fallback if the model call fails
        return {
            "answer": f"(debug) Model call failed: {e}. Returning snippets only.",
            "citations": [{"n": i+1, "hash": s["hash"], "source": s["source"], "namespace": s["namespace"]} for i, s in enumerate(snippets)],
            "snippets": snippets,
            "used_top_k": len(snippets),
            "namespace": effective_namespace
        }

    return {
        "answer": answer_text,
        "citations": [{"n": i+1, "hash": s["hash"], "source": s["source"], "namespace": s["namespace"]} for i, s in enumerate(snippets)],
        "snippets": snippets,
        "used_top_k": len(snippets),
        "namespace": effective_namespace
    }
