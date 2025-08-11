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
app = FastAPI(title="Rob Forever Brain API", version="2.2.5")

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
        {"url": "https://rob-brain-api.onrender.com", "description": "prod"}
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

# ==== /fees (CSV rollup across common fee columns + borrow/overnight; safe math) ====
@app.post("/fees")
def fees(body: FeesBody, authorization: Optional[str] = Header(default=None)):
    _check_bearer(authorization)

    start_dt = _parse_any_date(body.start) if body.start else None
    end_dt   = _parse_any_date(body.end)   if body.end   else None

    def in_range(dstr: str) -> bool:
        if not (start_dt or end_dt):
            return True
        d = _parse_any_date(dstr)
        if not d:
            return False
        if start_dt and d < start_dt: return False
        if end_dt   and d > end_dt:   return False
        return True

    ns = body.namespace or "trading"
    # Keep query broad; we filter by file/date below
    vec = embed_query("broker activity trade rows fee columns borrow overnight")
    res = index.query(vector=vec, top_k=body.top_k, namespace=ns, include_metadata=True)
    matches = getattr(res, "matches", []) or []

    fee_cols = ["TransFee", "ECNTaker", "ECNMaker", "ORFFee", "TAFFee", "SECFee", "CATFee", "Commissions"]

    by_col: Dict[str, float] = {c: 0.0 for c in fee_cols}
    borrow_fees = 0.0
    overnight_fees = 0.0
    rows_scanned = 0

    for m in matches:
        md = getattr(m, "metadata", {}) or {}

        # Optional file filter
        if body.file and str(md.get("file", "")).strip() != str(body.file).strip():
            continue

        # Date filter
        if not in_range(str(md.get("date", ""))):
            continue

        # Sum explicit fee columns safely (nulls, strings like "-3,000", "$12.34", etc.)
        for c in fee_cols:
            by_col[c] += _safe_float(md.get(c, 0.0), 0.0)

        # Borrow / overnight come from dedicated rows with fee_type + amount
        ft  = str(md.get("fee_type") or "").strip().lower()
        amt = _safe_float(md.get("amount", 0.0), 0.0)
        if ft == "borrow_fee":
            borrow_fees += amt
        elif ft == "overnight_borrow_fee":
            overnight_fees += amt

        rows_scanned += 1

    by_column_sum = sum(by_col.values())
    grand_total = by_column_sum + borrow_fees + overnight_fees

    pretty_lines = []
    pretty_lines.append(f"Rows scanned: {rows_scanned}")
    pretty_lines.append("")
    pretty_lines.append("--- Fee columns ---")
    for c in fee_cols:
        pretty_lines.append(f"{c:<10}: ${by_col[c]:,.2f}")
    pretty_lines.append("")
    pretty_lines.append(f"Borrow fees: ${borrow_fees:,.2f}")
    pretty_lines.append(f"Overnight fees: ${overnight_fees:,.2f}")
    pretty_lines.append("")
    pretty_lines.append("=== ALL-IN FEES (fee columns + borrow + overnight) ===")
    pretty_lines.append(f"${grand_total:,.2f}")

    return {
        "ok": True,
        "filters": {
            "namespace": ns,
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
            "by_column_sum": round(by_column_sum, 2),
            "borrow_fees": round(borrow_fees, 2),
            "overnight_fees": round(overnight_fees, 2),
            "grand_total_all_in": round(grand_total, 2),
        },
        "pretty": "\n".join(pretty_lines),
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

# ---- shared worker so GET and POST do identical work ----
def _short_pnl_core(
    start_date: Optional[str],
    end_date: Optional[str],
    top_k: int,
    file: Optional[str],
    authorization: Optional[str],
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
        if start_dt and d < start_dt:
            return False
        if end_dt and d > end_dt:
            return False
        return True

    ns = "trading"
    vec = embed_query("Type SHORT short trades sell buy cover quantity price FIFO PnL")
    res = index.query(vector=vec, top_k=top_k, namespace=ns, include_metadata=True)
    matches = getattr(res, "matches", []) or []

    # Per-symbol FIFO open lots
    open_lots: Dict[str, List[Dict[str, Any]]] = {}
    realized_by_symbol: Dict[str, float] = {}
    rows_scanned = 0
    trades_used = 0

    def get_num(*vals, default=0.0):
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

    for m in matches:
        md = (getattr(m, "metadata", {}) or {})

        if file and str(md.get("file", "")).strip() != file.strip():
            continue

        dstr = str(md.get("date", "")).strip()
        if not dstr or not in_range(dstr):
            continue

        tval = (md.get("Type") or md.get("type") or "").strip().upper()
        if tval != "SHORT":
            continue

        sym = (md.get("symbol") or md.get("Symbol") or "").strip().upper()
        if not sym or sym == "MARKET":
            continue

        qty = get_num(md.get("Quantity"), md.get("Qty"), md.get("quantity"), md.get("qty"))
        if qty == 0:
            continue

        price = get_num(md.get("Price"), md.get("price"))
        if price == 0:
            continue

        rows_scanned += 1

        if sym not in open_lots:
            open_lots[sym] = []
        if sym not in realized_by_symbol:
            realized_by_symbol[sym] = 0.0

        if qty < 0:
            open_lots[sym].append({
                "shares": int(abs(qty)),
                "price": float(price),
                "date": dstr,
            })
            trades_used += 1
        else:
            cover = int(qty)
            trades_used += 1
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

    realized_list = [
        {"symbol": s, "realized_pnl": round(v, 2)}
        for s, v in realized_by_symbol.items()
        if abs(v) > 1e-9
    ]
    realized_list.sort(key=lambda x: x["realized_pnl"], reverse=True)
    total_realized = round(sum(x["realized_pnl"] for x in realized_list), 2)

    return {
        "ok": True,
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "top_k": top_k,
            "file": file,
            "namespace": ns
        },
        "rows_scanned": rows_scanned,
        "trades_used": trades_used,
        "realized_by_symbol": realized_list,
        "total_realized_pnl": total_realized,
    }

# ==== /short_pnl (realized P&L for SHORT trades, FIFO) ====

@app.post("/short_pnl")
def short_pnl(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str]   = Query(None, description="End date YYYY-MM-DD"),
    top_k: int                = Query(5000, description="How many rows to scan"),
    file: Optional[str]       = Query(None, description="Optional: restrict to a specific uploaded CSV filename"),
    authorization: Optional[str] = Header(default=None),
):
    """
    Realized P&L for SHORT trades over a date window (FIFO):
    - Uses vector rows where Type == SHORT, qty != 0, symbol != MARKET
    - qty < 0 = open short (sell); qty > 0 = cover (buy)
    - PnL = (open_price - cover_price) * matched_shares
    - Returns realized PnL by symbol and any open lots still outstanding
    """
    _check_bearer(authorization)

    start_dt = _parse_any_date(start_date) if start_date else None
    end_dt   = _parse_any_date(end_date)   if end_date   else None

    def in_range(dstr: str) -> bool:
        if not (start_dt or end_dt):
            return True
        d = _parse_any_date(dstr)
        if not d:
            return False
        if start_dt and d < start_dt:
            return False
        if end_dt and d > end_dt:
            return False
        return True

    ns = "trading"
    vec = embed_query("Type SHORT short trades sell buy cover quantity price FIFO PnL")
    res = index.query(vector=vec, top_k=top_k, namespace=ns, include_metadata=True)
    matches = getattr(res, "matches", []) or []

    # Per-symbol FIFO open lots: list of dicts {shares:int, price:float, date:str}
    open_lots: Dict[str, List[Dict[str, Any]]] = {}
    realized_by_symbol: Dict[str, float] = {}
    rows_scanned = 0
    trades_used = 0

    def get_num(*vals, default=0.0):
        for v in vals:
            if v is None:
                continue
            try:
                # handle strings like "-3,000"
                if isinstance(v, str):
                    v = v.replace(",", "").strip()
                return float(v)
            except Exception:
                pass
        return float(default)

    for m in matches:
        md = (getattr(m, "metadata", {}) or {})
        # Optional: restrict to a specific uploaded CSV
        if file and str(md.get("file", "")).strip() != file.strip():
            continue

        # Date filter
        dstr = str(md.get("date", "")).strip()
        if not dstr or not in_range(dstr):
            continue

        # Type must be SHORT
        tval = (md.get("Type") or md.get("type") or "").strip().upper()
        if tval != "SHORT":
            continue

        # Symbol present and not MARKET
        sym = (md.get("symbol") or md.get("Symbol") or "").strip().upper()
        if not sym or sym == "MARKET":
            continue

        # Quantity and Price
        qty = get_num(md.get("Quantity"), md.get("Qty"), md.get("quantity"), md.get("qty"))
        if qty == 0:
            continue
        price = get_num(md.get("Price"), md.get("price"))
        if price == 0:
            # skip rows without a usable price
            continue

        rows_scanned += 1

        # Initialize symbol buckets
        if sym not in open_lots:
            open_lots[sym] = []
        if sym not in realized_by_symbol:
            realized_by_symbol[sym] = 0.0

        if qty < 0:
            # Open short: store positive share count for convenience
            open_lots[sym].append({
                "shares": int(abs(qty)),
                "price": float(price),
                "date": dstr,
            })
            trades_used += 1
        else:
            # Cover: FIFO match against open lots
            cover = int(qty)
            trades_used += 1

            lots = open_lots[sym]
            i = 0
            while cover > 0 and i < len(lots):
                lot = lots[i]
                match_shares = min(cover, lot["shares"])
                # Short PnL = (open - cover) * shares
                pnl = (lot["price"] - price) * match_shares
                realized_by_symbol[sym] += pnl

                lot["shares"] -= match_shares
                cover -= match_shares
                if lot["shares"] == 0:
                    lots.pop(i)
                else:
                    i += 1
            # If cover > 0 here, unmatched cover (no open lot) is ignored.

    # Prepare outputs
    realized_list = [
        {"symbol": s, "realized_pnl": round(v, 2)}
        for s, v in realized_by_symbol.items()
        if abs(v) > 1e-9
    ]
    realized_list.sort(key=lambda x: x["realized_pnl"], reverse=True)

    total_realized = round(sum(x["realized_pnl"] for x in realized_list), 2)

    open_short_shares_by_symbol = {
        s: sum(lot["shares"] for lot in lots) for s, lots in open_lots.items()
        if sum(lot["shares"] for lot in lots) != 0
    }

    return {
        "ok": True,
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "top_k": top_k,
            "file": file,
            "namespace": ns
        },
        "rows_scanned": rows_scanned,
        "trades_used": trades_used,
        "realized_by_symbol": realized_list,
        "total_realized_pnl": total_realized,
        "open_short_shares_by_symbol": open_short_shares_by_symbol,
    }


# GET alias so tools that prefer GET can call it via query params
@app.get("/short_pnl", summary="Short PnL (GET)")
def short_pnl_get(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str]   = Query(None, description="End date YYYY-MM-DD"),
    top_k: int                = Query(5000, description="How many rows to scan"),
    file: Optional[str]       = Query(None, description="Optional: restrict to a specific uploaded CSV filename"),
    authorization: Optional[str] = Header(default=None),
):
    # Reuse the same implementation by calling the POST handler
    return short_pnl(
        start_date=start_date,
        end_date=end_date,
        top_k=top_k,
        file=file,
        authorization=authorization,
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
