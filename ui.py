import os
import json
import datetime as dt
from typing import Optional, Dict, Any, List

import requests
import streamlit as st

# --- Config (read from env if present) ---
DEFAULT_BASE_URL = os.getenv("APP_BASE_URL", "https://rob-brain-api-1.onrender.com")
DEFAULT_TOKEN    = os.getenv("ROB_BRAIN_TOKEN", "")
DEFAULT_FILE     = os.getenv("ROB_BRAIN_FILE", "cobra_activity_2025-07-01_to_2025-08-10.csv.csv")

# --- Helpers ---
def auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}

def get_json(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(url, headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def post_json(url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
    hdrs = {**headers, "Content-Type": "application/json"}
    r = requests.post(url, headers=hdrs, json=body, timeout=60)
    r.raise_for_status()
    return r.json()

def date_to_str(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")

def section_divider():
    st.markdown("---")

# --- UI ---
st.set_page_config(page_title="Rob Brain – Trading Analytics", layout="wide")

st.title("🧠 Rob Brain – Trading Analytics")

with st.sidebar:
    st.header("Settings")
    base_url = st.text_input("API Base URL", value=DEFAULT_BASE_URL, help="Your Render service URL")
    token    = st.text_input("Auth Token", value=DEFAULT_TOKEN, type="password", help="ROB_BRAIN_TOKEN")
    file_in  = st.text_input("CSV Filename", value=DEFAULT_FILE, help="Exact filename stored in the vector index")

    st.caption("Tip: set env vars APP_BASE_URL / ROB_BRAIN_TOKEN / ROB_BRAIN_FILE to prefill these.")

    # New fields
    namespace = st.text_input("Namespace", value="trading", help="e.g., trading, nonfiction, short-selling")
    top_k_search = st.number_input("Top K (search)", min_value=1, max_value=5000, value=8, step=1)

# Health check
col_h1, col_h2 = st.columns([1,3])
with col_h1:
    if st.button("✅ Health check"):
        try:
            resp = requests.get(f"{base_url}/healthz", headers=auth_headers(token), timeout=20)
            st.success(resp.text)
        except Exception as e:
            st.error(f"Health check failed: {e}")

with col_h2:
    if st.button("📜 Show OpenAPI"):
        try:
            resp = requests.get(f"{base_url}/openapi.json", headers=auth_headers(token), timeout=20)
            resp.raise_for_status()
            st.json(resp.json().get("info", {}))
        except Exception as e:
            st.error(f"OpenAPI fetch failed: {e}")

section_divider()

# Short P&L Breakdown
st.subheader("📉 Short P&L Breakdown")
c1, c2, c3, c4 = st.columns(4)
with c1:
    start_b = st.date_input("Start date (breakdown)", value=dt.date(2025, 8, 1))
with c2:
    end_b   = st.date_input("End date (breakdown)", value=dt.date(2025, 8, 31))
with c3:
    symbol  = st.text_input("Symbol (optional)", value="")
with c4:
    allow_carry = st.checkbox("Allow carry (opens before start can close inside window)", value=False)

c5, c6 = st.columns(2)
with c5:
    top_k_b = st.number_input("Top K (vector rows to scan)", min_value=100, max_value=20000, value=8000, step=100)
with c6:
    limit_b = st.number_input("Max legs to return", min_value=100, max_value=200000, value=100000, step=1000)

if st.button("Run P&L Breakdown"):
    try:
        params = {
            "start_date": date_to_str(start_b),
            "end_date":   date_to_str(end_b),
            "file":       file_in,
            "top_k":      top_k_b,
            "limit":      limit_b,
            "allow_carry": str(allow_carry).lower(),
        }
        if symbol.strip():
            params["symbol"] = symbol.strip().upper()

        data = get_json(f"{base_url}/short_pnl_breakdown", auth_headers(token), params)
        st.success("Breakdown fetched.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Realized P&L", f"{data.get('total_realized_pnl', 0):,.2f}")
        m2.metric("Legs Count", f"{data.get('legs_count', 0)}")
        m3.metric("Rows Scanned", f"{data.get('rows_scanned', 0)}" if "rows_scanned" in data else "-")

        # Legs table
        legs = data.get("legs", [])
        if legs:
            st.caption("Legs")
            st.dataframe(legs, use_container_width=True)
        else:
            st.info("No legs returned for this window/filters.")

        with st.expander("Raw JSON"):
            st.code(json.dumps(data, indent=2))
    except Exception as e:
        st.error(f"Breakdown error: {e}")

section_divider()

# Short P&L (aggregate)
st.subheader("💰 Short P&L (Aggregate)")
c1, c2, c3 = st.columns(3)
with c1:
    start_p = st.date_input("Start date (P&L)", value=dt.date(2025, 8, 1), key="startp")
with c2:
    end_p   = st.date_input("End date (P&L)", value=dt.date(2025, 8, 31), key="endp")
with c3:
    top_k_p = st.number_input("Top K", min_value=100, max_value=20000, value=8000, step=100, key="topkp")

if st.button("Run Short P&L Aggregate"):
    try:
        body = {
            "start_date": date_to_str(start_p),
            "end_date":   date_to_str(end_p),
            "top_k":      top_k_p,
            "file":       file_in
        }
        data = post_json(f"{base_url}/short_pnl", auth_headers(token), body)
        st.success("P&L fetched.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Realized P&L", f"{data.get('total_realized_pnl', 0):,.2f}")
        m2.metric("Trades Used", f"{data.get('trades_used', 0)}")
        m3.metric("Rows Scanned", f"{data.get('rows_scanned', 0)}")

        by_symbol = data.get("realized_by_symbol", [])
        if by_symbol:
            st.caption("By Symbol")
            st.dataframe(by_symbol, use_container_width=True)

        with st.expander("Raw JSON"):
            st.code(json.dumps(data, indent=2))
    except Exception as e:
        st.error(f"Short P&L error: {e}")

section_divider()

# Fees
st.subheader("🧾 Fees")
c1, c2, c3 = st.columns(3)
with c1:
    start_f = st.date_input("Start date (fees)", value=dt.date(2025, 8, 1), key="startf")
with c2:
    end_f   = st.date_input("End date (fees)", value=dt.date(2025, 8, 31), key="endf")
with c3:
    top_k_f = st.number_input("Top K", min_value=100, max_value=20000, value=8000, step=100, key="topkf")

if st.button("Run Fees"):
    try:
        body = {
            "file":   file_in,
            "start":  date_to_str(start_f),
            "end":    date_to_str(end_f),
            "top_k":  top_k_f
        }
        data = post_json(f"{base_url}/fees", auth_headers(token), body)
        st.success("Fees fetched.")

        totals = data.get("totals", {})
        by_col = totals.get("by_column", {})
        m1, m2, m3 = st.columns(3)
        m1.metric("Txn Fees (sum)",
                  f"{totals.get('by_column_sum', 0):,.2f}")
        m2.metric("Borrow Fees",
                  f"{totals.get('borrow_fees', 0):,.2f}")
        m3.metric("Overnight Fees",
                  f"{totals.get('overnight_fees', 0):,.2f}")

        st.metric("Grand Total (all-in)", f"{totals.get('grand_total_all_in', 0):,.2f}")

        st.caption("By Column")
        st.json(by_col)

        by_symbol = data.get("by_symbol", {})
        if by_symbol:
            # convert dict to list of rows
            rows = [{"symbol": k, **v} for k, v in by_symbol.items()]
            st.caption("By Symbol")
            st.dataframe(rows, use_container_width=True)

        with st.expander("Raw JSON"):
            st.code(json.dumps(data, indent=2))
    except Exception as e:
        st.error(f"Fees error: {e}")

section_divider()
st.subheader("🔎 Vector Search (any namespace)")

qcol1, qcol2 = st.columns([3,1])
with qcol1:
    query_text = st.text_input("Query", value="*")
with qcol2:
    st.caption("Use * to list rows")

if st.button("Run Search"):
    try:
        body = {
            "query": query_text,
            "top_k": int(top_k_search),
            "namespace": namespace
        }
        data = post_json(f"{base_url}/search", auth_headers(token), body)
        st.success("Search complete.")
        hits = data.get("hits", [])
        st.write(f"Results: {len(hits)}")
        # Show the metadata for inspection
        rows = [h.get("metadata", {}) for h in hits]
        if rows:
            st.dataframe(rows, use_container_width=True)
        with st.expander("Raw JSON"):
            st.code(json.dumps(data, indent=2))
    except Exception as e:
        st.error(f"Search error: {e}")
