"""Debug console: headline KB numbers + full stage-by-stage logs (ingest, retrieval,
gate, revert, GC, HTTP) — everything the backend saw, in order."""
import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Logs & Stats", layout="wide")
st.title("Logs & Stats")

auto = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
if auto:
    st.sidebar.caption("Refreshing…")

# --- headline numbers -------------------------------------------------- #
try:
    s = requests.get(f"{BACKEND}/logs/stats", timeout=10).json()
except Exception as exc:
    st.error(f"backend unreachable at {BACKEND}: {exc}")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Nodes", s["nodes_total"])
c2.metric("Dependencies (edges)", s["dependencies"])
c3.metric("Documents ingested", s["documents_ingested"])
c4.metric("Semantic keys", s["semantic_keys"])
c5.metric("Pending SYSTEM_2", s["queue_pending"])

with st.expander("Full breakdown", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.write("**Nodes by state**")
    col1.json(s["nodes_by_state"])
    col2.write("**Edges by kind**")
    col2.json(s["edges_by_kind"])
    col3.write("**Records by tier**")
    col3.json(s["records_by_tier"])
    st.write(f"Total immutable records (incl. superseded/archived): **{s['records_total']}** · "
            f"Audit rows: **{s['audit_rows']}** · Queue by kind:")
    st.json(s["queue_by_kind"])

st.divider()

# --- log viewer ---------------------------------------------------------- #
st.subheader("Live log stream")
f1, f2, f3, f4 = st.columns([1, 1, 2, 1])
level = f1.selectbox("min level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
logger = f2.text_input("logger prefix", placeholder="dagkb.pipeline")
contains = f3.text_input("contains", placeholder="e.g. a semantic_key, txn id, or query text")
limit = f4.number_input("lines", min_value=50, max_value=5000, value=300, step=50)

params = {"limit": int(limit), "level": level}
if logger:
    params["logger"] = logger
if contains:
    params["contains"] = contains

logs = requests.get(f"{BACKEND}/logs", params=params, timeout=15).json()
st.caption(f"{logs['count']} lines shown (buffer holds the most recent 5000 across the process)")

text = "\n".join(f"{r['ts']}  {r['level']:<7}  {r['logger']:<30}  {r['message']}"
                for r in logs["lines"])
st.text_area("log", text, height=520, label_visibility="collapsed")

if auto:
    time.sleep(5)
    st.rerun()
