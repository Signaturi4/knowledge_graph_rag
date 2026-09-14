"""Deterministic DAG Knowledgebase — Streamlit console (home / ingestion).

Run:  streamlit run ui.py     (expects the backend at $BACKEND_URL, default :8000)
"""
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="DAG Knowledgebase", layout="wide")
st.title("Deterministic DAG Knowledgebase")

# --- health ---------------------------------------------------------------- #
try:
    h = requests.get(f"{BACKEND}/health", timeout=5).json()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("status", h.get("status", "?"))
    c2.metric("LLM provider", h.get("provider", "?"))
    c3.metric("semantic keys", h.get("semantic_keys", 0))
    c4.metric("SYSTEM_2 queue", h.get("pending_system_2", 0))
except Exception as exc:
    st.error(f"backend unreachable at {BACKEND}: {exc}")
    st.stop()

st.divider()
st.subheader("Ingest knowledge (Path A)")

tab_text, tab_dir = st.tabs(["Paste text", "PDF directory"])

with tab_text:
    src = st.text_input("source_id", value="municipal:res-402",
                        help="prefix drives the authority tier: statute/contract/municipal=T0, "
                             "policy/sop=T1, vendor_doc/arxiv=T2, derived=T3, conversation/email=T4")
    text = st.text_area("text", height=180,
                        placeholder="City Council Resolution 402 sets the Sector 4 zoning limit to 30 stories.")
    if st.button("Ingest text", type="primary", disabled=not text.strip()):
        r = requests.post(f"{BACKEND}/ingest/text", json={"text": text, "source_id": src}, timeout=180)
        if r.ok:
            st.success("done")
            st.json(r.json())
        else:
            st.error(r.text)

with tab_dir:
    d = st.text_input("directory (absolute path with *.pdf)")
    src2 = st.text_input("source_id ", value="arxiv")
    if st.button("Ingest directory", disabled=not d.strip()):
        r = requests.post(f"{BACKEND}/ingest/directory", json={"directory": d, "source_id": src2}, timeout=600)
        st.json(r.json() if r.ok else {"error": r.text})

st.caption("Use the pages in the sidebar: Ask · Memory · Review Queue · Visualization")
