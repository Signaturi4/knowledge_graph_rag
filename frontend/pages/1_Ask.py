"""KG-only Q&A over the ACTIVE view. No vector search."""
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Ask", layout="wide")
st.title("Ask the knowledge graph")

col1, col2 = st.columns([3, 1])
with col1:
    q = st.text_input("question", placeholder="What supplement does PT-7714 take and what happened to their diagnosis?")
with col2:
    depth = st.slider("multi-hop depth", 1, 4, 2)

include_history = st.checkbox("Include historical / superseded facts (EHR timeline)", value=False)

with st.expander("PLANFENCE-gate this answer (optional)"):
    root = st.text_input("lineage_root record_id")
    deps = st.text_input("declared_deps (comma-separated semantic keys)")

if st.button("Ask", type="primary", disabled=not q.strip()):
    body = {"query": q, "depth": depth, "include_history": include_history}
    if root and deps:
        body["lineage_root"] = root
        body["declared_deps"] = [d.strip() for d in deps.split(",") if d.strip()]
    r = requests.post(f"{BACKEND}/qa", json=body, timeout=180)
    if not r.ok:
        st.error(r.text)
    else:
        data = r.json()
        if data.get("gate"):
            st.info(f"lineage gate: {data['gate']}")
        st.markdown(f"### {data.get('answer', '')}")
        st.caption("entities: " + ", ".join(data.get("entities", []) or ["—"]))
        with st.expander(f"Grounding Facts ({len(data.get('context', []))})", expanded=True):
            for line in data.get("context", []):
                if line.startswith("[ARCHIVED"):
                    st.markdown(f"⚠️ *`{line}`*")
                else:
                    st.markdown(f"✅ **{line}**")
