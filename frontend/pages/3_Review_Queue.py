"""SYSTEM_2 human-in-the-loop review queue."""
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Review Queue", layout="wide")
st.title("SYSTEM_2 review queue")

items = requests.get(f"{BACKEND}/review/queue", timeout=10).json()
if not items:
    st.success("queue is empty")
    st.stop()

for it in items:
    with st.container(border=True):
        st.markdown(f"**{it['kind']}** · `{it['semantic_key']}`")
        st.json(it["payload"], expanded=False)
        c1, c2, c3 = st.columns(3)
        if c1.button("Approve", key=f"a{it['id']}", type="primary"):
            r = requests.post(f"{BACKEND}/review/{it['id']}", json={"verdict": "approve"}, timeout=30)
            st.toast(str(r.json()))
            st.rerun()
        if c2.button("Reject", key=f"r{it['id']}"):
            requests.post(f"{BACKEND}/review/{it['id']}", json={"verdict": "reject"}, timeout=30)
            st.rerun()
        if c3.button("Defer", key=f"d{it['id']}"):
            requests.post(f"{BACKEND}/review/{it['id']}", json={"verdict": "defer"}, timeout=30)
            st.rerun()
