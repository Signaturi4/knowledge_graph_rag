"""Inspect a semantic key: current head, version history, lineage; revert."""
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Memory", layout="wide")
st.title("Memory inspector")

key = st.text_input("semantic_key", placeholder="sector_4::zoning_limit")
if not key:
    st.stop()

head = requests.get(f"{BACKEND}/memory/head/{key}", timeout=10)
if not head.ok:
    st.warning(head.json().get("detail", head.text))
    st.stop()

h = head.json()
st.subheader("Active head")
st.json(h)

vers = requests.get(f"{BACKEND}/memory/versions/{key}", timeout=10).json()
st.subheader(f"Versions ({len(vers)})")
st.dataframe(vers, use_container_width=True)

st.subheader("Revert")
target = st.selectbox("target record_id",
                      [v["record_id"] for v in vers if v["state"] in ("ARCHIVED", "DELETED", "EVICTED")])
actor = st.radio("actor", ["SYSTEM_2", "SYSTEM_1"], horizontal=True)
if st.button("Revert head to this version", type="primary", disabled=not target):
    r = requests.post(f"{BACKEND}/memory/revert",
                      json={"semantic_key": key, "target_record_id": target, "actor": actor}, timeout=30)
    st.success(r.json()) if r.ok else st.error(r.json().get("detail", r.text))
