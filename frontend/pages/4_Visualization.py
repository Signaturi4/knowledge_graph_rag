"""Visualize the live knowledge DAG — interactive, colored by lifecycle state, with patient subgraph selection."""
import os

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="DAG Subgraph Visualization", layout="wide")
st.title("Patient Subgraph & Knowledge DAG Explorer")
st.caption(
    "🟢 **ACTIVE** (ground truth) · 🟡 **TBD** (re-deriving) · 🔴 **FLAGGED** (contradiction) · "
    "⚪ **ARCHIVED** (superseded history) · 🟠 **PATIENT** (identity anchor) · 🔵 **ENTITY** (concept/drug/disease)"
)

# Fetch available patients from backend
@st.cache_data(ttl=5)
def get_patients():
    try:
        r = requests.get(f"{BACKEND}/visualization/patients", timeout=5)
        if r.ok:
            return r.json().get("patients", [])
    except Exception:
        pass
    return []

patients = get_patients()

# Filter Controls Bar
col_sel, col_hist, col_phys, col_ref = st.columns([3, 2, 2, 1])

with col_sel:
    patient_options = ["All Patients (Full Graph)"] + [
        f"{p['label']} ({p['id']})" for p in patients
    ]
    # If no patients detected yet from endpoint, provide predefined shortcuts
    if len(patient_options) == 1:
        patient_options += ["PT-1049 (pt_1049)", "PT-3392 (pt_3392)", "PT-7714 (pt_7714)"]

    selected_option = st.selectbox(
        "Select Subgraph View",
        options=patient_options,
        index=0,
        help="Select an individual patient to isolate their local clinical subgraph (diagnoses, treatments, symptoms, and supersessions).",
    )

with col_hist:
    st.write("")
    st.write("")
    include_history = st.checkbox("Show Superseded History (ARCHIVED)", value=True, help="Include prior superseded medication and diagnosis versions linked by dashed SUPERSEDES edges.")

with col_phys:
    st.write("")
    st.write("")
    enable_physics = st.checkbox("Physics Layout (ForceAtlas2)", value=True, help="Enable dynamic force-directed physics clustering.")

with col_ref:
    st.write("")
    st.write("")
    if st.button("Refresh", type="secondary"):
        st.cache_data.clear()
        st.rerun()

# Extract selected patient ID
selected_patient_id = None
if selected_option != "All Patients (Full Graph)":
    # Parse e.g. "PT-1049 (pt_1049)" -> "pt_1049"
    if "(" in selected_option and ")" in selected_option:
        selected_patient_id = selected_option.split("(")[-1].replace(")", "").strip()
    else:
        selected_patient_id = selected_option.strip()

# Patient Summary Banner if patient selected
if selected_patient_id:
    try:
        sum_resp = requests.get(f"{BACKEND}/visualization/patient/{selected_patient_id}/summary", timeout=5)
        if sum_resp.ok:
            summary = sum_resp.json()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Active Facts (Ground Truth)", summary.get("active_count", 0))
            m2.metric("Superseded Facts (Archived)", summary.get("archived_count", 0))
            m3.metric("Flagged Contradictions", summary.get("flagged_count", 0))
            m4.metric("Supersession Links", len(summary.get("supersessions", [])))

            # Expandable Clinical Details
            with st.expander("Clinical Trajectory & Fact Breakdown", expanded=False):
                t_active, t_hist, t_super = st.tabs(["Active Knowledge", "Archived History", "Supersession Chains"])

                with t_active:
                    if summary.get("active_facts"):
                        st.dataframe(
                            [
                                {
                                    "Predicate": f["predicate"],
                                    "Object": f["object"],
                                    "Mentions": f.get("mention_count", 1),
                                    "Source Documents": ", ".join(f.get("source_ids", [])) or f.get("source_id", "N/A"),
                                    "Document Dates": ", ".join(f.get("doc_dates", [])) or f.get("valid_from", "N/A"),
                                    "Authority Tier": f["tier"],
                                }
                                for f in summary["active_facts"]
                            ],
                            use_container_width=True,
                        )
                    else:
                        st.info("No active facts recorded.")

                with t_hist:
                    if summary.get("archived_facts"):
                        st.dataframe(
                            [
                                {
                                    "Predicate": f["predicate"],
                                    "Superseded Object": f["object"],
                                    "Mentions": f.get("mention_count", 1),
                                    "Source Documents": ", ".join(f.get("source_ids", [])) or f.get("source_id", "N/A"),
                                    "Document Dates": ", ".join(f.get("doc_dates", [])) or f.get("valid_from", "N/A"),
                                    "Authority Tier": f["tier"],
                                }
                                for f in summary["archived_facts"]
                            ],
                            use_container_width=True,
                        )
                    else:
                        st.info("No archived historical facts.")

                with t_super:
                    if summary.get("supersessions"):
                        for s in summary["supersessions"]:
                            st.markdown(
                                f"- 🟢 **Current**: `{s['new_record']['predicate']}` = **{s['new_record']['obj']}** "
                                f"*(ID: `{s['new_record']['id'][:10]}`)*  \n"
                                f"  ↳ ⬅️ **Supersedes**: `{s['superseded_record']['predicate']}` = *{s['superseded_record']['obj']}* "
                                f"*(ID: `{s['superseded_record']['id'][:10]}`)*"
                            )
                    else:
                        st.info("No supersession chains recorded.")
    except Exception as e:
        st.warning(f"Could not load patient summary: {e}")

st.divider()

# Graph Rendering
tab_graph, tab_gephi = st.tabs(["Interactive Subgraph", "Gephi-Lite (External)"])

with tab_graph:
    @st.cache_data(ttl=10)
    def _fetch_html(pid, hist, phys):
        params = {"include_history": hist, "physics": phys}
        if pid:
            params["patient_id"] = pid
        r = requests.get(f"{BACKEND}/visualization/graph.html", params=params, timeout=30)
        r.raise_for_status()
        return r.text

    try:
        html_data = _fetch_html(selected_patient_id, include_history, enable_physics)
        components.html(html_data, height=780, scrolling=True)
    except Exception as exc:
        st.error(f"Could not render subgraph from {BACKEND}: {exc}")
        st.info("Make sure the backend is running (`uvicorn service.main:app --port 8000`).")

with tab_gephi:
    st.write("Export `data/dag.graphml` and import it here for advanced topological analysis.")
    components.iframe("https://gephi.org/gephi-lite/", height=700, scrolling=True)
