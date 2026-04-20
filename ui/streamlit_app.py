"""Streamlit dashboard.

    streamlit run ui/streamlit_app.py

Calls the FastAPI service (RAG_API_URL, default http://localhost:8000). Shows the
answer, verified citations, confidence breakdown, retrieved chunks, and a
dense-vs-hybrid toggle.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Support Knowledge Copilot", layout="wide")
st.title("📚 Support Knowledge Copilot")
st.caption("Hybrid RAG with verified citations")

with st.sidebar:
    st.header("Settings")
    hybrid = st.toggle("Hybrid retrieval (dense + BM25)", value=True)
    top_k = st.slider("Top-k chunks", 1, 10, 5)
    if st.button("Re-ingest corpus"):
        r = requests.post(f"{API}/ingest", json={"rebuild": True})
        st.success(f"Indexed {r.json().get('indexed_chunks')} chunks")
    try:
        h = requests.get(f"{API}/health", timeout=3).json()
        st.info(f"LLM: {h['llm_provider']} · Embeddings: {h['embedding_provider']}")
    except Exception:
        st.error(f"API not reachable at {API}")

question = st.text_input("Ask a question", "How do I reset my password?")

if st.button("Ask", type="primary") and question:
    resp = requests.post(f"{API}/ask",
                         json={"question": question, "hybrid": hybrid, "top_k": top_k})
    if resp.status_code != 200:
        st.error(resp.text)
    else:
        a = resp.json()
        if a["refused"]:
            st.warning(a["answer"])
        else:
            st.subheader("Answer")
            st.write(a["answer"])

        c = a["confidence"]
        cols = st.columns(4)
        cols[0].metric("Confidence", c["overall"])
        cols[1].metric("Max dense", c["max_dense"])
        cols[2].metric("Max BM25", c["max_bm25"])
        cols[3].metric("Citation support", c["citation_support_rate"])

        if a["citations"]:
            st.subheader("✅ Verified citations")
            for cit in a["citations"]:
                st.write(f"- `{cit['chunk_id']}` ({cit['source']}) — "
                         f"support {cit['support_score']}")
        if a["unverified"]:
            st.subheader("⚠️ Could not verify")
            for u in a["unverified"]:
                st.write(f"- {u}")

        with st.expander("Retrieved chunks"):
            for h in a["retrieved"]:
                st.markdown(f"**{h['chunk_id']}** · score={round(h['score'],4)} "
                            f"· dense_rank={h['dense_rank']} · sparse_rank={h['sparse_rank']}")
                st.caption(h["text"][:300])
