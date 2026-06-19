"""Research Agent — Streamlit UI.

Ask a question; a LangGraph pipeline plans searches, reads the web, writes a cited
answer, and self-critiques it. Run: `streamlit run app.py`
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from agent.config import GEMINI_CHAT_MODEL, OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL
from agent.graph import build_graph
from agent.llm import GeminiLLM, OllamaLLM
from agent.tracing import tracing_enabled

load_dotenv()

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")
st.title("🔎 Research Agent")
st.caption("Multi-agent (LangGraph): plan → web search → write → self-critique → cited answer")


def _friendly(exc, fallback):
    if "11434" in str(exc) or "Connection refused" in str(exc):
        return ("Couldn't reach Ollama at localhost:11434. On the hosted demo, choose **Gemini** and "
                "paste a free key — Ollama only runs when you host the app locally.")
    return f"{fallback}: {exc}"


def make_llm():
    if provider.startswith("Gemini"):
        if not api_key:
            raise ValueError("Enter your Gemini API key (free at aistudio.google.com/apikey).")
        from google import genai
        return GeminiLLM(genai.Client(api_key=api_key))
    return OllamaLLM(model=gen_model, base_url=base_url)


with st.sidebar:
    st.header("Settings")
    provider = st.radio("Model provider", ["Gemini (cloud)", "Ollama (local)"])
    if provider.startswith("Gemini"):
        api_key = st.text_input("Gemini API key", type="password",
                                value=os.getenv("GEMINI_API_KEY", ""),
                                help="Free key: aistudio.google.com/apikey. Never stored or logged.")
        st.caption(f"Model: `{GEMINI_CHAT_MODEL}`")
    else:
        base_url = st.text_input("Ollama base URL", OLLAMA_BASE_URL)
        gen_model = st.text_input("Chat model", OLLAMA_CHAT_MODEL)
    st.caption("🔎 Web search: DuckDuckGo (free, no key).")
    st.caption("🔭 LangSmith tracing: " + ("on ✅" if tracing_enabled() else "off (set LANGSMITH_* in .env)"))

question = st.text_input("Ask a research question",
                         placeholder="e.g. What is retrieval-augmented generation and why is it used?")

if st.button("Research", type="primary") and question:
    try:
        llm = make_llm()
        graph = build_graph(llm)
        with st.spinner("Planning → searching → writing → self-critiquing…"):
            final = graph.invoke({"question": question, "revisions": 0})

        st.markdown(final.get("answer") or "_No answer produced._")

        sources = final.get("sources", [])
        with st.expander(f"Sources ({len(sources)})"):
            for i, s in enumerate(sources, 1):
                st.markdown(f"**[{i}] [{s['title'] or s['url']}]({s['url']})**")
                st.caption(s["text"][:300] + ("…" if len(s["text"]) > 300 else ""))

        with st.expander("How it ran (agent trace)"):
            st.write("**Search queries:**", final.get("queries", []))
            st.write("**Sources gathered:**", len(sources))
            st.write("**Revisions:**", final.get("revisions", 0))
            issues = final.get("issues", [])
            st.write("**Critic notes (unresolved):**", issues) if issues else st.write("**Critic:** approved ✅")
            cost = sum(c["cost_usd"] for c in llm.calls)
            st.caption(f"⚙️ {len(llm.calls)} LLM calls · {llm.provider} · ${cost:.5f}")
    except Exception as exc:
        st.error(_friendly(exc, "Error"))
