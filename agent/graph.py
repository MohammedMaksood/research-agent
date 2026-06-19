"""LangGraph multi-agent pipeline: plan → research → write → critique → (revise) → end.

Each node is a single responsibility; the critic can send the draft back to the
reviser once (bounded by MAX_REVISIONS) before the run ends. Search/fetch are
free; only plan/write/critique/revise call the LLM (3–5 calls per run, capped).
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from .config import MAX_QUERIES, MAX_SOURCES, MAX_REVISIONS
from .prompts import PLANNER_SYS, WRITER_SYS, CRITIC_SYS, REVISER_SYS
from .search import web_search, fetch_text


class Plan(BaseModel):
    queries: list[str] = Field(default_factory=list)


class Draft(BaseModel):
    answer: str
    citations: list[int] = Field(default_factory=list)


class Critique(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)


class State(TypedDict, total=False):
    question: str
    queries: list[str]
    sources: list[dict]   # {title, url, text}
    answer: str
    citations: list[int]
    issues: list[str]
    revisions: int


def _sources_block(sources: list[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] {s['title']} ({s['url']})\n{s['text']}" for i, s in enumerate(sources))


def build_graph(llm):
    """Compile the agent graph around a given LLM backend (GeminiLLM / OllamaLLM)."""

    def plan(state: State) -> dict:
        result = llm.structured("plan", PLANNER_SYS, state["question"], Plan)
        return {"queries": result.queries[:MAX_QUERIES] or [state["question"]]}

    def research(state: State) -> dict:
        sources, seen = [], set()
        for query in state["queries"]:
            for hit in web_search(query):
                if hit["url"] in seen:
                    continue
                seen.add(hit["url"])
                text = fetch_text(hit["url"])
                if text:
                    sources.append({"title": hit["title"], "url": hit["url"], "text": text})
                if len(sources) >= MAX_SOURCES:
                    break
            if len(sources) >= MAX_SOURCES:
                break
        return {"sources": sources}

    def write(state: State) -> dict:
        user = f"Question: {state['question']}\n\nSources:\n{_sources_block(state['sources'])}"
        draft = llm.structured("write", WRITER_SYS, user, Draft)
        return {"answer": draft.answer, "citations": draft.citations}

    def critique(state: State) -> dict:
        user = (f"Question: {state['question']}\n\nDraft:\n{state['answer']}\n\n"
                f"Sources:\n{_sources_block(state['sources'])}")
        result = llm.structured("critique", CRITIC_SYS, user, Critique)
        return {"issues": [] if result.approved else result.issues}

    def revise(state: State) -> dict:
        issues = "\n- ".join(state["issues"])
        user = (f"Question: {state['question']}\n\nDraft:\n{state['answer']}\n\n"
                f"Issues to fix:\n- {issues}\n\nSources:\n{_sources_block(state['sources'])}")
        draft = llm.structured("revise", REVISER_SYS, user, Draft)
        return {"answer": draft.answer, "citations": draft.citations,
                "revisions": state.get("revisions", 0) + 1}

    def route_after_critique(state: State) -> str:
        if state.get("issues") and state.get("revisions", 0) < MAX_REVISIONS:
            return "revise"
        return END

    g = StateGraph(State)
    for name, fn in [("plan", plan), ("research", research), ("write", write),
                     ("critique", critique), ("revise", revise)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "research")
    g.add_edge("research", "write")
    g.add_edge("write", "critique")
    g.add_conditional_edges("critique", route_after_critique, {"revise": "revise", END: END})
    g.add_edge("revise", "critique")
    return g.compile()


def make_graph():
    """Factory used by LangGraph CLI / Studio (see langgraph.json).

    Picks Gemini if GEMINI_API_KEY is set, otherwise the free local Ollama backend.
    """
    import os
    if os.getenv("GEMINI_API_KEY"):
        from google import genai
        from .llm import GeminiLLM
        return build_graph(GeminiLLM(genai.Client(api_key=os.environ["GEMINI_API_KEY"])))
    from .llm import OllamaLLM
    return build_graph(OllamaLLM())
