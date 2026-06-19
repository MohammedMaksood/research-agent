"""Offline graph-wiring test: stub LLM + stub search, so it costs nothing and needs
no network (Rules 3 & 9)."""
import agent.graph as graph_mod
from agent.graph import build_graph, Plan, Draft, Critique


class StubLLM:
    provider = "stub"

    def __init__(self):
        self.calls = []

    def structured(self, node, system, user, schema):
        if schema is Plan:
            return Plan(queries=["q1", "q2"])
        if schema is Draft:
            return Draft(answer="RAG combines retrieval with generation [1].", citations=[1])
        if schema is Critique:
            return Critique(approved=True, issues=[])
        raise AssertionError(f"unexpected schema at node {node}")


def test_graph_runs_end_to_end(monkeypatch):
    monkeypatch.setattr(graph_mod, "web_search",
                        lambda q, **k: [{"title": "T", "url": f"http://x/{q}", "snippet": "s"}])
    monkeypatch.setattr(graph_mod, "fetch_text", lambda url, **k: "source text about RAG")

    final = build_graph(StubLLM()).invoke({"question": "what is RAG?", "revisions": 0})

    assert "RAG" in final["answer"]
    assert final["sources"]                 # research node collected sources
    assert final.get("issues", []) == []    # critic approved → no unresolved issues
