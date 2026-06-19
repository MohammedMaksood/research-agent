"""Web research tools: DuckDuckGo search + readable-text extraction.

No LLM and no API key here — this is the cheap part of the pipeline. Extracted
text is truncated per source to keep downstream prompt tokens bounded (Rule 3).
"""
from __future__ import annotations

from ddgs import DDGS

from .config import RESULTS_PER_QUERY, SOURCE_CHARS


def web_search(query: str, max_results: int = RESULTS_PER_QUERY) -> list[dict]:
    """Return [{title, url, snippet}] for a query (no LLM)."""
    results = []
    for r in DDGS().text(query, max_results=max_results):
        url = r.get("href", "")
        if url:
            results.append({"title": r.get("title", ""), "url": url, "snippet": r.get("body", "")})
    return results


def fetch_text(url: str, max_chars: int = SOURCE_CHARS) -> str:
    """Fetch a page and extract its main text (boilerplate stripped), truncated."""
    import trafilatura
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    return (trafilatura.extract(downloaded) or "")[:max_chars]
