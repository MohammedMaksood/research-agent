"""Optional LangSmith tracing/observability.

Tracing is OFF unless `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set in the
environment (e.g. via .env). When off, `@traceable` is a no-op and nothing is sent,
so the app runs identically with or without a LangSmith account.

With it on, each decorated call (and LangGraph's nodes) appears as a span in the
LangSmith UI with inputs, outputs, latency, and token usage.
"""
from __future__ import annotations

import os

try:
    from langsmith import traceable  # type: ignore
except Exception:  # langsmith not installed -> graceful no-op decorator
    def traceable(*args, **kwargs):
        if args and callable(args[0]):       # used as @traceable
            return args[0]
        def deco(fn):                         # used as @traceable(...)
            return fn
        return deco


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))
