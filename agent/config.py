"""Pinned models, pricing, and agent bounds — single source of truth.

The bounds exist to cap how many LLM calls and web fetches a single run can make,
so cost and latency stay predictable (Rule 3).
"""

# --- Models (pinned, GA — verified against ai.google.dev, June 2026) ---
GEMINI_CHAT_MODEL = "gemini-2.5-flash"   # cheapest capable model
OLLAMA_CHAT_MODEL = "qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Pricing, USD per 1M tokens (paid Standard tier) ---
PRICING = {"gemini-2.5-flash": {"input": 0.30, "output": 2.50}}

# --- Agent bounds (cap cost/latency) ---
MAX_QUERIES = 3          # search queries the planner may emit
RESULTS_PER_QUERY = 3    # web results per query
MAX_SOURCES = 6          # total sources kept for writing
SOURCE_CHARS = 4000      # chars of extracted text kept per source (caps prompt tokens)
MAX_REVISIONS = 1        # self-critique → revise loops
