"""LLM backends with schema-validated structured output + per-call logging.

Both expose the same `.structured(node, system, user, schema)` returning a validated
Pydantic object, so the graph nodes are backend-agnostic. Every call is logged and
appended to `.calls` so a run's total cost and call count can be shown.
"""
from __future__ import annotations

import re
import time

import requests
from pydantic import BaseModel, ValidationError

from .config import GEMINI_CHAT_MODEL, OLLAMA_CHAT_MODEL, OLLAMA_BASE_URL
from .obs import log_call
from .prompts import PROMPT_VERSION
from .retry import retry_transient
from .tracing import traceable


def _validate(text: str, schema: type[BaseModel]) -> BaseModel:
    try:
        return schema.model_validate_json(text)
    except ValidationError:
        match = re.search(r"\{.*\}", text, re.DOTALL)  # salvage JSON wrapped in prose
        if match:
            return schema.model_validate_json(match.group(0))
        raise ValueError("Model did not return valid JSON.")


class GeminiLLM:
    provider = "gemini"

    def __init__(self, client, model: str = GEMINI_CHAT_MODEL):
        from google.genai import types
        self._client = client
        self._types = types
        self.model = model
        self.calls: list[dict] = []

    @traceable(run_type="llm", name="gemini.structured")
    def structured(self, node: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        cfg = self._types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json",
            response_schema=schema, temperature=0.2,
        )
        t0 = time.time()
        resp = retry_transient(
            lambda: self._client.models.generate_content(model=self.model, contents=user, config=cfg)
        )
        um = getattr(resp, "usage_metadata", None)
        self.calls.append(log_call(
            node=node, provider="gemini", model=self.model, prompt_version=PROMPT_VERSION,
            in_tok=getattr(um, "prompt_token_count", 0) or 0,
            out_tok=getattr(um, "candidates_token_count", 0) or 0,
            latency_ms=int((time.time() - t0) * 1000),
        ))
        parsed = getattr(resp, "parsed", None)
        return parsed if isinstance(parsed, schema) else _validate(resp.text, schema)


class OllamaLLM:
    provider = "ollama"

    def __init__(self, model: str = OLLAMA_CHAT_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.calls: list[dict] = []

    @traceable(run_type="llm", name="ollama.structured")
    def structured(self, node: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        payload = {
            "model": self.model, "system": system, "prompt": user, "stream": False,
            "think": False, "format": schema.model_json_schema(), "options": {"temperature": 0.2},
        }
        t0 = time.time()
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        self.calls.append(log_call(
            node=node, provider="ollama", model=self.model, prompt_version=PROMPT_VERSION,
            in_tok=data.get("prompt_eval_count", 0), out_tok=data.get("eval_count", 0),
            latency_ms=int((time.time() - t0) * 1000),
        ))
        return _validate(data["response"], schema)
