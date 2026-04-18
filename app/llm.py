"""LLM providers behind one interface.

Production: openai (gpt-4o-mini), anthropic (claude haiku), or ollama (local
llama). CI/offline: a deterministic `test` provider so the whole suite runs
without API keys — it produces grounded answers from the provided context and
valid JSON for the citation judge, so downstream parsing is exercised for real.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from .config import get_settings


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    def __init__(self):
        self.s = get_settings()

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> LLMResult:
        p = self.s.llm_provider
        if p == "openai":
            return self._openai(system, user, json_mode)
        if p == "anthropic":
            return self._anthropic(system, user)
        if p == "ollama":
            return self._ollama(system, user, json_mode)
        return self._test(system, user, json_mode)

    # --- production providers ---
    def _openai(self, system, user, json_mode):
        from openai import OpenAI
        client = OpenAI()
        kwargs = dict(model=self.s.llm_model,
                      messages=[{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                      temperature=self.s.llm_temperature,
                      max_tokens=self.s.llm_max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**kwargs)
        u = r.usage
        return LLMResult(r.choices[0].message.content, u.prompt_tokens, u.completion_tokens)

    def _anthropic(self, system, user):
        import anthropic
        client = anthropic.Anthropic()
        r = client.messages.create(model=self.s.llm_model, system=system,
                                    messages=[{"role": "user", "content": user}],
                                    max_tokens=self.s.llm_max_tokens,
                                    temperature=self.s.llm_temperature)
        return LLMResult(r.content[0].text, r.usage.input_tokens, r.usage.output_tokens)

    def _ollama(self, system, user, json_mode):
        import requests
        payload = {"model": self.s.llm_model, "stream": False,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "options": {"temperature": self.s.llm_temperature}}
        if json_mode:
            payload["format"] = "json"
        r = requests.post(f"{self.s.ollama_host}/api/chat", json=payload, timeout=120)
        return LLMResult(r.json()["message"]["content"])

    # --- deterministic test/CI provider ---
    def _test(self, system, user, json_mode):
        if json_mode:
            # citation judge expects {"supported": bool, "score": float}
            return LLMResult(json.dumps({"supported": True, "score": 0.9}))
        ctx = re.search(r"context:\s*\n(.+?)(?:\n\nquestion:|\Z)",
                        user, re.IGNORECASE | re.DOTALL)
        if ctx:
            block = ctx.group(1).strip()
            cite = re.search(r"\[([^\]]+)\]", block)
            tag = f" [{cite.group(1)}]" if cite else ""
            body = re.sub(r"\[[^\]]+\]\s*", "", block)
            body = re.sub(r"\([^)]*\)\s*", "", body, count=1)
            return LLMResult(f"{body[:300].strip()}{tag}")
        return LLMResult("I could not find this in the documentation.")


@lru_cache
def get_llm() -> LLMClient:
    return LLMClient()
