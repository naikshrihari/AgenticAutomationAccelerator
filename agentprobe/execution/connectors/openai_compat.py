"""OpenAI-compatible connector.

Targets that speak the OpenAI Chat Completions protocol (including agents fronted
by an OpenAI-compatible gateway, or another Ollama instance) are reached here.
The request is a standard ``/chat/completions`` call and the answer is read from
``choices[0].message.content``.
"""

from __future__ import annotations

import time

from ...models import AgentResponse
from .base import BaseConnector


class OpenAICompatConnector(BaseConnector):
    async def send(self, question: str) -> AgentResponse:
        body = {
            "model": self.config.request_template.get("model", "default"),
            "messages": [{"role": "user", "content": question}],
            "temperature": self.config.request_template.get("temperature", 0.0),
        }
        endpoint = self.config.endpoint or "/v1/chat/completions"
        started = time.perf_counter()
        payload = await self._request("POST", endpoint, self._auth_headers(), body)
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            answer = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            answer = ""
        return AgentResponse(case_id="", answer=str(answer), latency_ms=latency_ms, ok=True, raw=payload)
