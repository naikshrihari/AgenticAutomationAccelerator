"""ServiceNow Virtual Agent connector.

ServiceNow's conversational APIs identify a conversation by a request id echoed
on each turn. Like the Oracle connector this reuses the generic REST send but
lets the target YAML supply ServiceNow-specific headers (instance tokens) and the
response mapping needed to pull the reply text out of the messages array.
"""

from __future__ import annotations

import time

from ...models import AgentResponse
from .base import BaseConnector


class ServiceNowConnector(BaseConnector):
    async def send(self, question: str) -> AgentResponse:
        body = self._build_body(question)
        started = time.perf_counter()
        payload = await self._request(
            self.config.method, self.config.endpoint, self._auth_headers(), body
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return self._parse("", payload, latency_ms)
