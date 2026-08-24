"""Generic REST connector.

The baseline connector for any agent exposed over HTTP: it posts the target's
request template to a single endpoint and parses the reply with the configured
response mapping. Oracle Fusion and ServiceNow specialise this only where their
session or payload conventions differ.
"""

from __future__ import annotations

import time

from ...models import AgentResponse
from .base import BaseConnector


class GenericRESTConnector(BaseConnector):
    async def send(self, question: str) -> AgentResponse:
        body = self._build_body(question)
        started = time.perf_counter()
        payload = await self._request(
            self.config.method, self.config.endpoint, self._auth_headers(), body
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return self._parse("", payload, latency_ms)
