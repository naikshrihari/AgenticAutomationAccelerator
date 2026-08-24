"""Oracle Fusion AI Agent Studio connector.

Oracle Fusion agents are session-oriented: you open a conversation, exchange
turns carrying the conversation id, then close it. This connector specialises the
generic REST flow to thread the session id from ``start_session`` into every
send, reading the answer and any grounding citations via the target's response
mapping.
"""

from __future__ import annotations

import time

from ...models import AgentResponse
from .base import BaseConnector


class OracleFusionConnector(BaseConnector):
    async def send(self, question: str) -> AgentResponse:
        # request_template carries {question} and {session_id}; the session id
        # was captured by BaseConnector.start_session via response.session_id_path.
        body = self._build_body(question)
        started = time.perf_counter()
        payload = await self._request(
            self.config.method, self.config.endpoint, self._auth_headers(), body
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return self._parse("", payload, latency_ms)
