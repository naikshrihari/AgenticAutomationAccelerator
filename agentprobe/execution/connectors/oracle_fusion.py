"""Oracle Fusion AI Agent Studio connector.

Oracle Fusion AI Agent Studio exposes agent teams through an **asynchronous**
REST API, so this connector implements the real two-step contract rather than a
single request/response:

1. POST ``/api/fusion-ai/orchestrator/agent/v2/{agentCode}/invokeAsync`` with the
   user message; the call returns a ``jobId`` (and a ``conversationId``) instead
   of an answer.
2. Poll GET ``/api/fusion-ai/orchestrator/agent/v2/{agentCode}/status/{jobId}``
   until ``status`` reaches a terminal value (``COMPLETE``), then read the
   ``answer`` field.

The ``conversationId`` returned by the first turn is threaded into later turns so
multi-turn sessions work. To use it you only need three things in the target
YAML: the Fusion ``base_url``, the agent's code/name (``options.agent_code``),
and credentials (Basic username/password, or a bearer token).

Contract confirmed against Oracle's documentation and Oracle A-Team / Fusion CoE
integration guides for ``invokeAsync``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import httpx

from ...models import AgentResponse
from .base import BaseConnector, _dig

# Default Oracle Fusion AI Agent Studio API paths. {agent_code} and {job_id} are
# filled per request. Override via config.options if your instance differs.
_DEFAULT_INVOKE_PATH = "/api/fusion-ai/orchestrator/agent/v2/{agent_code}/invokeAsync"
_DEFAULT_STATUS_PATH = "/api/fusion-ai/orchestrator/agent/v2/{agent_code}/status/{job_id}"

# Status values that mean the job is finished (success or failure).
_TERMINAL_OK = {"COMPLETE", "COMPLETED", "SUCCESS", "SUCCEEDED"}
_TERMINAL_FAIL = {"FAILED", "ERROR", "CANCELLED", "CANCELED"}


class OracleFusionConnector(BaseConnector):
    """Async invoke + poll connector for Oracle Fusion AI Agent Studio."""

    def __init__(self, config, client=None):
        super().__init__(config, client)
        opts: dict[str, Any] = config.options or {}
        # The agent team code / name — the only Oracle-specific value you must set.
        self.agent_code: str = str(opts.get("agent_code") or config.name)
        self.invoke_path: str = opts.get("invoke_path", _DEFAULT_INVOKE_PATH)
        self.status_path: str = opts.get("status_path", _DEFAULT_STATUS_PATH)
        self.invocation_mode: str = opts.get("invocation_mode", "END_USER")
        self.conversational: bool = bool(opts.get("conversational", True))
        self.extra_parameters: dict[str, Any] = opts.get("parameters", {}) or {}
        # Polling controls: how often to check the status endpoint, and for how long.
        self.poll_interval_s: float = float(opts.get("poll_interval_s", 2.0))
        self.poll_max_attempts: int = int(opts.get("poll_max_attempts", 60))
        # Where the answer lives in the status payload (kept configurable).
        self.answer_path: str = config.response.answer_path or "answer"
        self.citations_path: Optional[str] = config.response.citations_path
        # Conversation continuity across turns.
        self.conversation_id: Optional[str] = None

    async def start_session(self) -> None:
        # Oracle creates the conversation implicitly on the first invoke (with a
        # null conversationId), so there is no separate session-open call.
        self.conversation_id = None

    async def close_session(self) -> None:
        # No explicit close endpoint in the invokeAsync contract.
        return

    async def send(self, question: str) -> AgentResponse:
        started = time.perf_counter()
        job_id = await self._invoke(question)
        payload = await self._poll(job_id)
        latency_ms = (time.perf_counter() - started) * 1000

        # Remember the conversation id so the next question continues the session.
        conv = _dig(payload, "conversationId")
        if isinstance(conv, str) and conv:
            self.conversation_id = conv

        answer = _dig(payload, self.answer_path)
        citations = _dig(payload, self.citations_path) if self.citations_path else None
        if isinstance(citations, str):
            citations = [citations]
        return AgentResponse(
            case_id="",
            answer="" if answer is None else str(answer),
            cited_sources=list(citations) if isinstance(citations, list) else [],
            latency_ms=latency_ms,
            ok=True,
            raw=payload if isinstance(payload, dict) else {"value": payload},
        )

    # -- internals -------------------------------------------------------- #
    async def _invoke(self, question: str) -> str:
        """POST invokeAsync and return the job id."""
        body: dict[str, Any] = {
            "message": question,
            "conversational": self.conversational,
            "conversationId": self.conversation_id,  # null on the first turn
            "invocationMode": self.invocation_mode,
            "parameters": self.extra_parameters,
        }
        path = self.invoke_path.format(agent_code=self.agent_code)
        payload = await self._request("POST", path, self._auth_headers(), body)
        job_id = _dig(payload, "jobId") or _dig(payload, "id")
        if not job_id:
            raise httpx.RemoteProtocolError(
                f"invokeAsync did not return a jobId (got keys: {list(payload)[:8]})"
            )
        return str(job_id)

    async def _poll(self, job_id: str) -> dict[str, Any]:
        """Poll the status endpoint until the job reaches a terminal state."""
        assert self._client is not None
        path = self.status_path.format(agent_code=self.agent_code, job_id=job_id)
        url = self.config.base_url.rstrip("/") + "/" + path.lstrip("/")
        for _ in range(self.poll_max_attempts):
            resp = await self._client.get(url, headers=self._auth_headers())
            resp.raise_for_status()
            payload = resp.json()
            status = str(_dig(payload, "status") or "").upper()
            if status in _TERMINAL_OK:
                return payload
            if status in _TERMINAL_FAIL:
                # Agent-level failure: return the payload so grading records a
                # failed answer (this is not an infrastructure error).
                return payload
            await asyncio.sleep(self.poll_interval_s)
        # Ran out of polling attempts — treat as a timeout so the runner
        # classifies it as an infrastructure error, not a wrong answer.
        raise httpx.TimeoutException(
            f"Oracle Fusion job {job_id} did not complete after "
            f"{self.poll_max_attempts} polls"
        )
