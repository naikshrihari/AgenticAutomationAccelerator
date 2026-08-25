"""Oracle Fusion AI Agent Studio connector.

Oracle Fusion AI Agent Studio agents are invoked with a **bearer token**, not
Basic auth, and the flow is asynchronous. This connector implements the real
three-step contract:

1. Obtain a bearer token. Two supported ways (see ``options.auth_mode``):
     * ``token``  — a bearer token supplied directly (paste it; short-lived).
     * ``relay``  — call ``/fscmRestApi/tokenrelay`` with the browser session
                    cookie + xsrf token to mint a fresh ``access_token``. The
                    token is refreshed automatically if a call returns 401.
   (An OAuth2 client-credentials mode can layer on top of the same design.)
2. POST ``/api/fusion-ai/orchestrator/agent/v2/{agentCode}/invokeAsync`` with
   the user message and ``Authorization: Bearer <token>`` → returns a ``jobId``.
3. Poll GET ``…/status/{jobId}?invocationMode=ADMIN`` until ``status`` is
   terminal, then read the ``answer`` field.

Contract confirmed against a live Fusion environment's browser dev-tools capture
(tokenrelay → invokeAsync → status).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import httpx

from ...models import AgentResponse
from .base import BaseConnector, _dig

_DEFAULT_INVOKE_PATH = "/api/fusion-ai/orchestrator/agent/v2/{agent_code}/invokeAsync"
_DEFAULT_STATUS_PATH = "/api/fusion-ai/orchestrator/agent/v2/{agent_code}/status/{job_id}"
_DEFAULT_RELAY_PATH = "/fscmRestApi/tokenrelay"

_TERMINAL_OK = {"COMPLETE", "COMPLETED", "SUCCESS", "SUCCEEDED"}
_TERMINAL_FAIL = {"FAILED", "ERROR", "CANCELLED", "CANCELED"}


class OracleFusionConnector(BaseConnector):
    """Async invoke + poll connector with bearer-token / token-relay auth."""

    def __init__(self, config, client=None):
        super().__init__(config, client)
        opts: dict[str, Any] = config.options or {}
        self.agent_code: str = str(opts.get("agent_code") or config.name)
        self.invoke_path: str = opts.get("invoke_path", _DEFAULT_INVOKE_PATH)
        self.status_path: str = opts.get("status_path", _DEFAULT_STATUS_PATH)
        self.invocation_mode: str = opts.get("invocation_mode", "END_USER")
        self.status_invocation_mode: str = opts.get("status_invocation_mode", "ADMIN")
        self.conversational: bool = bool(opts.get("conversational", True))
        self.extra_parameters: dict[str, Any] = opts.get("parameters", {}) or {}
        self.poll_interval_s: float = float(opts.get("poll_interval_s", 2.0))
        self.poll_max_attempts: int = int(opts.get("poll_max_attempts", 60))
        self.answer_path: str = config.response.answer_path or "answer"
        self.citations_path: Optional[str] = config.response.citations_path

        # Auth: "token" (direct bearer) or "relay" (mint via tokenrelay).
        self.auth_mode: str = opts.get("auth_mode", "token")
        self.relay_path: str = opts.get("relay_path", _DEFAULT_RELAY_PATH)
        self.relay_cookie: str = opts.get("relay_cookie", "") or ""
        self.relay_xsrf: str = opts.get("relay_xsrf", "") or ""

        # The active bearer token (from config.auth.token, or fetched via relay).
        self._bearer: Optional[str] = config.auth.token
        self.conversation_id: Optional[str] = None

    # -- auth ------------------------------------------------------------- #
    def _auth_headers(self) -> dict[str, str]:
        """Send the bearer token if we have one; else defer to the base logic."""
        headers = dict(self.config.extra_headers)
        if self._bearer:
            headers["Authorization"] = f"Bearer {self._bearer}"
            return headers
        return super()._auth_headers()

    async def _fetch_relay_token(self) -> None:
        """Mint a fresh bearer token from the tokenrelay endpoint."""
        assert self._client is not None
        url = self.config.base_url.rstrip("/") + "/" + self.relay_path.lstrip("/")
        headers = {
            "Accept": "*/*",
            "Referer": self.config.base_url.rstrip("/") + "/",
        }
        if self.relay_cookie:
            headers["Cookie"] = self.relay_cookie
        if self.relay_xsrf:
            headers["x-xsrf-token"] = self.relay_xsrf
        resp = await self._client.get(url, headers=headers)
        resp.raise_for_status()
        try:
            data = resp.json()
            token = data.get("access_token") or data.get("token")
        except ValueError:
            token = resp.text.strip()
        if not token:
            raise httpx.RemoteProtocolError("tokenrelay did not return an access_token")
        self._bearer = token

    async def _ensure_token(self) -> None:
        if self.auth_mode == "relay" and not self._bearer:
            await self._fetch_relay_token()

    # -- lifecycle -------------------------------------------------------- #
    async def start_session(self) -> None:
        self.conversation_id = None
        await self._ensure_token()

    async def close_session(self) -> None:
        return

    async def send(self, question: str) -> AgentResponse:
        started = time.perf_counter()
        try:
            job_id = await self._invoke(question)
            payload = await self._poll(job_id)
        except httpx.HTTPStatusError as exc:
            # A 401 mid-run usually means the token expired — refresh once (relay).
            if exc.response.status_code == 401 and self.auth_mode == "relay":
                await self._fetch_relay_token()
                job_id = await self._invoke(question)
                payload = await self._poll(job_id)
            else:
                raise
        latency_ms = (time.perf_counter() - started) * 1000

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
        body: dict[str, Any] = {
            "message": question,
            "conversational": self.conversational,
            "conversationId": self.conversation_id,
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
        assert self._client is not None
        path = self.status_path.format(agent_code=self.agent_code, job_id=job_id)
        url = self.config.base_url.rstrip("/") + "/" + path.lstrip("/")
        params = {"invocationMode": self.status_invocation_mode} if self.status_invocation_mode else None
        for _ in range(self.poll_max_attempts):
            resp = await self._client.get(url, headers=self._auth_headers(), params=params)
            resp.raise_for_status()
            payload = resp.json()
            status = str(_dig(payload, "status") or "").upper()
            if status in _TERMINAL_OK or status in _TERMINAL_FAIL:
                return payload
            await asyncio.sleep(self.poll_interval_s)
        raise httpx.TimeoutException(
            f"Oracle Fusion job {job_id} did not complete after {self.poll_max_attempts} polls"
        )
