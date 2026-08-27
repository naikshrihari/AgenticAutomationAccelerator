"""The connector interface that hides platform differences.

Every target platform — Oracle Fusion AI Agent Studio, ServiceNow, a generic
REST agent, an OpenAI-compatible endpoint — is reached through the same three
operations: start a session, send a question, close the session. The execution
engine only ever sees this interface, so adding a platform means adding a
connector, not touching the runner.

Response parsing (pulling the answer and citations out of the reply) and auth
are shared here and driven by the target YAML, so concrete connectors mostly
differ in how they shape the request and manage sessions.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx

from ...config import TargetConfig
from ...models import AgentResponse


class BaseConnector:
    """Shared HTTP plumbing for platform connectors."""

    def __init__(self, config: TargetConfig, client: Optional[httpx.AsyncClient] = None):
        self.config = config
        self._client = client
        self._owns_client = client is None
        self.session_id: Optional[str] = None

    # -- lifecycle -------------------------------------------------------- #
    async def __aenter__(self) -> "BaseConnector":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout_s)
        await self.start_session()
        return self

    async def __aexit__(self, *exc: object) -> None:
        try:
            await self.close_session()
        finally:
            if self._owns_client and self._client is not None:
                await self._client.aclose()

    # -- interface (override as needed) ----------------------------------- #
    async def start_session(self) -> None:
        """Open a session if the platform requires one. Default: no-op."""
        if not self.config.session_start_endpoint:
            return
        resp = await self._request("POST", self.config.session_start_endpoint, self._auth_headers(), {})
        self.session_id = _dig(resp, self.config.response.session_id_path) if self.config.response.session_id_path else None

    async def send(self, question: str) -> AgentResponse:
        """Send one question and return a normalised response."""
        raise NotImplementedError

    async def close_session(self) -> None:
        if not self.config.session_close_endpoint:
            return
        try:
            await self._request("POST", self.config.session_close_endpoint, self._auth_headers(), {})
        except Exception:  # noqa: BLE001 - closing is best-effort
            pass

    # -- helpers ---------------------------------------------------------- #
    def _build_body(self, question: str) -> dict[str, Any]:
        """Fill the target's request_template with the question/session id."""
        return _fill_template(
            self.config.request_template,
            {"question": question, "session_id": self.session_id or ""},
        )

    def _auth_headers(self) -> dict[str, str]:
        headers = dict(self.config.extra_headers)
        auth = self.config.auth
        # Literal credentials (e.g. typed into the UI) win over *_env variables.
        token = auth.token or (os.environ.get(auth.token_env, "") if auth.token_env else "")
        username = auth.username or (os.environ.get(auth.username_env, "") if auth.username_env else "")
        password = auth.password or (os.environ.get(auth.password_env, "") if auth.password_env else "")

        if auth.type == "bearer" and token:
            headers["Authorization"] = f"Bearer {token}"
        elif auth.type == "api_key" and token:
            headers[auth.header_name] = token
        elif auth.type == "basic" and username and password:
            import base64

            raw = f"{username}:{password}"
            headers["Authorization"] = "Basic " + base64.b64encode(raw.encode()).decode()
        return headers

    def _parse(self, case_id: str, payload: Any, latency_ms: float) -> AgentResponse:
        # Some agents return the answer as a bare JSON string (the whole body is
        # the reply), rather than an object with an answer field. Handle both:
        # a string payload, or an empty answer_path, means "use the whole body".
        if isinstance(payload, str):
            answer: Any = payload
            citations = None
        else:
            path = self.config.response.answer_path
            answer = payload if not path else _dig(payload, path)
            citations = _dig(payload, self.config.response.citations_path) if self.config.response.citations_path else None
        if isinstance(citations, str):
            citations = [citations]
        return AgentResponse(
            case_id=case_id,
            answer="" if answer is None else (answer if isinstance(answer, str) else str(answer)),
            cited_sources=list(citations) if isinstance(citations, list) else [],
            latency_ms=latency_ms,
            ok=True,
            raw=payload if isinstance(payload, dict) else {"value": payload},
        )

    async def _request(
        self, method: str, path: str, headers: dict[str, str], body: dict[str, Any]
    ) -> dict[str, Any]:
        assert self._client is not None
        url = self.config.base_url.rstrip("/") + "/" + path.lstrip("/")
        resp = await self._client.request(method, url, headers=headers, json=body)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return {"text": resp.text}


# --------------------------------------------------------------------------- #
# Small utilities shared by connectors
# --------------------------------------------------------------------------- #
def _fill_template(template: Any, values: dict[str, str]) -> Any:
    """Recursively substitute {placeholders} in a request template."""
    if isinstance(template, str):
        try:
            return template.format(**values)
        except (KeyError, IndexError):
            return template
    if isinstance(template, dict):
        return {k: _fill_template(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [_fill_template(v, values) for v in template]
    return template


def _dig(obj: Any, path: Optional[str]) -> Any:
    """Resolve a dotted path with optional ``[index]`` segments (e.g. a[-1].b)."""
    if not path:
        return None
    cur = obj
    for token in path.split("."):
        key, _, idx = token.partition("[")
        if key:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        if idx:
            try:
                i = int(idx.rstrip("]"))
                cur = cur[i]
            except (ValueError, IndexError, TypeError):
                return None
    return cur
