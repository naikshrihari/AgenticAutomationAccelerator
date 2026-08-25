"""Tests for the Oracle Fusion async invoke + poll connector.

Uses a fake async HTTP client so no real Oracle endpoint is needed: the fake
returns a jobId from invokeAsync, then a couple of RUNNING statuses, then a
COMPLETE status carrying the answer — exercising the polling loop.
"""

from __future__ import annotations

import asyncio

from agentprobe.config import AuthConfig, TargetConfig
from agentprobe.execution.connectors.oracle_fusion import OracleFusionConnector


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    """Minimal async stand-in for httpx.AsyncClient."""

    def __init__(self, invoke_payload, status_sequence):
        self._invoke_payload = invoke_payload
        self._status_sequence = list(status_sequence)
        self.invoked = []
        self.polls = 0

    async def request(self, method, url, headers=None, json=None):
        self.invoked.append((method, url, json, headers))
        return _FakeResponse(self._invoke_payload)

    async def get(self, url, headers=None, params=None):
        # A relay token fetch hits the tokenrelay URL; everything else is a poll.
        if "tokenrelay" in url:
            return _FakeResponse({"access_token": "relayed-token-xyz"})
        idx = min(self.polls, len(self._status_sequence) - 1)
        self.polls += 1
        self.last_poll_params = params
        return _FakeResponse(self._status_sequence[idx])


def _make_config() -> TargetConfig:
    return TargetConfig(
        name="oracle_test",
        connector="oracle_fusion",
        base_url="https://fusion.example.com",
        auth=AuthConfig(type="basic", username_env="U", password_env="P"),
        options={"agent_code": "HR_AGENT", "poll_interval_s": 0, "poll_max_attempts": 5},
    )


def test_oracle_invoke_and_poll_returns_answer():
    fake = _FakeClient(
        invoke_payload={"jobId": "job-123", "conversationId": "conv-1", "status": "RUNNING"},
        status_sequence=[
            {"status": "RUNNING"},
            {"status": "RUNNING"},
            {"status": "COMPLETE", "answer": "You have 12 leave days.",
             "conversationId": "conv-1"},
        ],
    )
    conn = OracleFusionConnector(_make_config(), client=fake)

    async def run():
        await conn.start_session()
        return await conn.send("How many leave days do I have?")

    resp = asyncio.run(run())
    assert resp.ok is True
    assert resp.answer == "You have 12 leave days."
    # It polled until COMPLETE (3 status calls).
    assert fake.polls == 3
    # The invoke body carried the message and a null conversationId on turn one.
    _, url, body, _hdrs = fake.invoked[0]
    assert url.endswith("/api/fusion-ai/orchestrator/agent/v2/HR_AGENT/invokeAsync")
    assert body["message"] == "How many leave days do I have?"
    assert body["conversationId"] is None
    # Conversation id is captured for subsequent turns.
    assert conn.conversation_id == "conv-1"
    # The status poll carried ?invocationMode=ADMIN.
    assert fake.last_poll_params == {"invocationMode": "ADMIN"}


def test_oracle_relay_mints_bearer_token_and_sends_it():
    """auth_mode=relay fetches a token and sends it as a Bearer header."""
    cfg = TargetConfig(
        name="oracle_test", connector="oracle_fusion", base_url="https://fusion.example.com",
        options={"agent_code": "HR_AGENT", "poll_interval_s": 0, "poll_max_attempts": 5,
                 "auth_mode": "relay", "relay_cookie": "SESSION=abc", "relay_xsrf": "xsrf-1"},
    )
    fake = _FakeClient(
        invoke_payload={"jobId": "job-1"},
        status_sequence=[{"status": "COMPLETE", "answer": "42 days"}],
    )
    conn = OracleFusionConnector(cfg, client=fake)

    async def run():
        await conn.start_session()  # should mint the token via relay
        return await conn.send("q")

    resp = asyncio.run(run())
    assert resp.answer == "42 days"
    assert conn._bearer == "relayed-token-xyz"
    # The invoke request carried the relayed bearer token.
    _, _, _, headers = fake.invoked[0]
    assert headers["Authorization"] == "Bearer relayed-token-xyz"


def test_oracle_failed_status_returns_ok_empty_answer():
    """An agent-level FAILED status is a wrong answer, not an infra error."""
    fake = _FakeClient(
        invoke_payload={"jobId": "job-9"},
        status_sequence=[{"status": "FAILED", "error": "tool blew up"}],
    )
    conn = OracleFusionConnector(_make_config(), client=fake)

    async def run():
        await conn.start_session()
        return await conn.send("x")

    resp = asyncio.run(run())
    assert resp.ok is True
    assert resp.answer == ""
