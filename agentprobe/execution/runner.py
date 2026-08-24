"""Async execution runner.

Sends each golden-set question to the target agent concurrently, with bounded
parallelism, retries with exponential backoff, and session handling delegated to
the connector. Crucially, infrastructure failures (timeouts, auth errors,
connection resets) are classified as errors and reported separately from wrong
answers, so they never distort the agent's pass rate.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

try:  # tenacity is the documented choice, but keep the runner importable without it
    from tenacity import (
        AsyncRetrying,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    _HAS_TENACITY = True
except ImportError:  # pragma: no cover
    _HAS_TENACITY = False

from ..config import TargetConfig
from ..models import AgentResponse, TestCase
from .connectors import build_connector

logger = logging.getLogger(__name__)

# Exceptions we treat as retriable infrastructure faults.
_INFRA_ERRORS = (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)


def _classify_error(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (401, 403):
            return "auth"
        return "http"
    if isinstance(exc, _INFRA_ERRORS):
        return "connection"
    return "connector"


class ExecutionEngine:
    """Runs a list of test cases against one configured target."""

    def __init__(self, config: TargetConfig):
        self.config = config

    async def run(self, cases: list[TestCase]) -> list[AgentResponse]:
        """Execute all cases and return one response per case, in input order."""
        semaphore = asyncio.Semaphore(self.config.concurrency)
        async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
            connector = build_connector(self.config)
            connector._client = client  # share one pooled client across sends
            connector._owns_client = False
            await connector.start_session()
            try:
                tasks = [self._run_one(connector, case, semaphore) for case in cases]
                return await asyncio.gather(*tasks)
            finally:
                await connector.close_session()

    async def _run_one(self, connector, case: TestCase, semaphore: asyncio.Semaphore) -> AgentResponse:
        async with semaphore:
            try:
                response = await self._send_with_retries(connector, case.question)
                response.case_id = case.case_id
                return response
            except Exception as exc:  # noqa: BLE001 - convert into a classified error result
                kind = _classify_error(exc)
                logger.warning("case %s failed (%s): %s", case.case_id, kind, exc)
                return AgentResponse(
                    case_id=case.case_id,
                    ok=False,
                    error_kind=kind,
                    error_detail=str(exc),
                )

    async def _send_with_retries(self, connector, question: str) -> AgentResponse:
        if not _HAS_TENACITY:
            return await self._send_manual_retry(connector, question)
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=16),
            retry=retry_if_exception_type(_INFRA_ERRORS),
            reraise=True,
        ):
            with attempt:
                return await connector.send(question)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _send_manual_retry(self, connector, question: str) -> AgentResponse:
        """Fallback retry loop when tenacity is not installed."""
        delay = 1.0
        last: Optional[Exception] = None
        for _ in range(self.config.max_retries):
            try:
                return await connector.send(question)
            except _INFRA_ERRORS as exc:
                last = exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, 16.0)
        raise last if last else RuntimeError("send failed")
