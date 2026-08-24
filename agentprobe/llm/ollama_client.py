"""Thin OpenAI-compatible client pointed at a local Ollama server.

The accelerator's whole LLM surface — generation, self-consistency, and the
LLM-as-judge — goes through this one class. It speaks the OpenAI Chat
Completions protocol, which Ollama serves at ``/v1/chat/completions``. That is
what makes the core provider-agnostic: swap the base URL and a hosted model can
stand in for Ollama without touching the pipeline.

Structured output is handled here rather than via a heavyweight dependency: we
ask the model for JSON, extract the first JSON object, and validate it against a
Pydantic model, retrying with the validation error fed back to the model. This
is the same loop libraries like Instructor implement, kept small and explicit.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ..config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


class LLMError(RuntimeError):
    """Raised when the model server cannot be reached or returns an error."""


class OllamaClient:
    """Synchronous chat client for Ollama's OpenAI-compatible endpoint."""

    def __init__(self, settings: Optional[Settings] = None, model: Optional[str] = None):
        self.settings = settings or Settings.from_env()
        self.model = model or self.settings.generation_model
        self._client = httpx.Client(
            base_url=self.settings.ollama_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.settings.ollama_api_key}"},
            timeout=self.settings.request_timeout_s,
        )

    # -- lifecycle -------------------------------------------------------- #
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- core call -------------------------------------------------------- #
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.4,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion and return the assistant text."""
        payload: dict[str, object] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            # Ollama honours OpenAI's response_format for JSON-only output
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network dependent
            raise LLMError(f"Ollama request failed: {exc}") from exc
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Ollama response shape: {data}") from exc

    # -- structured output ------------------------------------------------ #
    def structured(
        self,
        messages: list[dict[str, str]],
        schema: Type[T],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_retries: int = 2,
    ) -> T:
        """Return a validated ``schema`` instance from the model.

        Retries feed the Pydantic validation error back to the model so it can
        correct its own JSON — a lightweight, dependency-free ``Instructor``.
        """
        schema_hint = json.dumps(schema.model_json_schema(), indent=2)
        convo = list(messages)
        convo.append(
            {
                "role": "system",
                "content": "Respond with a single JSON object that conforms to this "
                f"JSON schema. Do not wrap it in prose or markdown.\n\n{schema_hint}",
            }
        )
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            raw = self.chat(convo, model=model, temperature=temperature, json_mode=True)
            try:
                return schema.model_validate(_extract_json(raw))
            except (ValidationError, ValueError) as exc:
                last_err = exc
                logger.debug("structured() attempt %d failed: %s", attempt, exc)
                convo.append({"role": "assistant", "content": raw})
                convo.append(
                    {
                        "role": "user",
                        "content": f"That did not validate: {exc}. "
                        "Return corrected JSON only.",
                    }
                )
        raise LLMError(f"Could not obtain valid {schema.__name__}: {last_err}")


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply, tolerating stray prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(text)
        if not match:
            raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
        return json.loads(match.group(0))
