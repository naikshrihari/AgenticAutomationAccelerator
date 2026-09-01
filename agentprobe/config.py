"""Configuration models and loaders.

Two kinds of config exist in AgentProbe:

* :class:`Settings` — global knobs (Ollama endpoint, model names, paths). Read
  from environment variables so the same code runs on a laptop and in CI.
* :class:`TargetConfig` — one YAML file per agent under test, describing its
  endpoint, auth, request/response mapping, and pass thresholds. This is what
  lets the same suite run against Oracle Fusion, ServiceNow, or a REST agent
  by changing only the config.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    """Load a local .env file into the environment, once, if python-dotenv is
    installed. Values already set in the real environment take precedence."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    try:
        from dotenv import load_dotenv

        load_dotenv()  # reads ./.env; does not override existing env vars
    except ImportError:
        pass


class Settings(BaseModel):
    """Global runtime settings, populated from the environment.

    All LLM work is routed to Ollama's OpenAI-compatible endpoint. Because
    inference is local, no document or HR data leaves the network.
    """

    # Ollama exposes an OpenAI-compatible API at /v1
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: str = "ollama"  # placeholder; Ollama ignores it

    # A stronger model for judging, a lighter one for generation, plus embeddings
    generation_model: str = "llama3.1"
    judge_model: str = "llama3.1"
    embedding_model: str = "nomic-embed-text"

    # Sampling
    generation_temperature: float = 0.4
    judge_temperature: float = 0.0  # deterministic grading

    # Generous default: local judging on CPU can take minutes per call.
    request_timeout_s: float = 600.0

    # Paths
    data_dir: Path = Path("data")

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv_once()
        env = os.environ
        return cls(
            ollama_base_url=env.get("OLLAMA_BASE_URL", cls.model_fields["ollama_base_url"].default),
            ollama_api_key=env.get("OLLAMA_API_KEY", "ollama"),
            generation_model=env.get("AGENTPROBE_GEN_MODEL", cls.model_fields["generation_model"].default),
            judge_model=env.get("AGENTPROBE_JUDGE_MODEL", cls.model_fields["judge_model"].default),
            embedding_model=env.get("AGENTPROBE_EMBED_MODEL", cls.model_fields["embedding_model"].default),
            request_timeout_s=float(env.get("AGENTPROBE_TIMEOUT", cls.model_fields["request_timeout_s"].default)),
            data_dir=Path(env.get("AGENTPROBE_DATA_DIR", "data")),
        )

    @property
    def golden_dir(self) -> Path:
        return self.data_dir / "golden"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"


class AuthConfig(BaseModel):
    """How the connector authenticates to the target agent."""

    type: str = "none"  # none | bearer | basic | api_key | oauth2
    token_env: Optional[str] = None  # env var holding a bearer/api token
    username_env: Optional[str] = None
    password_env: Optional[str] = None
    header_name: str = "Authorization"  # for api_key style
    # Literal credentials. When set (e.g. entered in the UI at runtime) these
    # take precedence over the *_env variables above. Not written to disk by the
    # app; kept in memory for the duration of a run.
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    # oauth2
    token_url: Optional[str] = None
    client_id_env: Optional[str] = None
    client_secret_env: Optional[str] = None

    def resolve(self) -> dict[str, str]:
        """Resolve secrets from the environment into concrete values."""
        out: dict[str, str] = {}
        for attr in ("token_env", "username_env", "password_env", "client_id_env", "client_secret_env"):
            var = getattr(self, attr)
            if var:
                out[attr.removesuffix("_env")] = os.environ.get(var, "")
        return out


class ResponseMapping(BaseModel):
    """JSON paths used to pull the answer and citations out of the agent reply.

    Paths are simple dotted paths with optional ``[index]`` segments, e.g.
    ``result.messages[-1].content``. This keeps every connector's response
    parsing declarative in the target YAML instead of hard-coded.
    """

    answer_path: str = "answer"
    citations_path: Optional[str] = None
    session_id_path: Optional[str] = None


class Thresholds(BaseModel):
    """Pass/partial thresholds applied by the verdict resolver."""

    pass_score: float = 0.8
    partial_score: float = 0.5
    require_groundedness: bool = False


class TargetConfig(BaseModel):
    """One agent under test. Loaded from ``config/<target>.yaml``."""

    name: str
    connector: str = "generic_rest"  # oracle_fusion | servicenow | generic_rest | openai_compat
    base_url: str = ""
    endpoint: str = "/"  # path appended to base_url for send
    session_start_endpoint: Optional[str] = None
    session_close_endpoint: Optional[str] = None
    method: str = "POST"
    auth: AuthConfig = Field(default_factory=AuthConfig)

    # Request shaping: a template whose {question} / {session_id} are filled in
    request_template: dict[str, Any] = Field(default_factory=lambda: {"message": "{question}"})
    extra_headers: dict[str, str] = Field(default_factory=dict)

    response: ResponseMapping = Field(default_factory=ResponseMapping)
    thresholds: Thresholds = Field(default_factory=Thresholds)

    # Connector-specific options (e.g. Oracle Fusion agent code, poll settings).
    # Kept as a free-form dict so per-platform knobs don't clutter the shared model.
    options: dict[str, Any] = Field(default_factory=dict)

    # Execution tuning
    concurrency: int = 4
    max_retries: int = 3
    timeout_s: float = 60.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TargetConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)
