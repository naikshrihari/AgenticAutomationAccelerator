"""Connector registry.

Maps the ``connector`` name in a target YAML to a concrete connector class. New
platforms are added by registering here — the execution engine never learns
about individual platforms.
"""

from __future__ import annotations

from typing import Type

from ...config import TargetConfig
from .base import BaseConnector
from .generic_rest import GenericRESTConnector
from .oracle_fusion import OracleFusionConnector
from .servicenow import ServiceNowConnector
from .openai_compat import OpenAICompatConnector

_REGISTRY: dict[str, Type[BaseConnector]] = {
    "generic_rest": GenericRESTConnector,
    "oracle_fusion": OracleFusionConnector,
    "servicenow": ServiceNowConnector,
    "openai_compat": OpenAICompatConnector,
}


def get_connector_class(name: str) -> Type[BaseConnector]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown connector '{name}'. Available: {', '.join(sorted(_REGISTRY))}"
        ) from None


def build_connector(config: TargetConfig) -> BaseConnector:
    return get_connector_class(config.connector)(config)


__all__ = [
    "BaseConnector",
    "GenericRESTConnector",
    "OracleFusionConnector",
    "ServiceNowConnector",
    "OpenAICompatConnector",
    "get_connector_class",
    "build_connector",
]
