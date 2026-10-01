"""The provider and model interfaces.

This is the whole contract between Sine and the outside world's model APIs. A
provider owns transport, authentication, and translation; a model binds a
provider to a model identifier and its capabilities. The recommendation engine
depends on these two types and on nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sine.llm.capabilities import ModelCapabilities, ModelInfo
from sine.llm.generation import GenerationRequest, GenerationResponse


@runtime_checkable
class Provider(Protocol):
    """A service that can be asked to generate text with one or more models."""

    @property
    def provider_id(self) -> str:
        """Stable identifier for this provider, as used in configuration."""

    def capabilities(self, model_id: str) -> ModelCapabilities:
        """Describe what ``model_id`` supports, without calling the model."""

    def generate(self, model_id: str, request: GenerationRequest) -> GenerationResponse:
        """Generate a completion, or raise a :class:`~sine.llm.errors.ProviderError`."""

    def list_models(self) -> Sequence[ModelInfo]:
        """Return the provider's model catalogue, as far as it exposes one."""


@dataclass(frozen=True, slots=True)
class Model:
    """A provider paired with a specific model identifier."""

    provider: Provider
    model_id: str
    capabilities: ModelCapabilities

    @property
    def qualified_id(self) -> str:
        return f"{self.provider.provider_id}/{self.model_id}"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return self.provider.generate(self.model_id, request)
