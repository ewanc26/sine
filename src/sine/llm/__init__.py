"""Provider-neutral LLM abstraction.

Sine talks to models through :class:`Provider` and :class:`Model`. Nothing above
this package knows which vendor is in use; provider-specific behaviour lives in
:mod:`sine.llm.adapters` and provider-specific facts live in
:mod:`sine.llm.registry`.
"""

from sine.llm.capabilities import (
    ModelCapabilities,
    ModelInfo,
    StructuredOutputDialect,
)
from sine.llm.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUnsupportedError,
)
from sine.llm.generation import (
    FinishReason,
    GenerationRequest,
    GenerationResponse,
    JsonObjectFormat,
    JsonSchemaFormat,
    ResponseFormat,
    TextFormat,
    TokenUsage,
)
from sine.llm.messages import Message, MessageRole
from sine.llm.provider import Model, Provider
from sine.llm.registry import (
    PROFILES,
    ProviderFamily,
    ProviderProfile,
    build_model,
    build_provider,
    known_providers,
    profile_for,
)
from sine.llm.structured import describe_schema, ensure_strict

__all__ = [
    "PROFILES",
    "FinishReason",
    "GenerationRequest",
    "GenerationResponse",
    "JsonObjectFormat",
    "JsonSchemaFormat",
    "Message",
    "MessageRole",
    "Model",
    "ModelCapabilities",
    "ModelInfo",
    "Provider",
    "ProviderAuthenticationError",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderFamily",
    "ProviderProfile",
    "ProviderRateLimitError",
    "ProviderRequestError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "ProviderUnsupportedError",
    "ResponseFormat",
    "StructuredOutputDialect",
    "TextFormat",
    "TokenUsage",
    "build_model",
    "build_provider",
    "describe_schema",
    "ensure_strict",
    "known_providers",
    "profile_for",
]
