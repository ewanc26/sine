"""Provider registry: the one place that knows specific providers exist.

This table is the only module in Sine that names vendors. The recommendation
engine selects a provider through configuration and a
:class:`~sine.llm.provider.Model`; it never asks which vendor is in use, and
there is no ``if provider == ...`` anywhere above this layer.

Each entry records where a provider lives, how it is authenticated, and how it
differs from the OpenAI-compatible baseline. Adding a provider is a table entry
when it speaks an implemented API family, and a new adapter when it does not.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import httpx

from sine.llm.adapters.anthropic import DEFAULT_BASE_URL as ANTHROPIC_BASE_URL
from sine.llm.adapters.anthropic import AnthropicProvider
from sine.llm.adapters.gemini import DEFAULT_BASE_URL as GEMINI_BASE_URL
from sine.llm.adapters.gemini import GeminiProvider
from sine.llm.adapters.http import RetryPolicy
from sine.llm.adapters.openai_compatible import OpenAICompatibleProvider
from sine.llm.adapters.structured_output import NvidiaGuidedJsonStrategy
from sine.llm.capabilities import ModelCapabilities, StructuredOutputDialect
from sine.llm.errors import ProviderConfigurationError
from sine.llm.provider import Model, Provider

DEFAULT_TIMEOUT = 120.0


class ProviderFamily(StrEnum):
    """The API shapes Sine can currently talk to."""

    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    """Everything Sine needs to reach a provider, and its known deviations."""

    provider_id: str
    family: ProviderFamily
    description: str
    base_url: str | None
    api_key_env: str | None
    requires_api_key: bool
    structured_output: StructuredOutputDialect = StructuredOutputDialect.JSON_SCHEMA
    max_tokens_field: str = "max_tokens"
    self_hosted: bool = False
    notes: str | None = None

    def resolved_base_url(self, override: str | None = None) -> str | None:
        return override or self.base_url


_JSON = StructuredOutputDialect.JSON_SCHEMA

PROFILES: Mapping[str, ProviderProfile] = {
    profile.provider_id: profile
    for profile in (
        ProviderProfile(
            provider_id="openai",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description="OpenAI chat completions.",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            requires_api_key=True,
            # Reasoning models reject ``max_tokens`` outright and require
            # ``max_completion_tokens``, which current chat models also accept.
            max_tokens_field="max_completion_tokens",
        ),
        ProviderProfile(
            provider_id="openai-compatible",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description=(
                "Any endpoint speaking the OpenAI chat completions API. Use this "
                "for OpenAI-compatible gateways and self-hosted servers not listed "
                "below; set base_url."
            ),
            base_url=None,
            api_key_env="SINE_LLM_API_KEY",
            requires_api_key=False,
            notes="base_url is required for this entry.",
        ),
        ProviderProfile(
            provider_id="xai",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description="xAI Grok.",
            base_url="https://api.x.ai/v1",
            api_key_env="XAI_API_KEY",
            requires_api_key=True,
            notes="Documented schema limits: depth 2048, 64 properties.",
        ),
        ProviderProfile(
            provider_id="mistral",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description="Mistral AI.",
            base_url="https://api.mistral.ai/v1",
            api_key_env="MISTRAL_API_KEY",
            requires_api_key=True,
        ),
        ProviderProfile(
            provider_id="deepseek",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description="DeepSeek.",
            base_url="https://api.deepseek.com/v1",
            api_key_env="DEEPSEEK_API_KEY",
            requires_api_key=True,
            # DeepSeek answers json_schema with HTTP 400 on chat completions.
            structured_output=StructuredOutputDialect.JSON_OBJECT,
            notes=(
                "json_object only; thinking-enabled models can return empty content "
                "with finish_reason=length."
            ),
        ),
        ProviderProfile(
            provider_id="openrouter",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description="OpenRouter multi-provider gateway.",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            requires_api_key=True,
            notes=(
                "Structured output support depends on the routed model and upstream "
                "endpoint. Mid-stream failures arrive as HTTP 200 with an error chunk."
            ),
        ),
        ProviderProfile(
            provider_id="groq",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description="Groq LPU inference.",
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
            requires_api_key=True,
            notes=(
                "Constrained decoding is model-dependent; other models need "
                "structured_output = 'json_object'."
            ),
        ),
        ProviderProfile(
            provider_id="together",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description="Together AI.",
            base_url="https://api.together.ai/v1",
            api_key_env="TOGETHER_API_KEY",
            requires_api_key=True,
        ),
        ProviderProfile(
            provider_id="fireworks",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description="Fireworks AI.",
            base_url="https://api.fireworks.ai/inference/v1",
            api_key_env="FIREWORKS_API_KEY",
            requires_api_key=True,
        ),
        ProviderProfile(
            provider_id="cerebras",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description="Cerebras inference.",
            base_url="https://api.cerebras.ai/v1",
            api_key_env="CEREBRAS_API_KEY",
            requires_api_key=True,
        ),
        ProviderProfile(
            provider_id="nvidia",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description="NVIDIA NIM, hosted or self-hosted.",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key_env="NVIDIA_API_KEY",
            requires_api_key=True,
            notes="Uses extra_body.guided_json rather than response_format.",
        ),
        ProviderProfile(
            provider_id="cohere",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description="Cohere Command, v2 chat API.",
            base_url="https://api.cohere.ai/v2",
            api_key_env="COHERE_API_KEY",
            requires_api_key=True,
            notes=(
                "Shares this family but returns content as a list of content "
                "blocks; requires a dedicated response reader. Not usable yet."
            ),
        ),
        ProviderProfile(
            provider_id="ollama",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description="Ollama, local or Ollama Cloud.",
            base_url="http://localhost:11434/v1",
            api_key_env="OLLAMA_API_KEY",
            requires_api_key=False,
            self_hosted=True,
            notes=(
                "A key is sent but ignored locally. Structured outputs are a local "
                "capability; Ollama Cloud does not support them."
            ),
        ),
        ProviderProfile(
            provider_id="lm-studio",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description="LM Studio local server.",
            base_url="http://localhost:1234/v1",
            api_key_env="LM_STUDIO_API_KEY",
            requires_api_key=False,
            self_hosted=True,
            notes="No authentication is enforced. Small models often cannot do structured output.",
        ),
        ProviderProfile(
            provider_id="vllm",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description="vLLM OpenAI-compatible server.",
            base_url="http://localhost:8000/v1",
            api_key_env=None,
            requires_api_key=False,
            self_hosted=True,
        ),
        ProviderProfile(
            provider_id="llamacpp",
            family=ProviderFamily.OPENAI_COMPATIBLE,
            description="llama.cpp server.",
            base_url="http://127.0.0.1:8080/v1",
            api_key_env="LLAMA_API_KEY",
            requires_api_key=False,
            self_hosted=True,
            notes="Chat requests require a usable chat template.",
        ),
        ProviderProfile(
            provider_id="anthropic",
            family=ProviderFamily.ANTHROPIC,
            description="Anthropic Claude, Messages API.",
            base_url=ANTHROPIC_BASE_URL,
            api_key_env="ANTHROPIC_API_KEY",
            requires_api_key=True,
            notes="Sampling parameters are deprecated on current models and are not sent.",
        ),
        ProviderProfile(
            provider_id="gemini",
            family=ProviderFamily.GEMINI,
            description="Google Gemini, generateContent API.",
            base_url=GEMINI_BASE_URL,
            api_key_env="GEMINI_API_KEY",
            requires_api_key=True,
            notes=(
                "Accepts a subset of JSON Schema. Responses can be empty when a "
                "safety filter blocks the prompt or the reply."
            ),
        ),
    )
}

#: Providers whose response reader is not implemented yet. Listed so that
#: configuration errors are explicit instead of surfacing as confusing HTTP 404s.
_PENDING = frozenset({"cohere"})


def profile_for(provider_id: str) -> ProviderProfile:
    """Look up a provider profile, or explain what is available."""

    try:
        return PROFILES[provider_id.strip().lower()]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise ProviderConfigurationError(
            "config", f"unknown provider {provider_id!r}; known providers: {known}"
        ) from None


def known_providers() -> tuple[ProviderProfile, ...]:
    return tuple(PROFILES[key] for key in sorted(PROFILES))


def build_provider(
    provider_id: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    http_client: httpx.Client | None = None,
    retry: RetryPolicy | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    structured_output: StructuredOutputDialect | None = None,
) -> Provider:
    """Instantiate the adapter for ``provider_id``.

    ``structured_output`` overrides the profile's declared dialect, which is the
    escape hatch for a backend that is more capable in practice than documented,
    or less capable on a particular model.
    """

    profile = profile_for(provider_id)

    if profile.provider_id in _PENDING:
        raise ProviderConfigurationError(
            profile.provider_id,
            "no response reader is implemented for this provider yet",
        )

    resolved_url = profile.resolved_base_url(base_url)
    if not resolved_url:
        raise ProviderConfigurationError(
            profile.provider_id, "base_url is required for this provider"
        )
    if profile.requires_api_key and not api_key:
        raise ProviderConfigurationError(
            profile.provider_id,
            f"an API key is required; set {profile.api_key_env or 'the api key'} in configuration",
        )

    dialect = structured_output or profile.structured_output

    match profile.family:
        case ProviderFamily.OPENAI_COMPATIBLE:
            strategy = None
            if (
                dialect is StructuredOutputDialect.JSON_SCHEMA
                and profile.provider_id == "nvidia"
            ):
                strategy = NvidiaGuidedJsonStrategy()
            return OpenAICompatibleProvider(
                provider_id=profile.provider_id,
                base_url=resolved_url,
                api_key=api_key,
                structured_output=dialect,
                max_tokens_field=profile.max_tokens_field,
                http_client=http_client,
                retry=retry,
                timeout=timeout,
                strategy=strategy,
                notes=profile.notes,
            )
        case ProviderFamily.ANTHROPIC:
            return AnthropicProvider(
                provider_id=profile.provider_id,
                base_url=resolved_url,
                api_key=api_key,
                structured_output=dialect,
                http_client=http_client,
                retry=retry,
                timeout=timeout,
            )
        case ProviderFamily.GEMINI:
            return GeminiProvider(
                provider_id=profile.provider_id,
                base_url=resolved_url,
                api_key=api_key,
                structured_output=dialect,
                http_client=http_client,
                retry=retry,
                timeout=timeout,
            )
    raise ProviderConfigurationError(  # pragma: no cover - exhaustive above
        profile.provider_id, f"unsupported provider family {profile.family}"
    )


def build_model(
    provider_id: str,
    model_id: str,
    *,
    capabilities: ModelCapabilities | None = None,
    **kwargs: object,
) -> Model:
    """Instantiate a provider and bind a model to it."""

    provider = build_provider(provider_id, **kwargs)  # type: ignore[arg-type]
    resolved = capabilities or provider.capabilities(model_id)
    return Model(provider=provider, model_id=model_id, capabilities=resolved)


__all__ = [
    "PROFILES",
    "ProviderFamily",
    "ProviderProfile",
    "build_model",
    "build_provider",
    "known_providers",
    "profile_for",
]
