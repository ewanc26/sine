"""Adapter for the Cohere v2 Chat API.

Materially different from the OpenAI-compatible shape it otherwise resembles,
which is why ``cohere`` names this adapter rather than the ``openai_compatible``
family despite the request body being close to a chat-completions call:

* a reply's content is a list of typed content blocks, not a message string;
* finish reasons use Cohere's own vocabulary (``COMPLETE``, ``MAX_TOKENS``, ...);
* usage is reported under ``usage.tokens``, with ``usage.billed_units`` kept
  separate for billing and not used for Sine's own accounting;
* the model catalogue lives one API version behind the chat endpoint, at
  ``/v1/models``, and lists non-chat models (embed, rerank) that must be
  filtered out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from sine.llm.adapters.http import (
    HttpTransport,
    RetryPolicy,
    build_client,
    require_mapping,
)
from sine.llm.adapters.structured_output import (
    CohereJsonSchemaStrategy,
    StructuredOutputStrategy,
)
from sine.llm.capabilities import ModelCapabilities, ModelInfo, StructuredOutputDialect
from sine.llm.errors import (
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderResponseError,
)
from sine.llm.generation import (
    FinishReason,
    GenerationRequest,
    GenerationResponse,
    TokenUsage,
)
from sine.llm.messages import MessageRole, collapse_consecutive, split_system

DEFAULT_BASE_URL = "https://api.cohere.ai/v2"

_FINISH_REASONS: dict[str, FinishReason] = {
    "complete": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_call": FinishReason.OTHER,
    "error": FinishReason.ERROR,
}


def _text_from_content(content: Any) -> str:
    """Read a reply's content blocks, Cohere's own list-of-parts shape."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        ]
        return "".join(parts)
    return ""


def _models_url(base_url: str) -> str:
    """The model catalogue lives at ``/v1/models``, one version behind chat."""

    if base_url.endswith("/v2"):
        return f"{base_url[: -len('/v2')]}/v1/models"
    return f"{base_url}/models"


class CohereProvider:
    """Provider backed by the Cohere v2 Chat API."""

    def __init__(
        self,
        *,
        provider_id: str = "cohere",
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        structured_output: StructuredOutputDialect = StructuredOutputDialect.JSON_SCHEMA,
        http_client: httpx.Client | None = None,
        transport: HttpTransport | None = None,
        timeout: float = 120.0,
        retry: RetryPolicy | None = None,
        strategy: StructuredOutputStrategy | None = None,
    ) -> None:
        if not base_url:
            raise ProviderConfigurationError(provider_id, "base_url is required")
        self._provider_id = provider_id
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._capabilities = ModelCapabilities(
            structured_output=structured_output,
            supports_system_messages=True,
            reports_token_usage=True,
        )
        self._strategy = strategy or CohereJsonSchemaStrategy()
        self._owns_client = http_client is None and transport is None
        self._transport = transport or HttpTransport(
            provider_id=provider_id,
            client=http_client or build_client(timeout),
            retry=retry,
        )

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def base_url(self) -> str:
        return self._base_url

    def capabilities(self, model_id: str) -> ModelCapabilities:
        return self._capabilities

    def generate(self, model_id: str, request: GenerationRequest) -> GenerationResponse:
        body = self._build_body(model_id, request)
        payload = self._transport.post_json(
            f"{self._base_url}/chat", headers=self._headers(), payload=body
        )
        return self._parse(
            require_mapping(payload, self._provider_id, "response"), model_id
        )

    def list_models(self) -> Sequence[ModelInfo]:
        payload = self._transport.get_json(
            _models_url(self._base_url), headers=self._headers()
        )
        data = require_mapping(payload, self._provider_id, "model list")
        entries = data.get("models")
        if not isinstance(entries, list):
            raise ProviderResponseError(
                self._provider_id, "model list did not contain a list of models"
            )

        models: list[ModelInfo] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            endpoints = entry.get("endpoints")
            if isinstance(endpoints, list) and "chat" not in endpoints:
                continue
            model_id = entry.get("name")
            if not isinstance(model_id, str) or not model_id:
                continue
            models.append(
                ModelInfo(
                    model_id=model_id,
                    owned_by="cohere",
                    context_window=entry.get("context_length"),
                )
            )
        return tuple(sorted(models, key=lambda info: info.model_id))

    def close(self) -> None:
        if self._owns_client:
            self._transport.close()

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise ProviderConfigurationError(
                self._provider_id, "an API key is required; set it in configuration"
            )
        return {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
            "accept": "application/json",
        }

    def _build_body(self, model_id: str, request: GenerationRequest) -> dict[str, Any]:
        system, conversation = split_system(request.messages)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": MessageRole.SYSTEM.value, "content": system})
        for message in collapse_consecutive(conversation):
            messages.append({"role": message.role.value, "content": message.content})
        if not messages:
            raise ProviderRequestError(
                self._provider_id, "request contained no messages"
            )

        body: dict[str, Any] = {"model": model_id, "messages": messages}
        if request.max_output_tokens is not None:
            body["max_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.stop_sequences:
            body["stop_sequences"] = list(request.stop_sequences)

        self._strategy.apply(body, request.response_format)
        return body

    def _parse(self, payload: dict[str, Any], model_id: str) -> GenerationResponse:
        message = payload.get("message")
        text = _text_from_content(
            message.get("content") if isinstance(message, dict) else None
        )
        if not text:
            raise ProviderResponseError(
                self._provider_id,
                "response contained no text content; the model may have produced "
                "only a tool call or been cut off",
                status_code=200,
            )

        return GenerationResponse(
            text=text,
            provider=self._provider_id,
            model=str(payload.get("model") or model_id),
            finish_reason=self._finish_reason(payload.get("finish_reason")),
            usage=self._usage(payload.get("usage")),
            request_id=payload.get("id")
            if isinstance(payload.get("id"), str)
            else None,
        )

    def _finish_reason(self, raw: Any) -> FinishReason:
        if not isinstance(raw, str):
            return FinishReason.UNKNOWN
        return _FINISH_REASONS.get(raw.lower(), FinishReason.OTHER)

    def _usage(self, raw: Any) -> TokenUsage | None:
        if not isinstance(raw, Mapping):
            return None
        tokens = raw.get("tokens")
        if not isinstance(tokens, Mapping):
            return None
        return TokenUsage(
            input_tokens=_as_int(tokens.get("input_tokens")),
            output_tokens=_as_int(tokens.get("output_tokens")),
        )


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = ["DEFAULT_BASE_URL", "CohereProvider"]
