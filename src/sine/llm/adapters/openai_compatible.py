"""Adapter for the OpenAI-compatible ``/v1/chat/completions`` family.

One adapter serves the large group of backends that share this request and
response shape, including hosted providers (OpenAI, xAI, Mistral, OpenRouter,
Groq, Together, Fireworks, Cerebras, DeepSeek) and self-hosted servers (Ollama,
LM Studio, vLLM, llama.cpp, LocalAI). The places where those backends genuinely
differ are parameters, not a second class:

* ``structured_output`` selects the constraint mechanism (see
  :mod:`sine.llm.adapters.structured_output`);
* ``max_tokens_field`` selects the output-token key, because OpenAI's own
  reasoning models reject ``max_tokens`` and require ``max_completion_tokens``;
* ``requires_api_key`` and ``base_url`` describe deployment.

Everything else is assumed, not guaranteed, to behave identically — which is
exactly the assumption the registry documents as risky.
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
from sine.llm.adapters.structured_output import StructuredOutputStrategy, strategy_for
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

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "end_turn": FinishReason.STOP,
    "eos": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "max_tokens": FinishReason.LENGTH,
    "model_length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
    "insufficient_system_resource": FinishReason.OTHER,
}


def _text_from_content(content: Any) -> str:
    """Read message content, which is a string or a list of typed parts.

    The list form is not part of the OpenAI specification, but gateways and
    self-hosted servers emit it, and treating an unexpected shape as a parse
    failure would make Sine fail on responses that are actually valid.
    """

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "".join(parts)
    return ""


class OpenAICompatibleProvider:
    """Provider backed by an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        *,
        provider_id: str = "openai-compatible",
        base_url: str,
        api_key: str | None = None,
        structured_output: StructuredOutputDialect = StructuredOutputDialect.JSON_SCHEMA,
        max_tokens_field: str = "max_tokens",
        http_client: httpx.Client | None = None,
        transport: HttpTransport | None = None,
        timeout: float = 120.0,
        retry: RetryPolicy | None = None,
        strategy: StructuredOutputStrategy | None = None,
        notes: str | None = None,
    ) -> None:
        if not base_url:
            raise ProviderConfigurationError(provider_id, "base_url is required")
        self._provider_id = provider_id
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_tokens_field = max_tokens_field
        self._notes = notes
        self._capabilities = ModelCapabilities(
            structured_output=structured_output,
            supports_system_messages=True,
            reports_token_usage=True,
            notes=notes,
        )
        self._strategy = strategy or strategy_for(structured_output)
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
            f"{self._base_url}/chat/completions", headers=self._headers(), payload=body
        )
        return self._parse(
            require_mapping(payload, self._provider_id, "response"), model_id
        )

    def list_models(self) -> Sequence[ModelInfo]:
        payload = self._transport.get_json(
            f"{self._base_url}/models", headers=self._headers()
        )
        data = require_mapping(payload, self._provider_id, "model list")

        entries: Any = data.get("data")
        if entries is None:
            # Ollama's native listing and some gateways use a plain "models" key.
            entries = data.get("models", [])
        if not isinstance(entries, list):
            raise ProviderResponseError(
                self._provider_id, "model list did not contain a list of models"
            )

        models: list[ModelInfo] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id") or entry.get("name")
            if not isinstance(model_id, str) or not model_id:
                continue
            models.append(
                ModelInfo(
                    model_id=model_id,
                    display_name=entry.get("display_name") or entry.get("displayName"),
                    owned_by=entry.get("owned_by") or entry.get("ownedBy"),
                    context_window=entry.get("context_length")
                    or entry.get("context_window"),
                )
            )
        return tuple(sorted(models, key=lambda info: info.model_id))

    def close(self) -> None:
        if self._owns_client:
            self._transport.close()

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", "accept": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        return headers

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
            body[self._max_tokens_field] = request.max_output_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.stop_sequences:
            body["stop"] = list(request.stop_sequences)

        self._strategy.apply(body, request.response_format)
        return body

    def _parse(self, payload: dict[str, Any], model_id: str) -> GenerationResponse:
        in_band = payload.get("error")
        if in_band:
            # Some gateways report mid-stream failures and quota errors with a
            # 200 status and an error object in the body.
            raise ProviderResponseError(
                self._provider_id,
                f"provider returned an error in a successful response: {in_band!r}"[
                    :300
                ],
                status_code=200,
            )

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderResponseError(
                self._provider_id, "response contained no choices"
            )
        first = choices[0]
        if not isinstance(first, dict):
            raise ProviderResponseError(
                self._provider_id, "response choice was not an object"
            )

        message = first.get("message")
        text = _text_from_content(
            message.get("content") if isinstance(message, dict) else None
        )
        if not text:
            raise ProviderResponseError(
                self._provider_id,
                "response contained no content; the model may have produced only "
                "reasoning tokens or been cut off",
                status_code=200,
            )

        return GenerationResponse(
            text=text,
            provider=self._provider_id,
            model=str(payload.get("model") or model_id),
            finish_reason=self._finish_reason(first.get("finish_reason")),
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
        details = raw.get("prompt_tokens_details")
        cached = None
        if isinstance(details, Mapping):
            value = details.get("cached_tokens")
            cached = value if isinstance(value, int) else None
        completion_details = raw.get("completion_tokens_details")
        reasoning = None
        if isinstance(completion_details, Mapping):
            value = completion_details.get("reasoning_tokens")
            reasoning = value if isinstance(value, int) else None
        return TokenUsage(
            input_tokens=_as_int(raw.get("prompt_tokens")),
            output_tokens=_as_int(raw.get("completion_tokens")),
            cached_input_tokens=cached,
            reasoning_tokens=reasoning,
        )


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = ["OpenAICompatibleProvider"]
