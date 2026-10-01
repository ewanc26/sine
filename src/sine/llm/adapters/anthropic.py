"""Adapter for the Anthropic Messages API.

Materially different from the OpenAI shape, which is why it is a separate adapter
rather than a flag on the OpenAI-compatible one:

* the system prompt is a top-level ``system`` field, not a message;
* ``max_tokens`` is required, and newer models deprecate ``temperature``,
  ``top_p``, and ``top_k``, so none of the sampling knobs are sent;
* structured output is requested through ``output_config.format``, not
  ``response_format``;
* the reply is a list of typed content blocks rather than a message object;
* usage is reported as ``input_tokens``/``output_tokens`` with cache reads
  tracked separately.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from sine.llm.adapters.http import (
    HttpTransport,
    RetryPolicy,
    build_client,
    require_mapping,
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
    JsonObjectFormat,
    JsonSchemaFormat,
    TokenUsage,
)
from sine.llm.messages import MessageRole, collapse_consecutive, split_system
from sine.llm.structured import ensure_strict

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com"
#: ``max_tokens`` has no default on this API, so Sine supplies one. Recent models
#: cap output at 128k tokens; this is a conservative request ceiling, not a claim
#: about the model's limit.
DEFAULT_MAX_OUTPUT_TOKENS = 4096

_FINISH_REASONS: dict[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "model_context_window_exceeded": FinishReason.LENGTH,
    "refusal": FinishReason.CONTENT_FILTER,
    "tool_use": FinishReason.OTHER,
    "pause_turn": FinishReason.OTHER,
}


class AnthropicProvider:
    """Provider backed by the Anthropic Messages API."""

    def __init__(
        self,
        *,
        provider_id: str = "anthropic",
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        structured_output: StructuredOutputDialect = StructuredOutputDialect.JSON_SCHEMA,
        http_client: httpx.Client | None = None,
        transport: HttpTransport | None = None,
        timeout: float = 120.0,
        retry: RetryPolicy | None = None,
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
        payload = self._transport.post_json(
            f"{self._base_url}/v1/messages",
            headers=self._headers(),
            payload=self._build_body(model_id, request),
        )
        return self._parse(
            require_mapping(payload, self._provider_id, "response"), model_id
        )

    def list_models(self) -> Sequence[ModelInfo]:
        payload = self._transport.get_json(
            f"{self._base_url}/v1/models", headers=self._headers()
        )
        data = require_mapping(payload, self._provider_id, "model list")
        entries = data.get("data")
        if not isinstance(entries, list):
            raise ProviderResponseError(
                self._provider_id, "model list did not contain a list of models"
            )
        return tuple(
            sorted(
                (
                    ModelInfo(
                        model_id=entry["id"],
                        display_name=entry.get("display_name"),
                        owned_by="anthropic",
                    )
                    for entry in entries
                    if isinstance(entry, dict) and isinstance(entry.get("id"), str)
                ),
                key=lambda info: info.model_id,
            )
        )

    def close(self) -> None:
        if self._owns_client:
            self._transport.close()

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise ProviderConfigurationError(
                self._provider_id, "an API key is required; set it in configuration"
            )
        return {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            "accept": "application/json",
        }

    def _build_body(self, model_id: str, request: GenerationRequest) -> dict[str, Any]:
        system, conversation = split_system(request.messages)
        collapsed = collapse_consecutive(conversation)
        if not collapsed:
            raise ProviderRequestError(
                self._provider_id, "request contained no messages"
            )
        if collapsed[0].role is MessageRole.ASSISTANT:
            # Assistant prefill is rejected by current Claude models.
            raise ProviderRequestError(
                self._provider_id, "the conversation must start with a user turn"
            )

        body: dict[str, Any] = {
            "model": model_id,
            "max_tokens": request.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in collapsed
            ],
        }
        if system:
            body["system"] = system
        if request.stop_sequences:
            body["stop_sequences"] = list(request.stop_sequences)
        # ``temperature`` is intentionally omitted: Anthropic deprecates the
        # sampling parameters on current models and returns 400 for
        # non-default values, so the configured value is not forwarded.

        response_format = request.response_format
        if isinstance(response_format, JsonSchemaFormat):
            body["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": (
                        ensure_strict(response_format.json_schema)
                        if response_format.strict
                        else dict(response_format.json_schema)
                    ),
                }
            }
        elif isinstance(response_format, JsonObjectFormat):
            # There is no "any JSON object" mode; an unconstrained object schema is
            # the closest equivalent this API offers.
            body["output_config"] = {
                "format": {"type": "json_schema", "schema": {"type": "object"}}
            }

        return body

    def _parse(self, payload: dict[str, Any], model_id: str) -> GenerationResponse:
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise ProviderResponseError(
                self._provider_id, "response contained no content blocks"
            )

        text = "".join(
            block["text"]
            for block in blocks
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
        if not text:
            types = sorted(
                {str(block.get("type")) for block in blocks if isinstance(block, dict)}
            )
            raise ProviderResponseError(
                self._provider_id,
                f"response contained no text block (saw: {', '.join(types) or 'none'})",
            )

        usage_raw = payload.get("usage")
        usage = None
        if isinstance(usage_raw, dict):
            usage = TokenUsage(
                input_tokens=_as_int(usage_raw.get("input_tokens")),
                output_tokens=_as_int(usage_raw.get("output_tokens")),
                cached_input_tokens=_as_int(usage_raw.get("cache_read_input_tokens")),
            )

        return GenerationResponse(
            text=text,
            provider=self._provider_id,
            model=str(payload.get("model") or model_id),
            finish_reason=self._finish_reason(payload.get("stop_reason")),
            usage=usage,
            request_id=payload.get("id")
            if isinstance(payload.get("id"), str)
            else None,
        )

    def _finish_reason(self, raw: Any) -> FinishReason:
        if not isinstance(raw, str):
            return FinishReason.UNKNOWN
        return _FINISH_REASONS.get(raw, FinishReason.OTHER)


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = ["ANTHROPIC_VERSION", "AnthropicProvider"]
