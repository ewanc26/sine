"""Adapter for the Google Gemini Developer API.

The most divergent of the three families, which is the point of a separate
adapter:

* the URL carries the model and the action (``models/{model}:generateContent``);
* the system prompt is ``systemInstruction``, and the model's own turns use the
  role ``model`` rather than ``assistant``;
* generation settings live in a camelCase ``generationConfig``;
* structured output is ``responseMimeType`` plus ``responseSchema``, and the
  schema is a subset of JSON Schema rather than full JSON Schema;
* usage is ``usageMetadata`` with different field names again;
* a response may arrive with no candidates at all when a safety filter blocks
  it, which must surface as a content-filter error rather than a parse failure.
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

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
API_VERSION = "v1beta"

_ROLE_OUT = {MessageRole.USER.value: "user", MessageRole.ASSISTANT.value: "model"}

_FINISH_REASONS: dict[str, FinishReason] = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,
    "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
    "SPII": FinishReason.CONTENT_FILTER,
    "BLOCKLIST": FinishReason.CONTENT_FILTER,
    "IMAGE_PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
    "MALFORMED_FUNCTION_CALL": FinishReason.ERROR,
}


class GeminiProvider:
    """Provider backed by the Gemini ``generateContent`` API."""

    def __init__(
        self,
        *,
        provider_id: str = "gemini",
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
            f"{self._base_url}/{API_VERSION}/models/{model_id}:generateContent",
            headers=self._headers(),
            payload=self._build_body(model_id, request),
        )
        return self._parse(
            require_mapping(payload, self._provider_id, "response"), model_id
        )

    def list_models(self) -> Sequence[ModelInfo]:
        payload = self._transport.get_json(
            f"{self._base_url}/{API_VERSION}/models", headers=self._headers()
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
            name = entry.get("name")
            if not isinstance(name, str):
                continue
            methods = entry.get("supportedGenerationMethods")
            if isinstance(methods, list) and "generateContent" not in methods:
                continue
            models.append(
                ModelInfo(
                    model_id=name.removeprefix("models/"),
                    display_name=entry.get("displayName"),
                    owned_by="google",
                    context_window=_as_int(entry.get("inputTokenLimit")),
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
            "x-goog-api-key": self._api_key,
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

        body: dict[str, Any] = {
            "contents": [
                {
                    "role": _ROLE_OUT[message.role.value],
                    "parts": [{"text": message.content}],
                }
                for message in collapsed
            ]
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        config: dict[str, Any] = {}
        if request.max_output_tokens is not None:
            config["maxOutputTokens"] = request.max_output_tokens
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.stop_sequences:
            config["stopSequences"] = list(request.stop_sequences)

        response_format = request.response_format
        if isinstance(response_format, JsonSchemaFormat):
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = (
                ensure_strict(response_format.json_schema)
                if response_format.strict
                else dict(response_format.json_schema)
            )
        elif isinstance(response_format, JsonObjectFormat):
            # ``responseMimeType`` alone is only a strong hint, but the request has
            # no schema-only mode; the engine also describes the shape in the prompt.
            config["responseMimeType"] = "application/json"

        if config:
            body["generationConfig"] = config
        return body

    def _parse(self, payload: dict[str, Any], model_id: str) -> GenerationResponse:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            feedback = payload.get("promptFeedback")
            reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
            raise ProviderResponseError(
                self._provider_id,
                f"response contained no candidates (prompt feedback: {reason or 'none'})",
                status_code=200,
            )

        first = candidates[0]
        content = first.get("content") if isinstance(first, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        text = "".join(
            part["text"]
            for part in (parts or [])
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        if not text:
            raise ProviderResponseError(
                self._provider_id, "response contained no text parts", status_code=200
            )

        usage_raw = payload.get("usageMetadata")
        usage = None
        if isinstance(usage_raw, dict):
            usage = TokenUsage(
                input_tokens=_as_int(usage_raw.get("promptTokenCount")),
                output_tokens=_as_int(usage_raw.get("candidatesTokenCount")),
                cached_input_tokens=_as_int(usage_raw.get("cachedContentTokenCount")),
                reasoning_tokens=_as_int(usage_raw.get("thoughtsTokenCount")),
            )

        finish = first.get("finishReason") if isinstance(first, dict) else None
        return GenerationResponse(
            text=text,
            provider=self._provider_id,
            model=str(payload.get("modelVersion") or model_id),
            finish_reason=self._finish_reason(finish),
            usage=usage,
            request_id=payload.get("responseId")
            if isinstance(payload.get("responseId"), str)
            else None,
        )

    def _finish_reason(self, raw: Any) -> FinishReason:
        if not isinstance(raw, str):
            return FinishReason.UNKNOWN
        return _FINISH_REASONS.get(raw.upper(), FinishReason.OTHER)


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = ["DEFAULT_BASE_URL", "GeminiProvider"]
