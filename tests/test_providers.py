"""Provider adapters and registry.

Nothing here reaches the network. Each test installs a mock transport, runs one
generation, and inspects the bytes that would have gone on the wire. That is the
only way to assert the thing that actually matters about a provider adapter: that
Sine's provider-neutral request becomes the request shape that provider documents.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from sine.llm.adapters.anthropic import ANTHROPIC_VERSION, AnthropicProvider
from sine.llm.adapters.gemini import GeminiProvider
from sine.llm.adapters.http import RetryPolicy, error_for_response
from sine.llm.adapters.openai_compatible import OpenAICompatibleProvider
from sine.llm.capabilities import ModelCapabilities, StructuredOutputDialect
from sine.llm.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from sine.llm.generation import (
    FinishReason,
    GenerationRequest,
    JsonObjectFormat,
    JsonSchemaFormat,
)
from sine.llm.messages import Message
from sine.llm.registry import PROFILES, build_provider, known_providers, profile_for

#: Retries must not slow the suite down; the policy itself is tested separately.
FAST = RetryPolicy(attempts=3, initial_delay=0.0, max_delay=0.0)

REQUEST = GenerationRequest(
    messages=(Message.system("be brief"), Message.user("hello")),
    response_format=JsonObjectFormat(),
    max_output_tokens=256,
    temperature=0.5,
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"a": {"type": "string"}},
    "required": ["a"],
}
SCHEMA_REQUEST = REQUEST.model_copy(
    update={"response_format": JsonSchemaFormat(json_schema=SCHEMA)}
)


def openai_text(content: str = "hi", **extra: Any) -> dict[str, Any]:
    return {
        "id": "req-1",
        "model": "test-model",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        **extra,
    }


def anthropic_text(content: str = "hi", **extra: Any) -> dict[str, Any]:
    return {
        "id": "msg-1",
        "model": "claude-x",
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        **extra,
    }


def gemini_text(content: str = "hi", **extra: Any) -> dict[str, Any]:
    return {
        "responseId": "resp-1",
        "modelVersion": "gemini-x",
        "candidates": [
            {"content": {"parts": [{"text": content}]}, "finishReason": "STOP"}
        ],
        **extra,
    }


class Wire:
    """Captures the requests an adapter would send and replays queued responses.

    A callable stays available for every attempt, so it can describe "always fails"
    and let the retry policy be observed; a fixed response is consumed once, which
    is what sequencing a retry-then-succeed needs.
    """

    def __init__(
        self, *responses: httpx.Response | Callable[[httpx.Request], httpx.Response]
    ):
        self.requests: list[httpx.Request] = []
        self._queue = list(responses)
        self._sticky: Callable[[httpx.Request], httpx.Response] | None = None
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if callable(self._sticky):
            return self._sticky(request)
        if not self._queue:
            raise AssertionError(f"unexpected extra request to {request.url}")
        nxt = self._queue.pop(0)
        if callable(nxt):
            self._sticky = nxt
            return nxt(request)
        return nxt

    def json(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)

    @property
    def url(self) -> str:
        return str(self.requests[-1].url)


def client_for(wire: Wire) -> httpx.Client:
    return httpx.Client(transport=wire.transport)


def openai_provider(wire: Wire, **kwargs: Any) -> OpenAICompatibleProvider:
    kwargs = {"api_key": "sk-test", **kwargs}
    return OpenAICompatibleProvider(
        base_url="https://api.example.com/v1",
        http_client=client_for(wire),
        retry=FAST,
        **kwargs,
    )


def anthropic_provider(wire: Wire, **kwargs: Any) -> AnthropicProvider:
    return AnthropicProvider(
        api_key="sk-ant-test", http_client=client_for(wire), retry=FAST, **kwargs
    )


def gemini_provider(wire: Wire, **kwargs: Any) -> GeminiProvider:
    return GeminiProvider(
        api_key="gm-test", http_client=client_for(wire), retry=FAST, **kwargs
    )


# ------------------------------------------------------- openai-compatible body


def test_openai_sends_the_documented_chat_completion_shape() -> None:
    wire = Wire(httpx.Response(200, json=openai_text()))
    provider = openai_provider(wire)

    provider.generate("test-model", REQUEST)

    assert wire.url == "https://api.example.com/v1/chat/completions"
    assert wire.requests[0].headers["authorization"] == "Bearer sk-test"
    body = wire.json()
    assert body["model"] == "test-model"
    assert body["max_tokens"] == 256
    assert body["temperature"] == 0.5
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]
    assert body["response_format"] == {"type": "json_object"}


def test_openai_uses_the_completion_token_field_where_required() -> None:
    """OpenAI renamed the field; the reasoning models reject the old one outright."""

    wire = Wire(httpx.Response(200, json=openai_text()))
    openai_provider(wire, max_tokens_field="max_completion_tokens").generate(
        "test-model", REQUEST
    )

    body = wire.json()
    assert body["max_completion_tokens"] == 256
    assert "max_tokens" not in body


def test_openai_sends_a_strict_schema_when_one_is_requested() -> None:
    wire = Wire(httpx.Response(200, json=openai_text("{}")))
    openai_provider(wire).generate("test-model", SCHEMA_REQUEST)

    fmt = wire.json()["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "response"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False


def test_openai_does_not_send_an_api_level_constraint_for_a_prompt_only_model() -> None:
    wire = Wire(httpx.Response(200, json=openai_text()))
    provider = openai_provider(wire, structured_output=StructuredOutputDialect.PROMPT)
    provider.generate("test-model", SCHEMA_REQUEST)

    assert "response_format" not in wire.json()


def test_openai_sends_stop_sequences_when_asked() -> None:
    wire = Wire(httpx.Response(200, json=openai_text()))
    request = REQUEST.model_copy(update={"stop_sequences": ("END",)})
    openai_provider(wire).generate("test-model", request)

    assert wire.json()["stop"] == ["END"]


def test_openai_merges_consecutive_turns_of_the_same_role() -> None:
    """Two adjacent user turns are rejected by several backends."""

    wire = Wire(httpx.Response(200, json=openai_text()))
    request = GenerationRequest(messages=(Message.user("a"), Message.user("b")))
    openai_provider(wire).generate("test-model", request)

    assert wire.json()["messages"] == [{"role": "user", "content": "a\n\nb"}]


def test_openai_refuses_an_empty_conversation() -> None:
    with pytest.raises(ProviderRequestError, match="no messages"):
        openai_provider(Wire()).generate(
            "test-model", REQUEST.model_copy(update={"messages": ()})
        )


def test_openai_can_send_no_authorization_header_when_there_is_no_key() -> None:
    """Self-hosted servers commonly run without authentication."""

    wire = Wire(httpx.Response(200, json=openai_text()))
    provider = OpenAICompatibleProvider(
        base_url="http://localhost:1234/v1", http_client=client_for(wire), retry=FAST
    )
    provider.generate("test-model", REQUEST)

    assert "authorization" not in wire.requests[0].headers


def test_openai_requires_a_base_url() -> None:
    with pytest.raises(ProviderConfigurationError, match="base_url"):
        OpenAICompatibleProvider(base_url="")


# ------------------------------------------------------------- openai parsing


def test_openai_usage_is_normalised_including_the_detail_fields() -> None:
    payload = openai_text(
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 2},
        }
    )
    response = openai_provider(Wire(httpx.Response(200, json=payload))).generate(
        "test-model", REQUEST
    )

    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 15
    assert response.usage.cached_input_tokens == 4
    assert response.usage.reasoning_tokens == 2


def test_openai_absent_usage_is_reported_as_absent_not_zero() -> None:
    response = openai_provider(Wire(httpx.Response(200, json=openai_text()))).generate(
        "test-model", REQUEST
    )
    assert response.usage is None


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("stop", FinishReason.STOP),
        ("stop_sequence", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("max_tokens", FinishReason.LENGTH),
        ("model_length", FinishReason.LENGTH),
        ("content_filter", FinishReason.CONTENT_FILTER),
        ("insufficient_system_resource", FinishReason.OTHER),
        ("something_new", FinishReason.OTHER),
        (None, FinishReason.UNKNOWN),
    ],
)
def test_openai_finish_reasons_are_mapped(
    reported: str | None, expected: FinishReason
) -> None:
    choice: dict[str, Any] = {"message": {"content": "hi"}}
    if reported is not None:
        choice["finish_reason"] = reported
    payload = {"choices": [choice]}

    response = openai_provider(Wire(httpx.Response(200, json=payload))).generate(
        "test-model", REQUEST
    )
    assert response.finish_reason is expected


def test_openai_reads_content_sent_as_a_list_of_parts() -> None:
    """Some gateways and self-hosted servers emit the list form."""

    payload = {
        "choices": [
            {"message": {"content": [{"type": "text", "text": "he"}, {"text": "llo"}]}}
        ]
    }
    response = openai_provider(Wire(httpx.Response(200, json=payload))).generate(
        "test-model", REQUEST
    )
    assert response.text == "hello"


def test_openai_reports_an_in_band_error_that_arrives_with_status_200() -> None:
    """OpenRouter returns HTTP 200 with an error object for quota and mid-stream failures."""

    payload = {"error": {"message": "model is overloaded", "code": 503}}
    provider = openai_provider(Wire(httpx.Response(200, json=payload)))

    with pytest.raises(ProviderResponseError, match="overloaded") as caught:
        provider.generate("test-model", REQUEST)
    assert caught.value.status_code == 200


def test_openai_empty_content_is_an_error_rather_than_an_empty_answer() -> None:
    """DeepSeek's thinking models return empty content when they hit the token cap."""

    payload = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
    with pytest.raises(ProviderResponseError, match="no content"):
        openai_provider(Wire(httpx.Response(200, json=payload))).generate(
            "test-model", REQUEST
        )


def test_openai_response_without_choices_is_reported() -> None:
    with pytest.raises(ProviderResponseError, match="no choices"):
        openai_provider(Wire(httpx.Response(200, json={"nope": 1}))).generate(
            "test-model", REQUEST
        )


def test_openai_records_the_request_id_when_the_provider_gives_one() -> None:
    response = openai_provider(Wire(httpx.Response(200, json=openai_text()))).generate(
        "test-model", REQUEST
    )
    assert response.request_id == "req-1"
    assert response.model == "test-model"


def test_openai_reads_a_model_catalogue() -> None:
    payload = {
        "data": [
            {"id": "model-b"},
            {
                "id": "model-a",
                "display_name": "Model A",
                "owned_by": "acme",
                "context_length": 8192,
            },
            {"not-a-model": True},
        ]
    }
    models = openai_provider(Wire(httpx.Response(200, json=payload))).list_models()

    assert [info.model_id for info in models] == ["model-a", "model-b"]
    assert models[0].label == "Model A"
    assert models[0].context_window == 8192


def test_openai_reads_ollama_style_model_listings() -> None:
    payload = {"models": [{"name": "llama3.2:3b"}]}
    models = openai_provider(Wire(httpx.Response(200, json=payload))).list_models()
    assert [info.model_id for info in models] == ["llama3.2:3b"]


def test_openai_rejects_a_model_list_that_is_not_a_list() -> None:
    with pytest.raises(ProviderResponseError, match="did not contain a list"):
        openai_provider(
            Wire(httpx.Response(200, json={"data": {"a": 1}}))
        ).list_models()


# ------------------------------------------------------------------- anthropic


def test_anthropic_sends_the_messages_api_shape() -> None:
    wire = Wire(httpx.Response(200, json=anthropic_text()))
    anthropic_provider(wire).generate("claude-x", REQUEST)

    assert wire.url == "https://api.anthropic.com/v1/messages"
    request = wire.requests[0]
    assert request.headers["x-api-key"] == "sk-ant-test"
    assert request.headers["anthropic-version"] == ANTHROPIC_VERSION
    body = wire.json()
    assert body["model"] == "claude-x"
    assert body["max_tokens"] == 256
    # The system prompt is a top-level field here, not a message.
    assert body["system"] == "be brief"
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_anthropic_omits_the_deprecated_sampling_parameters() -> None:
    """Current Claude models return 400 for a non-default temperature."""

    wire = Wire(httpx.Response(200, json=anthropic_text()))
    anthropic_provider(wire).generate(
        "claude-x", REQUEST.model_copy(update={"temperature": 1.4})
    )

    body = wire.json()
    assert "temperature" not in body
    assert "top_p" not in body


def test_anthropic_supplies_a_max_tokens_ceiling_when_none_is_requested() -> None:
    """``max_tokens`` is required on this API and has no default."""

    wire = Wire(httpx.Response(200, json=anthropic_text()))
    request = REQUEST.model_copy(update={"max_output_tokens": None})
    anthropic_provider(wire).generate("claude-x", request)

    assert wire.json()["max_tokens"] == 4096


def test_anthropic_flattens_typed_content_blocks_and_skips_thinking() -> None:
    payload = {
        "content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "the answer"},
        ],
        "stop_reason": "end_turn",
    }
    response = anthropic_provider(Wire(httpx.Response(200, json=payload))).generate(
        "claude-x", REQUEST
    )
    assert response.text == "the answer"


def test_anthropic_reports_a_response_with_no_text_block() -> None:
    payload = {
        "content": [{"type": "thinking", "thinking": "hmm"}],
        "stop_reason": "max_tokens",
    }
    with pytest.raises(ProviderResponseError, match="no text block"):
        anthropic_provider(Wire(httpx.Response(200, json=payload))).generate(
            "claude-x", REQUEST
        )


def test_anthropic_records_cache_read_tokens() -> None:
    payload = anthropic_text(
        usage={"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 6}
    )
    response = anthropic_provider(Wire(httpx.Response(200, json=payload))).generate(
        "claude-x", REQUEST
    )
    assert response.usage is not None
    assert response.usage.cached_input_tokens == 6


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("end_turn", FinishReason.STOP),
        ("stop_sequence", FinishReason.STOP),
        ("max_tokens", FinishReason.LENGTH),
        ("model_context_window_exceeded", FinishReason.LENGTH),
        ("refusal", FinishReason.CONTENT_FILTER),
        ("tool_use", FinishReason.OTHER),
        ("unheard_of", FinishReason.OTHER),
        (None, FinishReason.UNKNOWN),
    ],
)
def test_anthropic_stop_reasons_are_mapped(
    reported: str | None, expected: FinishReason
) -> None:
    payload = {"content": [{"type": "text", "text": "x"}], "stop_reason": reported}
    response = anthropic_provider(Wire(httpx.Response(200, json=payload))).generate(
        "claude-x", REQUEST
    )
    assert response.finish_reason is expected


def test_anthropic_requests_schema_output_through_output_config() -> None:
    wire = Wire(httpx.Response(200, json=anthropic_text("{}")))
    anthropic_provider(wire).generate("claude-x", SCHEMA_REQUEST)

    fmt = wire.json()["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False


def test_anthropic_json_object_mode_becomes_an_unconstrained_schema() -> None:
    """There is no "any JSON object" mode on this API."""

    wire = Wire(httpx.Response(200, json=anthropic_text("{}")))
    anthropic_provider(wire).generate("claude-x", REQUEST)

    assert wire.json()["output_config"]["format"] == {
        "type": "json_schema",
        "schema": {"type": "object"},
    }


def test_anthropic_refuses_a_conversation_that_starts_with_the_model() -> None:
    """Assistant prefill is rejected by current Claude models."""

    request = GenerationRequest(
        messages=(Message.assistant("draft"), Message.user("go on"))
    )
    with pytest.raises(ProviderRequestError, match="start with a user turn"):
        anthropic_provider(Wire()).generate("claude-x", request)


def test_anthropic_requires_an_api_key_before_sending_anything() -> None:
    wire = Wire()
    provider = AnthropicProvider(http_client=client_for(wire), retry=FAST)

    with pytest.raises(ProviderConfigurationError, match="API key"):
        provider.generate("claude-x", REQUEST)
    assert wire.requests == []


def test_anthropic_reads_its_own_model_catalogue() -> None:
    payload = {
        "data": [{"id": "claude-b"}, {"id": "claude-a", "display_name": "Claude A"}]
    }
    models = anthropic_provider(Wire(httpx.Response(200, json=payload))).list_models()

    assert [info.model_id for info in models] == ["claude-a", "claude-b"]
    assert models[0].owned_by == "anthropic"


# ----------------------------------------------------------------------- gemini


def test_gemini_sends_camel_case_contents_and_a_generation_config() -> None:
    wire = Wire(httpx.Response(200, json=gemini_text()))
    gemini_provider(wire).generate("test-model", REQUEST)

    assert (
        wire.url
        == "https://generativelanguage.googleapis.com/v1beta/models/test-model:generateContent"
    )
    assert wire.requests[0].headers["x-goog-api-key"] == "gm-test"
    body = wire.json()
    assert body["systemInstruction"] == {"parts": [{"text": "be brief"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert body["generationConfig"]["maxOutputTokens"] == 256
    assert body["generationConfig"]["temperature"] == 0.5
    assert body["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_uses_the_model_role_for_the_models_own_turns() -> None:
    wire = Wire(httpx.Response(200, json=gemini_text()))
    request = GenerationRequest(
        messages=(Message.user("a"), Message.assistant("b"), Message.user("c"))
    )
    gemini_provider(wire).generate("test-model", request)

    assert [entry["role"] for entry in wire.json()["contents"]] == [
        "user",
        "model",
        "user",
    ]


def test_gemini_normalises_usage_and_finish_reason() -> None:
    # ``finishReason`` belongs to the candidate, not the envelope.
    payload = gemini_text(
        usageMetadata={
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "cachedContentTokenCount": 3,
            "thoughtsTokenCount": 2,
        }
    )
    payload["candidates"][0]["finishReason"] = "MAX_TOKENS"
    response = gemini_provider(Wire(httpx.Response(200, json=payload))).generate(
        "gemini-x", REQUEST
    )

    assert response.finish_reason is FinishReason.LENGTH
    assert response.model == "gemini-x"
    assert response.request_id == "resp-1"
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.cached_input_tokens == 3
    assert response.usage.reasoning_tokens == 2


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("STOP", FinishReason.STOP),
        ("MAX_TOKENS", FinishReason.LENGTH),
        ("SAFETY", FinishReason.CONTENT_FILTER),
        ("RECITATION", FinishReason.CONTENT_FILTER),
        ("PROHIBITED_CONTENT", FinishReason.CONTENT_FILTER),
        ("MALFORMED_FUNCTION_CALL", FinishReason.ERROR),
        ("surprise", FinishReason.OTHER),
    ],
)
def test_gemini_finish_reasons_are_mapped(
    reported: str, expected: FinishReason
) -> None:
    payload = {
        "candidates": [
            {"content": {"parts": [{"text": "x"}]}, "finishReason": reported}
        ]
    }
    response = gemini_provider(Wire(httpx.Response(200, json=payload))).generate(
        "gemini-x", REQUEST
    )
    assert response.finish_reason is expected


def test_gemini_sends_the_schema_as_a_response_schema() -> None:
    wire = Wire(httpx.Response(200, json=gemini_text("{}")))
    gemini_provider(wire).generate("test-model", SCHEMA_REQUEST)

    config = wire.json()["generationConfig"]
    assert config["responseSchema"]["additionalProperties"] is False


def test_gemini_reports_a_safety_block_that_leaves_no_candidates() -> None:
    payload = {"promptFeedback": {"blockReason": "SAFETY"}}
    with pytest.raises(ProviderResponseError, match="SAFETY"):
        gemini_provider(Wire(httpx.Response(200, json=payload))).generate(
            "gemini-x", REQUEST
        )


def test_gemini_reports_an_empty_parts_list() -> None:
    payload = {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}
    with pytest.raises(ProviderResponseError, match="no text parts"):
        gemini_provider(Wire(httpx.Response(200, json=payload))).generate(
            "gemini-x", REQUEST
        )


def test_gemini_skips_models_that_cannot_generate_content() -> None:
    payload = {
        "models": [
            {
                "name": "models/embed-001",
                "supportedGenerationMethods": ["embedContent"],
            },
            {"name": "models/gemini-x", "inputTokenLimit": 1048576},
            {"no_name": True},
        ]
    }
    models = gemini_provider(Wire(httpx.Response(200, json=payload))).list_models()

    assert [info.model_id for info in models] == ["gemini-x"]
    assert models[0].context_window == 1048576


def test_gemini_requires_an_api_key_before_sending_anything() -> None:
    wire = Wire()
    provider = GeminiProvider(http_client=client_for(wire), retry=FAST)

    with pytest.raises(ProviderConfigurationError, match="API key"):
        provider.generate("gemini-x", REQUEST)
    assert wire.requests == []


# ------------------------------------------------------- status code mapping


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, ProviderAuthenticationError),
        (403, ProviderAuthenticationError),
        (400, ProviderRequestError),
        (404, ProviderRequestError),
        (422, ProviderRequestError),
        (429, ProviderRateLimitError),
        (408, ProviderTimeoutError),
        (504, ProviderTimeoutError),
        (500, ProviderUnavailableError),
        (503, ProviderUnavailableError),
        (529, ProviderUnavailableError),
    ],
)
def test_status_codes_map_to_typed_errors(
    status: int, expected: type[Exception]
) -> None:
    error = error_for_response("openai", httpx.Response(status, text="nope"))
    assert isinstance(error, expected)


def test_the_providers_own_message_vocabulary_is_understood() -> None:
    """Otherwise every real failure surfaces as generic text and loses its cause."""

    assert "overloaded" in str(
        error_for_response(
            "openrouter", httpx.Response(503, json={"error": {"message": "overloaded"}})
        )
    )
    assert "invalid api key" in str(
        error_for_response(
            "anthropic",
            httpx.Response(
                401, json={"type": "error", "error": {"message": "invalid api key"}}
            ),
        )
    )
    assert "PERMISSION_DENIED" in str(
        error_for_response(
            "gemini",
            httpx.Response(
                403, json={"error": {"code": 403, "message": "PERMISSION_DENIED"}}
            ),
        )
    )
    assert "credits exhausted" in str(
        error_for_response(
            "openrouter",
            httpx.Response(
                402,
                json={
                    "error": {
                        "message": "",
                        "metadata": {"error_type": "credits exhausted"},
                    }
                },
            ),
        )
    )


def test_a_rate_limit_error_carries_the_retry_after_hint() -> None:
    error = error_for_response(
        "openai", httpx.Response(429, text="slow down", headers={"retry-after": "12"})
    )
    assert isinstance(error, ProviderRateLimitError)
    assert error.retry_after_seconds == 12


def test_a_rate_limit_with_an_unparseable_retry_after_is_still_a_rate_limit() -> None:
    error = error_for_response(
        "openai",
        httpx.Response(429, text="slow down", headers={"retry-after": "soonish"}),
    )
    assert isinstance(error, ProviderRateLimitError)
    assert error.retry_after_seconds is None


def test_a_non_json_error_body_is_still_reported_with_its_status() -> None:
    error = error_for_response(
        "openai", httpx.Response(502, text="<html>bad gateway</html>")
    )
    assert isinstance(error, ProviderUnavailableError)
    assert "bad gateway" in str(error)
    assert error.status_code == 502


# --------------------------------------------------------------------- retries


def test_a_retryable_failure_is_retried_and_can_succeed() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=openai_text())

    wire = Wire(handler)
    response = openai_provider(wire).generate("test-model", REQUEST)

    assert response.text == "hi"
    assert attempts == 2


def test_retries_are_bounded_and_then_the_failure_surfaces() -> None:
    wire = Wire(lambda r: httpx.Response(503, text="unavailable"))
    with pytest.raises(ProviderUnavailableError):
        openai_provider(wire).generate("test-model", REQUEST)
    assert len(wire.requests) == FAST.attempts


def test_a_rate_limit_is_retried() -> None:
    wire = Wire(
        httpx.Response(429, text="slow down"),
        httpx.Response(200, json=openai_text()),
    )
    assert openai_provider(wire).generate("test-model", REQUEST).text == "hi"


def test_an_authentication_failure_is_not_retried() -> None:
    """Replaying a rejected credential cannot fix it and may lock the account."""

    wire = Wire(lambda r: httpx.Response(401, text="bad key"))
    with pytest.raises(ProviderAuthenticationError):
        openai_provider(wire).generate("test-model", REQUEST)
    assert len(wire.requests) == 1


def test_a_rejected_request_is_not_retried() -> None:
    wire = Wire(lambda r: httpx.Response(400, text="bad request"))
    with pytest.raises(ProviderRequestError):
        openai_provider(wire).generate("test-model", REQUEST)
    assert len(wire.requests) == 1


def test_a_connection_failure_is_reported_as_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    wire = Wire(handler)
    with pytest.raises(ProviderUnavailableError, match="no route to host"):
        openai_provider(wire).generate("test-model", REQUEST)


def test_a_timeout_is_reported_as_a_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(ProviderTimeoutError):
        openai_provider(Wire(handler)).generate("test-model", REQUEST)


def test_a_success_that_is_not_json_is_reported_not_swallowed() -> None:
    wire = Wire(httpx.Response(200, text="<html>hello</html>"))
    with pytest.raises(ProviderResponseError, match="not JSON"):
        openai_provider(wire).generate("test-model", REQUEST)


def test_an_api_key_never_appears_in_an_error_message() -> None:
    """Sine authenticates by header, so a failure must not surface the credential.

    The provider here says nothing useful, which is the worst case: whatever the
    fallback message is, it must come from the status and not from the request.
    """

    wire = Wire(lambda r: httpx.Response(401, text="Unauthorized"))
    with pytest.raises(ProviderAuthenticationError) as caught:
        openai_provider(wire).generate("test-model", REQUEST)
    assert "sk-test" not in str(caught.value)
    assert str(caught.value) == "openai-compatible: Unauthorized"


def test_a_credential_is_not_forwarded_into_the_request_body() -> None:
    wire = Wire(httpx.Response(200, json=openai_text()))
    openai_provider(wire).generate("test-model", REQUEST)
    assert "sk-test" not in wire.requests[0].content.decode()


def test_an_error_body_that_echoes_the_request_is_truncated() -> None:
    """Several providers echo the request body, which for Sine means listening context."""

    context = "x" * 5000
    wire = Wire(httpx.Response(400, json={"error": {"message": context}}))
    with pytest.raises(ProviderRequestError) as caught:
        openai_provider(wire).generate("test-model", REQUEST)
    assert len(str(caught.value)) < 500


# -------------------------------------------------------------------- registry


def test_unknown_providers_are_rejected_with_the_known_list() -> None:
    with pytest.raises(ProviderConfigurationError, match="unknown provider.*anthropic"):
        profile_for("not-a-provider")


def test_provider_lookup_is_case_insensitive_and_trims() -> None:
    assert profile_for("  Anthropic ").provider_id == "anthropic"


def test_every_profile_has_an_endpoint_except_the_one_that_demands_one() -> None:
    for profile in PROFILES.values():
        if profile.provider_id == "openai-compatible":
            continue
        assert profile.base_url, f"{profile.provider_id} has no endpoint"


def test_the_generic_profile_explains_that_it_needs_an_endpoint() -> None:
    assert PROFILES["openai-compatible"].base_url is None
    assert "base_url" in (PROFILES["openai-compatible"].notes or "")


def test_profiles_record_their_own_limitations() -> None:
    """A provider that cannot do something must say so rather than fail at runtime."""

    assert PROFILES["deepseek"].structured_output is StructuredOutputDialect.JSON_OBJECT
    assert "guided_json" in (PROFILES["nvidia"].notes or "")
    assert "Sampling parameters are deprecated" in (PROFILES["anthropic"].notes or "")
    assert PROFILES["cohere"].notes
    assert "not usable yet" in PROFILES["cohere"].notes.lower()


def test_profiles_that_require_authentication_name_their_environment_variable() -> None:
    for profile in PROFILES.values():
        if profile.requires_api_key:
            assert profile.api_key_env, (
                f"{profile.provider_id} requires a key but names no env var"
            )


def test_local_profiles_point_at_a_loopback_endpoint_and_need_no_key() -> None:
    for name in ("ollama", "lm-studio", "vllm", "llamacpp"):
        profile = PROFILES[name]
        assert profile.self_hosted is True
        assert profile.requires_api_key is False
        assert "localhost" in profile.base_url or "127.0.0.1" in profile.base_url


def test_known_providers_are_listed_in_a_stable_order() -> None:
    ids = [profile.provider_id for profile in known_providers()]
    assert ids == sorted(ids)
    assert {"anthropic", "gemini", "openai", "ollama"} <= set(ids)


def test_profiles_are_immutable() -> None:
    with pytest.raises(AttributeError):
        profile_for("anthropic").base_url = "https://elsewhere"


def test_building_a_provider_applies_its_endpoint_and_dialect() -> None:
    provider = build_provider("anthropic", api_key="sk-ant-x")

    assert provider.provider_id == "anthropic"
    assert provider.base_url == "https://api.anthropic.com"
    assert (
        provider.capabilities("claude-x").structured_output
        is StructuredOutputDialect.JSON_SCHEMA
    )


def test_a_declared_base_url_override_wins() -> None:
    provider = build_provider(
        "openai", api_key="sk-x", base_url="https://proxy.internal/v1"
    )
    assert provider.base_url == "https://proxy.internal/v1"


def test_a_provider_that_requires_a_key_fails_early_with_a_usable_message() -> None:
    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
        build_provider("openai")


def test_a_provider_without_a_key_can_still_be_built() -> None:
    assert build_provider("ollama").provider_id == "ollama"


def test_the_generic_provider_requires_an_endpoint_before_a_key_check() -> None:
    with pytest.raises(ProviderConfigurationError, match="base_url"):
        build_provider("openai-compatible", api_key="k")


def test_nvidia_gets_the_guided_json_strategy_rather_than_response_format() -> None:
    """NVIDIA documents ``extra_body.guided_json``, and prefers it to ``response_format``."""

    wire = Wire(httpx.Response(200, json=openai_text("{}")))
    provider = build_provider(
        "nvidia", api_key="k", http_client=client_for(wire), retry=FAST
    )
    provider.generate("meta/llama-3.1-70b", SCHEMA_REQUEST)

    body = wire.json()
    assert body["extra_body"]["guided_json"]["additionalProperties"] is False
    assert "response_format" not in body


def test_a_backend_can_be_asked_for_a_stronger_dialect_than_it_documents() -> None:
    """The documented dialect is a default, not a ceiling."""

    wire = Wire(httpx.Response(200, json=openai_text("{}")))
    provider = build_provider(
        "deepseek",
        api_key="k",
        structured_output=StructuredOutputDialect.JSON_SCHEMA,
        http_client=client_for(wire),
        retry=FAST,
    )
    provider.generate("deepseek-chat", SCHEMA_REQUEST)

    assert wire.json()["response_format"]["type"] == "json_schema"


def test_the_pending_provider_is_refused_with_an_explanation() -> None:
    """Cohere needs its own reader; failing at config time beats a confusing 404 later."""

    with pytest.raises(ProviderConfigurationError, match="cohere.*no response reader"):
        build_provider("cohere", api_key="k")


def test_capabilities_default_to_not_supporting_structured_output() -> None:
    """Fail closed: an undeclared dialect must not be treated as support."""

    assert ModelCapabilities().structured_output is StructuredOutputDialect.NONE
    assert ModelCapabilities().supports_structured_output() is False
