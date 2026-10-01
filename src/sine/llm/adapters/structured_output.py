"""Structured-output payload strategies for the OpenAI-compatible family.

Most of the ~15 backends that expose ``POST /v1/chat/completions`` accept
``response_format``, but not identically: NVIDIA expects a schema under
``extra_body.guided_json`` and recommends it over ``response_format``; DeepSeek
rejects ``json_schema`` with a 400 and only offers ``json_object``; and any
endpoint can be driven by prompt alone. Composing the request is therefore a
small strategy object rather than a branch in the adapter.
"""

from __future__ import annotations

from typing import Any, Protocol

from sine.llm.capabilities import StructuredOutputDialect
from sine.llm.generation import JsonObjectFormat, JsonSchemaFormat, TextFormat
from sine.llm.structured import ensure_strict

ResponseFormat = TextFormat | JsonObjectFormat | JsonSchemaFormat


class StructuredOutputStrategy(Protocol):
    """Mutates a request body in place to express the requested output format."""

    def apply(self, body: dict[str, Any], response_format: ResponseFormat) -> None: ...


class PromptOnlyStrategy:
    """Describe the format in the prompt; send no API-level constraint."""

    def apply(self, body: dict[str, Any], response_format: ResponseFormat) -> None:
        return None


class JsonObjectStrategy:
    """``response_format: {"type": "json_object"}``."""

    def apply(self, body: dict[str, Any], response_format: ResponseFormat) -> None:
        body["response_format"] = {"type": "json_object"}


class OpenAIJsonSchemaStrategy:
    """``response_format: {"type": "json_schema", "json_schema": {...}}``."""

    def __init__(self, *, strict: bool = True) -> None:
        self._strict = strict

    def apply(self, body: dict[str, Any], response_format: ResponseFormat) -> None:
        if not isinstance(response_format, JsonSchemaFormat):
            JsonObjectStrategy().apply(body, response_format)
            return
        schema = (
            ensure_strict(response_format.json_schema)
            if response_format.strict
            else dict(response_format.json_schema)
        )
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.name,
                "schema": schema,
                "strict": response_format.strict,
            },
        }


class NvidiaGuidedJsonStrategy:
    """``extra_body: {"guided_json": schema}``, as NVIDIA NIM documents."""

    def apply(self, body: dict[str, Any], response_format: ResponseFormat) -> None:
        if not isinstance(response_format, JsonSchemaFormat):
            JsonObjectStrategy().apply(body, response_format)
            return
        extra = body.setdefault("extra_body", {})
        extra["guided_json"] = (
            ensure_strict(response_format.json_schema)
            if response_format.strict
            else dict(response_format.json_schema)
        )


class CohereJsonSchemaStrategy:
    """``response_format: {"type": "json_object", "json_schema": schema}``."""

    def apply(self, body: dict[str, Any], response_format: ResponseFormat) -> None:
        if not isinstance(response_format, JsonSchemaFormat):
            JsonObjectStrategy().apply(body, response_format)
            return
        body["response_format"] = {
            "type": "json_object",
            "json_schema": response_format.json_schema,
        }


_STRATEGIES: dict[StructuredOutputDialect, StructuredOutputStrategy] = {
    StructuredOutputDialect.NONE: PromptOnlyStrategy(),
    StructuredOutputDialect.PROMPT: PromptOnlyStrategy(),
    StructuredOutputDialect.JSON_OBJECT: JsonObjectStrategy(),
    StructuredOutputDialect.JSON_SCHEMA: OpenAIJsonSchemaStrategy(),
}


def strategy_for(dialect: StructuredOutputDialect) -> StructuredOutputStrategy:
    return _STRATEGIES[dialect]
