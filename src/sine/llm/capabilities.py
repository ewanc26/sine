"""Capability discovery.

The core needs to know, before it calls a model, what that model can be asked to
do. Expressing structured output as a *dialect* rather than a boolean matters in
practice: DeepSeek rejects ``json_schema`` outright, Groq separates models that
support constrained decoding from those that do not, and NVIDIA expects
``guided_json`` under ``extra_body`` rather than ``response_format``. A boolean
cannot represent that; a dialect can.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class StructuredOutputDialect(StrEnum):
    """How a model can be constrained to produce JSON.

    Ordered from strongest to weakest. An adapter advertises the strongest
    dialect it can actually deliver.
    """

    JSON_SCHEMA = "json_schema"
    """The provider constrains decoding to the supplied JSON Schema."""

    JSON_OBJECT = "json_object"
    """The provider guarantees a JSON object, but not its shape.

    The schema must also be described in the prompt.
    """

    PROMPT = "prompt"
    """No API-level constraint. The schema is described in the prompt and the
    response is parsed leniently."""

    NONE = "none"
    """The model cannot be asked for JSON. Sine will not attempt structured
    recommendations against such a model."""


class ModelCapabilities(BaseModel):
    """What a specific model on a specific provider supports."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    structured_output: StructuredOutputDialect = StructuredOutputDialect.NONE
    supports_system_messages: bool = True
    reports_token_usage: bool = True
    context_window: int | None = Field(default=None, ge=1)
    notes: str | None = Field(
        default=None, description="Human-readable caveat shown by the CLI."
    )

    def supports_structured_output(self) -> bool:
        return self.structured_output is not StructuredOutputDialect.NONE


class ModelInfo(BaseModel):
    """A model as reported by a provider's catalogue endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    display_name: str | None = None
    owned_by: str | None = None
    context_window: int | None = Field(default=None, ge=1)

    @property
    def label(self) -> str:
        return self.display_name or self.model_id
