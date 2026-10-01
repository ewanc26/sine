"""Provider-neutral generation requests, responses, and token accounting.

Every field here is the intersection of what the researched API families can
express. Anything a family does not support is discovered through
:class:`~sine.llm.capabilities.ModelCapabilities` rather than being encoded in
this module.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from sine.llm.messages import Message


class FinishReason(StrEnum):
    """Normalised stop reasons.

    Providers disagree on vocabulary and case: ``end_turn``, ``STOP``, ``length``,
    ``MAX_TOKENS``, ``content_filter``, ``SAFETY``. Adapters map their own
    vocabulary into this enum so that callers never branch on provider strings.
    """

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    OTHER = "other"
    UNKNOWN = "unknown"


class TokenUsage(BaseModel):
    """Token accounting, where the provider reports it.

    Providers count differently and some report nothing. Every field is optional
    and the total is computed rather than trusted, because cached and reasoning
    tokens are counted inconsistently across providers.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(
        default=None, ge=0, description="Subset of input tokens served from cache."
    )
    reasoning_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Output tokens spent on reasoning, when reported separately.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int | None:
        known = [
            value
            for value in (self.input_tokens, self.output_tokens)
            if value is not None
        ]
        if not known:
            return None
        return sum(known)

    def __bool__(self) -> bool:
        return self.total_tokens is not None


class TextFormat(BaseModel):
    """Ask for unconstrained text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["text"] = "text"


class JsonObjectFormat(BaseModel):
    """Ask for a JSON object, without constraining its shape."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["json_object"] = "json_object"


class JsonSchemaFormat(BaseModel):
    """Ask for JSON conforming to ``schema``.

    The provider is expected to enforce the schema during decoding. Sine validates
    the result regardless, because enforcement is not guaranteed and some
    providers accept the field without honouring it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["json_schema"] = "json_schema"
    json_schema: dict[str, Any] = Field(
        description="JSON Schema the output must satisfy."
    )
    name: str = Field(default="response", min_length=1)
    strict: bool = True


ResponseFormat = Annotated[
    TextFormat | JsonObjectFormat | JsonSchemaFormat,
    Field(discriminator="kind"),
]


class GenerationRequest(BaseModel):
    """A provider-neutral generation request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: tuple[Message, ...] = Field(min_length=1)
    response_format: ResponseFormat = Field(default_factory=TextFormat)
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stop_sequences: tuple[str, ...] = Field(default_factory=tuple)

    def with_messages(self, messages: tuple[Message, ...]) -> GenerationRequest:
        return self.model_copy(update={"messages": messages})


class GenerationResponse(BaseModel):
    """A provider-neutral generation response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    provider: str
    model: str
    finish_reason: FinishReason = FinishReason.UNKNOWN
    usage: TokenUsage | None = None
    request_id: str | None = Field(
        default=None, description="Provider request identifier, for support and logs."
    )
