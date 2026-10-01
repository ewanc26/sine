"""The recommendation engine.

Takes a listening profile, a request, and any
:class:`~sine.llm.provider.Model`; returns validated recommendations. It contains
no provider knowledge at all: what it sends and how it reads the reply is decided
by the model's advertised capabilities, and every failure is a
:class:`~sine.llm.errors.ProviderError` or a :class:`RecommendationError`.

Three things happen here that a model cannot be trusted to do itself:

1. the response is parsed and validated against a Pydantic model;
2. a bounded repair attempt is made if the reply was unusable;
3. novelty labels and play counts are computed from the history, not taken from the
   model's word for it.
"""

from __future__ import annotations

from typing import Any

from sine.llm.capabilities import StructuredOutputDialect
from sine.llm.errors import ProviderError
from sine.llm.generation import (
    GenerationRequest,
    JsonObjectFormat,
    JsonSchemaFormat,
    ResponseFormat,
    TextFormat,
)
from sine.llm.messages import Message
from sine.llm.provider import Model
from sine.llm.structured import describe_schema, ensure_strict
from sine.models.music import Track
from sine.models.profile import ListeningProfile
from sine.models.recommendation import (
    Novelty,
    Recommendation,
    RecommendationRequest,
    RecommendationSet,
)
from sine.models.statistics import ListeningStatistics
from sine.recommend.prompts import (
    build_json_reminder,
    build_system_prompt,
    build_user_prompt,
)
from sine.recommend.validation import ResponseParseError, parse_model_payload

DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_OUTPUT_TOKENS = 4096


class RecommendationError(Exception):
    """Recommendations could not be produced or validated."""


def recommendation_schema() -> dict[str, Any]:
    """The JSON Schema requested from the model.

    Built from the validated domain model rather than hand-written, so the request
    and the response can never drift apart.
    """

    return ensure_strict(RecommendationSet.model_json_schema())


def _response_format_for(
    dialect: StructuredOutputDialect, schema: dict[str, Any]
) -> tuple[ResponseFormat, bool]:
    """Choose a request format, and whether the prompt must also carry the schema."""

    match dialect:
        case StructuredOutputDialect.JSON_SCHEMA:
            return JsonSchemaFormat(json_schema=schema, name="recommendations"), False
        case StructuredOutputDialect.JSON_OBJECT:
            return JsonObjectFormat(), True
        case _:
            return TextFormat(), True


class RecommendationEngine:
    """Produce validated recommendations from a profile."""

    def __init__(
        self,
        model: Model,
        *,
        temperature: float | None = DEFAULT_TEMPERATURE,
        max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS,
        max_repairs: int = 1,
    ) -> None:
        if model.capabilities.structured_output is StructuredOutputDialect.NONE:
            raise RecommendationError(
                f"{model.qualified_id} cannot be asked for JSON, so Sine cannot use it "
                "for recommendations; choose a model with structured output support"
            )
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._max_repairs = max(0, max_repairs)

    @property
    def model(self) -> Model:
        return self._model

    def recommend(
        self,
        profile: ListeningProfile,
        request: RecommendationRequest,
    ) -> RecommendationSet:
        """Recommend tracks for ``request``, using ``profile`` as grounding."""

        schema = recommendation_schema()
        response_format, schema_in_prompt = _response_format_for(
            self._model.capabilities.structured_output, schema
        )

        messages: tuple[Message, ...] = (
            build_system_prompt(
                include_schema=describe_schema(schema) if schema_in_prompt else None
            ),
            build_user_prompt(profile, request),
        )

        generation = GenerationRequest(
            messages=messages,
            response_format=response_format,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )

        last_error: Exception | None = None
        for attempt in range(self._max_repairs + 1):
            try:
                response = self._model.generate(generation)
            except ProviderError as exc:
                # Transport and provider failures are not the model's fault to fix,
                # so there is nothing to repair; surface them as they are.
                raise RecommendationError(
                    f"{self._model.qualified_id} could not be used: {exc}"
                ) from exc

            try:
                proposed = parse_model_payload(response.text, RecommendationSet)
            except ResponseParseError as exc:
                last_error = exc
                if attempt == self._max_repairs:
                    break
                generation = generation.with_messages(
                    messages
                    + (Message.assistant(response.text), build_json_reminder(schema))
                )
                continue

            return self._finalise(proposed, profile, request)

        raise RecommendationError(
            f"the model did not return a usable recommendation set after "
            f"{self._max_repairs + 1} attempt(s): {last_error}"
        )

    def _finalise(
        self,
        proposed: RecommendationSet,
        profile: ListeningProfile,
        request: RecommendationRequest,
    ) -> RecommendationSet:
        """Apply the rules the model was asked to follow but cannot be trusted on.

        Novelty, play counts, exclusions, and the requested count are all decided
        here, from the history, so that a model cannot overstate how well it
        followed the brief.
        """

        excluded = {artist.identity for artist in request.exclude_artists}
        stats = profile.statistics

        seen: set[str] = set()
        accepted: list[Recommendation] = []
        dropped = 0

        for recommendation in proposed.recommendations:
            track = recommendation.track
            identity = track.identity
            artist_identity = track.artist.identity

            if identity in seen:
                dropped += 1
                continue
            seen.add(identity)

            if artist_identity in excluded:
                dropped += 1
                continue

            known_plays = _known_plays(stats, track)
            if known_plays and not request.allow_replays:
                dropped += 1
                continue

            if known_plays:
                novelty = Novelty.REPLAY
            elif _artist_plays(stats, artist_identity) > 0:
                novelty = Novelty.KNOWN_ARTIST_NEW_TRACK
            else:
                novelty = Novelty.NEW_ARTIST

            accepted.append(
                recommendation.model_copy(
                    update={"novelty": novelty, "known_plays": known_plays}
                )
            )
            if len(accepted) == request.limit:
                break

        notes = proposed.notes
        if dropped:
            reason = (
                f"{dropped} suggestion(s) were removed by Sine: already played, "
                "excluded, or duplicated."
            )
            notes = f"{notes}\n{reason}" if notes else reason

        if len(proposed.recommendations) < request.limit and not dropped:
            shortfall = request.limit - len(proposed.recommendations)
            reason = (
                f"The model returned {len(proposed.recommendations)} of the "
                f"{request.limit} tracks requested, {shortfall} short."
            )
            notes = f"{notes}\n{reason}" if notes else reason

        return RecommendationSet(recommendations=tuple(accepted), notes=notes)


def _artist_plays(stats: ListeningStatistics, artist_identity: str) -> int:
    """Plays for an artist, from the uncapped tally rather than the top-N list."""

    return stats.artist_plays_by_identity.get(artist_identity, 0)


def _known_plays(stats: ListeningStatistics, track: Track) -> int:
    """Plays for a track. Must not come from ``top_tracks``, which is truncated:

    a track outside the top N was still played, and labelling it "new" would be a
    claim the data contradicts.
    """

    return stats.track_plays.get(track.identity, 0)


__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_TEMPERATURE",
    "RecommendationEngine",
    "RecommendationError",
    "recommendation_schema",
]
