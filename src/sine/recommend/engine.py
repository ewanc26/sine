"""The recommendation engine.

Takes a listening profile, a request, and any
:class:`~sine.llm.provider.Model`; returns validated recommendations. It contains
no provider knowledge at all: what it sends and how it reads the reply is decided
by the model's advertised capabilities, and every failure is a
:class:`~sine.llm.errors.ProviderError` or a :class:`RecommendationError`.

Four things happen here that a model cannot be trusted to do itself:

1. the response is parsed and validated against a Pydantic model;
2. a bounded repair attempt is made if the reply was unusable;
3. novelty labels and play counts are computed from the history, not taken from the
   model's word for it;
4. a playlist's positions are renumbered after screening, so the sequence Sine
   returns is contiguous whatever the model claimed.
"""

from __future__ import annotations

from collections.abc import Sequence
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
from sine.models.base import HistoryModel
from sine.models.music import Track
from sine.models.profile import ListeningProfile
from sine.models.recommendation import (
    Novelty,
    Playlist,
    PlaylistRequest,
    Recommendation,
    RecommendationRequest,
    RecommendationSet,
)
from sine.models.statistics import ListeningStatistics
from sine.recommend.prompts import (
    build_json_reminder,
    build_playlist_prompt,
    build_system_prompt,
    build_user_prompt,
)
from sine.recommend.validation import ResponseParseError, parse_model_payload

DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_OUTPUT_TOKENS = 4096

#: Roughly how many tracks fit in a given number of minutes, used only to size a
#: request. It is an estimate of conventional song length, not knowledge about any
#: particular track: Sine is told never to state durations.
MINUTES_PER_TRACK = 4.5


class RecommendationError(Exception):
    """Recommendations could not be produced or validated."""


def recommendation_schema() -> dict[str, Any]:
    """The JSON Schema requested from the model.

    Built from the validated domain model rather than hand-written, so the request
    and the response can never drift apart.
    """

    return ensure_strict(RecommendationSet.model_json_schema())


def playlist_schema() -> dict[str, Any]:
    """The JSON Schema requested when the caller wants an ordered sequence."""

    return ensure_strict(Playlist.model_json_schema())


def _response_format_for(
    dialect: StructuredOutputDialect, schema: dict[str, Any], shape: str
) -> tuple[ResponseFormat, bool]:
    """Choose a request format, and whether the prompt must also carry the schema."""

    match dialect:
        case StructuredOutputDialect.JSON_SCHEMA:
            return JsonSchemaFormat(json_schema=schema, name=shape), False
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

        proposed = self._ask(
            user_prompt=build_user_prompt(profile, request),
            schema=recommendation_schema(),
            response_model=RecommendationSet,
            shape="recommendations",
            label="recommendation set",
        )
        accepted, dropped = _screen(
            proposed.recommendations, profile.statistics, request
        )
        return RecommendationSet(
            recommendations=accepted,
            notes=_set_notes(
                proposed.notes, dropped, len(proposed.recommendations), request
            ),
        )

    def playlist(
        self,
        profile: ListeningProfile,
        request: PlaylistRequest,
    ) -> Playlist:
        """Build an ordered sequence for ``request``, using ``profile`` as grounding."""

        proposed = self._ask(
            user_prompt=build_playlist_prompt(profile, request),
            schema=playlist_schema(),
            response_model=Playlist,
            shape="playlist",
            label="playlist",
            playlist=True,
        )
        # The stated positions are the model's ordering claim; sorting by them makes
        # that claim the order Sine reasons about, whatever order the tracks arrived
        # in. Screening then removes what the history rules out, and the positions
        # are renumbered so the sequence the caller receives is contiguous.
        ordered = sorted(
            enumerate(proposed.tracks), key=lambda pair: (pair[1].position, pair[0])
        )
        accepted, dropped = _screen(
            [track for _, track in ordered], profile.statistics, request
        )
        tracks = tuple(
            track.model_copy(update={"position": position})
            for position, track in enumerate(accepted, start=1)
        )
        return Playlist(
            title=proposed.title,
            intent=proposed.intent,
            tracks=tracks,
            notes=_playlist_notes(
                proposed.notes,
                dropped,
                len(proposed.tracks),
                request,
                transitions=sum(1 for track in tracks if track.transition),
            ),
        )

    def _ask[OutT: HistoryModel](
        self,
        *,
        user_prompt: Message,
        schema: dict[str, Any],
        response_model: type[OutT],
        shape: str,
        label: str,
        playlist: bool = False,
    ) -> OutT:
        """Send one request and return a validated response, repairing once at most.

        The response format is chosen from what the model advertises, and the schema
        travels in the prompt only when the API cannot be constrained with it.
        """

        response_format, schema_in_prompt = _response_format_for(
            self._model.capabilities.structured_output, schema, shape
        )

        messages: tuple[Message, ...] = (
            build_system_prompt(
                include_schema=describe_schema(schema) if schema_in_prompt else None,
                playlist=playlist,
            ),
            user_prompt,
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
                return parse_model_payload(response.text, response_model)  # type: ignore[return-value]
            except ResponseParseError as exc:
                last_error = exc
                if attempt == self._max_repairs:
                    break
                generation = generation.with_messages(
                    messages
                    + (Message.assistant(response.text), build_json_reminder(schema))
                )
                continue

        raise RecommendationError(
            f"the model did not return a usable {label} after "
            f"{self._max_repairs + 1} attempt(s): {last_error}"
        )


def _screen[ItemT: Recommendation](
    candidates: Sequence[ItemT],
    stats: ListeningStatistics,
    request: RecommendationRequest,
) -> tuple[list[ItemT], int]:
    """Apply the rules the model was asked to follow but cannot be trusted on.

    Novelty, play counts, exclusions, and the requested count are all decided here,
    from the history, so that a model cannot overstate how well it followed the
    brief. Returns the accepted recommendations in the order given, and the number
    dropped.
    """

    excluded = {artist.identity for artist in request.exclude_artists}

    seen: set[str] = set()
    accepted: list[ItemT] = []
    dropped = 0

    for recommendation in candidates:
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

    return accepted, dropped


def _set_notes(
    notes: str | None, dropped: int, proposed_count: int, request: RecommendationRequest
) -> str | None:
    """Explain, in Sine's own words, anything the model did not deliver."""

    additions: list[str] = []
    if dropped:
        additions.append(
            f"{dropped} suggestion(s) were removed by Sine: already played, "
            "excluded, or duplicated."
        )
    if proposed_count < request.limit and not dropped:
        additions.append(
            f"The model returned {proposed_count} of the {request.limit} tracks "
            f"requested, {request.limit - proposed_count} short."
        )
    return _join_notes(notes, additions)


def _playlist_notes(
    notes: str | None,
    dropped: int,
    proposed_count: int,
    request: PlaylistRequest,
    *,
    transitions: int,
) -> str | None:
    """Same accounting as a set, plus what the sequence cannot be checked against.

    A length target the model was given and Sine cannot measure is the one number in
    a playlist most likely to be wrong, so it is stated as an unverified aim rather
    than quietly presented as the result.
    """

    additions: list[str] = []
    if dropped:
        additions.append(
            f"{dropped} track(s) were removed by Sine: already played, excluded, or "
            "duplicated. The remaining sequence has been renumbered."
        )
    if proposed_count < request.limit and not dropped:
        additions.append(
            f"The model returned {proposed_count} of the {request.limit} tracks "
            f"requested, {request.limit - proposed_count} short; Sine will not pad "
            "the gap with tracks of its own."
        )
    if transitions < max(0, proposed_count - dropped - 1):
        additions.append(
            f"{transitions} transition note(s) were supplied for "
            f"{max(0, proposed_count - dropped)} track(s); the sequence is ordered, "
            "but not every placement was explained."
        )
    if request.target_minutes:
        additions.append(
            f"The {request.target_minutes}-minute target is an aim, not a "
            "measurement: Sine has no duration data for these tracks, so the "
            "running time is unknown."
        )
    return _join_notes(notes, additions)


def _join_notes(model_notes: str | None, additions: list[str]) -> str | None:
    if not additions:
        return model_notes
    return "\n".join([model_notes, *additions]) if model_notes else "\n".join(additions)


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
    "MINUTES_PER_TRACK",
    "RecommendationEngine",
    "RecommendationError",
    "playlist_schema",
    "recommendation_schema",
]
