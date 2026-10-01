"""Recommendation requests and validated recommendations.

A :class:`RecommendationRequest` states what the caller wants. A
:class:`RecommendationSet` is the validated result. The recommendation engine is
responsible for labelling its own output: a model may propose a track, but Sine
decides whether that track is new, familiar, or a replay, because those are
facts about the history rather than claims about taste.

A playlist is the same set with an order attached: :class:`PlaylistRequest` asks
for a sequence, and :class:`Playlist` is the validated result of that. Ordering is
the model's contribution; the positions that survive validation are Sine's.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from sine.models.base import HistoryModel
from sine.models.music import Artist, Track


class RecommendationFocus(StrEnum):
    """The intent a caller is asking the engine to serve."""

    DISCOVERY = "discovery"
    DEEPENING = "deepening"
    RECENT_ROTATION = "recent_rotation"
    FAMILIARITY = "familiarity"
    SURPRISE = "surprise"


class Novelty(StrEnum):
    """Relationship between a recommendation and the observed history."""

    NEW_ARTIST = "new_artist"
    KNOWN_ARTIST_NEW_TRACK = "known_artist_new_track"
    REPLAY = "replay"


class Confidence(StrEnum):
    """The recommender's own stated uncertainty."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EvidenceKind(StrEnum):
    """Whether a claim rests on recorded behaviour or on interpretation."""

    OBSERVED = "observed"
    INFERRED = "inferred"


class RecommendationRequest(HistoryModel):
    """What the caller wants recommended, and on what terms."""

    limit: int = Field(default=10, ge=1, le=50)
    focus: RecommendationFocus = RecommendationFocus.DISCOVERY
    allow_replays: bool = Field(
        default=False,
        description="Permit tracks already present in the listening history.",
    )
    seed_artists: tuple[Artist, ...] = Field(
        default_factory=tuple,
        description="Optional artists the caller wants recommendations anchored to.",
    )
    exclude_artists: tuple[Artist, ...] = Field(default_factory=tuple)
    guidance: str | None = Field(
        default=None,
        description="Free-text steer from the caller, passed to the model verbatim.",
    )


class RecommendationEvidence(HistoryModel):
    """A supporting claim, explicitly marked as observation or inference."""

    kind: EvidenceKind
    statement: str = Field(min_length=1)
    references: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Names of history entities the claim refers to.",
    )


class Recommendation(HistoryModel):
    """A single validated recommendation."""

    track: Track
    rationale: str = Field(
        min_length=1, description="Why this track, in one paragraph."
    )
    evidence: tuple[RecommendationEvidence, ...] = Field(default_factory=tuple)
    confidence: Confidence = Confidence.MEDIUM
    genre_hints: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Genre or style descriptors offered by the model. Advisory only; "
            "Sine is not a music database and does not verify these."
        ),
    )
    novelty: Novelty = Novelty.NEW_ARTIST
    known_plays: int = Field(
        default=0,
        ge=0,
        description="Plays of this track in the source history, computed by Sine.",
    )


class RecommendationSet(HistoryModel):
    """The validated output of one recommendation request.

    This is the shape requested from the model and validated on return.
    """

    recommendations: tuple[Recommendation, ...] = Field(default_factory=tuple)
    notes: str | None = Field(
        default=None, description="Model commentary on the set as a whole."
    )


class PlaylistRequest(RecommendationRequest):
    """A request for an ordered sequence rather than an unordered set.

    Everything a recommendation request carries still applies. A playlist adds two
    things: an intent for the sequence as a whole, and an optional length target.

    The length target is a request, not a measurement. Sine does not know how long
    the recommended tracks are, so it neither states nor verifies their durations; it
    sizes the request and tells the model how many tracks to aim for.
    """

    title: str | None = Field(
        default=None, description="Title or theme the listener asked for."
    )
    target_minutes: int | None = Field(
        default=None,
        ge=1,
        le=600,
        description=(
            "Approximate length to aim for. Sine cannot verify track durations, so "
            "this sizes the request rather than being enforced."
        ),
    )


class PlaylistTrack(Recommendation):
    """One recommendation placed at a position in a sequence.

    The position is the model's ordering claim. Sine renumbers the positions it
    keeps, so the sequence it returns is always contiguous from one, whatever the
    model returned or whatever Sine dropped.
    """

    position: int = Field(ge=1, description="One-based place in the sequence.")
    transition: str | None = Field(
        default=None,
        description=(
            "Why this track follows the previous one. Omit for the opening track, "
            "where there is nothing to transition from."
        ),
    )


class Playlist(HistoryModel):
    """The validated output of one playlist request."""

    title: str = Field(min_length=1)
    intent: str | None = Field(
        default=None, description="What the sequence is trying to do, in a paragraph."
    )
    tracks: tuple[PlaylistTrack, ...] = Field(default_factory=tuple)
    notes: str | None = Field(
        default=None, description="Model commentary on the playlist as a whole."
    )
