"""Listening events and listening history.

A :class:`ListeningEvent` is a single observation: "this track was played at this
instant, as reported by this source". The source label and the source's own
event identifier belong here, at the ingestion boundary, rather than on the
track, so that the domain model stays service-neutral.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from pydantic import Field, field_validator, model_validator

from sine.models.base import HistoryModel, TimeWindow, aware_utc
from sine.models.music import Track


class ListeningEvent(HistoryModel):
    """One observed play."""

    track: Track
    played_at: datetime = Field(description="When the play occurred, timezone-aware.")
    source: str = Field(
        min_length=1,
        description="Ingestion source that reported the play, e.g. 'apple-music'.",
    )
    source_event_id: str | None = Field(
        default=None,
        description="Identifier assigned by the source, when it provides one.",
    )
    play_count: int = Field(
        default=1,
        ge=1,
        description="Number of plays represented by this observation.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Source-supplied attributes Sine does not model, retained verbatim "
            "as strings so that nothing observed is discarded or invented."
        ),
    )

    @field_validator("played_at")
    @classmethod
    def _require_aware(cls, value: datetime) -> datetime:
        return aware_utc(value)

    @property
    def identity(self) -> str:
        """Source-aware identity, used for de-duplication."""

        if self.source_event_id is not None:
            return f"{self.source}\x1f{self.source_event_id}"
        return f"{self.source}\x1f{self.track.identity}\x1f{self.played_at.isoformat()}"

    @property
    def key(self) -> tuple[str, str]:
        return (self.track.identity, self.played_at.isoformat())


def sort_events(events: Iterable[ListeningEvent]) -> tuple[ListeningEvent, ...]:
    """Return events ordered by play time, then by source and track identity.

    Sorting is total and stable so that downstream statistics are deterministic
    regardless of the order a source happened to return events in.
    """

    return tuple(
        sorted(
            events,
            key=lambda event: (event.played_at, event.source, event.track.identity),
        )
    )


def deduplicate_events(events: Iterable[ListeningEvent]) -> tuple[ListeningEvent, ...]:
    """Collapse duplicate observations, summing repeated ``play_count``.

    A source that re-exports overlapping windows reports the same play twice. When
    the same play is reported more than once, the counts are added rather than
    discarded, because two exports of ``play_count=1`` may genuinely be two plays
    that the source chose not to give a stable identifier for.
    """

    merged: dict[str, ListeningEvent] = {}
    order: list[str] = []
    for event in events:
        existing = merged.get(event.identity)
        if existing is None:
            merged[event.identity] = event
            order.append(event.identity)
            continue
        merged[event.identity] = existing.model_copy(
            update={"play_count": existing.play_count + event.play_count}
        )
    return sort_events(merged[identity] for identity in order)


def normalise_events(events: Iterable[ListeningEvent]) -> tuple[ListeningEvent, ...]:
    """De-duplicate and order a batch of observations into a canonical sequence."""

    return deduplicate_events(events)


def same_listen(
    left: ListeningEvent,
    right: ListeningEvent,
    *,
    tolerance_seconds: int = 60,
) -> bool:
    """Whether two observations plausibly describe one listen.

    Two services watching the same player report the same play with slightly
    different timestamps: Last.fm may round to the second, a radio stream start may
    lag the track start, and one service may be a minute into the delay. Comparing
    within a tolerance recognises those, whereas exact-timestamp equality does not.
    """

    if left.track.identity != right.track.identity:
        return False
    delta = abs((left.played_at - right.played_at).total_seconds())
    return delta <= tolerance_seconds


def _richness(event: ListeningEvent) -> tuple[int, int, int]:
    """Rank how much metadata an observation carries.

    Ordered so the most trustworthy fields dominate: a MusicBrainz ID identifies
    the recording outright, then the number of credits and release, then the
    source's own event ID.
    """

    track = event.track
    return (
        len(track.artists) * 4
        + bool(track.recording_mbid) * 8
        + bool(track.album) * 2
        + bool(track.duration_seconds)
        + bool(track.release_mbid)
        + bool(track.isrc)
        + bool(event.source_event_id),
        1 if event.source_event_id else 0,
        event.play_count,
    )


def _merge_pair(primary: ListeningEvent, secondary: ListeningEvent) -> ListeningEvent:
    """Combine two observations of one listen, preferring populated values."""

    primary, secondary = sorted((primary, secondary), key=_richness, reverse=True)
    left, right = primary.track, secondary.track

    def pick(attribute: str) -> object:
        value = getattr(left, attribute)
        return value if value else getattr(right, attribute)

    merged_track = left.model_copy(
        update={
            "artists": left.artists or right.artists,
            "album": pick("album"),
            "duration_seconds": pick("duration_seconds"),
            "isrc": pick("isrc"),
            "recording_mbid": pick("recording_mbid"),
            "release_mbid": pick("release_mbid"),
        }
    )
    return primary.model_copy(
        update={
            "track": merged_track,
            "play_count": primary.play_count + secondary.play_count,
            "source_event_id": primary.source_event_id or secondary.source_event_id,
            "metadata": {**secondary.metadata, **primary.metadata},
        }
    )


def merge_duplicate_events(
    events: Iterable[ListeningEvent], *, tolerance_seconds: int = 60
) -> tuple[ListeningEvent, ...]:
    """Collapse observations of the same listen, combining their metadata.

    Unlike :func:`deduplicate_events`, which needs a stable source identifier,
    this matches on track and a time window, so it also reconciles overlapping
    exports from different services. Play counts are summed, because two services
    reporting the same play are two reports of one play, not two plays.
    """

    ordered = sort_events(events)
    merged: list[ListeningEvent] = []

    for event in ordered:
        match_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if same_listen(existing, event, tolerance_seconds=tolerance_seconds)
            ),
            None,
        )
        if match_index is None:
            merged.append(event)
        else:
            merged[match_index] = _merge_pair(merged[match_index], event)

    return tuple(merged)


class ListeningHistory(HistoryModel):
    """An ordered, de-duplicated set of listening observations."""

    events: tuple[ListeningEvent, ...] = Field(default_factory=tuple)
    notes: str | None = Field(
        default=None,
        description="Provenance note about the import, kept with the data.",
    )

    @model_validator(mode="after")
    def _canonical_order(self) -> ListeningHistory:
        if self.events and self.events != sort_events(self.events):
            return self.model_copy(update={"events": sort_events(self.events)})
        return self

    @classmethod
    def from_events(
        cls, events: Iterable[ListeningEvent], *, notes: str | None = None
    ) -> ListeningHistory:
        """Build a history from raw observations, normalising as it goes."""

        return cls(events=normalise_events(events), notes=notes)

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def total_plays(self) -> int:
        return sum(event.play_count for event in self.events)

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted({event.source for event in self.events}))

    @property
    def window(self) -> TimeWindow | None:
        """The observed time span, or ``None`` for an empty history."""

        if not self.events:
            return None
        first = self.events[0].played_at
        last = self.events[-1].played_at
        return TimeWindow(start=min(first, last), end=max(first, last))

    def in_window(self, window: TimeWindow) -> tuple[ListeningEvent, ...]:
        return tuple(event for event in self.events if window.contains(event.played_at))

    def slice(self, start: datetime, end: datetime) -> ListeningHistory:
        """Return a history restricted to ``[start, end]``."""

        window = TimeWindow(start=start, end=end)
        return ListeningHistory(events=self.in_window(window), notes=self.notes)

    def is_empty(self) -> bool:
        return not self.events
