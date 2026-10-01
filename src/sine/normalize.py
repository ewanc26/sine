"""Deterministic normalisation and deduplication for listening events."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import re
import unicodedata

from sine.models import ListeningEvent, Track


def canonicalize_timestamp(value: datetime | str) -> datetime:
    """Return a timezone-aware UTC datetime with microseconds removed."""

    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        value = datetime.fromisoformat(text)

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0)


def normalize_string(value: str) -> str:
    """Normalise a display string for case-, punctuation-, and Unicode-insensitive matching."""

    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(char if char.isalnum() or char.isspace() else " " for char in value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_name(value: str) -> str:
    """Apply Unicode compatibility normalisation without changing display casing."""

    return unicodedata.normalize("NFKC", value)


def first_artist(event: ListeningEvent) -> str:
    return normalize_string(event.track.artists[0]) if event.track.artists else ""


def exact_key(event: ListeningEvent) -> tuple[str, str, datetime]:
    """Return the canonical artist/track/timestamp key used for exact deduplication."""

    return (
        first_artist(event),
        normalize_string(event.track.title),
        canonicalize_timestamp(event.played_at),
    )


def are_duplicates(
    left: ListeningEvent,
    right: ListeningEvent,
    *,
    tolerance_seconds: int = 60,
) -> bool:
    """Return whether two events represent the same listen."""

    if first_artist(left) != first_artist(right):
        return False
    if normalize_string(left.track.title) != normalize_string(right.track.title):
        return False
    delta = abs(
        (canonicalize_timestamp(left.played_at) - canonicalize_timestamp(right.played_at)).total_seconds()
    )
    return delta <= tolerance_seconds


def _richness(event: ListeningEvent) -> int:
    track = event.track
    return (
        len(track.artists) * 2
        + bool(track.album)
        + bool(track.duration_seconds)
        + bool(track.recording_mbid)
        + bool(track.release_mbid)
        + bool(track.artist_mbids)
        + bool(track.isrc)
        + bool(event.source_id)
    )


def merge_events(events: Iterable[ListeningEvent], *, tolerance_seconds: int = 60) -> list[ListeningEvent]:
    """Deduplicate and merge events while retaining the richest known metadata."""

    ordered = sorted(events, key=lambda event: canonicalize_timestamp(event.played_at))
    unique: list[ListeningEvent] = []

    for event in ordered:
        event = event.model_copy(update={"played_at": canonicalize_timestamp(event.played_at)})
        match_index = next(
            (
                index
                for index, existing in enumerate(unique)
                if are_duplicates(existing, event, tolerance_seconds=tolerance_seconds)
            ),
            None,
        )
        if match_index is None:
            unique.append(event)
            continue

        existing = unique[match_index]
        unique[match_index] = _merge_event(existing, event)

    return unique


def _merge_event(left: ListeningEvent, right: ListeningEvent) -> ListeningEvent:
    """Merge non-conflicting metadata, preferring populated values."""

    primary, secondary = sorted((left, right), key=_richness, reverse=True)
    track = primary.track
    other = secondary.track

    artists = track.artists or other.artists
    if not artists:
        artists = tuple(dict.fromkeys((*track.artists, *other.artists)))

    merged_track = track.model_copy(
        update={
            "artists": artists,
            "album": track.album or other.album,
            "duration_seconds": track.duration_seconds or other.duration_seconds,
            "recording_mbid": track.recording_mbid or other.recording_mbid,
            "release_mbid": track.release_mbid or other.release_mbid,
            "artist_mbids": track.artist_mbids or other.artist_mbids,
            "isrc": track.isrc or other.isrc,
        }
    )
    metadata = {**secondary.source_metadata, **primary.source_metadata}
    return primary.model_copy(
        update={
            "track": merged_track,
            "source_metadata": metadata,
            "source_id": primary.source_id or secondary.source_id,
        }
    )


def deduplicate_events(events: Iterable[ListeningEvent]) -> list[ListeningEvent]:
    """Remove exact duplicates using the canonical artist/track/timestamp key."""

    seen: dict[tuple[str, str, datetime], ListeningEvent] = {}
    for event in events:
        key = exact_key(event)
        seen.setdefault(key, event)
    return list(seen.values())
