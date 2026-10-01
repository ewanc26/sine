"""Last.fm CSV ingestion."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

from sine.models import ListeningEvent, Track
from sine.normalize import canonicalize_timestamp, normalize_name


_COLUMN_MAP = {
    "uts": "uts",
    "date": "uts",
    "timestamp": "uts",
    "played_at": "uts",
    "time": "uts",
    "artist": "artist",
    "artist_name": "artist",
    "artistname": "artist",
    "artist_mbid": "artist_mbid",
    "artistmbid": "artist_mbid",
    "artist_id": "artist_mbid",
    "album": "album",
    "album_name": "album",
    "albumname": "album",
    "release": "album",
    "album_mbid": "album_mbid",
    "albummbid": "album_mbid",
    "album_id": "album_mbid",
    "track": "track",
    "track_name": "track",
    "trackname": "track",
    "song": "track",
    "title": "track",
    "track_mbid": "track_mbid",
    "trackmbid": "track_mbid",
    "track_id": "track_mbid",
    "utc_time": "utc_time",
    "utctime": "utc_time",
    "datetime": "utc_time",
}


def _delimiter(text: str) -> str:
    line = next((line for line in text.splitlines() if line.strip()), "")
    return max((",", ";", "\t", "|"), key=lambda item: line.count(item))


def _timestamp(row: dict[str, str]) -> object:
    raw = (row.get("uts") or "").strip()
    if raw:
        value = int(raw)
        if len(raw) >= 13:
            value //= 1000
        from datetime import datetime, timezone
        return datetime.fromtimestamp(value, tz=timezone.utc)
    return canonicalize_timestamp(row["utc_time"])


def parse(content: str) -> list[ListeningEvent]:
    """Parse a Last.fm CSV export into canonical listening events."""

    if content.startswith("\ufeff"):
        content = content[1:]
    reader = csv.DictReader(StringIO(content), delimiter=_delimiter(content))
    events: list[ListeningEvent] = []

    for raw in reader:
        normalised: dict[str, str] = {}
        for key, value in raw.items():
            if key is None:
                continue
            mapped = _COLUMN_MAP.get(key.lower().split("#", 1)[0].strip())
            if mapped:
                normalised[mapped] = (value or "").strip()

        if not normalised.get("artist") or not normalised.get("track"):
            continue
        if not normalised.get("uts") and not normalised.get("utc_time"):
            continue

        artist = normalize_name(normalised["artist"])
        track = Track(
            title=normalize_name(normalised["track"]),
            artists=(artist,),
            album=normalize_name(normalised["album"]) if normalised.get("album") else None,
            recording_mbid=normalised.get("track_mbid") or None,
            release_mbid=normalised.get("album_mbid") or None,
            artist_mbids=(normalised["artist_mbid"],) if normalised.get("artist_mbid") else (),
        )
        events.append(
            ListeningEvent(
                track=track,
                played_at=canonicalize_timestamp(_timestamp(normalised)),  # type: ignore[arg-type]
                source="lastfm",
                source_id=f"{artist}:{track.title}",
                source_metadata={"origin": "lastfm-export"},
            )
        )

    return events


def load(path: Path) -> list[ListeningEvent]:
    return parse(path.read_text(encoding="utf-8-sig"))
