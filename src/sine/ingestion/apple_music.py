"""Apple Music Play Activity CSV ingestion."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any

from sine.models import ListeningEvent, Track
from sine.normalize import canonicalize_timestamp, normalize_name


TITLE_COLUMNS = ("Content Name", "Song Name")
ARTIST_COLUMNS = ("Artist Name", "Container Artist Name")
ALBUM_COLUMNS = ("Album Name", "Container Album Name")


class AppleMusicSchemaError(ValueError):
    """Raised when a CSV is not an Apple Music Play Activity export."""


def daily_tracks_artist_map(rows: list[dict[str, str]]) -> dict[str, str]:
    """Build an unambiguous title-to-artist fallback from Apple's daily export."""

    candidates: dict[str, set[str]] = {}
    for row in rows:
        description = (row.get("Track Description") or "").strip()
        if " - " not in description:
            continue
        artist, title = (part.strip() for part in description.split(" - ", 1))
        if artist and title:
            key = normalize_name(title).casefold()
            candidates.setdefault(key, set()).add(artist)

    return {key: next(iter(values)) for key, values in candidates.items() if len(values) == 1}


def _first(row: dict[str, str], columns: tuple[str, ...]) -> str | None:
    for column in columns:
        value = row.get(column)
        if value and value.strip():
            return value.strip()
    return None


def parse(content: str, artist_lookup: dict[str, str] | None = None) -> list[ListeningEvent]:
    """Parse Apple's per-play CSV while tolerating export schema generations."""

    if content.startswith("\ufeff"):
        content = content[1:]
    reader = csv.DictReader(StringIO(content))
    rows = list(reader)
    if rows:
        fields = {field.strip().lstrip("\ufeff") for field in reader.fieldnames or ()}
        if not any(column in fields for column in TITLE_COLUMNS):
            raise AppleMusicSchemaError(
                f"Expected one of {TITLE_COLUMNS}; found {sorted(fields)}"
            )

    events: list[ListeningEvent] = []
    for row in rows:
        title = _first(row, TITLE_COLUMNS)
        timestamp = _first(row, ("Event End Timestamp", "Event Start Timestamp"))
        if not title or not timestamp:
            continue

        artist = _first(row, ARTIST_COLUMNS)
        if not artist and artist_lookup:
            artist = artist_lookup.get(normalize_name(title).casefold())

        duration_ms = _first(row, ("Media Duration In Milliseconds",))
        duration = int(float(duration_ms) / 1000) if duration_ms and float(duration_ms) > 0 else None

        track = Track(
            title=normalize_name(title),
            artists=(normalize_name(artist),) if artist else (),
            album=normalize_name(_first(row, ALBUM_COLUMNS) or "")
            if _first(row, ALBUM_COLUMNS)
            else None,
            duration_seconds=duration,
        )
        raw_time = timestamp.replace(" ", "T", 1) if "T" not in timestamp else timestamp
        if not raw_time.endswith("Z") and "+" not in raw_time:
            raw_time += "Z"

        events.append(
            ListeningEvent(
                track=track,
                played_at=canonicalize_timestamp(raw_time),
                source="apple_music",
                source_metadata={
                    "origin": "apple-music-play-activity",
                    "country": row.get("ISO Country"),
                    "item_type": row.get("Item Type"),
                },
            )
        )
    return events


def load(path: Path, artist_lookup: dict[str, str] | None = None) -> list[ListeningEvent]:
    return parse(path.read_text(encoding="utf-8-sig"), artist_lookup)
