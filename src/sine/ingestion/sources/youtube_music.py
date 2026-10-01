"""YouTube Music Google Takeout ingestion.

A Takeout archive mixes YouTube Music plays with ordinary YouTube video watches in
one array. They are told apart by the ``header`` field; the artist is the first
subtitle, except when that subtitle is a channel URL, which identifies a channel
rather than a performing artist and is therefore left unset rather than invented.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sine.ingestion.normalisation import FieldMapping
from sine.ingestion.records import RawRecord

_WATCHED_PREFIX = "Watched "


def _prepare(record: dict[str, Any]) -> tuple[dict[str, object] | None, str | None]:
    """Return ``(fields, rejection_reason)`` for one Takeout row."""

    header = record.get("header")
    if header != "YouTube Music":
        return None, "not a YouTube Music record"

    raw_title = record.get("title")
    if not isinstance(raw_title, str) or not raw_title.startswith(_WATCHED_PREFIX):
        return None, "not a play (no 'Watched' title)"

    title = raw_title[len(_WATCHED_PREFIX) :].strip()
    if not title:
        return None, "empty title after 'Watched' prefix"

    time = record.get("time")
    if not isinstance(time, str) or not time:
        return None, "no timestamp"

    subtitles = record.get("subtitles")
    artist = None
    if isinstance(subtitles, list) and subtitles and isinstance(subtitles[0], dict):
        name = subtitles[0].get("name")
        if isinstance(name, str) and name.strip():
            # A music.youtube.com subtitle is the channel, not the artist. Leaving
            # it unset is honest; recording it would attribute the track to a
            # channel that may not have performed it.
            artist = None if "music.youtube.com" in name else name.strip()

    prepared: dict[str, object] = {"track_title": title, "played_at": time}
    if artist:
        prepared["artist"] = artist
    url = record.get("titleUrl")
    if isinstance(url, str) and url:
        prepared["source_event_id"] = url
    return prepared, None


class YouTubeMusicSource:
    """Reads a YouTube Music activity export from Google Takeout."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def source_id(self) -> str:
        return "youtube-music"

    def field_mapping(self) -> FieldMapping:
        return FieldMapping(
            track_title="track_title",
            artist="artist",
            played_at="played_at",
            event_id="source_event_id",
        )

    def fetch(self) -> Iterable[RawRecord]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        rows = [row for row in payload if isinstance(row, dict)]

        records: list[RawRecord] = []
        for index, row in enumerate(rows):
            prepared, reason = _prepare(row)
            records.append(
                RawRecord(
                    source=self.source_id,
                    index=index,
                    line_number=index + 1,
                    source_path=self.path,
                    fields=prepared if prepared is not None else {},
                    error=reason,
                )
            )
        return records
