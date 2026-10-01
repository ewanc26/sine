"""Spotify listening-history JSON ingestion.

A Spotify account-data export mixes music with podcast episodes and audiobooks in
the same array, distinguished only by which metadata fields are populated. The
adapter keeps music and reports the rest as rejected records rather than
silently discarding them, so the count of non-music rows stays visible.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sine.ingestion.normalisation import FieldMapping
from sine.ingestion.records import RawRecord

_TRACK_NAME = "master_metadata_track_name"
_ARTIST_NAME = "master_metadata_album_artist_name"
_ALBUM_NAME = "master_metadata_album_album_name"


def _is_music(record: dict[str, Any]) -> bool:
    return bool(record.get(_TRACK_NAME) and record.get(_ARTIST_NAME))


def _reason_rejected(record: dict[str, Any]) -> str | None:
    if _is_music(record):
        return None
    if record.get("episode_name"):
        return "podcast episode, not a music play"
    if record.get("audiobook_title") or record.get("chapter"):
        return "audiobook, not a music play"
    if record.get("spotify_episode_uri"):
        return "podcast episode, not a music play"
    return "no track name or album artist in record"


class SpotifySource:
    """Reads a Spotify ``StreamingHistory*.json`` export."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def source_id(self) -> str:
        return "spotify"

    def field_mapping(self) -> FieldMapping:
        return FieldMapping(
            track_title=_TRACK_NAME,
            artist=_ARTIST_NAME,
            played_at="ts",
            album=_ALBUM_NAME,
            # Deliberately no duration: Spotify's "ms_played" is how long playback
            # lasted, not the track's length, so recording it as track duration
            # would state something the source did not report.
            event_id=("spotify_track_uri", "spotify_episode_uri"),
            extra_fields=("platform", "country", "episode_name", "reason_start"),
        )

    def fetch(self) -> Iterable[RawRecord]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        records = [row for row in payload if isinstance(row, dict)]

        results: list[RawRecord] = []
        for index, record in enumerate(records):
            reason = _reason_rejected(record)
            results.append(
                RawRecord(
                    source=self.source_id,
                    index=index,
                    line_number=index + 1,
                    source_path=self.path,
                    fields=record if reason is None else {},
                    error=reason,
                )
            )
        return results
