"""Historical listening-history import from JSON and JSON Lines files.

Sine's first ingestion path, and the template every other source follows: read
records, declare the field mapping, let normalisation do the rest.

Recognised layouts, selected with ``preset``:

``sine``
    Sine's own shape: ``{"title", "artist", "album", "played_at"}``.
``lastfm``
    Last.fm ``getRecentTracks`` style, where the track, artist, and album are
    nested objects and the time is ``date.uts`` (epoch seconds).
``listenbrainz``
    ListenBrainz listens, with ``ts`` in ISO 8601.
``spotify``
    A Spotify extended-history export, whose keys are long and self-describing.

The presets are field mappings and timestamp formats, not parsers. Any other
shape can be imported by passing a custom :class:`FieldMapping`; no code change
is needed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sine.ingestion.normalisation import FieldMapping
from sine.ingestion.records import RawRecord

PRESETS: dict[str, FieldMapping] = {
    "sine": FieldMapping(
        track_title="title",
        artist="artist",
        album="album",
        played_at="played_at",
        play_count="play_count",
    ),
    "lastfm": FieldMapping(
        # The Last.fm API nests names under "#text"; "name" is the shape used by
        # some third-party mirrors, so both are tried before giving up.
        track_title=("track.#text", "track.name"),
        artist=("artist.#text", "artist.name"),
        album=("album.#text", "album.name"),
        played_at=("date.uts", "date.#text"),
        event_id="date.uts",
        extra_fields=("artist.mbid", "album.mbid", "track.mbid"),
    ),
    "listenbrainz": FieldMapping(
        track_title="track_name",
        artist="artist_name",
        album="release_name",
        played_at="listened_at",
        event_id="inserted_at",
        extra_fields=("recording_mbid", "msid"),
    ),
    "spotify": FieldMapping(
        track_title="master_metadata_track_name",
        artist="master_metadata_album_artist_name",
        album="master_metadata_album_album_name",
        played_at="ts",
        duration_seconds="ms_played",
    ),
}


class JsonListeningHistorySource:
    """Read a listening history from a JSON array or JSON Lines file."""

    def __init__(
        self,
        path: str | Path,
        *,
        source_id: str = "json-import",
        preset: str = "sine",
        mapping: FieldMapping | None = None,
    ) -> None:
        self._path = Path(path)
        self._source_id = source_id
        if mapping is None:
            try:
                mapping = PRESETS[preset]
            except KeyError:
                known = ", ".join(sorted(PRESETS))
                raise ValueError(
                    f"unknown preset {preset!r}; known presets: {known}"
                ) from None
        self._mapping = mapping

    @property
    def source_id(self) -> str:
        return self._source_id

    def field_mapping(self) -> FieldMapping:
        return self._mapping

    def fetch(self) -> Iterator[RawRecord]:
        text = self._path.read_text(encoding="utf-8")
        stripped = text.lstrip()
        if stripped.startswith("["):
            yield from self._from_array(json.loads(text))
        else:
            yield from self._from_lines(text)

    def _from_array(self, payload: Any) -> Iterator[RawRecord]:
        if not isinstance(payload, list):
            # The file parsed as JSON but has the wrong shape; that is a bad input
            # file rather than a wrong type handed in by a caller.
            raise ValueError(  # noqa: TRY004
                f"{self._path}: expected a JSON array of records"
            )
        for index, entry in enumerate(payload):
            if isinstance(entry, dict):
                yield RawRecord(source=self._source_id, fields=entry, index=index)
            else:
                yield RawRecord(
                    source=self._source_id,
                    index=index,
                    error=f"array entry is {type(entry).__name__}, expected an object",
                )

    def _from_lines(self, text: str) -> Iterator[RawRecord]:
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except ValueError:
                yield RawRecord(
                    source=self._source_id,
                    line_number=line_number,
                    error="line is not valid JSON",
                )
                continue
            if isinstance(entry, dict):
                yield RawRecord(
                    source=self._source_id, fields=entry, line_number=line_number
                )
            else:
                yield RawRecord(
                    source=self._source_id,
                    line_number=line_number,
                    error=f"line is {type(entry).__name__}, expected a JSON object",
                )


__all__ = ["PRESETS", "JsonListeningHistorySource"]
