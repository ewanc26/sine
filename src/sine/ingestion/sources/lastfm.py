"""Last.fm CSV export ingestion.

Last.fm exports are CSV with a timestamp column that is variously named and is
either epoch seconds or epoch milliseconds depending on the export generation, so
the adapter normalises both into a single field and lets the shared timestamp parser
decide.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from sine.ingestion.normalisation import FieldMapping
from sine.ingestion.records import RawRecord
from sine.ingestion.sources.csv_records import raw_records, read_rows

#: Column spellings seen across Last.fm export generations, lower-cased.
_TIMESTAMP_COLUMNS = (
    "uts",
    "date",
    "timestamp",
    "played_at",
    "time",
    "utc_time",
    "utctime",
)
_TRACK_COLUMNS = ("track", "track_name", "trackname", "song", "title")
_ARTIST_COLUMNS = ("artist", "artist_name", "artistname")
_ALBUM_COLUMNS = ("album", "album_name", "albumname", "release")
_TRACK_MBID_COLUMNS = ("track_mbid", "trackmbid", "track_id", "recording_mbid")
_ARTIST_MBID_COLUMNS = ("artist_mbid", "artistmbid", "artist_id")
_ALBUM_MBID_COLUMNS = ("album_mbid", "albummbid", "album_id", "release_mbid")


def _first_present(
    row: dict[str, str], candidates: tuple[str, ...]
) -> tuple[str, str] | None:
    """Return the first candidate column that has a value, with its value."""

    for name in candidates:
        value = row.get(name)
        if value and value.strip():
            return name, value.strip()
    return None


def _prepare(row: dict[str, str]) -> tuple[dict[str, object], str | None]:
    """Collapse a Last.fm row onto Sine's field names.

    Rows missing an artist, title, or time are not plays. They are reported as
    rejections rather than dropped, so the number of unusable rows stays visible.
    """

    timestamp = _first_present(row, _TIMESTAMP_COLUMNS)
    track = _first_present(row, _TRACK_COLUMNS)
    artist = _first_present(row, _ARTIST_COLUMNS)
    if timestamp is None:
        return {}, "no timestamp column"
    if track is None:
        return {}, "no track column"
    if artist is None:
        return {}, "no artist column"

    prepared: dict[str, object] = {
        "played_at": timestamp[1],
        "track_title": track[1],
        "artist": artist[1],
    }

    album = _first_present(row, _ALBUM_COLUMNS)
    if album is not None:
        prepared["album"] = album[1]

    for field_name, candidates in (
        ("recording_mbid", _TRACK_MBID_COLUMNS),
        ("release_mbid", _ALBUM_MBID_COLUMNS),
        ("artist_mbid", _ARTIST_MBID_COLUMNS),
    ):
        found = _first_present(row, candidates)
        if found is not None:
            prepared[field_name] = found[1]

    # Last.fm has no stable per-play event ID, so leave event_id unset and let
    # domain de-duplication key on artist, title, and time instead.
    return prepared, None


class LastFmSource:
    """Reads a Last.fm CSV export."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def source_id(self) -> str:
        return "lastfm"

    def field_mapping(self) -> FieldMapping:
        return FieldMapping(
            track_title="track_title",
            artist="artist",
            played_at="played_at",
            album="album",
            recording_mbid="recording_mbid",
            release_mbid="release_mbid",
            artist_mbid="artist_mbid",
        )

    def fetch(self) -> Iterable[RawRecord]:
        rows = read_rows(self.path)
        prepared_rows: list[dict[str, object]] = []
        reasons: list[str | None] = []
        for row in rows:
            prepared, reason = _prepare(row)
            prepared_rows.append(prepared)
            reasons.append(reason)

        records = list(
            raw_records(prepared_rows, source=self.source_id, path=self.path)
        )
        return [
            RawRecord(
                source=record.source,
                index=record.index,
                line_number=record.line_number,
                source_path=record.source_path,
                fields=record.fields,
                error=reason,
            )
            for record, reason in zip(records, reasons, strict=True)
        ]
