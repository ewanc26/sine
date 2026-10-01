"""Apple Music Play Activity CSV ingestion.

Apple has shipped several generations of this export, which differ in column names
and in how much per-row detail they carry. A file that contains none of the known
title columns is rejected outright rather than parsed into empty events, because
that almost always means the user handed over the wrong file.

Where the per-play export omits the artist, Apple's separate daily-playlists export
carries it in ``Track Description`` as ``"Artist - Title"``. When that yields
exactly one candidate for a title it is used; when a title maps to several
different artists the guess is ambiguous, so no artist is asserted and the record
is rejected rather than mis-attributed.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from sine.ingestion.normalisation import FieldMapping
from sine.ingestion.records import RawRecord
from sine.ingestion.sources.csv_records import column, raw_records, read_rows

TITLE_COLUMNS = ("Content Name", "Song Name")
ARTIST_COLUMNS = ("Artist Name", "Container Artist Name")
ALBUM_COLUMNS = ("Album Name", "Container Album Name")
_TIMESTAMP_COLUMNS = ("Event End Timestamp", "Event Start Timestamp", "Event Timestamp")


class AppleMusicSchemaError(ValueError):
    """Raised when a CSV is not an Apple Music Play Activity export."""


def artist_fallbacks(rows: list[dict[str, str]]) -> dict[str, str]:
    """Build a title-to-artist map from Apple's ``Track Description`` column.

    Only unambiguous titles are included: a title credited to more than one artist
    across the export yields no entry, because picking one would be a guess.
    """

    candidates: dict[str, set[str]] = {}
    for row in rows:
        description = (row.get("Track Description") or "").strip()
        if " - " not in description:
            continue
        artist, title = (part.strip() for part in description.split(" - ", 1))
        if artist and title:
            candidates.setdefault(title.casefold(), set()).add(artist)
    return {
        title: next(iter(values))
        for title, values in candidates.items()
        if len(values) == 1
    }


def _prepare(
    row: dict[str, str], fallbacks: dict[str, str]
) -> tuple[dict[str, object] | None, str | None]:
    title = column(row, TITLE_COLUMNS)
    if not title:
        return None, "no title column"

    timestamp = column(row, _TIMESTAMP_COLUMNS)
    if not timestamp:
        return None, "no timestamp column"

    # Apple's timestamps are local wall-clock with no offset, written as
    # "2026-07-17 11:39:14". The offset is not in the file, so the caller supplies
    # the timezone the export was generated in.
    if "T" not in timestamp and " " in timestamp:
        timestamp = timestamp.replace(" ", "T", 1)

    prepared: dict[str, object] = {"track_title": title, "played_at": timestamp}

    artist = column(row, ARTIST_COLUMNS)
    if not artist:
        artist = fallbacks.get(title.casefold())
    if artist:
        prepared["artist"] = artist

    album = column(row, ALBUM_COLUMNS)
    if album:
        prepared["album"] = album

    duration_ms = column(row, ("Media Duration In Milliseconds",))
    if duration_ms:
        try:
            milliseconds = float(duration_ms)
        except ValueError:
            milliseconds = 0.0
        if milliseconds > 0:
            prepared["duration_seconds"] = int(milliseconds // 1000)

    country = (row.get("ISO Country") or "").strip()
    if country:
        prepared["iso_country"] = country
    item_type = (row.get("Item Type") or "").strip()
    if item_type:
        prepared["item_type"] = item_type
    return prepared, None


class AppleMusicSource:
    """Reads an Apple Music Play Activity CSV export."""

    def __init__(
        self, path: str | Path, *, artist_lookup: dict[str, str] | None = None
    ) -> None:
        self.path = Path(path)
        self._artist_lookup = artist_lookup

    @property
    def source_id(self) -> str:
        return "apple-music"

    def field_mapping(self) -> FieldMapping:
        return FieldMapping(
            track_title="track_title",
            artist="artist",
            played_at="played_at",
            album="album",
            duration_seconds="duration_seconds",
            extra_fields=("iso_country", "item_type"),
        )

    def fetch(self) -> Iterable[RawRecord]:
        rows = read_rows(self.path)
        if rows:
            fields = {field.casefold() for field in rows[0]}
            if not any(name.casefold() in fields for name in TITLE_COLUMNS):
                raise AppleMusicSchemaError(
                    f"expected one of {TITLE_COLUMNS} in {self.path.name}; "
                    f"found {sorted(fields)}"
                )

        fallbacks = self._artist_lookup or artist_fallbacks(rows)
        prepared_rows: list[dict[str, object]] = []
        errors: list[str | None] = []
        for row in rows:
            prepared, reason = _prepare(row, fallbacks)
            prepared_rows.append(prepared if prepared is not None else {})
            errors.append(reason)

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
            for record, reason in zip(records, errors, strict=True)
        ]
