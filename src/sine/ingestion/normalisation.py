"""Normalisation: source-shaped fields to domain listening events.

This module is the only place that reads a source's field names, and it does so
through a declarative :class:`FieldMapping`. Adding a source therefore means
declaring which of its keys mean what, not writing parsing code again.

Two rules are enforced throughout:

* nothing is invented — a value that the source did not supply stays absent on
  the domain object rather than being guessed at;
* nothing is discarded — a record that cannot be normalised is reported with a
  reason instead of being dropped.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, ValidationError

from sine.ingestion.records import RawRecord
from sine.models.base import HistoryModel, aware_utc, normalise_text
from sine.models.history import ListeningEvent
from sine.models.music import MBID_PATTERN, Album, Artist, Track


class RecordRejected(Exception):
    """A raw record cannot be normalised into a listening event."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class FieldMapping(HistoryModel):
    """Which keys in a source's payload carry which meaning.

    A value may be a single dotted path (``"artist.name"``) or a tuple of
    alternatives, tried in order. Keys that a source does not provide are left as
    ``None`` and the corresponding domain field stays unset.
    """

    track_title: str | tuple[str, ...]
    artist: str | tuple[str, ...]
    played_at: str | tuple[str, ...]
    artists: str | tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Path to a list of additional credited artist names. The mapped "
            "'artist' remains the primary credit and is always kept."
        ),
    )
    album: str | tuple[str, ...] | None = None
    album_artist: str | tuple[str, ...] | None = None
    year: str | tuple[str, ...] | None = None
    duration_seconds: str | tuple[str, ...] | None = None
    isrc: str | tuple[str, ...] | None = None
    recording_mbid: str | tuple[str, ...] | None = None
    release_mbid: str | tuple[str, ...] | None = None
    artist_mbid: str | tuple[str, ...] | None = None
    event_id: str | tuple[str, ...] | None = None
    play_count: str | tuple[str, ...] | None = None
    extra_fields: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Source fields to retain verbatim as event metadata.",
    )

    def paths(self) -> dict[str, str | tuple[str, ...] | None]:
        return self.model_dump()


class IngestOptions(HistoryModel):
    """How to interpret values the source left ambiguous.

    ``default_timezone`` exists because some exports record local time without an
    offset. Guessing that offset would fabricate data, so a naive timestamp is
    rejected unless the caller states the timezone the source used.
    """

    default_timezone: str | None = None
    played_at_format: str | None = Field(
        default=None, description="strptime format, when the source is not ISO 8601."
    )


_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    # Last.fm writes a comma before the time; some exports and hand-edits drop it.
    "%d %b %Y, %H:%M",
    "%d %b %Y %H:%M",
    "%d %b %Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
)

MBID_RE = re.compile(MBID_PATTERN)
_NUMERIC_RE = re.compile(r"[+-]?\d+(\.\d+)?")

#: Epoch values above this are milliseconds rather than seconds. 1e11 seconds is
#: the year 5138, so no plausible seconds timestamp reaches it.
_MILLISECOND_THRESHOLD = 1e11


def _lookup(
    payload: Mapping[str, object], path: str | tuple[str, ...] | None
) -> object | None:
    """Resolve a dotted path, or the first alternative that yields a value."""

    if path is None:
        return None
    candidates: Sequence[str] = (path,) if isinstance(path, str) else path

    for candidate in candidates:
        current: object = payload
        for part in candidate.split("."):
            if not isinstance(current, Mapping):
                current = None
                break
            current = current.get(part)
        # A container here means the caller wanted to go deeper, so keep trying
        # the remaining alternatives: "artist" against {"artist": {"name": ...}}
        # should fall through to "artist.name" rather than yield the whole object.
        if current is not None and not isinstance(current, Mapping):
            return current
    return None


def _text(value: object) -> str | None:
    if isinstance(value, str):
        stripped = " ".join(value.split())
        return stripped or None
    return None


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def parse_timestamp(value: object, options: IngestOptions) -> datetime:
    """Interpret a source timestamp as an aware UTC datetime.

    Supports epoch seconds, epoch milliseconds, ISO 8601, and a small set of
    export-specific formats. Naive values require ``options.default_timezone``.
    """

    if isinstance(value, datetime):
        return _localise(value, options)

    if isinstance(value, bool):
        raise RecordRejected(f"played_at {value!r} is not a timestamp")

    if isinstance(value, (int, float)):
        seconds = float(value)
        if abs(seconds) >= _MILLISECOND_THRESHOLD:
            seconds /= 1000.0
        try:
            return datetime.fromtimestamp(seconds, tz=UTC)
        except (OverflowError, OSError, ValueError) as exc:
            raise RecordRejected(f"played_at {value!r} is out of range") from exc

    if not isinstance(value, str):
        raise RecordRejected(f"played_at {value!r} is not a timestamp")

    raw = value.strip()
    if not raw:
        raise RecordRejected("played_at is empty")

    # CSV exports carry every value as text, so an epoch arrives as digits. The
    # threshold distinguishes milliseconds from seconds.
    if _NUMERIC_RE.fullmatch(raw):
        return parse_timestamp(float(raw), options)

    try:
        return _localise(datetime.fromisoformat(raw), options)
    except ValueError:
        pass

    formats = (
        (options.played_at_format,) if options.played_at_format else _TIMESTAMP_FORMATS
    )
    for fmt in formats:
        try:
            # The formats a source may use carry no zone; _localise decides whether a
            # default zone is available or the record must be rejected.
            return _localise(datetime.strptime(raw, fmt), options)  # noqa: DTZ007
        except ValueError:
            continue

    raise RecordRejected(f"played_at {raw!r} is not a recognised timestamp")


def _localise(value: datetime, options: IngestOptions) -> datetime:
    if value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None:
        return aware_utc(value)
    if options.default_timezone is None:
        raise RecordRejected(
            f"played_at {value.isoformat()} has no timezone; set default_timezone to the "
            "zone the source recorded"
        )
    try:
        zone = ZoneInfo(options.default_timezone)
    except ZoneInfoNotFoundError as exc:
        raise RecordRejected(f"unknown timezone {options.default_timezone!r}") from exc
    return aware_utc(value.replace(tzinfo=zone))


def _optional_int(value: object, *, minimum: int, field_name: str) -> int | None:
    parsed = _positive_int(value)
    if parsed is None:
        if value is None:
            return None
        raise RecordRejected(f"{field_name} {value!r} is not a positive integer")
    return max(parsed, minimum)


def build_event(
    payload: Mapping[str, object],
    *,
    mapping: FieldMapping,
    source: str,
    options: IngestOptions,
) -> ListeningEvent:
    """Normalise one source payload into a :class:`ListeningEvent`."""

    title = _text(_lookup(payload, mapping.track_title))
    if title is None:
        raise RecordRejected("missing or empty track title")
    artist_name = _text(_lookup(payload, mapping.artist))
    if artist_name is None:
        raise RecordRejected("missing or empty artist name")

    raw_played_at = _lookup(payload, mapping.played_at)
    if raw_played_at is None:
        raise RecordRejected("missing played_at")
    played_at = parse_timestamp(raw_played_at, options)

    artist = Artist(name=artist_name, mbid=_mbid(_lookup(payload, mapping.artist_mbid)))

    credits: list[Artist] = [artist]
    raw_extra_artists = _lookup(payload, mapping.artists)
    if isinstance(raw_extra_artists, (list, tuple)):
        for entry in raw_extra_artists:
            credit = _credit(entry)
            # A source may repeat the primary among the full credit list; keeping it
            # twice would overstate how many artists were credited.
            if credit is not None and credit.identity not in {
                c.identity for c in credits
            }:
                credits.append(credit)

    album: Album | None = None
    album_title = _text(_lookup(payload, mapping.album))
    if album_title is not None:
        album_artist_name = _text(_lookup(payload, mapping.album_artist)) or artist_name
        raw_year = _lookup(payload, mapping.year)
        year = None
        if raw_year is not None:
            try:
                year = int(str(raw_year).strip()[:4])
            except ValueError:
                year = None
        album = Album(
            title=album_title,
            artist=Artist(name=album_artist_name),
            year=year,
            mbid=_mbid(_lookup(payload, mapping.release_mbid)),
        )

    duration = None
    if mapping.duration_seconds is not None:
        duration = _optional_int(
            _lookup(payload, mapping.duration_seconds), minimum=0, field_name="duration"
        )

    isrc = _text(_lookup(payload, mapping.isrc))
    if isrc is not None:
        isrc = isrc.upper().replace("-", "").replace(" ", "")

    play_count = 1
    if mapping.play_count is not None:
        raw_count = _lookup(payload, mapping.play_count)
        if raw_count is not None:
            play_count = (
                _optional_int(raw_count, minimum=1, field_name="play_count") or 1
            )

    metadata: dict[str, str] = {}
    for key in mapping.extra_fields:
        value = _lookup(payload, key)
        text = _text(value)
        if text is not None and normalise_text(text) != "":
            metadata[key] = text

    return ListeningEvent(
        track=Track(
            title=title,
            artists=tuple(credits),
            album=album,
            duration_seconds=duration,
            isrc=isrc,
            recording_mbid=_mbid(_lookup(payload, mapping.recording_mbid)),
            release_mbid=_mbid(_lookup(payload, mapping.release_mbid)),
        ),
        played_at=played_at,
        source=source,
        source_event_id=_text(_lookup(payload, mapping.event_id)),
        play_count=play_count,
        metadata=metadata,
    )


def _credit(entry: object) -> Artist | None:
    """Build one credited artist from an entry in a source's credit list.

    Sources disagree on shape: some give bare names, others give objects carrying a
    name and a MusicBrainz ID. The Last.fm API uses ``#text``, so that is tried too.
    An entry whose name cannot be read is skipped rather than invented.
    """

    if isinstance(entry, Mapping):
        name = None
        for key in ("name", "#text", "artist", "title"):
            name = _text(entry.get(key))
            if name is not None:
                break
        if name is None:
            return None
        mbid = _mbid(entry.get("mbid") or entry.get("artist_mbid"))
        return Artist(name=name, mbid=mbid)
    name = _text(entry)
    return Artist(name=name) if name is not None else None


def _mbid(value: object) -> str | None:
    """Return a syntactically valid MusicBrainz ID, or ``None``.

    An identifier that is not a well-formed UUID is dropped rather than rejected:
    it is unusable for joining, and refusing the whole record would lose an
    otherwise valid play.
    """

    text = _text(value)
    if text is None or not MBID_RE.fullmatch(text):
        return None
    return text


class RejectedRecord(HistoryModel):
    """A record that could not be normalised.

    Rejections are reported rather than raised: real scrobble exports contain
    malformed rows, and a partial history with a visible rejection list is more
    useful than an exception that discards the whole import.
    """

    index: int | None = None
    line_number: int | None = None
    reason: str = Field(min_length=1)

    def describe(self) -> str:
        if self.line_number is not None:
            return f"line {self.line_number}: {self.reason}"
        if self.index is not None:
            return f"record {self.index}: {self.reason}"
        return self.reason


def normalise_records(
    records: Iterable[RawRecord],
    *,
    mapping: FieldMapping,
    source: str,
    options: IngestOptions | None = None,
) -> tuple[tuple[ListeningEvent, ...], tuple[RejectedRecord, ...]]:
    """Normalise a batch of raw records into events and rejections.

    Ordering and de-duplication are not applied here; they happen once, in the
    domain model, so that they are identical for every source.
    """

    resolved = options or IngestOptions()
    events: list[ListeningEvent] = []
    rejected: list[RejectedRecord] = []

    for record in records:
        if record.error is not None:
            rejected.append(
                RejectedRecord(
                    index=record.index,
                    line_number=record.line_number,
                    reason=record.error,
                )
            )
            continue
        try:
            events.append(
                build_event(
                    record.fields, mapping=mapping, source=source, options=resolved
                )
            )
        except RecordRejected as exc:
            rejected.append(
                RejectedRecord(
                    index=record.index,
                    line_number=record.line_number,
                    reason=exc.reason,
                )
            )
        except ValidationError as exc:
            # A source can supply values that pass the mapping but fail domain
            # validation, e.g. an out-of-range year. That is a bad record, not a bug.
            rejected.append(
                RejectedRecord(
                    index=record.index,
                    line_number=record.line_number,
                    reason="failed validation: " + _summarise(exc),
                )
            )

    return tuple(events), tuple(rejected)


def _summarise(error: ValidationError) -> str:
    problems = [
        f"{'.'.join(str(part) for part in detail['loc']) or '(root)'}: {detail['msg']}"
        for detail in error.errors(include_url=False)
    ]
    return "; ".join(problems[:4])


__all__ = [
    "FieldMapping",
    "IngestOptions",
    "RecordRejected",
    "build_event",
    "normalise_records",
    "parse_timestamp",
]
