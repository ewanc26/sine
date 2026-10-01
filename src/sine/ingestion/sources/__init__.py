"""Listening-history sources.

Each adapter handles one service's file format and nothing else. They are all
registered here so the CLI can offer them by name, but nothing depends on this
module: any object with ``source_id``, ``fetch``, and ``field_mapping`` is a valid
source.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sine.ingestion.normalisation import FieldMapping
from sine.ingestion.protocols import ListeningHistorySource
from sine.ingestion.sources.apple_music import AppleMusicSchemaError, AppleMusicSource
from sine.ingestion.sources.csv_records import (
    column,
    raw_records,
    read_rows,
    sniff_delimiter,
)
from sine.ingestion.sources.jsonl import JsonListeningHistorySource
from sine.ingestion.sources.lastfm import LastFmSource
from sine.ingestion.sources.listenbrainz import ListenBrainzSource
from sine.ingestion.sources.spotify import SpotifySource
from sine.ingestion.sources.youtube_music import YouTubeMusicSource

SourceFactory = Callable[..., ListeningHistorySource]

#: The generic JSON reader takes a field mapping, because its input shape is not a
#: service's own export format. Every other source knows its own layout.
JSON_PRESETS = ("sine", "lastfm", "listenbrainz", "spotify", "generic")


def _json_factory(
    path: Path, *, source_id: str = "json-import", preset: str = "sine"
) -> JsonListeningHistorySource:
    """Build the generic JSON source.

    ``preset`` selects a field mapping rather than a parser; ``generic`` means the
    mapping Sine's own shape uses.
    """

    return JsonListeningHistorySource(
        path, source_id=source_id, preset="sine" if preset == "generic" else preset
    )


#: Every file-based source Sine can read, by the name the CLI accepts.
SOURCES: dict[str, SourceFactory] = {
    "lastfm": LastFmSource,
    "spotify": SpotifySource,
    "apple-music": AppleMusicSource,
    "youtube-music": YouTubeMusicSource,
    "listenbrainz": ListenBrainzSource,
    "json": _json_factory,
}

#: Aliases for spellings people reasonably try.
ALIASES = {
    "apple": "apple-music",
    "apple_music": "apple-music",
    "applemusic": "apple-music",
    "last.fm": "lastfm",
    "last-fm": "lastfm",
    "listen-brainz": "listenbrainz",
    "youtube": "youtube-music",
    "youtube_music": "youtube-music",
    "ytmusic": "youtube-music",
    "jsonl": "json",
    "sine": "json",
}


def resolve_source_name(name: str) -> str:
    """Map a user-supplied source name onto a registered source."""

    key = name.strip().casefold()
    key = ALIASES.get(key, key)
    if key not in SOURCES:
        raise ValueError(
            f"unknown source {name!r}; choose one of {', '.join(sorted(SOURCES))}"
        )
    return key


def open_source(name: str, path: str | Path, **options: Any) -> ListeningHistorySource:
    """Build the named source for a local export path.

    ``options`` are passed to the factory; only the generic JSON source uses them.
    """

    return SOURCES[resolve_source_name(name)](Path(path), **options)


def describe_sources() -> tuple[tuple[str, str], ...]:
    """One line per source for ``sine import --help`` and shell completion."""

    descriptions = {
        "json": "JSON array or JSON Lines; uses a field-mapping preset",
        "lastfm": "Last.fm CSV export",
        "spotify": "Spotify extended-history CSV (skips podcasts)",
        "listenbrainz": "ListenBrainz JSON, JSON Lines, or export ZIP",
        "apple-music": "Apple Music play-history CSV",
        "youtube-music": "YouTube Music watch-history CSV",
    }
    return tuple((name, descriptions[name]) for name in sorted(SOURCES))


__all__ = [
    "ALIASES",
    "JSON_PRESETS",
    "SOURCES",
    "AppleMusicSchemaError",
    "AppleMusicSource",
    "FieldMapping",
    "JsonListeningHistorySource",
    "LastFmSource",
    "ListenBrainzSource",
    "ListeningHistorySource",
    "SpotifySource",
    "YouTubeMusicSource",
    "column",
    "describe_sources",
    "open_source",
    "raw_records",
    "read_rows",
    "resolve_source_name",
    "sniff_delimiter",
]
