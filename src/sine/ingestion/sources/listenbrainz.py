"""ListenBrainz JSON, JSONL, and archive ingestion.

ListenBrainz hands back several shapes depending on how the export was requested:
a bare array, ``{"listens": [...]}``, ``{"payload": {"listens": [...]}}``, or one
JSON object per line. Full exports arrive as a ZIP archive alongside unrelated
files, so those are skipped rather than parsed as if they were listens.

ListenBrainz is also the richest source for MusicBrainz data, which is what lets
Sine recognise the same recording reported by a different service.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from sine.ingestion.normalisation import FieldMapping
from sine.ingestion.records import RawRecord

_ARTIFACT_SUFFIXES = (".json", ".jsonl", ".json.gz")
_SKIP_MEMBERS = ("user.json", "feedback.jsonl")


def _records(value: Any) -> list[dict[str, Any]]:
    """Unwrap any of the shapes ListenBrainz exports."""

    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    for key in ("listens",):
        nested = value.get(key)
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
    payload = value.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("listens"), list):
        return [item for item in payload["listens"] if isinstance(item, dict)]
    return [value]


def _parse_text(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []

    try:
        return _records(json.loads(stripped))
    except json.JSONDecodeError:
        pass

    collected: list[dict[str, Any]] = []
    for line in stripped.splitlines():
        try:
            collected.extend(_records(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return collected


def _prepare(listen: dict[str, Any]) -> dict[str, object] | None:
    """Flatten one listen onto Sine's field names."""

    timestamp = listen.get("listened_at")
    metadata = listen.get("track_metadata")
    if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
        return None
    if not isinstance(metadata, dict):
        return None

    title = metadata.get("track_name")
    artist = metadata.get("artist_name")
    if not isinstance(title, str) or not isinstance(artist, str):
        return None

    prepared: dict[str, object] = {
        "played_at": timestamp,
        "track_title": title,
        "artist": artist,
    }

    release = metadata.get("release_name")
    if isinstance(release, str) and release:
        prepared["album"] = release

    mapping = metadata.get("mbid_mapping")
    additional = metadata.get("additional_info")
    mapping = mapping if isinstance(mapping, dict) else {}
    additional = additional if isinstance(additional, dict) else {}

    # A matched MBID mapping is authoritative and carries the full credit list,
    # including featured artists that "artist_name" omits.
    credits = mapping.get("artists")
    if isinstance(credits, list) and credits:
        names: list[str] = []
        for entry in credits:
            if isinstance(entry, dict) and isinstance(
                entry.get("artist_credit_name"), str
            ):
                names.append(entry["artist_credit_name"])
        if names:
            prepared["artists"] = names

    recording_mbid = mapping.get("recording_mbid") or additional.get("recording_mbid")
    if isinstance(recording_mbid, str):
        prepared["recording_mbid"] = recording_mbid
    release_mbid = mapping.get("release_mbid") or additional.get("release_mbid")
    if isinstance(release_mbid, str):
        prepared["release_mbid"] = release_mbid
    artist_mbid = mapping.get("artist_mbid") or additional.get("artist_mbid")
    if isinstance(artist_mbid, str):
        prepared["artist_mbid"] = artist_mbid
    isrc = additional.get("isrc")
    if isinstance(isrc, str) and isrc:
        prepared["isrc"] = isrc

    origin = additional.get("origin_url")
    if isinstance(origin, str) and origin:
        prepared["source_event_id"] = origin
    music_service = additional.get("music_service")
    if isinstance(music_service, str):
        prepared["music_service"] = music_service
    duration = additional.get("duration_ms")
    if (
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and duration > 0
    ):
        prepared["duration_seconds"] = int(duration // 1000)
    return prepared


class ListenBrainzSource:
    """Reads a ListenBrainz export, including a full ZIP archive."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def source_id(self) -> str:
        return "listenbrainz"

    def field_mapping(self) -> FieldMapping:
        return FieldMapping(
            track_title="track_title",
            artist="artist",
            artists="artists",
            played_at="played_at",
            album="album",
            duration_seconds="duration_seconds",
            isrc="isrc",
            recording_mbid="recording_mbid",
            release_mbid="release_mbid",
            artist_mbid="artist_mbid",
            event_id="source_event_id",
            extra_fields=("music_service",),
        )

    def _members(self) -> list[tuple[str, str]]:
        """Return ``(member_name, text)`` pairs for every listen-bearing member."""

        if self.path.suffix.lower() != ".zip":
            return [(self.path.name, self.path.read_text(encoding="utf-8"))]

        found: list[tuple[str, str]] = []
        with ZipFile(self.path) as archive:
            for info in archive.infolist():
                name = info.filename
                lowered = name.lower()
                if not lowered.endswith(_ARTIFACT_SUFFIXES):
                    continue
                if lowered.endswith(_SKIP_MEMBERS):
                    continue
                found.append((name, archive.read(info).decode("utf-8")))
        return found

    def fetch(self) -> Iterable[RawRecord]:
        records: list[RawRecord] = []
        index = 0
        for member, text in self._members():
            for listen in _parse_text(text):
                prepared = _prepare(listen)
                index += 1
                records.append(
                    RawRecord(
                        source=self.source_id,
                        index=index,
                        source_path=self.path,
                        fields=prepared if prepared is not None else {},
                        error=None
                        if prepared is not None
                        else "listen has no track metadata",
                    )
                )
        return records
