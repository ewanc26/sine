"""ListenBrainz JSON, JSONL, and archive ingestion."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from sine.models import ListeningEvent, Track
from sine.normalize import canonicalize_timestamp, normalize_name


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        if isinstance(value.get("listens"), list):
            return [item for item in value["listens"] if isinstance(item, dict)]
        payload = value.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("listens"), list):
            return [item for item in payload["listens"] if isinstance(item, dict)]
        return [value]
    return []


def parse(content: str) -> list[ListeningEvent]:
    text = content.strip()
    if not text:
        return []

    try:
        values = _records(json.loads(text))
    except json.JSONDecodeError:
        values = []
        for line in text.splitlines():
            try:
                values.extend(_records(json.loads(line)))
            except json.JSONDecodeError:
                continue

    events: list[ListeningEvent] = []
    for record in values:
        timestamp = record.get("listened_at")
        metadata = record.get("track_metadata")
        if not isinstance(timestamp, (int, float)) or not isinstance(metadata, dict):
            continue

        title = metadata.get("track_name")
        artist = metadata.get("artist_name")
        if not isinstance(title, str) or not isinstance(artist, str):
            continue

        additional = metadata.get("additional_info") or {}
        mapping = metadata.get("mbid_mapping") or {}
        mapped_artists = mapping.get("artists") if isinstance(mapping, dict) else None
        artists = (
            tuple(normalize_name(item["artist_credit_name"]) for item in mapped_artists if item.get("artist_credit_name"))
            if isinstance(mapped_artists, list) and mapped_artists
            else (normalize_name(artist),)
        )

        track = Track(
            title=normalize_name(title),
            artists=artists,
            album=normalize_name(metadata["release_name"]) if metadata.get("release_name") else None,
            recording_mbid=(mapping.get("recording_mbid") or additional.get("recording_mbid"))
            if isinstance(mapping, dict)
            else additional.get("recording_mbid"),
            release_mbid=(mapping.get("release_mbid") or additional.get("release_mbid"))
            if isinstance(mapping, dict)
            else additional.get("release_mbid"),
            artist_mbids=tuple(
                item["artist_mbid"] for item in mapped_artists
                if isinstance(item, dict) and item.get("artist_mbid")
            ) if isinstance(mapped_artists, list) else (),
            isrc=additional.get("isrc") if isinstance(additional, dict) else None,
        )
        events.append(
            ListeningEvent(
                track=track,
                played_at=datetime.fromtimestamp(timestamp, tz=timezone.utc),
                source="listenbrainz",
                source_id=additional.get("origin_url") if isinstance(additional, dict) else None,
                source_metadata={
                    "origin": "listenbrainz-export",
                    "music_service": additional.get("music_service")
                    if isinstance(additional, dict)
                    else None,
                },
            )
        )

    return events


def load(path: Path) -> list[ListeningEvent]:
    if path.suffix.lower() != ".zip":
        return parse(path.read_text(encoding="utf-8"))

    events: list[ListeningEvent] = []
    with ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename.lower()
            if not (name.endswith(".json") or name.endswith(".jsonl")):
                continue
            if name.endswith("/user.json") or name.endswith("/feedback.jsonl"):
                continue
            events.extend(parse(archive.read(info).decode("utf-8")))
    return events
