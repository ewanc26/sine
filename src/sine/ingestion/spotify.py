"""Spotify listening-history JSON ingestion."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sine.models import ListeningEvent, Track
from sine.normalize import canonicalize_timestamp, normalize_name


def parse(records: list[dict[str, Any]]) -> list[ListeningEvent]:
    """Convert Spotify export records, excluding podcasts and non-track entries."""

    events: list[ListeningEvent] = []
    for record in records:
        title = record.get("master_metadata_track_name")
        artist = record.get("master_metadata_album_artist_name")
        if not title or not artist:
            continue

        track = Track(
            title=normalize_name(title),
            artists=(normalize_name(artist),),
            album=normalize_name(record["master_metadata_album_album_name"])
            if record.get("master_metadata_album_album_name")
            else None,
        )
        uri = record.get("spotify_track_uri")
        events.append(
            ListeningEvent(
                track=track,
                played_at=canonicalize_timestamp(datetime.fromisoformat(record["ts"].replace("Z", "+00:00"))),
                source="spotify",
                source_id=uri.removeprefix("spotify:track:") if uri else None,
                source_metadata={"origin": "spotify-export", "platform": record.get("platform")},
            )
        )
    return events


def load(path: Path) -> list[ListeningEvent]:
    return parse(json.loads(path.read_text(encoding="utf-8")))
