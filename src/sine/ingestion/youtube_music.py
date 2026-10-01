"""YouTube Music Google Takeout ingestion."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sine.models import ListeningEvent, Track
from sine.normalize import canonicalize_timestamp, normalize_name


def parse(records: list[dict[str, Any]]) -> list[ListeningEvent]:
    events: list[ListeningEvent] = []
    for record in records:
        if (
            record.get("header") != "YouTube Music"
            or not record.get("title", "").startswith("Watched ")
            or not record.get("subtitles")
        ):
            continue

        title = record["title"][8:]
        if not title:
            continue
        subtitle = record["subtitles"][0].get("name", "")
        artist = None if "music.youtube.com" in subtitle else subtitle

        events.append(
            ListeningEvent(
                track=Track(
                    title=normalize_name(title),
                    artists=(normalize_name(artist),) if artist else (),
                ),
                played_at=canonicalize_timestamp(
                    datetime.fromisoformat(record["time"].replace("Z", "+00:00"))
                ),
                source="youtube_music",
                source_id=record.get("titleUrl"),
                source_metadata={"origin": "google-takeout"},
            )
        )
    return events


def load(path: Path) -> list[ListeningEvent]:
    return parse(json.loads(path.read_text(encoding="utf-8")))
