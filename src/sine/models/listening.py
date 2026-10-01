"""Canonical listening-history domain models.

These models deliberately know nothing about AT Protocol or a particular music
service. Source-specific identifiers and metadata stay on ListeningEvent.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Track(BaseModel):
    """A music track with metadata that is useful across sources."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    artists: tuple[str, ...] = ()
    album: str | None = None
    duration_seconds: int | None = Field(default=None, gt=0)
    recording_mbid: str | None = None
    release_mbid: str | None = None
    artist_mbids: tuple[str, ...] = ()
    isrc: str | None = None


class ListeningEvent(BaseModel):
    """One observed listen, independent of its originating service."""

    model_config = ConfigDict(extra="forbid")

    track: Track
    played_at: datetime
    source: str
    source_id: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)
