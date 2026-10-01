"""Descriptive statistics over a listening history.

These are facts about what was observed. Nothing here claims a preference: a
play count says how often something was played, nothing more. Interpretation is
the profile's job, and it is kept separate for that reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import Field

from sine.models.base import HistoryModel, TimeWindow
from sine.models.music import Album, Artist, Track


class PlayCount[ItemT: HistoryModel](HistoryModel):
    """How often one entity was played, over which span and on how many days."""

    item: ItemT
    plays: int = Field(ge=1)
    event_count: int = Field(ge=1, description="Observations backing this count.")
    first_played: datetime
    last_played: datetime
    active_days: int = Field(ge=1)


class ListeningStatistics(HistoryModel):
    """A deterministic summary of observed listening behaviour."""

    window: TimeWindow
    recent_window_start: datetime = Field(
        description="Start of the recent-listening window, or the window start."
    )
    event_count: int = Field(ge=0)
    total_plays: int = Field(ge=0)
    active_days: int = Field(ge=0)
    max_plays_per_day: int = Field(ge=0)

    unique_tracks: int = Field(ge=0)
    unique_artists: int = Field(ge=0)
    unique_albums: int = Field(ge=0)

    top_artists: tuple[PlayCount[Artist], ...] = Field(default_factory=tuple)
    top_tracks: tuple[PlayCount[Track], ...] = Field(default_factory=tuple)
    top_albums: tuple[PlayCount[Album], ...] = Field(default_factory=tuple)

    track_plays: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Plays per track identity, uncapped. The top_* lists are truncated for "
            "readability; these counts are not, so novelty labelling never mistakes "
            "a rarely played track for one never played."
        ),
    )
    artist_plays_by_identity: dict[str, int] = Field(
        default_factory=dict,
        description="Plays per artist identity, uncapped. See track_plays.",
    )

    repeated_tracks: int = Field(ge=0, description="Tracks played on 2+ separate days.")
    one_off_tracks: int = Field(ge=0, description="Tracks played on exactly one day.")
    repeat_ratio: float = Field(ge=0.0, le=1.0)
    one_off_ratio: float = Field(ge=0.0, le=1.0)
    top_artist_share: float = Field(
        ge=0.0, le=1.0, description="Share of plays attributable to the top 10 artists."
    )
    artist_diversity: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalised Shannon entropy of play share across artists.",
    )
    new_artist_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Share of plays by artists whose only recorded plays fall inside the "
            "recent window. This is not proof of discovery: plays before the start "
            "of the history are not visible."
        ),
    )

    recent_plays: int = Field(ge=0, description="Plays inside the recent window.")
    long_term_plays: int = Field(ge=0, description="Plays before the recent window.")
    recent_artist_count: int = Field(ge=0)
    long_term_artist_count: int = Field(ge=0)
    recent_only_artists: tuple[Artist, ...] = Field(default_factory=tuple)
    long_term_only_artists: tuple[Artist, ...] = Field(default_factory=tuple)

    def artist_plays(self, artist: Artist) -> int:
        return self.artist_plays_by_identity.get(artist.identity, 0)

    def is_tracked(self, track: Track) -> bool:
        return self.track_plays.get(track.identity, 0) > 0

    def artists(self) -> Sequence[Artist]:
        return [entry.item for entry in self.top_artists]
