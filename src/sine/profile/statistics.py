"""Deterministic statistics over a listening history.

Everything here is arithmetic on observations: how often, how recently, how
concentrated. No model is involved, and nothing infers taste. Interpretation
lives in :mod:`sine.profile.signals`, which consumes this output.

Determinism is a requirement, not a convenience. Counts are ordered by
``(-plays, identity)`` so that a history yields identical statistics regardless of
the order its events arrived in.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timedelta

from sine.models.base import TimeWindow
from sine.models.history import ListeningEvent, ListeningHistory
from sine.models.music import Album, Artist, Track
from sine.models.statistics import ListeningStatistics, PlayCount

DEFAULT_RECENT_WINDOW_DAYS = 28
DEFAULT_TOP_N = 25


class _Tally:
    """Accumulates plays, event count, first and last play, and active days."""

    def __init__(self) -> None:
        self.plays = 0
        self.events = 0
        self.first: datetime | None = None
        self.last: datetime | None = None
        self.days: set[datetime] = set()

    def add(self, event: ListeningEvent) -> None:
        self.plays += event.play_count
        self.events += 1
        if self.first is None or event.played_at < self.first:
            self.first = event.played_at
        if self.last is None or event.played_at > self.last:
            self.last = event.played_at
        self.days.add(event.played_at.date())


def _play_counts[T](
    tallies: dict[str, tuple[T, _Tally]], *, limit: int | None
) -> tuple[PlayCount[T], ...]:
    """Convert tallies into an ordered, capped tuple of play counts."""

    entries = [
        PlayCount[T](
            item=item,
            plays=tally.plays,
            event_count=tally.events,
            first_played=tally.first,
            last_played=tally.last,
            active_days=len(tally.days),
        )
        for item, tally in tallies.values()
        if tally.first is not None and tally.last is not None
    ]
    # Sort on a stable secondary key so equal counts never reorder between runs.
    entries.sort(key=lambda entry: (-entry.plays, _identity_of(entry.item)))
    return tuple(entries if limit is None else entries[:limit])


def _identity_of(item: object) -> str:
    identity = getattr(item, "identity", None)
    return identity if isinstance(identity, str) else str(item)


def normalise_entropy(weights: Sequence[float]) -> float:
    """Shannon entropy of a weight distribution, normalised to 0..1.

    0 means all listening went to one entity; 1 means it was spread evenly.
    Normalising by ``log(n)`` keeps the value comparable across histories of
    different sizes, which a raw entropy does not.
    """

    total = sum(weights)
    if total <= 0 or len(weights) <= 1:
        return 0.0
    entropy = 0.0
    for weight in weights:
        if weight <= 0:
            continue
        share = weight / total
        entropy -= share * math.log(share)
    return round(entropy / math.log(len(weights)), 6)


def compute_statistics(
    history: ListeningHistory,
    *,
    now: datetime,
    recent_window_days: int = DEFAULT_RECENT_WINDOW_DAYS,
    top_n: int = DEFAULT_TOP_N,
) -> ListeningStatistics:
    """Summarise a listening history as of ``now``.

    ``now`` is supplied rather than read from the clock so that "recent" is a
    function of the data, which keeps the whole pipeline reproducible.
    """

    window = history.window or TimeWindow(start=now, end=now)
    events = history.events

    artist_tallies: dict[str, tuple[Artist, _Tally]] = {}
    album_tallies: dict[str, tuple[Album, _Tally]] = {}
    track_tallies: dict[str, tuple[Track, _Tally]] = {}

    plays_per_day: Counter[object] = Counter()
    recent_start = now - timedelta(days=recent_window_days)

    recent_artist_plays: Counter[str] = Counter()
    long_term_artist_plays: Counter[str] = Counter()
    recent_plays = 0
    long_term_plays = 0

    for event in events:
        track = event.track
        artist = track.artist
        plays_per_day[event.played_at.date()] += event.play_count

        for store, item, key in (
            (artist_tallies, artist, artist.identity),
            (track_tallies, track, track.identity),
        ):
            if key not in store:
                store[key] = (item, _Tally())
            store[key][1].add(event)

        if track.album is not None:
            key = track.album.identity
            if key not in album_tallies:
                album_tallies[key] = (track.album, _Tally())
            album_tallies[key][1].add(event)

        if event.played_at >= recent_start:
            recent_plays += event.play_count
            recent_artist_plays[artist.identity] += event.play_count
        else:
            long_term_plays += event.play_count
            long_term_artist_plays[artist.identity] += event.play_count

    all_top_artists = _play_counts(artist_tallies, limit=None)
    top_artists = all_top_artists[:top_n]
    top_tracks = _play_counts(track_tallies, limit=top_n)
    top_albums = _play_counts(album_tallies, limit=top_n)

    total_plays = history.total_plays
    top_share_plays = sum(entry.plays for entry in all_top_artists[:10])

    all_top_tracks = _play_counts(track_tallies, limit=None)
    repeated_tracks = sum(1 for entry in all_top_tracks if entry.active_days >= 2)
    one_off_tracks = sum(1 for entry in all_top_tracks if entry.active_days == 1)
    repeated_plays = sum(
        entry.plays for entry in all_top_tracks if entry.active_days >= 2
    )

    recent_only = sorted(set(recent_artist_plays) - set(long_term_artist_plays))
    long_term_only = sorted(set(long_term_artist_plays) - set(recent_artist_plays))
    by_identity = {entry.item.identity: entry.item for entry in all_top_artists}

    # Plays by artists heard only in the recent window. Within this history that is
    # the observable part of "new to me"; it cannot account for plays predating it.
    recent_only_plays = sum(recent_artist_plays[identity] for identity in recent_only)

    return ListeningStatistics(
        window=window,
        recent_window_start=recent_start,
        event_count=history.event_count,
        total_plays=total_plays,
        active_days=len(plays_per_day),
        max_plays_per_day=max(plays_per_day.values(), default=0),
        unique_tracks=len(track_tallies),
        unique_artists=len(artist_tallies),
        unique_albums=len(album_tallies),
        top_artists=top_artists,
        top_tracks=top_tracks,
        top_albums=top_albums,
        track_plays={entry.item.identity: entry.plays for entry in all_top_tracks},
        artist_plays_by_identity={
            entry.item.identity: entry.plays for entry in all_top_artists
        },
        repeated_tracks=repeated_tracks,
        one_off_tracks=one_off_tracks,
        repeat_ratio=round(repeated_plays / total_plays, 6) if total_plays else 0.0,
        one_off_ratio=round(one_off_tracks / len(track_tallies), 6)
        if track_tallies
        else 0.0,
        top_artist_share=(
            round(top_share_plays / total_plays, 6) if total_plays else 0.0
        ),
        artist_diversity=normalise_entropy([entry.plays for entry in all_top_artists]),
        new_artist_ratio=(
            round(recent_only_plays / total_plays, 6) if total_plays else 0.0
        ),
        recent_plays=recent_plays,
        long_term_plays=long_term_plays,
        recent_artist_count=len(recent_artist_plays),
        long_term_artist_count=len(long_term_artist_plays),
        recent_only_artists=tuple(
            by_identity[key] for key in recent_only if key in by_identity
        ),
        long_term_only_artists=tuple(
            by_identity[key] for key in long_term_only if key in by_identity
        ),
    )
