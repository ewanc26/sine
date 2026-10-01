"""Signals and gaps: reading a statistics summary without overclaiming.

The rule this module exists to enforce is that frequency is not preference. A
track played twelve times is a *measurement*; "this listener loves this track" is
a *judgement* that a measurement cannot settle on its own. Every signal therefore
states its observation first, and phrases any inference as a possibility bounded by
the data — including, where relevant, the fact that only one source was consulted.

Signals are emitted in a fixed order and all numbers are rounded, so the same
history always yields the same profile.
"""

from __future__ import annotations

from sine.models.base import EntityKind
from sine.models.history import ListeningHistory
from sine.models.profile import (
    GapKind,
    ProfileGap,
    ProfileSignal,
    SignalKind,
)
from sine.models.statistics import ListeningStatistics

#: Artists with at least this many plays on at least this many days are reported
#: individually, to keep the rendered context bounded.
REPORTED_ARTISTS = 12
THIN_HISTORY_EVENTS = 30
THIN_HISTORY_DAYS = 30.0


def build_signals(statistics: ListeningStatistics) -> tuple[ProfileSignal, ...]:
    """Derive profile signals from statistics, in a deterministic order."""

    signals: list[ProfileSignal] = []
    stats = statistics

    for entry in stats.top_artists[:REPORTED_ARTISTS]:
        if entry.active_days >= 2 and entry.plays >= 3:
            signals.append(
                ProfileSignal(
                    kind=SignalKind.REPEATED_LISTENING,
                    subject=entry.item.name,
                    subject_kind=EntityKind.ARTIST,
                    support=entry.plays,
                    observation=(
                        f"{entry.item.name} was played {entry.plays} times across "
                        f"{entry.active_days} distinct days"
                    ),
                    inference=(
                        "Returning to an artist on separate days is the strongest "
                        "signal available that it is a genuine preference rather "
                        "than a single impulse, but it is still not proof of taste"
                    ),
                )
            )
        elif entry.active_days == 1 and stats.unique_artists > 1:
            signals.append(
                ProfileSignal(
                    kind=SignalKind.ONE_OFF_LISTENING,
                    subject=entry.item.name,
                    subject_kind=EntityKind.ARTIST,
                    support=entry.plays,
                    observation=f"{entry.item.name} was played once and not returned to",
                    inference=(
                        "A single play is consistent with curiosity, coincidence, or "
                        "background listening, and does not establish a preference"
                    ),
                )
            )

    for entry in stats.top_tracks[:REPORTED_ARTISTS]:
        if entry.active_days >= 2:
            signals.append(
                ProfileSignal(
                    kind=SignalKind.REPEATED_LISTENING,
                    subject=entry.item.display_name,
                    subject_kind=EntityKind.TRACK,
                    support=entry.plays,
                    observation=(
                        f"the track {entry.item.display_name} was played "
                        f"{entry.plays} times on {entry.active_days} days"
                    ),
                    inference="repeated listening to a specific track suggests familiarity",
                )
            )

    for artist in stats.recent_only_artists[:REPORTED_ARTISTS]:
        signals.append(
            ProfileSignal(
                kind=SignalKind.RECENT_FOCUS,
                subject=artist.name,
                subject_kind=EntityKind.ARTIST,
                observation=(
                    f"{artist.name} appears only within the recent window, not earlier "
                    "in the history"
                ),
                inference=(
                    "recent-only listening may be a current interest, but a short "
                    "history cannot distinguish that from a change in what was available"
                ),
            )
        )

    for artist in stats.long_term_only_artists[:REPORTED_ARTISTS]:
        signals.append(
            ProfileSignal(
                kind=SignalKind.FADED,
                subject=artist.name,
                subject_kind=EntityKind.ARTIST,
                observation=f"{artist.name} was played earlier but not in the recent window",
                inference=(
                    "the data records absence, not a change of mind; the listener may "
                    "simply not have played it lately"
                ),
            )
        )

    if stats.recent_artist_count and stats.long_term_artist_count:
        overlap = (
            stats.recent_artist_count
            + stats.long_term_artist_count
            - stats.unique_artists
        )
        if overlap == 0:
            signals.append(
                ProfileSignal(
                    kind=SignalKind.RECENCY_SHIFT,
                    observation=(
                        f"none of the {stats.recent_artist_count} artists heard recently "
                        f"appear in the {stats.long_term_artist_count} heard earlier"
                    ),
                    inference=(
                        "the recent and long-term halves of this history barely overlap, "
                        "so the profile may reflect a change in listening rather than a "
                        "settled taste"
                    ),
                )
            )

    signals.extend(_breadth_signals(stats))

    signals.append(_concentration_signal(stats))
    signals.append(_underdetermined_signal(stats))

    return tuple(signals)


def _breadth_signals(stats: ListeningStatistics) -> list[ProfileSignal]:
    """Signals about how wide or narrow the observed listening is."""

    signals: list[ProfileSignal] = []

    if stats.unique_artists >= 10 and stats.one_off_ratio >= 0.5:
        signals.append(
            ProfileSignal(
                kind=SignalKind.BREADTH,
                observation=(
                    f"{stats.one_off_tracks} of {stats.unique_tracks} tracks "
                    f"({stats.one_off_ratio:.0%}) were played on a single day, across "
                    f"{stats.unique_artists} artists"
                ),
                inference=(
                    "this listener explores widely, so a narrow set of recommendations "
                    "would under-serve the observed behaviour"
                ),
            )
        )

    if stats.repeat_ratio >= 0.5 and stats.unique_artists <= 5:
        signals.append(
            ProfileSignal(
                kind=SignalKind.CONCENTRATION,
                observation=(
                    f"{stats.repeat_ratio:.0%} of plays went to tracks repeated on "
                    f"multiple days, across only {stats.unique_artists} artists"
                ),
                inference=(
                    "listening is narrow and habitual; novelty may not be what this "
                    "listener wants even when the request asks for it"
                ),
            )
        )

    return signals


def _concentration_signal(stats: ListeningStatistics) -> ProfileSignal:
    if stats.artist_diversity >= 0.0:
        return ProfileSignal(
            kind=SignalKind.CONCENTRATION,
            observation=(
                f"the top 10 artists account for {stats.top_artist_share:.0%} of plays; "
                f"artist diversity across the whole history is "
                f"{stats.artist_diversity:.2f} on a 0-1 scale"
            ),
            inference=(
                "a high top-10 share may suggest a core set the listener returns to, "
                "though heavy single-artist sessions can look the same"
            ),
        )
    return ProfileSignal(
        kind=SignalKind.CONCENTRATION,
        observation="no plays were recorded, so concentration cannot be measured",
    )


def _underdetermined_signal(stats: ListeningStatistics) -> ProfileSignal:
    if stats.event_count < THIN_HISTORY_EVENTS or stats.window.days < THIN_HISTORY_DAYS:
        return ProfileSignal(
            kind=SignalKind.UNDERDETERMINED,
            observation=(
                f"the history holds {stats.event_count} plays across "
                f"{stats.window.days:.0f} days"
            ),
            inference=(
                "this is a small sample; conclusions about preference should be "
                "treated as provisional and varied deliberately"
            ),
        )
    return ProfileSignal(
        kind=SignalKind.UNDERDETERMINED,
        observation=(
            f"{stats.event_count} plays across {stats.window.days:.0f} days, with "
            f"{stats.unique_artists} artists heard"
        ),
        inference="the sample is large enough to describe habits, not to explain them",
    )


def build_gaps(
    statistics: ListeningStatistics, history: ListeningHistory
) -> tuple[ProfileGap, ...]:
    """Record what the available data cannot support."""

    stats = statistics
    gaps: list[ProfileGap] = []

    if stats.window.days < THIN_HISTORY_DAYS:
        gaps.append(
            ProfileGap(
                kind=GapKind.SHORT_HISTORY,
                detail=(
                    f"the history spans only {stats.window.days:.0f} days, so "
                    "long-term listening cannot be separated from recent listening"
                ),
            )
        )

    if stats.event_count < THIN_HISTORY_EVENTS:
        gaps.append(
            ProfileGap(
                kind=GapKind.SPARSE_HISTORY,
                detail=(
                    f"only {stats.event_count} plays were recorded, which is too few for "
                    "reliable ranking or diversity estimates"
                ),
            )
        )

    sources = history.sources
    if len(sources) == 1:
        gaps.append(
            ProfileGap(
                kind=GapKind.SINGLE_SOURCE,
                detail=(
                    f"every play comes from {sources[0]!r}; listening that the source does "
                    "not record is invisible to this profile"
                ),
            )
        )

    if stats.event_count and stats.repeated_tracks == 0:
        gaps.append(
            ProfileGap(
                kind=GapKind.NO_REPEAT_PLAYS,
                detail=(
                    "no track was played on two separate days, so there is no evidence "
                    "of habit or familiarity in this history"
                ),
            )
        )

    missing_album = sum(1 for event in history.events if event.track.album is None)
    if missing_album and missing_album < stats.event_count:
        gaps.append(
            ProfileGap(
                kind=GapKind.MISSING_ALBUM_METADATA,
                detail=(
                    f"{missing_album} of {stats.event_count} plays have no album, so "
                    "album-level listening cannot be summarised for them"
                ),
            )
        )

    missing_duration = sum(
        1 for event in history.events if event.track.duration_seconds is None
    )
    if missing_duration and missing_duration < stats.event_count:
        gaps.append(
            ProfileGap(
                kind=GapKind.MISSING_DURATION_METADATA,
                detail=(
                    f"{missing_duration} of {stats.event_count} plays have no duration, so "
                    "session structure and total listening time are unknown"
                ),
            )
        )

    return tuple(gaps)
