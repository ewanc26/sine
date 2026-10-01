"""Recommendation guidance that adapts to what the history actually measures.

A fixed instruction per focus tells the model what kind of answer was asked for. It
tells the model nothing about *this* listener: "recommend deeper cuts" is a different
task for someone with a hundred plays by five artists than for someone who plays
forty artists once each, and asking for discovery from a listener already exploring
broadly is not the same request as asking for it from one who plays the same four
records on repeat.

So the base instruction is followed by a few lines derived from the deterministic
statistics, which means the brief Sine sends is shaped by the data rather than
repeated regardless of it.

Two rules govern every line here:

* **Only measured facts.** The numbers come from
  :class:`~sine.models.statistics.ListeningStatistics`, so they are observations, not
  guesses. No line asserts anything about taste.
* **Bounded by the history.** A measurement covers recorded listening only, so the
  lines say what the data shows and what it cannot show, in the same hedged register
  as the rest of the profile.

Output is deterministic: the same profile and focus always produce the same lines in
the same order.
"""

from __future__ import annotations

from collections.abc import Mapping

from sine.models.recommendation import RecommendationFocus
from sine.models.statistics import ListeningStatistics

#: The instruction that says what was asked for, independent of the listener.
BASE_GUIDANCE: Mapping[RecommendationFocus, str] = {
    RecommendationFocus.DISCOVERY: (
        "Recommend tracks by artists the listener has not played before, drawn from "
        "the musical neighbourhood of what they do play."
    ),
    RecommendationFocus.DEEPENING: (
        "Recommend deeper cuts by artists the listener already plays, including "
        "rarer material, and work that sits alongside their established listening."
    ),
    RecommendationFocus.RECENT_ROTATION: (
        "Weight the listener's most recent listening most heavily. They are asking "
        "what suits the current phase, not the long run."
    ),
    RecommendationFocus.FAMILIARITY: (
        "Favour well-known, accessible tracks in the listener's own territory. "
        "Reliability beats novelty here."
    ),
    RecommendationFocus.SURPRISE: (
        "Stretch beyond the obvious. Take the listener's listening seriously but "
        "reach past its edge, and explain the connection you are making."
    ),
}

#: Below this many distinct artists the history cannot show breadth, whatever the
#: play share looks like: four artists played evenly is still four artists.
BROAD_ENOUGH_ARTISTS = 10

#: At or above this share of tracks played on a single day, the listener is not
#: returning to individual tracks.
ONE_OFF_IS_BROAD = 0.5

#: Above this share of plays, listening sits inside a core group, and a brief that
#: leans on "more of the same" would not widen it.
CONCENTRATED_SHARE = 0.6

#: At or below this mean plays per artist, the history records about one play per
#: artist and cannot show whether the listener has heard the rest of that work.
SHALLOW_PLAYS_PER_ARTIST = 1.5

#: At or above this share of tracks played once, a brief built on a favourite track
#: would not fit the listener's recorded habits.
ONE_OFF_SHARE = 0.6

#: Below this share of plays inside the recent window, the history is mostly
#: long-term, so a recent-rotation brief has little recent evidence to work from.
RECENT_SHARE = 0.25


def focus_guidance(focus: RecommendationFocus, stats: ListeningStatistics) -> list[str]:
    """Return the base instruction for ``focus``, plus measured adjustments.

    The base line is always first so the model's primary task is unambiguous; the
    adjustments follow, each one a consequence of a measurement rather than a
    personality assumption about the listener.
    """

    lines = [BASE_GUIDANCE[focus]]
    match focus:
        case RecommendationFocus.DISCOVERY:
            lines.extend(_discovery(stats))
        case RecommendationFocus.DEEPENING:
            lines.extend(_deepening(stats))
        case RecommendationFocus.RECENT_ROTATION:
            lines.extend(_recent_rotation(stats))
        case RecommendationFocus.FAMILIARITY:
            lines.extend(_familiarity(stats))
        case RecommendationFocus.SURPRISE:
            lines.extend(_surprise(stats))
    return lines


def _discovery(stats: ListeningStatistics) -> list[str]:
    """Discovery means widening what is already there, so measure how wide it is."""

    if _is_broad(stats):
        return [
            (
                f"The listener already plays {stats.unique_artists} artists, and "
                f"{stats.one_off_ratio:.0%} of the tracks they play were played on a "
                "single day. More of the same kind of thing is not discovery: name "
                "specific artists rather than broad genres, and prefer the unfamiliar "
                "over the merely adjacent."
            )
        ]
    if stats.top_artist_share >= CONCENTRATED_SHARE:
        return [
            (
                f"The listener's plays are concentrated: {stats.top_artist_share:.0%} "
                f"come from their top 10 artists, across {stats.unique_artists} in "
                "total. Reach further out than you otherwise would, and expect the "
                "connection to be thinner than usual."
            )
        ]
    return []


def _deepening(stats: ListeningStatistics) -> list[str]:
    """Deeper cuts need to know how much of an artist the history has already seen."""

    if not stats.unique_artists:
        return []
    plays_per_artist = stats.total_plays / stats.unique_artists
    if plays_per_artist <= SHALLOW_PLAYS_PER_ARTIST:
        return [
            (
                f"The history records about {plays_per_artist:.1f} plays per artist "
                f"({stats.total_plays} plays across {stats.unique_artists} artists), so "
                "it cannot show which of their material the listener has already "
                "heard. Propose work adjacent to what is recorded, and say plainly "
                "which part of it is a guess."
            )
        ]
    return [
        (
            f"The listener averages {plays_per_artist:.1f} plays per artist, so some "
            "of each artist's catalogue is likely already covered. Avoid material "
            "adjacent to what is recorded, and do not repeat a track the history "
            "already contains."
        )
    ]


def _recent_rotation(stats: ListeningStatistics) -> list[str]:
    """Recent rotation is only as good as the recent evidence behind it."""

    if not stats.total_plays:
        return []
    recent_share = stats.recent_plays / stats.total_plays
    if recent_share >= RECENT_SHARE:
        return [
            (
                f"{recent_share:.0%} of the recorded plays fall inside the recent "
                "window, so recent listening is well evidenced. Weight it heavily, and "
                "do not treat older material as the listener's settled taste."
            )
        ]
    return [
        (
            f"Only {recent_share:.0%} of the recorded plays fall inside the recent "
            f"window ({stats.recent_plays} of {stats.total_plays}). The listener's "
            "current phase is thinly evidenced, so do not overstate what it is."
        )
    ]


def _familiarity(stats: ListeningStatistics) -> list[str]:
    """Familiarity only fits a listener who returns to things."""

    if stats.one_off_ratio >= ONE_OFF_SHARE:
        return [
            (
                f"{stats.one_off_ratio:.0%} of the tracks played were played on a "
                "single day, so the history records no settled favourite track to build "
                "on. Recommend artists the listener already plays rather than one "
                "particular song."
            )
        ]
    return [
        (
            f"{stats.repeat_ratio:.0%} of plays went to tracks repeated on more than "
            "one day, so the listener does return to tracks. Lead with material this "
            "history shows them playing."
        )
    ]


def _surprise(stats: ListeningStatistics) -> list[str]:
    """Surprise has to stretch something; where it can stretch depends on breadth."""

    if _is_broad(stats):
        return [
            (
                f"The listener already plays {stats.unique_artists} artists, so a jump "
                "across genres would not surprise them. Stretch within this territory "
                "instead: unfamiliar artists working in a similar register, or "
                "familiar artists in a mode the history does not record."
            )
        ]
    return [
        (
            f"The history records {stats.unique_artists} artists and "
            f"{stats.unique_tracks} tracks, so the listener's listening is not broad. "
            "Surprise means leaving that territory, and the connection you make is a "
            "long way from the data — say so rather than implying it follows from it."
        )
    ]


def _is_broad(stats: ListeningStatistics) -> bool:
    """True when the history shows genuine breadth, not just even play share."""

    return (
        stats.unique_artists >= BROAD_ENOUGH_ARTISTS
        and stats.one_off_ratio >= ONE_OFF_IS_BROAD
    )


__all__ = [
    "BASE_GUIDANCE",
    "BROAD_ENOUGH_ARTISTS",
    "CONCENTRATED_SHARE",
    "ONE_OFF_IS_BROAD",
    "ONE_OFF_SHARE",
    "RECENT_SHARE",
    "SHALLOW_PLAYS_PER_ARTIST",
    "focus_guidance",
]
