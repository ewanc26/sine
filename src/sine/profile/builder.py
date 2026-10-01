"""The listening-profile pipeline.

Assembles a :class:`~sine.models.profile.ListeningProfile` from a history. Entirely
deterministic: no model, no clock, no network. ``now`` is passed in by the caller
so that "recent" is a function of the inputs.
"""

from __future__ import annotations

from datetime import datetime

from sine.models.history import ListeningHistory
from sine.models.profile import ListeningProfile
from sine.profile.signals import build_gaps, build_signals
from sine.profile.statistics import (
    DEFAULT_RECENT_WINDOW_DAYS,
    DEFAULT_TOP_N,
    compute_statistics,
)

DEFAULT_MAX_CONTEXT_ENTITIES = 20


def build_profile(
    history: ListeningHistory,
    *,
    now: datetime,
    recent_window_days: int = DEFAULT_RECENT_WINDOW_DAYS,
    top_n: int = DEFAULT_TOP_N,
    max_context_entities: int = DEFAULT_MAX_CONTEXT_ENTITIES,
) -> ListeningProfile:
    """Construct the deterministic listening profile for a history.

    An empty history produces a valid profile with zero statistics rather than an
    error, so that callers can render a profile before importing anything.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    statistics = compute_statistics(
        history, now=now, recent_window_days=recent_window_days, top_n=top_n
    )
    return ListeningProfile(
        statistics=statistics,
        signals=build_signals(statistics),
        gaps=build_gaps(statistics, history),
        sources=history.sources,
        recent_window_days=recent_window_days,
        max_context_entities=max_context_entities,
    )


__all__ = [
    "DEFAULT_MAX_CONTEXT_ENTITIES",
    "DEFAULT_RECENT_WINDOW_DAYS",
    "DEFAULT_TOP_N",
    "build_profile",
]
