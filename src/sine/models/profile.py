"""The listening profile: observations, cautious inferences, and known gaps.

A profile is what Sine knows about a listener. It is built deterministically
from a listening history, with no model involved, and it always keeps three
things separate:

* ``observation`` — what was measured, which is a fact about the data;
* ``inference``   — a reading of that observation, which is a judgement;
* ``gaps``        — what the data cannot support.

Frequency is not treated as proof of preference anywhere in this module.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from sine.models.base import EntityKind, HistoryModel, TimeWindow
from sine.models.statistics import ListeningStatistics


class SignalKind(StrEnum):
    """Categories of pattern the deterministic profile can surface."""

    REPEATED_LISTENING = "repeated_listening"
    ONE_OFF_LISTENING = "one_off_listening"
    RECENT_FOCUS = "recent_focus"
    LONG_TERM_FOCUS = "long_term_focus"
    RECENCY_SHIFT = "recency_shift"
    FADED = "faded"
    BREADTH = "breadth"
    CONCENTRATION = "concentration"
    STEADY_ARTIST = "steady_artist"
    UNDERDETERMINED = "underdetermined"


class GapKind(StrEnum):
    """Ways in which the available data is insufficient."""

    SHORT_HISTORY = "short_history"
    SPARSE_HISTORY = "sparse_history"
    SINGLE_SOURCE = "single_source"
    NO_REPEAT_PLAYS = "no_repeat_plays"
    MISSING_ALBUM_METADATA = "missing_album_metadata"
    MISSING_DURATION_METADATA = "missing_duration_metadata"


class ProfileSignal(HistoryModel):
    """One pattern in the history, with its evidence and its limits."""

    kind: SignalKind
    observation: str = Field(
        min_length=1,
        description="Measured fact about the listening data. No interpretation.",
    )
    inference: str | None = Field(
        default=None,
        description=(
            "Reading of the observation. Phrase as a possibility, never as a fact."
        ),
    )
    subject: str | None = Field(default=None, description="Entity the signal is about.")
    subject_kind: EntityKind | None = None
    support: int | None = Field(
        default=None, ge=0, description="Observation count supporting the signal."
    )


class ProfileGap(HistoryModel):
    """A limitation that constrains how far the profile can be trusted."""

    kind: GapKind
    detail: str = Field(min_length=1)


class ListeningProfile(HistoryModel):
    """A deterministic, model-free picture of observed listening behaviour."""

    statistics: ListeningStatistics
    signals: tuple[ProfileSignal, ...] = Field(default_factory=tuple)
    gaps: tuple[ProfileGap, ...] = Field(default_factory=tuple)
    sources: tuple[str, ...] = Field(default_factory=tuple)
    recent_window_days: int = Field(ge=1)
    max_context_entities: int = Field(
        ge=1, le=200, description="Entity count included when rendering LLM context."
    )

    @property
    def window(self) -> TimeWindow:
        return self.statistics.window

    @property
    def is_thin(self) -> bool:
        """True when the profile is too sparse to support strong conclusions."""

        return self.statistics.event_count < 10 or self.window.days < 7

    def signals_of(self, kind: SignalKind) -> tuple[ProfileSignal, ...]:
        return tuple(signal for signal in self.signals if signal.kind == kind)
