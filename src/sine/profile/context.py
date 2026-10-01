"""Rendering a profile as text for a language model.

The rendering is a projection of the profile, not a new source of claims. It
states the measurements, marks every interpretation as an interpretation, and
carries the gaps through verbatim, so that the model is told what the data cannot
support rather than being invited to fill the silence.

Output is deterministic: the same profile always produces byte-identical text, so
prompts are reproducible and diffable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from sine.models.base import HistoryModel
from sine.models.profile import ListeningProfile
from sine.models.recommendation import RecommendationRequest
from sine.models.statistics import PlayCount

ItemT = TypeVar("ItemT", bound=HistoryModel)

HEADER = "LISTENING DATA (computed by Sine from the listener's own history)"


def render_profile_context(profile: ListeningProfile) -> str:
    """Render a profile as model-facing context text."""

    stats = profile.statistics
    limit = profile.max_context_entities
    lines: list[str] = [HEADER, ""]

    lines.append("MEASURED")
    if stats.event_count == 0:
        lines.append("- No listening events were recorded.")
    else:
        lines.append(
            f"- {stats.event_count} plays recorded over {stats.window.days:.0f} days "
            f"({stats.window.start.date().isoformat()} to {stats.window.end.date().isoformat()})"
        )
        lines.append(
            f"- {stats.unique_artists} artists, {stats.unique_albums} albums, "
            f"{stats.unique_tracks} tracks; listening spread across "
            f"{stats.active_days} distinct days"
        )
        lines.append(
            f"- {stats.repeated_tracks} tracks were played on 2+ separate days; "
            f"{stats.one_off_tracks} tracks were played once"
        )
        lines.append(
            f"- {stats.recent_plays} plays in the last {profile.recent_window_days} days "
            f"versus {stats.long_term_plays} before that"
        )

    lines.extend(
        _entity_lines("MOST PLAYED ARTISTS", stats.top_artists, limit, "plays")
    )
    lines.extend(_entity_lines("MOST PLAYED ALBUMS", stats.top_albums, limit, "plays"))
    lines.extend(_entity_lines("MOST PLAYED TRACKS", stats.top_tracks, limit, "plays"))

    lines.append("")
    lines.append("OBSERVED PATTERNS (inference, not fact)")
    if profile.signals:
        for signal in profile.signals:
            lines.append(f"- {signal.observation}.")
            if signal.inference:
                lines.append(f"  Possible reading: {signal.inference}.")
    else:
        lines.append("- None derived from the available data.")

    lines.append("")
    lines.append("LIMITS OF THIS DATA")
    if profile.gaps:
        for gap in profile.gaps:
            lines.append(f"- {gap.detail}.")
    else:
        lines.append("- None recorded.")

    lines.append("")
    lines.append("SOURCES")
    if profile.sources:
        lines.append("- " + ", ".join(profile.sources) + ".")
    else:
        lines.append("- None.")

    return "\n".join(lines)


def _entity_lines[ItemT](
    title: str, entries: Sequence[PlayCount[ItemT]], limit: int, unit: str
) -> list[str]:
    lines = ["", title]
    if not entries:
        lines.append("- None recorded.")
        return lines
    for entry in entries[:limit]:
        lines.append(
            f"- {entry.item.display_name}: {entry.plays} {unit} over "
            f"{entry.active_days} day(s) (first {entry.first_played.date().isoformat()}, "
            f"last {entry.last_played.date().isoformat()})"
        )
    return lines


def render_request_context(request: RecommendationRequest) -> str:
    """Render the caller's request as model-facing context text."""

    lines = [
        "REQUEST",
        f"- focus: {request.focus.value}",
        f"- number of tracks: {request.limit}",
    ]
    lines.append(
        "- may recommend tracks already played: "
        + ("yes" if request.allow_replays else "no")
    )
    if request.seed_artists:
        lines.append(
            "- anchor on these artists: "
            + ", ".join(a.name for a in request.seed_artists)
        )
    if request.exclude_artists:
        lines.append(
            "- do not recommend these artists: "
            + ", ".join(a.name for a in request.exclude_artists)
        )
    if request.guidance:
        lines.append(f"- additional guidance from the listener: {request.guidance}")
    return "\n".join(lines)


__all__ = ["HEADER", "render_profile_context", "render_request_context"]
