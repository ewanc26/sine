"""Recommendation guidance adapted to the listener's own measurements.

A focus is not a fixed instruction: "recommend deeper cuts" means something different
depending on whether the history records one play per artist or twenty. These tests
pin down that each focus adapts to the shape of the history, that every adaptation
cites a measurement rather than an assumption about taste, and that the brief Sine
sends stays deterministic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sine.models import (
    Artist,
    ListeningEvent,
    ListeningHistory,
    RecommendationFocus,
    RecommendationRequest,
    Track,
)
from sine.profile.builder import build_profile
from sine.recommend.focus import (
    BASE_GUIDANCE,
    CONCENTRATED_SHARE,
    ONE_OFF_SHARE,
    SHALLOW_PLAYS_PER_ARTIST,
    focus_guidance,
)

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def plays(artist: str, title: str, days_ago: float, count: int = 1) -> ListeningEvent:
    return ListeningEvent(
        track=Track(title=title, artists=(Artist(name=artist),)),
        played_at=NOW - timedelta(days=days_ago),
        source="test",
        play_count=count,
    )


def narrow_history() -> ListeningHistory:
    """Three artists, played hard: the shape of a settled, habitual listener."""

    return ListeningHistory.from_events(
        tuple(
            plays(f"Artist {artist}", f"Track {track}", day, count=4)
            for artist in range(3)
            for track in range(3)
            for day in (1, 2, 3, 4)
        )
    )


def broad_history() -> ListeningHistory:
    """Twenty artists, one play each: the shape of a listener who explores."""

    return ListeningHistory.from_events(
        tuple(
            plays(f"Artist {artist}", f"Track {artist}", day)
            for artist, day in enumerate(range(20))
        )
    )


def old_history() -> ListeningHistory:
    """Listening that all happened before the recent window."""

    return ListeningHistory.from_events(
        tuple(
            plays(f"Artist {index}", f"Track {index}", 60 + index, count=2)
            for index in range(10)
        )
    )


def recent_history() -> ListeningHistory:
    """Listening that all happened inside the recent window."""

    return ListeningHistory.from_events(
        tuple(
            plays(f"Artist {index}", f"Track {index}", index % 20, count=2)
            for index in range(20)
        )
    )


def guidance_for(events: ListeningHistory, focus: RecommendationFocus) -> list[str]:
    profile = build_profile(events, now=NOW)
    return focus_guidance(focus, profile.statistics)


def even_history() -> ListeningHistory:
    """Thirty artists played evenly: neither narrow nor habitually repeated."""

    return ListeningHistory.from_events(
        tuple(
            plays(f"Artist {artist}", f"Track {artist}-{track}", day, count=2)
            for artist in range(30)
            for track in range(2)
            for day in (1, 2)
        )
    )


# ------------------------------------------------------------------- base lines

#: The opening words of each focus's fixed instruction.
BASE_WORD = {
    RecommendationFocus.DISCOVERY: "Recommend tracks by artists",
    RecommendationFocus.DEEPENING: "Recommend deeper cuts",
    RecommendationFocus.RECENT_ROTATION: "Weight the listener",
    RecommendationFocus.FAMILIARITY: "Favour well-known",
    RecommendationFocus.SURPRISE: "Stretch beyond the obvious",
}


def test_the_base_instruction_is_always_first() -> None:
    for focus in RecommendationFocus:
        lines = guidance_for(broad_history(), focus)
        assert lines[0].startswith(BASE_WORD[focus])


def test_an_empty_history_still_gets_a_task() -> None:
    """Nothing measured means the base instruction and nothing invented."""

    lines = guidance_for(
        ListeningHistory.from_events(()), RecommendationFocus.DISCOVERY
    )
    assert lines == [BASE_GUIDANCE[RecommendationFocus.DISCOVERY]]


# -------------------------------------------------------------------- discovery


def test_discovery_reaches_further_out_for_a_concentrated_listener() -> None:
    lines = " ".join(guidance_for(narrow_history(), RecommendationFocus.DISCOVERY))
    assert "are concentrated" in lines
    assert "Reach further out" in lines
    assert "more of the same kind of thing is not discovery" not in lines


def test_discovery_is_specificated_rather_than_broadened_for_an_explorer() -> None:
    lines = " ".join(guidance_for(broad_history(), RecommendationFocus.DISCOVERY))
    assert "More of the same kind of thing is not discovery" in lines
    assert "20 artists" in lines
    assert "Reach further out" not in lines


def test_even_play_share_across_few_artists_is_not_treated_as_breadth() -> None:
    """Four artists played evenly is still four artists, so only the measured fact
    about concentration is used."""

    stats = build_profile(narrow_history(), now=NOW).statistics
    assert stats.top_artist_share == 1.0
    assert stats.unique_artists < 10
    lines = " ".join(focus_guidance(RecommendationFocus.DISCOVERY, stats))
    assert "Reach further out" in lines
    assert "not discovery" not in lines


# ------------------------------------------------------------------- deepening


def test_deepening_admits_the_data_cannot_show_coverage() -> None:
    lines = " ".join(guidance_for(broad_history(), RecommendationFocus.DEEPENING))
    assert "cannot show which of their material" in lines
    assert "plays per artist" in lines


def test_deepening_asks_for_unheard_material_when_coverage_is_deep() -> None:
    lines = " ".join(guidance_for(narrow_history(), RecommendationFocus.DEEPENING))
    assert "already covered" in lines
    assert "do not repeat a track the history already contains" in lines


def test_the_shallow_threshold_is_used_as_documented() -> None:
    stats = build_profile(broad_history(), now=NOW).statistics
    assert stats.total_plays / stats.unique_artists <= SHALLOW_PLAYS_PER_ARTIST


# ----------------------------------------------------------- recent rotation


def test_recent_rotation_says_when_recent_listening_is_thin() -> None:
    lines = " ".join(guidance_for(old_history(), RecommendationFocus.RECENT_ROTATION))
    assert "thinly evidenced" in lines
    assert "do not overstate" in lines


def test_recent_rotation_leads_with_the_recent_share_when_it_is_well_evidenced() -> (
    None
):
    lines = " ".join(
        guidance_for(recent_history(), RecommendationFocus.RECENT_ROTATION)
    )
    assert "recent listening is well evidenced" in lines
    assert "settled taste" in lines


# ----------------------------------------------------------------- familiarity


def test_familiarity_does_not_promise_a_favourite_the_history_lacks() -> None:
    lines = " ".join(guidance_for(broad_history(), RecommendationFocus.FAMILIARITY))
    assert "no settled favourite track" in lines
    assert "rather than one particular song" in lines
    assert ONE_OFF_SHARE <= 1.0


def test_familiarity_leads_with_returned_to_tracks_for_a_habitual_listener() -> None:
    lines = " ".join(guidance_for(narrow_history(), RecommendationFocus.FAMILIARITY))
    assert "does return to tracks" in lines


# -------------------------------------------------------------------- surprise


def test_surprise_stretches_within_a_broad_listener_s_territory() -> None:
    lines = " ".join(guidance_for(broad_history(), RecommendationFocus.SURPRISE))
    assert "would not surprise them" in lines
    assert "Stretch within this territory" in lines


def test_surprise_leaves_a_narrow_territory_and_says_how_far() -> None:
    lines = " ".join(guidance_for(narrow_history(), RecommendationFocus.SURPRISE))
    assert "not broad" in lines
    assert "long way from the data" in lines
    assert "reach past its edge" in lines


def test_the_concentrated_threshold_is_used_as_documented() -> None:
    stats = build_profile(narrow_history(), now=NOW).statistics
    assert stats.top_artist_share >= CONCENTRATED_SHARE


# ----------------------------------------------------------------- determinism


def test_guidance_is_deterministic_for_the_same_history() -> None:
    for focus in RecommendationFocus:
        first = guidance_for(narrow_history(), focus)
        second = guidance_for(narrow_history(), focus)
        assert first == second


def test_guidance_adapts_to_the_history_rather_than_being_fixed() -> None:
    """The same focus must not produce the same brief for two different listeners."""

    narrow = " ".join(guidance_for(narrow_history(), RecommendationFocus.SURPRISE))
    broad = " ".join(guidance_for(broad_history(), RecommendationFocus.SURPRISE))
    assert narrow != broad
    assert narrow.startswith(BASE_WORD[RecommendationFocus.SURPRISE])
    assert broad.startswith(BASE_WORD[RecommendationFocus.SURPRISE])


# ------------------------------------------------------ the balance note


def discovery_prompt(events: ListeningHistory, *, seed: bool = False) -> str:
    from sine.recommend.prompts import build_user_prompt

    seeds = (Artist(name="Boards of Canada"),) if seed else ()
    return build_user_prompt(
        build_profile(events, now=NOW),
        RecommendationRequest(
            limit=5, focus=RecommendationFocus.DISCOVERY, seed_artists=seeds
        ),
    ).content


def test_the_balance_note_fills_the_gap_when_the_measurements_say_nothing() -> None:
    """Neither narrow nor habitually repeated: the nudge to stay near home stands."""

    assert "Balance familiarity and discovery" in discovery_prompt(even_history())


def test_the_balance_note_does_not_contradict_a_measured_instruction() -> None:
    """A concentrated history is told to reach further out, not to stay near home."""

    text = discovery_prompt(narrow_history())
    assert "Reach further out" in text
    assert "Balance familiarity and discovery" not in text


def test_a_seed_artist_suppresses_the_balance_note() -> None:
    """The caller named an anchor, so the nudge to stay generic is not wanted."""

    assert "Balance familiarity and discovery" not in discovery_prompt(
        even_history(), seed=True
    )
