"""Profile construction.

The profile is the deterministic half of Sine: it summarises what was actually
observed, separately from anything inferred. These tests pin down determinism, the
separation of observed from inferred, and that thin histories say so rather than
producing confident nonsense.
"""

from datetime import UTC, datetime, timedelta

import pytest

from sine.models import Artist, ListeningEvent, ListeningHistory, Track
from sine.profile.builder import build_profile
from sine.profile.context import render_profile_context, render_request_context
from sine.profile.statistics import DEFAULT_TOP_N, compute_statistics

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def play(
    title: str,
    artist: str,
    days_ago: float,
    *,
    album: str | None = None,
    source: str = "test",
    play_count: int = 1,
) -> ListeningEvent:
    return ListeningEvent(
        track=Track(
            title=title,
            artists=(Artist(name=artist),),
            album=None if album is None else _album(album, artist),
        ),
        played_at=NOW - timedelta(days=days_ago),
        source=source,
        play_count=play_count,
    )


#: Phrases that mark an inference as a reading rather than an established fact.
HEDGES = (
    "may",
    "might",
    "could",
    "suggests",
    "possible",
    "likely",
    "consistent with",
    "provisional",
    "does not establish",
    "cannot distinguish",
    "not to explain",
)


def _is_hedged(text: str) -> bool:
    return any(marker in text.casefold() for marker in HEDGES)


def _album(title: str, artist: str):
    from sine.models import Album

    return Album(title=title, artist=Artist(name=artist))


def history(*events: ListeningEvent) -> ListeningHistory:
    return ListeningHistory.from_events(events)


def test_play_counts_are_kept_beyond_the_truncated_top_lists() -> None:
    """The top-* lists are for readability; the counts must not be lost with them."""

    events = [
        play(f"Filler {index}", f"Artist {index}", index + 1, play_count=5)
        for index in range(DEFAULT_TOP_N + 10)
    ]
    events.append(play("Obscure", "Rarely Played", 1, play_count=1))
    stats = compute_statistics(history(*events), now=NOW)

    assert len(stats.top_tracks) == DEFAULT_TOP_N
    obscure = Track(title="Obscure", artists=(Artist(name="Rarely Played"),))
    assert stats.track_plays[obscure.identity] == 1
    assert stats.artist_plays(Artist(name="Rarely Played")) == 1
    assert stats.is_tracked(obscure)


def test_new_artist_ratio_measures_artists_only_recently_heard() -> None:
    stats = compute_statistics(
        history(
            play("Old", "Long Term", 90),
            play("Recent", "Long Term", 2),
            play("New", "Only Recent", 1),
        ),
        now=NOW,
    )
    # Only "Only Recent" is recent-only; "Long Term" is heard in both windows.
    assert stats.new_artist_ratio == pytest.approx(1 / 3, abs=1e-6)


# ------------------------------------------------------------------ statistics


def test_statistics_count_plays_artists_and_tracks() -> None:
    stats = compute_statistics(
        history(
            play("Roygbiv", "Boards of Canada", 1, album="MHTRTC"),
            play("Roygbiv", "Boards of Canada", 2, album="MHTRTC"),
            play("Xtal", "Aphex Twin", 3),
        ),
        now=NOW,
    )
    assert stats.event_count == 3
    assert stats.unique_artists == 2
    assert stats.unique_tracks == 2
    assert stats.top_artists[0].item.name == "Boards of Canada"
    assert stats.top_artists[0].plays == 2


def test_play_count_multipliers_are_honoured() -> None:
    stats = compute_statistics(history(play("Roygbiv", "B", 1, play_count=5)), now=NOW)
    assert stats.event_count == 1
    assert stats.total_plays == 5
    assert stats.top_tracks[0].plays == 5


def test_statistics_are_independent_of_input_order() -> None:
    events = [
        play("A", "Artist One", 5),
        play("B", "Artist Two", 3),
        play("C", "Artist One", 1),
    ]
    forward = compute_statistics(history(*events), now=NOW)
    backward = compute_statistics(history(*reversed(events)), now=NOW)
    assert forward.model_dump() == backward.model_dump()


def test_the_same_inputs_produce_the_same_profile_every_time() -> None:
    events = [play("A", "One", 4), play("B", "Two", 2)]
    first = build_profile(history(*events), now=NOW)
    second = build_profile(history(*events), now=NOW)
    assert first.model_dump() == second.model_dump()


def test_recent_and_long_term_windows_are_separate() -> None:
    stats = compute_statistics(
        history(
            play("New", "Recent Only", 2),
            play("Old", "Long Gone", 200),
        ),
        now=NOW,
        recent_window_days=28,
    )
    recent_identities = {artist.identity for artist in stats.recent_only_artists}
    all_identities = {entry.item.identity for entry in stats.top_artists}
    assert "recent only" in recent_identities
    assert {"recent only", "long gone"} == all_identities
    assert stats.recent_only_artists


def test_top_artist_share_covers_the_top_ten_artists() -> None:
    """The measure is concentration across the top 10, not a single artist."""

    concentrated = compute_statistics(
        history(
            *[play(f"T{i}", f"Favourite {i}", 1, play_count=9) for i in range(3)],
            *[play(f"O{i}", f"Other {i}", 2, play_count=1) for i in range(20)],
        ),
        now=NOW,
    )
    assert concentrated.top_artist_share < 1.0

    # Every artist here falls inside the top 10, so the share is total.
    narrow = compute_statistics(
        history(play("A", "One", 1, play_count=3), play("B", "Two", 2, play_count=1)),
        now=NOW,
    )
    assert narrow.top_artist_share == 1.0


def test_an_empty_history_produces_empty_statistics() -> None:
    stats = compute_statistics(ListeningHistory(), now=NOW)
    assert stats.event_count == 0
    assert stats.unique_artists == 0
    assert stats.top_artists == ()


# --------------------------------------------------------------------- signals


def test_signals_are_derived_from_observed_counts() -> None:
    profile = build_profile(
        history(play("A", "One", 1, play_count=9), play("B", "Two", 2)), now=NOW
    )
    kinds = {signal.kind for signal in profile.signals}
    assert kinds  # some signal was produced
    assert all(signal.observation for signal in profile.signals)
    assert all(
        signal.inference is None or _is_hedged(signal.inference)
        for signal in profile.signals
    )


def test_a_thin_history_is_reported_as_thin() -> None:
    profile = build_profile(history(play("A", "One", 1)), now=NOW)
    assert profile.is_thin is True


def test_a_substantial_history_is_not_thin() -> None:
    events = [
        play(f"T{index}", f"Artist {index % 5}", index * 0.5) for index in range(60)
    ]
    profile = build_profile(history(*events), now=NOW)
    assert profile.is_thin is False


def test_gaps_describe_what_the_source_cannot_see() -> None:
    profile = build_profile(history(play("A", "One", 1, source="lastfm")), now=NOW)
    assert profile.gaps
    assert any("lastfm" in gap.detail for gap in profile.gaps)


def test_gaps_do_not_repeat_a_single_source() -> None:
    profile = build_profile(history(play("A", "One", 1, source="lastfm")), now=NOW)
    assert len([gap for gap in profile.gaps if "lastfm" in gap.detail]) == 1


# --------------------------------------------------------------------- context


def test_context_labels_statistics_as_observed() -> None:
    profile = build_profile(
        history(play("Roygbiv", "Boards of Canada", 1, album="MHTRTC")), now=NOW
    )
    rendered = render_profile_context(profile)
    assert "Roygbiv" in rendered
    assert "Boards of Canada" in rendered
    assert "observed" in rendered.casefold()


def test_context_is_deterministic() -> None:
    profile = build_profile(history(play("A", "One", 1), play("B", "Two", 3)), now=NOW)
    assert render_profile_context(profile) == render_profile_context(profile)


def test_request_context_includes_the_focus_and_limit() -> None:
    from sine.models import RecommendationFocus, RecommendationRequest

    rendered = render_request_context(
        RecommendationRequest(limit=5, focus=RecommendationFocus.DISCOVERY)
    )
    assert "discovery" in rendered.casefold()
    assert "5" in rendered


def test_an_empty_history_yields_usable_context() -> None:
    profile = build_profile(ListeningHistory(), now=NOW)
    rendered = render_profile_context(profile)
    assert isinstance(rendered, str)
    assert rendered.strip()
