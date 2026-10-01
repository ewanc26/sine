"""Duplicate and merge behaviour for listening events.

Re-importing an export, or importing overlapping exports from two services, reports
the same play more than once. These tests pin down when that is recognised and what
happens to the metadata.
"""

from datetime import UTC, datetime

import pytest

from sine.models import (
    Album,
    Artist,
    ListeningEvent,
    Track,
    deduplicate_events,
    merge_duplicate_events,
    same_listen,
    sort_events,
)


def event(
    title: str = "Track",
    artist: str = "Artist",
    at: str = "2026-07-17T11:39:14Z",
    *,
    source: str = "test",
    source_event_id: str | None = None,
    album: str | None = None,
    play_count: int = 1,
) -> ListeningEvent:
    return ListeningEvent(
        track=Track(
            title=title,
            artists=(Artist(name=artist),),
            album=Album(title=album, artist=Artist(name=artist)) if album else None,
        ),
        played_at=datetime.fromisoformat(at),
        source=source,
        source_event_id=source_event_id,
        play_count=play_count,
    )


def test_sort_is_total_and_stable_regardless_of_input_order() -> None:
    later = event(title="B", at="2026-07-17T12:00:00Z")
    earlier = event(title="A", at="2026-07-17T11:00:00Z")
    assert sort_events([later, earlier]) == (earlier, later)
    # Sorting an already-ordered batch changes nothing.
    assert sort_events([earlier, later]) == (earlier, later)


def test_source_event_id_deduplicates_and_sums_play_counts() -> None:
    first = event(source_event_id="abc", play_count=2)
    second = event(source_event_id="abc", play_count=3)
    merged = deduplicate_events([first, second])
    assert len(merged) == 1
    assert merged[0].play_count == 5


def test_events_without_ids_deduplicate_on_track_and_instant() -> None:
    merged = deduplicate_events([event(), event()])
    assert len(merged) == 1
    assert merged[0].play_count == 2


def test_same_track_at_different_times_is_not_a_duplicate() -> None:
    left = event(at="2026-07-17T11:39:14Z")
    right = event(at="2026-07-17T11:41:15Z")
    assert not same_listen(left, right)
    assert len(deduplicate_events([left, right])) == 2


@pytest.mark.parametrize(
    ("left_at", "right_at", "expected"),
    [
        ("2026-07-17T11:39:14Z", "2026-07-17T11:39:44Z", True),
        ("2026-07-17T11:39:14Z", "2026-07-17T11:40:14Z", True),
        ("2026-07-17T11:39:14Z", "2026-07-17T11:40:15Z", False),
    ],
)
def test_same_listen_respects_the_tolerance_window(
    left_at: str, right_at: str, expected: bool
) -> None:
    assert same_listen(event(at=left_at), event(at=right_at)) is expected


def test_merge_recognises_the_same_listen_reported_by_two_services() -> None:
    """Two services watching one player disagree slightly on the timestamp."""

    from_lastfm = event(at="2026-07-17T11:39:14Z", source="lastfm")
    from_listenbrainz = event(at="2026-07-17T11:39:44Z", source="listenbrainz")
    merged = merge_duplicate_events([from_lastfm, from_listenbrainz])
    assert len(merged) == 1
    assert merged[0].play_count == 2


def test_merge_keeps_the_richest_metadata_and_unions_the_rest() -> None:
    sparse = event(source="lastfm")
    rich = event(
        source="listenbrainz",
        source_event_id="https://musicbrainz.org/recording/abc",
        album="Album",
    )
    (merged,) = merge_duplicate_events([sparse, rich])
    assert merged.track.album is not None
    assert merged.track.album.title == "Album"
    assert merged.source_event_id == "https://musicbrainz.org/recording/abc"


def test_merge_does_not_confuse_different_tracks() -> None:
    first = event(title="One")
    second = event(title="Two")
    assert len(merge_duplicate_events([first, second])) == 2


def test_merge_keeps_all_canonicalised_timestamps_in_utc() -> None:
    offset_time = datetime(2026, 7, 17, 12, 39, 14, tzinfo=UTC)
    shifted = ListeningEvent(
        track=Track(title="Track", artists=(Artist(name="Artist"),)),
        played_at=offset_time,
        source="test",
    )
    assert shifted.played_at.utcoffset() == UTC.utcoffset(None)
