from datetime import datetime, timezone

from sine.models import ListeningEvent, Track
from sine.normalize import are_duplicates, canonicalize_timestamp, merge_events, normalize_string


def event(title: str, artist: str, timestamp: str, *, album: str | None = None) -> ListeningEvent:
    return ListeningEvent(
        track=Track(title=title, artists=(artist,), album=album),
        played_at=datetime.fromisoformat(timestamp.replace("Z", "+00:00")),
        source="test",
    )


def test_timestamp_formats_canonicalise_to_one_instant() -> None:
    assert canonicalize_timestamp("2026-07-17T11:39:14Z") == datetime(
        2026, 7, 17, 11, 39, 14, tzinfo=timezone.utc
    )
    assert canonicalize_timestamp("2026-07-17T11:39:14.000Z") == datetime(
        2026, 7, 17, 11, 39, 14, tzinfo=timezone.utc
    )


def test_normalize_string_ignores_case_punctuation_and_unicode_forms() -> None:
    assert normalize_string("DAGames") == "dagames"
    assert normalize_string("ＡＢＣ") == "abc"
    assert normalize_string("Jack Stauber’s") == "jack stauber s"
    assert normalize_string("Bru‑C") == "bru c"


def test_duplicate_window_merges_same_listen_across_sources() -> None:
    left = event("Track", "Artist", "2026-07-17T11:39:14Z")
    right = event("track", "artist", "2026-07-17T11:39:44.000Z", album="Album")
    assert are_duplicates(left, right, tolerance_seconds=60)
    merged = merge_events([right, left])
    assert len(merged) == 1
    assert merged[0].track.album == "Album"


def test_events_outside_window_are_distinct() -> None:
    left = event("Track", "Artist", "2026-07-17T11:39:14Z")
    right = event("Track", "Artist", "2026-07-17T11:41:15Z")
    assert not are_duplicates(left, right, tolerance_seconds=60)
