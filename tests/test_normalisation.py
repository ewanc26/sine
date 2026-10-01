"""Normalisation of source payloads into listening events.

Normalisation is where source-shaped data becomes Sine's domain, so these tests
focus on the two obligations that come with it: never invent metadata the source
did not supply, and never silently discard a record.
"""

from datetime import UTC, datetime

import pytest

from sine.ingestion import (
    FieldMapping,
    IngestOptions,
    RawRecord,
    RecordRejected,
    build_event,
    normalise_records,
    parse_timestamp,
)

SINE = FieldMapping(track_title="title", artist="artist", played_at="played_at")


def build(payload: dict[str, object], mapping: FieldMapping = SINE, **options: str):
    return build_event(
        payload, mapping=mapping, source="test", options=IngestOptions(**options)
    )


# ------------------------------------------------------------------- timestamps


def test_iso_timestamps_are_normalised_to_utc() -> None:
    parsed = parse_timestamp("2026-07-17T12:39:14+02:00", IngestOptions())
    assert parsed == datetime(2026, 7, 17, 10, 39, 14, tzinfo=UTC)


def test_z_suffix_is_understood() -> None:
    assert parse_timestamp("2026-07-17T10:39:14Z", IngestOptions()).hour == 10


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1721213954, "2024-07-17T10:59:14+00:00"),
        (1721213954000, "2024-07-17T10:59:14+00:00"),
        ("1721213954", "2024-07-17T10:59:14+00:00"),
    ],
)
def test_epoch_values_are_read_as_seconds_or_milliseconds(
    value: object, expected: str
) -> None:
    assert parse_timestamp(value, IngestOptions()).isoformat() == expected


def test_naive_timestamps_are_refused_rather_than_guessed() -> None:
    """Assuming UTC would fabricate a time the source never stated."""

    with pytest.raises(RecordRejected, match="no timezone"):
        parse_timestamp("2026-07-17 11:39:14", IngestOptions())


def test_naive_timestamps_use_the_stated_timezone() -> None:
    parsed = parse_timestamp(
        "2026-07-17 11:39:14", IngestOptions(default_timezone="Europe/London")
    )
    assert parsed.isoformat() == "2026-07-17T10:39:14+00:00"


def test_a_declared_format_is_used_for_unusual_layouts() -> None:
    options = IngestOptions(
        default_timezone="Europe/London", played_at_format="%d/%m/%Y %H:%M"
    )
    assert (
        parse_timestamp("17/07/2026 11:39", options).isoformat()
        == "2026-07-17T10:39:00+00:00"
    )


@pytest.mark.parametrize("value", ["not-a-date", "", None, True, "2026-13-45"])
def test_unusable_timestamps_are_rejected(value: object) -> None:
    with pytest.raises(RecordRejected):
        parse_timestamp(value, IngestOptions())


# ---------------------------------------------------------------------- events


def test_only_supplied_fields_are_carried_over() -> None:
    event = build(
        {"title": "Roygbiv", "artist": "Boards of Canada", "played_at": 1721213954}
    )
    assert event.track.title == "Roygbiv"
    assert event.track.album is None
    assert event.track.duration_seconds is None
    assert event.track.isrc is None
    assert event.track.recording_mbid is None
    assert event.metadata == {}


def test_optional_metadata_is_captured_when_present() -> None:
    mapping = FieldMapping(
        track_title="title",
        artist="artist",
        played_at="played_at",
        album="album",
        duration_seconds="ms",
        isrc="isrc",
    )
    event = build(
        {
            "title": "Roygbiv",
            "artist": "Boards of Canada",
            "played_at": 1721213954,
            "album": "Music Has the Right to Children",
            "ms": 300000,
            "isrc": "gb-abc-05-00001",
        },
        mapping,
    )
    assert event.track.album is not None
    assert event.track.album.title == "Music Has the Right to Children"
    assert event.track.duration_seconds == 300000
    # ISRCs are stored in their canonical unhyphenated form.
    assert event.track.isrc == "GBABC0500001"


def test_nested_paths_and_alternatives_are_resolved() -> None:
    mapping = FieldMapping(
        track_title=("title", "name", "track.name"),
        artist=("artist", "artist.name"),
        played_at="played_at",
    )
    event = build(
        {
            "name": "Roygbiv",
            "artist": {"name": "Boards of Canada"},
            "played_at": 1721213954,
        },
        mapping,
    )
    assert event.track.title == "Roygbiv"
    assert event.track.artists[0].name == "Boards of Canada"


def test_additional_credits_are_kept_in_order() -> None:
    mapping = FieldMapping(
        track_title="title", artist="artist", artists="artists", played_at="played_at"
    )
    event = build(
        {
            "title": "Folded",
            "artist": "Burial",
            "artists": ["Burial", "Kode9"],
            "played_at": 1721213954,
        },
        mapping,
    )
    assert [artist.name for artist in event.track.artists] == ["Burial", "Kode9"]
    assert event.track.artist.name == "Burial"


def test_repeated_credits_are_collapsed() -> None:
    mapping = FieldMapping(
        track_title="title", artist="artist", artists="artists", played_at="played_at"
    )
    event = build(
        {
            "title": "Folded",
            "artist": "Burial",
            "artists": ["Burial", "burial"],
            "played_at": 1721213954,
        },
        mapping,
    )
    assert len(event.track.artists) == 1


CREDITS = FieldMapping(
    track_title="title", artist="artist", artists="artists", played_at="played_at"
)
CREDITED = {
    "title": "Folded",
    "artist": "Burial",
    "played_at": 1721213954,
}
BURIAL = "a74b1b7f-71a5-4011-9441-d0b5e4122711"
KODE9 = "b74b1b7f-71a5-4011-9441-d0b5e4122712"


def test_credits_given_as_objects_are_read_rather_than_dropped() -> None:
    """Several sources give the credit list as objects, not bare strings."""

    event = build(
        dict(CREDITED, artists=[{"name": "Burial"}, {"name": "Kode9"}]),
        CREDITS,
    )
    assert [artist.name for artist in event.track.artists] == ["Burial", "Kode9"]


def test_a_credit_object_keeps_its_musicbrainz_id() -> None:
    event = build(
        dict(CREDITED, artists=[{"name": "Kode9", "mbid": KODE9}]),
        CREDITS,
    )
    assert event.track.artists[1].mbid == KODE9


def test_lastfm_style_hash_text_credits_are_read() -> None:
    """The Last.fm API names every entity ``#text``, including the credit list."""

    event = build(
        dict(CREDITED, artists=[{"#text": "Kode9", "mbid": KODE9}]),
        CREDITS,
    )
    assert event.track.artists[1].name == "Kode9"
    assert event.track.artists[1].mbid == KODE9


def test_a_repeated_primary_credit_is_collapsed_even_with_its_own_id() -> None:
    event = build(
        dict(
            CREDITED,
            artist={"name": "Burial", "mbid": BURIAL},
            artists=[{"name": "Burial", "mbid": BURIAL}, {"name": "Kode9"}],
        ),
        FieldMapping(
            track_title="title",
            artist=("artist.name", "artist"),
            artists="artists",
            played_at="played_at",
            artist_mbid="artist.mbid",
        ),
    )
    assert [artist.name for artist in event.track.artists] == ["Burial", "Kode9"]
    assert event.track.artists[0].mbid == BURIAL


def test_unreadable_credit_entries_are_skipped_not_invented() -> None:
    event = build(
        dict(CREDITED, artists=[42, None, {"mbid": KODE9}, {"name": "Kode9"}]),
        CREDITS,
    )
    assert [artist.name for artist in event.track.artists] == ["Burial", "Kode9"]


def test_musicbrainz_ids_become_the_recording_identity() -> None:
    mapping = FieldMapping(
        track_title="title",
        artist="artist",
        played_at="played_at",
        recording_mbid="recording_mbid",
    )
    event = build(
        {
            "title": "Roygbiv",
            "artist": "Boards of Canada",
            "played_at": 1721213954,
            "recording_mbid": "89AD4AC3-09F9-4E6E-9D3D-0E0F2B0A5C88",
        },
        mapping,
    )
    assert event.track.identity == "mbid\x1f89ad4ac3-09f9-4e6e-9d3d-0e0f2b0a5c88"


def test_a_malformed_musicbrainz_id_is_dropped_not_trusted() -> None:
    mapping = FieldMapping(
        track_title="title",
        artist="artist",
        played_at="played_at",
        recording_mbid="recording_mbid",
    )
    event = build(
        {
            "title": "Roygbiv",
            "artist": "A",
            "played_at": 1721213954,
            "recording_mbid": "not-a-uuid",
        },
        mapping,
    )
    assert event.track.recording_mbid is None
    assert not event.track.identity.startswith("mbid")


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"artist": "A", "played_at": 1721213954}, "track title"),
        ({"title": "T", "played_at": 1721213954}, "artist"),
        ({"title": "T", "artist": "A"}, "played_at"),
        ({"title": "   ", "artist": "A", "played_at": 1721213954}, "track title"),
    ],
)
def test_incomplete_records_are_rejected_with_a_reason(
    payload: dict, reason: str
) -> None:
    with pytest.raises(RecordRejected, match=reason):
        build(payload)


def test_values_failing_domain_validation_do_not_abort_the_import() -> None:
    """A bad year is a bad record, not a reason to lose the whole file."""

    mapping = FieldMapping(
        track_title="title",
        artist="artist",
        played_at="played_at",
        album="album",
        year="year",
    )
    events, rejected = normalise_records(
        [
            RawRecord(
                source="t",
                fields={
                    "title": "A",
                    "artist": "B",
                    "played_at": 1721213954,
                    "album": "X",
                },
            ),
            RawRecord(
                source="t",
                fields={
                    "title": "C",
                    "artist": "D",
                    "played_at": 1721213954,
                    "album": "Y",
                    "year": "99",
                },
            ),
        ],
        mapping=mapping,
        source="t",
    )
    assert len(events) == 1
    assert len(rejected) == 1
    assert "validation" in rejected[0].reason


# --------------------------------------------------------------------- batches


def test_records_the_source_could_not_read_are_reported() -> None:
    events, rejected = normalise_records(
        [
            RawRecord(source="t", fields={}, error="podcast episode, not a music play"),
            RawRecord(
                source="t",
                fields={"title": "A", "artist": "B", "played_at": 1721213954},
            ),
        ],
        mapping=SINE,
        source="t",
    )
    assert len(events) == 1
    assert rejected[0].reason == "podcast episode, not a music play"


def test_rejections_identify_their_line() -> None:
    _, rejected = normalise_records(
        [RawRecord(source="t", fields={}, line_number=17, error="malformed")],
        mapping=SINE,
        source="t",
    )
    assert rejected[0].describe() == "line 17: malformed"


def test_extra_fields_are_kept_verbatim_as_metadata() -> None:
    mapping = FieldMapping(
        track_title="title",
        artist="artist",
        played_at="played_at",
        extra_fields=("device", "note"),
    )
    event = build(
        {
            "title": "A",
            "artist": "B",
            "played_at": 1721213954,
            "device": "Living Room",
            "note": "",
        },
        mapping,
    )
    assert event.metadata == {"device": "Living Room"}
