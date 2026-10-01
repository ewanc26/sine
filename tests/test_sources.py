"""Per-service source adapters.

Each adapter is responsible only for reading one export format and declaring what
its field names mean. These tests keep the format quirks that actually lose data if
they regress: podcast rows mixed into Spotify exports, schema generations in
Apple's CSV, Takeout's non-music rows, ListenBrainz archive members, and the
ambiguous-artist case that must not be guessed at.
"""

import json
from pathlib import Path

import pytest

from sine.ingestion import IngestOptions, ingest
from sine.ingestion.sources import (
    AppleMusicSchemaError,
    AppleMusicSource,
    JsonListeningHistorySource,
    LastFmSource,
    ListenBrainzSource,
    SpotifySource,
    YouTubeMusicSource,
    resolve_source_name,
)
from sine.ingestion.sources.apple_music import artist_fallbacks

LONDON = IngestOptions(default_timezone="Europe/London")


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, payload: object) -> Path:
    return write(path, json.dumps(payload))


# ------------------------------------------------------------------------ last.fm


def test_lastfm_reads_epoch_seconds_and_ignores_header_case(tmp_path: Path) -> None:
    path = write(
        tmp_path / "lastfm.csv",
        "Artist,Track,uts\nBoards of Canada,Roygbiv,1721213954\n",
    )
    result = ingest(LastFmSource(path))
    assert result.accepted == 1
    event = result.history.events[0]
    assert event.source == "lastfm"
    assert event.track.title == "Roygbiv"
    assert event.track.artists[0].name == "Boards of Canada"
    assert event.played_at.isoformat() == "2024-07-17T10:59:14+00:00"


def test_lastfm_sniffs_semicolon_delimiters(tmp_path: Path) -> None:
    path = write(
        tmp_path / "lastfm.csv",
        "artist;album;track;uts\nAphex Twin;Selected Ambient Works;Xtal;1721213954\n",
    )
    (event,) = ingest(LastFmSource(path)).history.events
    assert event.track.artists[0].name == "Aphex Twin"
    assert event.track.album is not None
    assert event.track.album.title == "Selected Ambient Works"


def test_lastfm_accepts_alternative_column_spellings(tmp_path: Path) -> None:
    path = write(
        tmp_path / "lastfm.csv",
        "artist_name,track_name,utc_time\nTalk Talk,After the Flood,2026-07-17 11:39:14\n",
    )
    (event,) = ingest(LastFmSource(path), options=LONDON).history.events
    assert event.track.title == "After the Flood"
    assert event.played_at.isoformat() == "2026-07-17T10:39:14+00:00"


def test_lastfm_rejects_rows_missing_a_title(tmp_path: Path) -> None:
    path = write(tmp_path / "lastfm.csv", "Artist,Track,uts\nA,,1721213954\n")
    result = ingest(LastFmSource(path))
    assert result.accepted == 0
    assert len(result.rejected) == 1


def test_lastfm_tolerates_a_byte_order_mark(tmp_path: Path) -> None:
    path = tmp_path / "lastfm.csv"
    path.write_text("\ufeffArtist,Track,uts\nA,B,1721213954\n", encoding="utf-8")
    assert ingest(LastFmSource(path)).accepted == 1


# ------------------------------------------------------------------------- spotify


def test_spotify_keeps_music_and_rejects_podcasts(tmp_path: Path) -> None:
    path = write_json(
        tmp_path / "spotify.json",
        [
            {
                "ts": "2026-07-17T11:39:14Z",
                "master_metadata_track_name": "Roygbiv",
                "master_metadata_album_artist_name": "Boards of Canada",
                "master_metadata_album_album_name": "Music Has the Right to Children",
                "spotify_track_uri": "spotify:track:abc123",
            },
            {
                "ts": "2026-07-17T11:40:00Z",
                "episode_name": "A Podcast Episode",
                "spotify_episode_uri": "spotify:episode:xyz",
            },
        ],
    )
    result = ingest(SpotifySource(path))
    assert result.accepted == 1
    (event,) = result.history.events
    assert event.track.title == "Roygbiv"
    # The service's own identifier is kept verbatim rather than rewritten.
    assert event.source_event_id == "spotify:track:abc123"
    assert len(result.rejected) == 1
    assert "podcast" in result.rejected[0].reason


def test_spotify_does_not_record_played_duration_as_track_duration(
    tmp_path: Path,
) -> None:
    """``ms_played`` is how long playback lasted, not how long the track is."""

    path = write_json(
        tmp_path / "spotify.json",
        [
            {
                "ts": "2026-07-17T11:39:14Z",
                "master_metadata_track_name": "Roygbiv",
                "master_metadata_album_artist_name": "Boards of Canada",
                "ms_played": 15000,
            }
        ],
    )
    (event,) = ingest(SpotifySource(path)).history.events
    assert event.track.duration_seconds is None


# --------------------------------------------------------------------- listenbrainz


@pytest.mark.parametrize(
    "payload",
    [
        [
            {
                "listened_at": 1721213954,
                "track_metadata": {"track_name": "X", "artist_name": "A"},
            }
        ],
        {
            "listens": [
                {
                    "listened_at": 1721213954,
                    "track_metadata": {"track_name": "X", "artist_name": "A"},
                }
            ]
        },
        {
            "payload": {
                "listens": [
                    {
                        "listened_at": 1721213954,
                        "track_metadata": {"track_name": "X", "artist_name": "A"},
                    }
                ]
            }
        },
    ],
)
def test_listenbrainz_reads_every_export_shape(tmp_path: Path, payload: object) -> None:
    path = write_json(tmp_path / "lb.json", payload)
    result = ingest(ListenBrainzSource(path))
    assert result.accepted == 1
    assert result.history.events[0].source == "listenbrainz"


def test_listenbrainz_reads_json_lines(tmp_path: Path) -> None:
    path = write(
        tmp_path / "lb.jsonl",
        "\n".join(
            json.dumps(
                {
                    "listened_at": 1721213954 + offset,
                    "track_metadata": {"track_name": f"T{offset}", "artist_name": "A"},
                }
            )
            for offset in range(3)
        ),
    )
    assert ingest(ListenBrainzSource(path)).accepted == 3


def test_listenbrainz_uses_mbid_mapping_for_identity_and_full_credits(
    tmp_path: Path,
) -> None:
    path = write_json(
        tmp_path / "lb.json",
        [
            {
                "listened_at": 1721213954,
                "track_metadata": {
                    "track_name": "Roygbiv",
                    "artist_name": "Boards of Canada",
                    "mbid_mapping": {
                        "recording_mbid": "89ad4ac3-09f9-4e6e-9d3d-0e0f2b0a5c88",
                        "release_mbid": "0e0f2b0a-5c88-4e6e-9d3d-89ad4ac309f9",
                        "artists": [
                            {
                                "artist_credit_name": "Boards of Canada",
                                "artist_mbid": "67f66082-79b5-4af2-a8c9-9cbf0d8a4d61",
                            },
                        ],
                    },
                    "additional_info": {"isrc": "GBAAA0000001"},
                },
            }
        ],
    )
    (event,) = ingest(ListenBrainzSource(path)).history.events
    assert event.track.recording_mbid == "89ad4ac3-09f9-4e6e-9d3d-0e0f2b0a5c88"
    assert event.track.isrc == "GBAAA0000001"
    assert event.track.identity.startswith("mbid")


def test_listenbrainz_reads_archives_and_skips_unrelated_members(
    tmp_path: Path,
) -> None:
    import zipfile

    listens = json.dumps(
        [
            {
                "listened_at": 1721213954,
                "track_metadata": {"track_name": "X", "artist_name": "A"},
            }
        ]
    )
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("2024/listens.json", listens)
        bundle.writestr("2024/user.json", json.dumps({"user_name": "someone"}))
        bundle.writestr("2024/feedback.jsonl", json.dumps({"created": 1}))
        bundle.writestr("README.txt", "not a listen")
    assert ingest(ListenBrainzSource(archive)).accepted == 1


def test_listenbrainz_rejects_listens_without_track_metadata(tmp_path: Path) -> None:
    path = write_json(tmp_path / "lb.json", [{"listened_at": 1721213954}])
    result = ingest(ListenBrainzSource(path))
    assert result.accepted == 0
    assert "track metadata" in result.rejected[0].reason


# ------------------------------------------------------------------ youtube music


def test_youtube_music_keeps_plays_and_rejects_plain_videos(tmp_path: Path) -> None:
    path = write_json(
        tmp_path / "ytm.json",
        [
            {
                "header": "YouTube Music",
                "title": "Watched Roygbiv",
                "subtitles": [{"name": "Boards of Canada"}],
                "time": "2026-07-17T11:39:14Z",
                "titleUrl": "https://music.youtube.com/watch?v=abc",
            },
            {
                "header": "YouTube",
                "title": "Watched A Video",
                "subtitles": [{"name": "A Channel"}],
            },
            {
                "header": "YouTube Music",
                "title": "Watched Untimed",
                "subtitles": [{"name": "X"}],
            },
        ],
    )
    result = ingest(YouTubeMusicSource(path))
    assert result.accepted == 1
    (event,) = result.history.events
    assert event.track.title == "Roygbiv"
    assert event.track.artists[0].name == "Boards of Canada"
    assert len(result.rejected) == 2


def test_youtube_music_leaves_artist_unset_for_a_channel_url(tmp_path: Path) -> None:
    """A channel is not necessarily the performing artist."""

    path = write_json(
        tmp_path / "ytm.json",
        [
            {
                "header": "YouTube Music",
                "title": "Watched Something",
                "subtitles": [{"name": "https://music.youtube.com/channel/UC123"}],
                "time": "2026-07-17T11:39:14Z",
            }
        ],
    )
    result = ingest(YouTubeMusicSource(path))
    assert result.accepted == 0
    assert "artist" in result.rejected[0].reason


# --------------------------------------------------------------------- apple music


def test_apple_music_converts_milliseconds_to_seconds(tmp_path: Path) -> None:
    path = write(
        tmp_path / "apple.csv",
        "Song Name,Artist Name,Album Name,Event End Timestamp,"
        "Media Duration In Milliseconds\n"
        "Roygbiv,Boards of Canada,Music Has the Right to Children,"
        "2026-07-17 11:39:14,300000\n",
    )
    (event,) = ingest(AppleMusicSource(path), options=LONDON).history.events
    assert event.track.duration_seconds == 300
    assert event.source == "apple-music"
    assert event.played_at.isoformat() == "2026-07-17T10:39:14+00:00"


def test_apple_music_accepts_older_column_generations(tmp_path: Path) -> None:
    path = write(
        tmp_path / "apple.csv",
        "Content Name,Container Artist Name,Container Album Name,Event End Timestamp\n"
        "Roygbiv,Boards of Canada,Music,2026-07-17 11:39:14\n",
    )
    (event,) = ingest(AppleMusicSource(path), options=LONDON).history.events
    assert event.track.artists[0].name == "Boards of Canada"
    assert event.track.album is not None
    assert event.track.album.title == "Music"


def test_apple_music_requires_a_recognised_schema(tmp_path: Path) -> None:
    path = write(tmp_path / "wrong.csv", "Wrong Column,Value\nfoo,bar\n")
    with pytest.raises(AppleMusicSchemaError):
        AppleMusicSource(path).fetch()


def test_apple_music_rejects_naive_timestamps_without_a_timezone(
    tmp_path: Path,
) -> None:
    """Apple writes local time with no offset; Sine will not invent one."""

    path = write(
        tmp_path / "apple.csv",
        "Song Name,Artist Name,Event End Timestamp\nRoygbiv,Boards of Canada,2026-07-17 11:39:14\n",
    )
    result = ingest(AppleMusicSource(path))
    assert result.accepted == 0
    assert "timezone" in result.rejected[0].reason


def test_apple_music_falls_back_to_track_description_for_the_artist(
    tmp_path: Path,
) -> None:
    path = write(
        tmp_path / "apple.csv",
        "Song Name,Event End Timestamp,Track Description\n"
        "Roygbiv,2026-07-17 11:39:14,Boards of Canada - Roygbiv\n",
    )
    (event,) = ingest(AppleMusicSource(path), options=LONDON).history.events
    assert event.track.artists[0].name == "Boards of Canada"


def test_ambiguous_track_description_is_not_guessed_at() -> None:
    """A title credited to two artists yields no fallback at all."""

    rows = [
        {"Track Description": "Artist One - Shared Title"},
        {"Track Description": "Artist Two - Shared Title"},
    ]
    assert artist_fallbacks(rows) == {}


# ------------------------------------------------------------------- json/jsonl


def test_a_json_array_is_read_record_by_record(tmp_path: Path) -> None:
    path = write_json(
        tmp_path / "history.json",
        [
            {"title": "Roygbiv", "artist": "Boards of Canada", "played_at": 1721213954},
            {"title": "Xtal", "artist": "Aphex Twin", "played_at": 1721214000},
        ],
    )
    result = ingest(JsonListeningHistorySource(path))
    assert result.accepted == 2
    assert [event.track.title for event in result.history.events] == ["Roygbiv", "Xtal"]


def test_repeated_plays_are_counted_as_plays_not_as_stored_events(
    tmp_path: Path,
) -> None:
    """``accepted`` reports plays; deduplication merges them into fewer events."""

    path = write_json(
        tmp_path / "history.json",
        [
            {
                "title": "Roygbiv",
                "artist": "Boards of Canada",
                "played_at": "2026-09-01T10:00:00+00:00",
            },
            {
                "title": "Roygbiv",
                "artist": "Boards of Canada",
                "played_at": "2026-09-01T10:00:00+00:00",
            },
            {
                "title": "Roygbiv",
                "artist": "Boards of Canada",
                "played_at": "2026-09-01T11:00:00+00:00",
            },
        ],
    )
    result = ingest(JsonListeningHistorySource(path))
    assert result.accepted == 3
    assert len(result.history.events) == 2
    assert result.history.total_plays == 3


def test_an_array_entry_that_is_not_an_object_is_reported_not_dropped(
    tmp_path: Path,
) -> None:
    """A malformed entry is visible in the report rather than silently lost."""

    path = write_json(
        tmp_path / "history.json",
        [
            {"title": "Roygbiv", "artist": "Boards of Canada", "played_at": 1721213954},
            "a bare string",
        ],
    )
    result = ingest(JsonListeningHistorySource(path))
    assert result.accepted == 1
    assert len(result.rejected) == 1
    assert "expected an object" in result.rejected[0].reason


def test_the_lastfm_preset_reads_the_real_api_shape(tmp_path: Path) -> None:
    """Last.fm's API nests every name under ``#text``, not ``name``."""

    path = write_json(
        tmp_path / "recent.json",
        [
            {
                "artist": {
                    "#text": "Radiohead",
                    "mbid": "a74b1b7f-71a5-4011-9441-d0b5e4122711",
                },
                "album": {
                    "#text": "OK Computer",
                    "mbid": "b74b1b7f-71a5-4011-9441-d0b5e4122712",
                },
                "track": {
                    "#text": "Airbag",
                    "mbid": "c74b1b7f-71a5-4011-9441-d0b5e4122713",
                },
                "date": {"uts": "1137991440", "#text": "21 Jan 2006, 17:24"},
            }
        ],
    )
    result = ingest(JsonListeningHistorySource(path, preset="lastfm"))
    assert result.accepted == 1
    track = result.history.events[0].track
    assert track.title == "Airbag"
    assert track.artist.name == "Radiohead"
    assert track.album is not None
    assert track.album.title == "OK Computer"


def test_an_unknown_preset_names_the_ones_that_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        JsonListeningHistorySource(tmp_path / "history.json", preset="genius")


# ------------------------------------------------------------------------ registry


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("lastfm", "lastfm"),
        ("last.fm", "lastfm"),
        ("apple_music", "apple-music"),
        ("Apple", "apple-music"),
        ("ytmusic", "youtube-music"),
        ("jsonl", "json"),
    ],
)
def test_source_names_resolve_through_aliases(supplied: str, expected: str) -> None:
    assert resolve_source_name(supplied) == expected


def test_unknown_source_names_are_reported() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        resolve_source_name("napster")
