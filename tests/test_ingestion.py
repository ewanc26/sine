from sine.ingestion.apple_music import AppleMusicSchemaError, parse as parse_apple
from sine.ingestion.lastfm import parse as parse_lastfm
from sine.ingestion.listenbrainz import parse as parse_listenbrainz
from sine.ingestion.spotify import parse as parse_spotify
from sine.ingestion.youtube_music import parse as parse_youtube


def test_lastfm_export() -> None:
    events = parse_lastfm("artist,album,track,uts\nArtist,Album,Track,1721213954\n")
    assert len(events) == 1
    assert events[0].track.artists == ("Artist",)


def test_spotify_filters_podcasts() -> None:
    records = [
        {
            "ts": "2026-07-17T11:39:14Z",
            "master_metadata_track_name": "Track",
            "master_metadata_album_artist_name": "Artist",
            "master_metadata_album_album_name": "Album",
            "spotify_track_uri": "spotify:track:abc",
        },
        {"ts": "2026-07-17T11:40:00Z", "episode_name": "Podcast"},
    ]
    assert len(parse_spotify(records)) == 1


def test_apple_schema_and_duration() -> None:
    rows = "Song Name,Event End Timestamp,Media Duration In Milliseconds\nTrack,2026-07-17 11:39:14,180000\n"
    events = parse_apple(rows)
    assert events[0].track.duration_seconds == 180


def test_apple_schema_error() -> None:
    try:
        parse_apple("Wrong Column,Value\nfoo,bar\n")
    except AppleMusicSchemaError:
        pass
    else:
        raise AssertionError("expected AppleMusicSchemaError")


def test_youtube_music_filters_non_music_rows() -> None:
    records = [
        {
            "header": "YouTube Music",
            "title": "Watched Track",
            "subtitles": [{"name": "Artist"}],
            "time": "2026-07-17T11:39:14Z",
        },
        {"header": "YouTube", "title": "Watched Video", "subtitles": [{"name": "Channel"}]},
    ]
    assert len(parse_youtube(records)) == 1


def test_listenbrainz_wrapped_export() -> None:
    raw = '{"listens":[{"listened_at":1721213954,"track_metadata":{"artist_name":"Artist","track_name":"Track"}}]}'
    events = parse_listenbrainz(raw)
    assert len(events) == 1
    assert events[0].source == "listenbrainz"
