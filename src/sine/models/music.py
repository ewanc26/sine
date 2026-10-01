"""Music entities: artists, albums, and tracks.

These are Sine's own representations, shared by every source.

Two kinds of identifier are treated differently, on purpose:

* **Service identifiers** — Apple Music catalogue IDs, Spotify URIs, Last.fm
  track IDs, YouTube URLs — stay on the :class:`~sine.models.history.ListeningEvent`
  that observed the play. They are meaningful only to the service that issued them.
* **MusicBrainz identifiers** are public, open, cross-service join keys rather than
  a service's private catalogue key, so ``recording_mbid`` and friends live on the
  track where they can be used to recognise the same recording arriving from two
  different services.

Sine is not a music database and performs no fuzzy identity resolution: artists are
matched by exact normalised name, and recordings by MusicBrainz ID when one is
present, otherwise by artist and title.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from sine.models.base import HistoryModel, identity_key

MBID_PATTERN = (
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class Artist(HistoryModel):
    """A performing artist or band.

    Sine's canonical artist identity is the normalised ``name`` only.
    """

    name: str = Field(min_length=1, description="Artist name as displayed.")
    sort_name: str | None = Field(
        default=None,
        description="Optional sorting name, when the source provides one.",
    )
    mbid: str | None = Field(
        default=None,
        pattern=MBID_PATTERN,
        description="MusicBrainz artist ID, when a source supplies one.",
    )

    @property
    def identity(self) -> str:
        return identity_key(self.name)

    @property
    def display_name(self) -> str:
        return self.name


class Album(HistoryModel):
    """A release that a track belongs to."""

    title: str = Field(min_length=1, description="Album title as displayed.")
    artist: Artist = Field(description="Primary credited artist for the release.")
    year: int | None = Field(
        default=None,
        ge=1860,
        le=2200,
        description="Release year, only when the source supplies it.",
    )
    mbid: str | None = Field(
        default=None, pattern=MBID_PATTERN, description="MusicBrainz release-group ID."
    )

    @property
    def identity(self) -> str:
        return identity_key(self.artist.name, self.title)

    @property
    def display_name(self) -> str:
        if self.year is None:
            return f"{self.title} ({self.artist.name})"
        return f"{self.title} ({self.artist.name}, {self.year})"


class Track(HistoryModel):
    """A recorded piece of music.

    All credited artists are kept, in credit order, because collaborations are
    common and dropping secondary credits would misrepresent what was played.
    """

    title: str = Field(min_length=1, description="Track title as displayed.")
    artists: tuple[Artist, ...] = Field(
        min_length=1, description="Credited artists, in credit order."
    )
    album: Album | None = Field(
        default=None,
        description="Containing release, only when the source supplies it.",
    )
    duration_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Track duration, only when the source supplies it.",
    )
    isrc: str | None = Field(
        default=None,
        pattern=r"^[A-Z]{2}[A-Z0-9]{3}\d{7}$",
        description="International Standard Recording Code, when supplied.",
    )
    recording_mbid: str | None = Field(
        default=None,
        pattern=MBID_PATTERN,
        description="MusicBrainz recording ID, used to join the same track across sources.",
    )
    release_mbid: str | None = Field(
        default=None,
        pattern=MBID_PATTERN,
        description="MusicBrainz release ID, when supplied.",
    )

    @field_validator("artists", mode="after")
    @classmethod
    def _dedupe_credits(cls, artists: tuple[Artist, ...]) -> tuple[Artist, ...]:
        seen: set[str] = set()
        unique: list[Artist] = []
        for artist in artists:
            if artist.identity not in seen:
                seen.add(artist.identity)
                unique.append(artist)
        return tuple(unique)

    @property
    def artist(self) -> Artist:
        """The primary credited artist."""

        return self.artists[0]

    @property
    def identity(self) -> str:
        """A stable identity for this recording.

        A MusicBrainz recording ID is authoritative when present because it is the
        one identifier two independent sources agree on. Otherwise fall back to the
        credited artists and title.
        """

        if self.recording_mbid:
            return f"mbid\x1f{self.recording_mbid.lower()}"
        credits = "\x1f".join(artist.identity for artist in self.artists)
        return identity_key(credits, self.title)

    @property
    def display_name(self) -> str:
        artists = ", ".join(artist.name for artist in self.artists)
        return f"{self.title} — {artists}"

    def to_source_dict(self) -> dict[str, Any]:
        """Return a plain mapping, useful for persisting or for LLM context."""

        return {
            "title": self.title,
            "artists": [artist.name for artist in self.artists],
            "album": self.album.title if self.album else None,
            "year": self.album.year if self.album else None,
            "duration_seconds": self.duration_seconds,
        }
