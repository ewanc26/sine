"""Shared primitives for Sine's music domain models.

Nothing in this package describes a listening-history source or an LLM provider.
Sine's domain model is Sine's own; services translate into and out of it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

IDENTITY_SEPARATOR = "\x1f"


class EntityKind(StrEnum):
    """The kinds of music entity Sine's domain model knows about."""

    ARTIST = "artist"
    ALBUM = "album"
    TRACK = "track"


class HistoryModel(BaseModel):
    """Base configuration for domain models.

    ``extra="forbid"`` keeps domain records closed so that a malformed or
    unexpected source field fails loudly instead of being silently absorbed.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


def normalise_text(value: str) -> str:
    """Return a comparison form for human-readable names and titles.

    Case and internal whitespace vary between listening-history sources for what
    is very often the same entity, so identity comparisons use this form. It is a
    display-preserving normalisation only: no stemming, transliteration, or
    punctuation stripping, because those can merge genuinely different artists.
    """

    return " ".join(value.split()).casefold()


def identity_key(*parts: str) -> str:
    """Build a stable, hashable identity from normalised identity parts."""

    return IDENTITY_SEPARATOR.join(normalise_text(part) for part in parts if part)


def aware_utc(value: datetime) -> datetime:
    """Normalise a datetime to UTC, rejecting naive datetimes.

    Listening sources record time with wildly different precision, and naive
    datetimes cannot be compared across sources without guessing a timezone.
    Callers that receive naive values must resolve the source timezone explicitly.
    """

    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC)


class TimeWindow(HistoryModel):
    """A closed, timezone-aware interval covered by observations."""

    start: datetime
    end: datetime

    @property
    def days(self) -> float:
        """Length of the window in days, inclusive of both endpoints."""

        return max((self.end - self.start).total_seconds() / 86_400.0, 0.0)

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment <= self.end


def dump(model: BaseModel, **kwargs: Any) -> dict[str, Any]:
    """JSON-mode dump helper used when persisting domain models."""

    return model.model_dump(mode="json", **kwargs)
