"""The raw record an ingestion source yields, before normalisation.

A source is responsible only for producing raw records and declaring which of its
own field names mean what. Turning those fields into domain objects is
:mod:`sine.ingestion.normalisation`'s job, and is identical for every source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RawRecord:
    """One source-shaped listening observation.

    ``fields`` holds the source's own payload without interpretation. Sine keeps
    it as a plain mapping so that a new source never requires a domain change.
    """

    source: str
    fields: dict[str, Any] = field(default_factory=dict)
    index: int | None = None
    line_number: int | None = None
    source_path: Path | None = field(
        default=None,
    )
    """The local file this record was read from, kept as provenance."""
    error: str | None = field(
        default=None,
    )
    """Set when the record could not even be read, so the reason survives."""

    def describe(self) -> str:
        """A short, bounded description used in rejection reasons."""

        if self.line_number is not None:
            return f"line {self.line_number}"
        if self.index is not None:
            return f"record {self.index}"
        return "record"
