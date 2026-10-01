"""The ingestion boundary.

A listening-history source is anything that can produce raw records. Adding
Apple Music, a historical export, or a new service means implementing
:class:`ListeningHistorySource` and nothing else: the recommendation engine, the
profile pipeline, and the domain model are unaffected.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from pydantic import Field

from sine.ingestion.normalisation import (
    FieldMapping,
    IngestOptions,
    RejectedRecord,
    normalise_records,
)
from sine.ingestion.records import RawRecord
from sine.models.base import HistoryModel
from sine.models.history import ListeningHistory


@runtime_checkable
class ListeningHistorySource(Protocol):
    """A source of listening observations.

    Implementations do source-specific I/O only. They must not construct domain
    objects, and they must not decide what a record means.
    """

    @property
    def source_id(self) -> str:
        """Stable label recorded on every event this source produces."""

    def fetch(self) -> Iterable[RawRecord]:
        """Yield raw records, in whatever order the source produces them."""

    def field_mapping(self) -> FieldMapping:
        """Declare which of this source's field names carry which meaning."""


class IngestionResult(HistoryModel):
    """The outcome of reading one source."""

    source: str
    history: ListeningHistory
    accepted: int = Field(
        ge=0,
        description=(
            "Records that normalised successfully. This counts plays, not stored "
            "events: repeated plays of one track merge into a single event whose "
            "play_count sums them."
        ),
    )
    rejected: tuple[RejectedRecord, ...] = Field(default_factory=tuple)
    notes: str | None = None

    @property
    def total_records(self) -> int:
        return self.accepted + len(self.rejected)


def ingest(
    source: ListeningHistorySource,
    *,
    options: IngestOptions | None = None,
    notes: str | None = None,
) -> IngestionResult:
    """Read a source and normalise it into a :class:`ListeningHistory`.

    Rejected records are returned rather than raised; see
    :class:`~sine.ingestion.normalisation.RejectedRecord`.
    """

    events, rejected = normalise_records(
        source.fetch(),
        mapping=source.field_mapping(),
        source=source.source_id,
        options=options,
    )
    return IngestionResult(
        source=source.source_id,
        history=ListeningHistory.from_events(events),
        accepted=len(events),
        rejected=rejected,
        notes=notes,
    )
