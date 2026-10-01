"""Listening-history ingestion.

Sources adapt into :mod:`sine.models`; nothing above this package knows which
service a history came from.
"""

from sine.ingestion.normalisation import (
    FieldMapping,
    IngestOptions,
    RecordRejected,
    RejectedRecord,
    build_event,
    normalise_records,
    parse_timestamp,
)
from sine.ingestion.protocols import (
    IngestionResult,
    ListeningHistorySource,
    ingest,
)
from sine.ingestion.records import RawRecord
from sine.ingestion.sources import (
    SOURCES,
    AppleMusicSchemaError,
    AppleMusicSource,
    LastFmSource,
    ListenBrainzSource,
    SpotifySource,
    YouTubeMusicSource,
    open_source,
    resolve_source_name,
)

__all__ = [
    "SOURCES",
    "AppleMusicSchemaError",
    "AppleMusicSource",
    "FieldMapping",
    "IngestOptions",
    "IngestionResult",
    "LastFmSource",
    "ListenBrainzSource",
    "ListeningHistorySource",
    "RawRecord",
    "RecordRejected",
    "RejectedRecord",
    "SpotifySource",
    "YouTubeMusicSource",
    "build_event",
    "ingest",
    "normalise_records",
    "open_source",
    "parse_timestamp",
    "resolve_source_name",
]
