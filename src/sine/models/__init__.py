"""Sine's music domain model.

This package is independent of both listening-history sources and LLM providers.
Sources adapt into these types; the recommendation engine reads them.
"""

from sine.models.base import (
    EntityKind,
    HistoryModel,
    TimeWindow,
    aware_utc,
    identity_key,
    normalise_text,
)
from sine.models.history import (
    ListeningEvent,
    ListeningHistory,
    deduplicate_events,
    merge_duplicate_events,
    normalise_events,
    same_listen,
    sort_events,
)
from sine.models.music import Album, Artist, Track
from sine.models.profile import (
    GapKind,
    ListeningProfile,
    ProfileGap,
    ProfileSignal,
    SignalKind,
)
from sine.models.recommendation import (
    Confidence,
    EvidenceKind,
    Novelty,
    Playlist,
    PlaylistRequest,
    PlaylistTrack,
    Recommendation,
    RecommendationEvidence,
    RecommendationFocus,
    RecommendationRequest,
    RecommendationSet,
)
from sine.models.statistics import ListeningStatistics, PlayCount

__all__ = [
    "Album",
    "Artist",
    "Confidence",
    "EntityKind",
    "EvidenceKind",
    "GapKind",
    "HistoryModel",
    "ListeningEvent",
    "ListeningHistory",
    "ListeningProfile",
    "ListeningStatistics",
    "Novelty",
    "PlayCount",
    "Playlist",
    "PlaylistRequest",
    "PlaylistTrack",
    "ProfileGap",
    "ProfileSignal",
    "Recommendation",
    "RecommendationEvidence",
    "RecommendationFocus",
    "RecommendationRequest",
    "RecommendationSet",
    "SignalKind",
    "TimeWindow",
    "Track",
    "aware_utc",
    "deduplicate_events",
    "identity_key",
    "merge_duplicate_events",
    "normalise_events",
    "normalise_text",
    "same_listen",
    "sort_events",
]
