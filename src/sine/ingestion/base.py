"""Interfaces shared by listening-history ingestion adapters."""

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from sine.models import ListeningEvent


class IngestionAdapter(Protocol):
    """Protocol implemented by source-specific history adapters."""

    source: str

    def load(self, path: Path) -> Iterable[ListeningEvent]:
        """Load observed listening events from a local export."""
