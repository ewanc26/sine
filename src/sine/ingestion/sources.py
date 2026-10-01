"""Source registry for local listening-history imports."""

from pathlib import Path
from collections.abc import Callable

from sine.models import ListeningEvent
from sine.ingestion import IngestionAdapter
from sine.ingestion import apple_music, lastfm, listenbrainz, spotify, youtube_music


Loader = Callable[[Path], list[ListeningEvent]]


LOADERS: dict[str, Loader] = {
    "lastfm": lastfm.load,
    "spotify": spotify.load,
    "apple_music": apple_music.load,
    "youtube_music": youtube_music.load,
    "listenbrainz": listenbrainz.load,
}


def load(source: str, path: Path) -> list[ListeningEvent]:
    """Load an export through the named service adapter."""

    try:
        loader = LOADERS[source]
    except KeyError as exc:
        raise ValueError(f"Unsupported listening source: {source}") from exc
    return loader(path)
