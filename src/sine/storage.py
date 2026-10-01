"""Local persistence for imported histories and computed profiles.

JSON files under the configured data directory. Listening history is
user-owned data, so this module writes only what the user asked it to write, under
a directory the user chose, and never uploads anything.
"""

from __future__ import annotations

import json
from pathlib import Path

from sine.config import DataConfig
from sine.models.history import ListeningHistory
from sine.models.profile import ListeningProfile

HISTORY_SUFFIX = ".history.json"
PROFILE_SUFFIX = ".profile.json"


class Store:
    """Read and write Sine's local artefacts."""

    def __init__(self, data: DataConfig) -> None:
        self._data = data.expand()

    @property
    def data_dir(self) -> Path:
        return self._data.data_dir

    def history_path(self, name: str) -> Path:
        return self._data.history_dir / f"{_safe_name(name)}{HISTORY_SUFFIX}"

    def profile_path(self, name: str) -> Path:
        return self._data.profile_dir / f"{_safe_name(name)}{PROFILE_SUFFIX}"

    def save_history(self, name: str, history: ListeningHistory) -> Path:
        return _write(self.history_path(name), history)

    def load_history(self, name: str) -> ListeningHistory:
        return ListeningHistory.model_validate(
            _read(self.history_path(name), "history")
        )

    def save_profile(self, name: str, profile: ListeningProfile) -> Path:
        return _write(self.profile_path(name), profile)

    def load_profile(self, name: str) -> ListeningProfile:
        return ListeningProfile.model_validate(
            _read(self.profile_path(name), "profile")
        )

    def list_histories(self) -> tuple[str, ...]:
        return _list(self._data.history_dir, HISTORY_SUFFIX)

    def list_profiles(self) -> tuple[str, ...]:
        return _list(self._data.profile_dir, PROFILE_SUFFIX)


def _safe_name(name: str) -> str:
    """Reject path separators so a name cannot escape the data directory."""

    cleaned = name.strip()
    if not cleaned:
        raise ValueError("name must not be empty")
    if "/" in cleaned or "\\" in cleaned or cleaned in (".", ".."):
        raise ValueError(f"name must not contain path separators: {name!r}")
    return cleaned


def _write(path: Path, model: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _read(path: Path, kind: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"no {kind} named {path.stem!r} in {path.parent}")
    return json.loads(path.read_text(encoding="utf-8"))


def _list(directory: Path, suffix: str) -> tuple[str, ...]:
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(path.name[: -len(suffix)] for path in directory.glob(f"*{suffix}"))
    )


__all__ = ["HISTORY_SUFFIX", "PROFILE_SUFFIX", "Store"]
