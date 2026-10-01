"""Typed application configuration.

Configuration is a TOML file plus environment overrides, read with the standard
library. No settings dependency is added.

The important property is that credentials are never stored in the file and never
appear in output. ``api_key_env`` names the environment variable to read; the
value itself lives in a :class:`~pydantic.SecretStr` so that a stray ``print``,
log line, or ``model_dump`` cannot leak it.

    [llm]
    provider = "anthropic"
    model = "claude-sonnet-4-5"
    api_key_env = "ANTHROPIC_API_KEY"

    [llm.settings]
    temperature = 0.6
    max_output_tokens = 4096

    [data]
    data_dir = "~/.sine"
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from sine.llm.capabilities import StructuredOutputDialect
from sine.llm.errors import ProviderConfigurationError
from sine.llm.registry import profile_for

CONFIG_ENV_VAR = "SINE_CONFIG"
DEFAULT_CONFIG_FILENAME = "sine.toml"
USER_CONFIG_PATHS: tuple[Path, ...] = (
    Path("~/.config/sine/sine.toml"),
    Path("~/.sine/sine.toml"),
)


class GenerationSettings(BaseModel):
    """Sampling settings forwarded to the provider.

    ``temperature`` is sent only where the provider still accepts it. Anthropic
    deprecates the sampling parameters on current models, so the Anthropic adapter
    does not forward this value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float | None = Field(default=0.6, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=4096, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    retry_attempts: int = Field(default=3, ge=1, le=10)


class LLMConfig(BaseModel):
    """Which model to use, and how to reach it."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    provider: str = Field(min_length=1, description="Provider id, e.g. 'openai'.")
    model: str = Field(min_length=1, description="Model identifier at that provider.")
    base_url: str | None = Field(
        default=None,
        description="Override the provider's default endpoint, for gateways and proxies.",
    )
    api_key_env: str | None = Field(
        default=None, description="Environment variable holding the credential."
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="Credential, if supplied directly via the environment.",
    )
    structured_output: StructuredOutputDialect | None = Field(
        default=None,
        description=(
            "Override the provider's declared structured-output dialect. Use when a "
            "model supports more, or less, than the provider does."
        ),
    )
    settings: GenerationSettings = Field(default_factory=GenerationSettings)

    @model_validator(mode="after")
    def _known_provider(self) -> Self:
        profile_for(self.provider)
        return self

    def resolve_api_key(self, env: Mapping[str, str] | None = None) -> str | None:
        """Return the credential, preferring the environment over the config file.

        An explicit ``api_key`` wins over ``api_key_env``; otherwise the named
        environment variable is read. The resolved value is never written back into
        the config object.
        """

        if self.api_key is not None:
            return self.api_key.get_secret_value()
        source = env if env is not None else os.environ
        if self.api_key_env:
            return source.get(self.api_key_env)
        profile = profile_for(self.provider)
        if profile.api_key_env:
            return source.get(profile.api_key_env)
        return None

    def describe(self) -> dict[str, Any]:
        """A safe-to-print description. Never contains the credential."""

        profile = profile_for(self.provider)
        return {
            "provider": self.provider,
            "family": str(profile.family),
            "model": self.model,
            "base_url": self.base_url or profile.base_url,
            "structured_output": str(
                self.structured_output or profile.structured_output
            ),
            "credential": "configured" if self.resolve_api_key() else "not configured",
            "settings": self.settings.model_dump(),
        }


class DataConfig(BaseModel):
    """Where Sine's local data lives."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_dir: Path = Field(default=Path("~/.sine"))

    @property
    def history_dir(self) -> Path:
        return self.data_dir / "history"

    @property
    def profile_dir(self) -> Path:
        return self.data_dir / "profiles"

    def ensure_dirs(self) -> None:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.profile_dir.mkdir(parents=True, exist_ok=True)

    def expand(self) -> DataConfig:
        return self.model_copy(update={"data_dir": self.data_dir.expanduser()})


class AppConfig(BaseModel):
    """Sine's application configuration."""

    model_config = ConfigDict(extra="forbid")

    llm: LLMConfig | None = None
    data: DataConfig = Field(default_factory=DataConfig)

    def require_llm(self) -> LLMConfig:
        """Return the LLM configuration, or explain that it is missing.

        Importing and profiling need no model, so ``llm`` may legitimately be
        absent; only the recommendation path requires it.
        """

        if self.llm is None:
            raise ProviderConfigurationError(
                "config",
                "no [llm] section configured. Set SINE_LLM_PROVIDER and SINE_LLM_MODEL, "
                "or add them to your config file.",
            )
        return self.llm

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, env: Mapping[str, str] | None = None
    ) -> AppConfig:
        """Build a config from parsed TOML, then apply environment overrides."""

        source = env if env is not None else os.environ
        # A config file need not describe a model at all: importing and profiling work
        # without one. The [llm] section may therefore come from the file, from the
        # environment, or from neither.
        llm: dict[str, Any] = dict(raw.get("llm", {}))
        if isinstance(llm.get("settings"), Mapping):
            llm["settings"] = dict(llm["settings"])
        _apply_env_overrides(llm, source)
        if not (llm.get("provider") and llm.get("model")):
            # Neither named in the file nor supplied by the environment. Settings on
            # their own are not a usable model, so this is an absent section rather
            # than a broken one.
            llm = {}

        data: dict[str, Any] = dict(raw.get("data", {}))
        _apply_data_env_overrides(data, source)
        return cls.model_validate({"llm": llm or None, "data": data})

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> AppConfig:
        """Load configuration from ``path``, or from the first config file found.

        Raises :class:`~sine.llm.errors.ProviderConfigurationError` when no
        configuration is available, rather than defaulting to a vendor.
        """

        source = env if env is not None else os.environ
        resolved = _resolve_path(path, source)
        if resolved is None:
            raise ProviderConfigurationError(
                "config",
                "no configuration found. Set SINE_CONFIG, create sine.toml, or pass "
                "--provider and --model.",
            )
        with resolved.open("rb") as handle:
            raw = tomllib.load(handle)
        return cls.from_mapping(raw, env=source)

    def describe(self) -> dict[str, Any]:
        """A safe-to-print summary of the whole configuration."""

        return {
            "llm": self.llm.describe() if self.llm is not None else None,
            "data_dir": str(self.data.expand().data_dir),
        }


def load_data_config(
    path: str | Path | None = None, *, env: Mapping[str, str] | None = None
) -> DataConfig:
    """Load only the data-directory configuration.

    Importing and profiling need no model, so they must work with no config file
    present. Returns defaults rather than raising.
    """

    source = env if env is not None else os.environ
    try:
        resolved = _resolve_path(path, source)
    except ProviderConfigurationError:
        resolved = None
    payload: dict[str, Any] = {}
    if resolved is not None:
        with resolved.open("rb") as handle:
            payload = dict(tomllib.load(handle).get("data", {}))
    _apply_data_env_overrides(payload, source)
    return DataConfig.model_validate(payload)


def _resolve_path(path: str | Path | None, env: Mapping[str, str]) -> Path | None:
    if path is not None:
        candidate = Path(path).expanduser()
        if not candidate.is_file():
            raise ProviderConfigurationError(
                "config", f"config file not found: {candidate}"
            )
        return candidate

    from_env = env.get(CONFIG_ENV_VAR)
    if from_env:
        candidate = Path(from_env).expanduser()
        if not candidate.is_file():
            raise ProviderConfigurationError(
                "config", f"{CONFIG_ENV_VAR} points at a missing file: {candidate}"
            )
        return candidate

    local = Path(DEFAULT_CONFIG_FILENAME)
    if local.is_file():
        return local
    for candidate in USER_CONFIG_PATHS:
        expanded = candidate.expanduser()
        if expanded.is_file():
            return expanded
    return None


#: Environment overrides, so a shell can select a model without editing a file.
_ENV_OVERRIDES: Mapping[str, str] = {
    "SINE_LLM_PROVIDER": "provider",
    "SINE_LLM_MODEL": "model",
    "SINE_LLM_BASE_URL": "base_url",
    "SINE_LLM_API_KEY": "api_key",
    "SINE_LLM_API_KEY_ENV": "api_key_env",
    "SINE_LLM_STRUCTURED_OUTPUT": "structured_output",
    "SINE_LLM_TEMPERATURE": "settings.temperature",
    "SINE_LLM_MAX_OUTPUT_TOKENS": "settings.max_output_tokens",
    "SINE_LLM_TIMEOUT": "settings.timeout_seconds",
    "SINE_LLM_RETRY_ATTEMPTS": "settings.retry_attempts",
}


def _apply_env_overrides(llm: dict[str, Any], env: Mapping[str, str]) -> None:
    settings = llm.setdefault("settings", {})
    for variable, key in _ENV_OVERRIDES.items():
        raw = env.get(variable)
        if raw is None or raw == "":
            continue
        if key == "api_key":
            llm["api_key"] = raw
        elif key.startswith("settings."):
            settings[key.split(".", 1)[1]] = _coerce(raw)
        else:
            llm[key] = raw


def _apply_data_env_overrides(data: dict[str, Any], env: Mapping[str, str]) -> None:
    raw = env.get("SINE_DATA_DIR")
    if raw:
        data["data_dir"] = raw


def _coerce(raw: str) -> Any:
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


__all__ = [
    "CONFIG_ENV_VAR",
    "DEFAULT_CONFIG_FILENAME",
    "AppConfig",
    "DataConfig",
    "GenerationSettings",
    "LLMConfig",
    "load_data_config",
]
