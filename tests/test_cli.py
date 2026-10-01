"""Configuration, local storage, and the command-line interface.

The CLI is tested through ``main()`` rather than by inspecting argparse internals:
what matters is which source gets read, what gets written, what reaches the model,
and what the user sees. Configuration tests care most about one property — the
credential never appears in output, and a config that omits the model still loads,
because importing and profiling do not need one.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sine.cli import main
from sine.config import AppConfig, DataConfig, LLMConfig, load_data_config
from sine.llm.errors import ProviderConfigurationError
from sine.models.history import ListeningHistory
from sine.models.music import Artist, Track
from sine.storage import HISTORY_SUFFIX, PROFILE_SUFFIX, Store

NOW = datetime(2026, 10, 1, tzinfo=UTC)


# ------------------------------------------------------------------ fixtures


def sample_events() -> tuple:
    from sine.models.history import ListeningEvent

    return tuple(
        ListeningEvent(
            track=Track(
                title=f"Track {index}",
                artists=(Artist(name=f"Artist {index % 3}"),),
            ),
            played_at=NOW.replace(day=1) - __import__("datetime").timedelta(days=index),
            source="test",
            play_count=3,
        )
        for index in range(12)
    )


def history() -> ListeningHistory:
    return ListeningHistory.from_events(sample_events())


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def simple_history_file(tmp_path: Path) -> Path:
    return write_json(
        tmp_path / "history.json",
        [
            {
                "title": f"Track {index}",
                "artist": f"Artist {index % 3}",
                "album": "Some Album",
                "played_at": (
                    NOW.replace(day=1) - __import__("datetime").timedelta(days=index)
                ).isoformat(),
            }
            for index in range(12)
        ],
    )


# ------------------------------------------------------------------- config


def test_a_config_file_without_an_llm_section_loads() -> None:
    """Importing and profiling need no model, so [llm] is genuinely optional."""

    config = AppConfig.from_mapping({"data": {"data_dir": "/tmp/sine"}}, env={})
    assert config.llm is None
    assert config.data.data_dir == Path("/tmp/sine")


def test_an_llm_section_that_names_only_settings_is_treated_as_absent() -> None:
    """Settings alone are not a usable model, and must not fail the whole load."""

    config = AppConfig.from_mapping({"llm": {"settings": {"temperature": 0.2}}}, env={})
    assert config.llm is None


def test_the_environment_can_supply_the_model_the_file_omits() -> None:
    config = AppConfig.from_mapping(
        {"data": {}},
        env={"SINE_LLM_PROVIDER": "ollama", "SINE_LLM_MODEL": "llama3.2"},
    )
    assert config.llm is not None
    assert config.llm.provider == "ollama"
    assert config.llm.model == "llama3.2"


def test_the_environment_overrides_the_file() -> None:
    config = AppConfig.from_mapping(
        {"llm": {"provider": "openai", "model": "gpt-4o"}},
        env={"SINE_LLM_PROVIDER": "anthropic", "SINE_LLM_MODEL": "claude-sonnet-4-5"},
    )
    assert config.llm is not None
    assert config.llm.provider == "anthropic"


def test_an_unknown_provider_is_rejected_at_load_time() -> None:
    with pytest.raises(ProviderConfigurationError, match="unknown provider"):
        AppConfig.from_mapping({"llm": {"provider": "nope", "model": "x"}}, env={})


def test_a_credential_is_never_present_in_the_printable_description() -> None:
    config = AppConfig.from_mapping(
        {"llm": {"provider": "openai", "model": "gpt-4o"}},
        env={"SINE_LLM_API_KEY": "sk-super-secret"},
    )
    described = json.dumps(config.describe())

    assert "sk-super-secret" not in described
    assert "not configured" not in described
    assert config.describe()["llm"]["credential"] == "configured"


def test_an_absent_credential_is_reported_rather_than_crashing() -> None:
    config = AppConfig.from_mapping(
        {"llm": {"provider": "openai", "model": "gpt-4o"}}, env={}
    )
    assert config.describe()["llm"]["credential"] == "not configured"


def test_a_configured_credential_is_readable_only_through_the_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_KEY", "sk-real")
    config = AppConfig.from_mapping(
        {"llm": {"provider": "openai", "model": "gpt-4o", "api_key_env": "MY_KEY"}},
        env={},
    )
    assert config.require_llm().resolve_api_key() == "sk-real"
    assert "sk-real" not in json.dumps(config.require_llm().model_dump(mode="json"))


def test_the_provider_default_environment_variable_is_used_when_none_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    config = AppConfig.from_mapping(
        {"llm": {"provider": "anthropic", "model": "claude-x"}}, env={}
    )
    assert config.require_llm().resolve_api_key() == "sk-ant"


def test_an_explicit_credential_wins_over_the_environment_variable() -> None:
    named = LLMConfig(provider="openai", model="gpt-4o", api_key_env="MY_KEY")
    assert named.resolve_api_key(env={"MY_KEY": "from-env"}) == "from-env"

    direct = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-direct")
    assert direct.resolve_api_key(env={"MY_KEY": "from-env"}) == "sk-direct"


def test_the_llm_section_is_required_only_where_it_is_used() -> None:
    with pytest.raises(ProviderConfigurationError, match=r"no \[llm\] section"):
        AppConfig.from_mapping({"data": {}}, env={}).require_llm()


def test_loading_with_no_config_file_is_an_error_rather_than_a_default_vendor() -> None:
    """Defaulting to a provider would send listening history somewhere unchosen."""

    with pytest.raises(ProviderConfigurationError, match="no configuration found"):
        AppConfig.load(env={"HOME": "/nonexistent-sine-home"})


def test_a_config_path_that_does_not_exist_is_named_in_the_error() -> None:
    with pytest.raises(ProviderConfigurationError, match="config file not found"):
        AppConfig.load("/tmp/definitely-not-a-sine-config.toml")


def test_the_data_directory_can_be_read_without_any_config_file() -> None:
    data = load_data_config(env={"HOME": "/nonexistent-sine-home"})
    assert isinstance(data, DataConfig)
    assert data.data_dir


def test_the_data_directory_is_overridable_by_the_environment() -> None:
    data = load_data_config(env={"SINE_DATA_DIR": "/tmp/sine-env"})
    assert data.data_dir == Path("/tmp/sine-env")


def test_sampling_settings_are_coerced_from_their_environment_strings() -> None:
    config = AppConfig.from_mapping(
        {"llm": {"provider": "openai", "model": "gpt-4o"}},
        env={"SINE_LLM_TEMPERATURE": "0.15", "SINE_LLM_MAX_OUTPUT_TOKENS": "2048"},
    )
    settings = config.require_llm().settings
    assert settings.temperature == 0.15
    assert settings.max_output_tokens == 2048


def test_an_out_of_range_setting_is_rejected() -> None:
    with pytest.raises(ValueError, match="temperature"):
        AppConfig.from_mapping(
            {"llm": {"provider": "openai", "model": "gpt-4o"}},
            env={"SINE_LLM_TEMPERATURE": "9"},
        )


# ------------------------------------------------------------------ storage


def test_a_history_round_trips_through_the_store(tmp_path: Path) -> None:
    store = Store(DataConfig(data_dir=tmp_path / "data"))
    original = history()

    path = store.save_history("demo", original)
    loaded = store.load_history("demo")

    assert path.name == f"demo{HISTORY_SUFFIX}"
    assert loaded == original


def test_a_profile_round_trips_through_the_store(tmp_path: Path) -> None:
    from sine.profile.builder import build_profile

    store = Store(DataConfig(data_dir=tmp_path / "data"))
    profile = build_profile(history(), now=NOW)

    path = store.save_profile("demo", profile)
    assert path.name == f"demo{PROFILE_SUFFIX}"
    assert store.load_profile("demo") == profile


def test_the_store_lists_what_it_holds(tmp_path: Path) -> None:
    store = Store(DataConfig(data_dir=tmp_path / "data"))
    store.save_history("beta", history())
    store.save_history("alpha", history())

    assert store.list_histories() == ("alpha", "beta")


def test_listing_an_empty_store_returns_nothing(tmp_path: Path) -> None:
    store = Store(DataConfig(data_dir=tmp_path / "data"))
    assert store.list_histories() == ()
    assert store.list_profiles() == ()


def test_a_name_cannot_escape_the_data_directory(tmp_path: Path) -> None:
    """Listening history is user-owned data; a name must not write outside it."""

    store = Store(DataConfig(data_dir=tmp_path / "data"))
    for hostile in ("../escape", "a/b", "a\\b", ".."):
        with pytest.raises(ValueError):
            store.save_history(hostile, history())


def test_an_empty_name_is_rejected(tmp_path: Path) -> None:
    store = Store(DataConfig(data_dir=tmp_path / "data"))
    with pytest.raises(ValueError, match="must not be empty"):
        store.save_history("   ", history())


def test_loading_something_that_was_never_saved_names_what_was_missing(
    tmp_path: Path,
) -> None:
    store = Store(DataConfig(data_dir=tmp_path / "data"))
    with pytest.raises(FileNotFoundError, match="no history named"):
        store.load_history("absent")


def test_the_store_expands_a_home_relative_data_directory() -> None:
    store = Store(DataConfig(data_dir=Path("~/.sine-test")))
    assert "~" not in str(store.data_dir)


# ---------------------------------------------------------------------- CLI


def test_no_command_prints_help_and_succeeds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out


def test_import_then_profile_then_context(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    source = simple_history_file(tmp_path)

    assert (
        main(["--data-dir", str(data_dir), "import", str(source), "--out", "demo"]) == 0
    )
    out = capsys.readouterr().out
    assert "imported 12 play(s)" in out
    assert "source: json" in out

    assert (
        main(["--data-dir", str(data_dir), "profile", "demo", "--now", NOW.isoformat()])
        == 0
    )
    assert "signal(s)" in capsys.readouterr().out

    assert main(["--data-dir", str(data_dir), "profile", "demo", "--context"]) == 0
    context = capsys.readouterr().out
    assert "Track 0" in context


def test_import_rejects_an_unknown_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = simple_history_file(tmp_path)
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--data-dir",
                str(tmp_path / "d"),
                "import",
                str(source),
                "--source",
                "napster",
            ]
        )
    assert caught.value.code == 2
    assert "unknown source" in capsys.readouterr().err


def test_import_refuses_a_preset_for_a_source_that_does_not_use_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A preset is a field mapping; a service export's layout is not negotiable."""

    source = simple_history_file(tmp_path)
    with pytest.raises(SystemExit):
        main(
            [
                "--data-dir",
                str(tmp_path / "d"),
                "import",
                str(source),
                "--source",
                "lastfm",
                "--preset",
                "spotify",
            ]
        )
    assert "--preset does not apply" in capsys.readouterr().err


def test_import_refuses_a_label_for_a_service_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = simple_history_file(tmp_path)
    with pytest.raises(SystemExit):
        main(
            [
                "--data-dir",
                str(tmp_path / "d"),
                "import",
                str(source),
                "--source",
                "spotify",
                "--label",
                "mine",
            ]
        )
    assert "--label only applies" in capsys.readouterr().err


def test_importing_an_unreadable_file_is_reported_not_raised(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(
            ["--data-dir", str(tmp_path / "d"), "import", str(tmp_path / "absent.json")]
        )
    assert caught.value.code == 2
    assert capsys.readouterr().err.strip()


def test_importing_an_empty_file_fails_with_the_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        main(["--data-dir", str(tmp_path / "d"), "import", str(empty)])
    assert caught.value.code == 2
    assert "no usable records" in capsys.readouterr().err


def test_rejected_records_are_visible_in_the_import_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = write_json(
        tmp_path / "mixed.json",
        [
            {"title": "Good", "artist": "A", "played_at": NOW.isoformat()},
            {"title": "No time", "artist": "A"},
            {"artist": "A", "played_at": NOW.isoformat()},
            "not an object",
        ],
    )

    assert (
        main(
            ["--data-dir", str(tmp_path / "d"), "import", str(source), "--out", "mixed"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "imported 1 play(s)" in out
    assert "rejected 3 record(s)" in out


def test_a_rejected_record_is_described_rather_than_counted_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = write_json(tmp_path / "bad.json", [{"title": "No time", "artist": "A"}])
    with pytest.raises(SystemExit):
        main(["--data-dir", str(tmp_path / "d"), "import", str(source)])
    assert "played_at" in capsys.readouterr().err


def test_profiling_requires_an_imported_history(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--data-dir", str(tmp_path / "d"), "profile", "absent"])
    assert caught.value.code == 2
    assert "no history named" in capsys.readouterr().err


def test_profiling_with_an_naive_now_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "--data-dir",
            str(tmp_path / "d"),
            "import",
            str(simple_history_file(tmp_path)),
            "--out",
            "x",
        ]
    )
    capsys.readouterr()

    with pytest.raises(SystemExit):
        main(
            [
                "--data-dir",
                str(tmp_path / "d"),
                "profile",
                "x",
                "--now",
                "2026-10-01T00:00:00",
            ]
        )
    assert "must include a timezone" in capsys.readouterr().err


def test_recommending_without_a_model_is_a_configuration_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "--data-dir",
            str(tmp_path / "d"),
            "import",
            str(simple_history_file(tmp_path)),
            "--out",
            "x",
        ]
    )
    capsys.readouterr()

    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--data-dir",
                str(tmp_path / "d"),
                "recommend",
                "x",
                "--provider",
                "openai",
                "--model",
                "gpt-4o",
            ]
        )
    assert caught.value.code == 2
    # No credential is configured, so the failure names the missing key rather than
    # reaching the network.
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_recommending_with_no_model_at_all_is_a_configuration_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a provider or model there is nothing to call."""

    data_dir = tmp_path / "d"
    main(
        [
            "--data-dir",
            str(data_dir),
            "import",
            str(simple_history_file(tmp_path)),
            "--out",
            "x",
        ]
    )
    capsys.readouterr()

    with pytest.raises(SystemExit) as caught:
        main(["--data-dir", str(data_dir), "recommend", "x"])
    assert caught.value.code == 2
    assert "--provider" in capsys.readouterr().err


def test_config_output_carries_no_secret(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "sine.toml"
    config_path.write_text(
        '[llm]\nprovider = "openai"\nmodel = "gpt-4o"\n\n[data]\ndata_dir = "'
        + str(tmp_path / "d")
        + '"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("SINE_LLM_API_KEY", "sk-secret-value")

    assert main(["--config", str(config_path), "config"]) == 0
    out = capsys.readouterr().out
    assert "sk-secret-value" not in out
    assert '"credential": "configured"' in out


def test_providers_lists_the_registry_without_needing_a_config(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", "/nonexistent-sine-home")
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "anthropic" in out
    assert "gemini" in out
    # Limitations are surfaced, not hidden behind a working-looking table.
    assert "note:" in out


def test_global_flags_work_before_the_subcommand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "--data-dir",
                str(tmp_path / "d"),
                "import",
                str(simple_history_file(tmp_path)),
                "--out",
                "before",
            ]
        )
        == 0
    )
    assert (tmp_path / "d" / "history" / f"before{HISTORY_SUFFIX}").is_file()


def test_global_flags_work_after_the_subcommand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "import",
                str(simple_history_file(tmp_path)),
                "--out",
                "after",
                "--data-dir",
                str(tmp_path / "d"),
            ]
        )
        == 0
    )
    assert (tmp_path / "d" / "history" / f"after{HISTORY_SUFFIX}").is_file()


def test_an_unspecified_subcommand_flag_does_not_clobber_a_global_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """argparse defaults on the subparser must not overwrite the top-level value."""

    assert (
        main(
            [
                "--data-dir",
                str(tmp_path / "d"),
                "import",
                str(simple_history_file(tmp_path)),
                "--out",
                "clobber",
            ]
        )
        == 0
    )
    assert (tmp_path / "d" / "history" / f"clobber{HISTORY_SUFFIX}").is_file()
    assert not (tmp_path / "d2").exists()


def test_a_config_file_supplies_the_data_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "sine.toml"
    data_dir = tmp_path / "from-config"
    config_path.write_text(f'[data]\ndata_dir = "{data_dir}"\n', encoding="utf-8")

    assert (
        main(
            [
                "--config",
                str(config_path),
                "import",
                str(simple_history_file(tmp_path)),
                "--out",
                "cfg",
            ]
        )
        == 0
    )
    assert (data_dir / "history" / f"cfg{HISTORY_SUFFIX}").is_file()


def test_an_explicit_data_dir_beats_the_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "sine.toml"
    config_path.write_text(
        f'[data]\ndata_dir = "{tmp_path / "from-config"}"\n', encoding="utf-8"
    )
    override = tmp_path / "override"

    assert (
        main(
            [
                "--config",
                str(config_path),
                "--data-dir",
                str(override),
                "import",
                str(simple_history_file(tmp_path)),
                "--out",
                "o",
            ]
        )
        == 0
    )
    assert (override / "history" / f"o{HISTORY_SUFFIX}").is_file()
    assert not (tmp_path / "from-config").exists()


def test_a_config_file_with_only_data_settings_still_imports(
    tmp_path: Path,
) -> None:
    """Importing needs no model, so a config without one must not be a failure."""

    config_path = tmp_path / "sine.toml"
    data_dir = tmp_path / "from-config"
    config_path.write_text(f'[data]\ndata_dir = "{data_dir}"\n', encoding="utf-8")

    assert (
        main(
            [
                "--config",
                str(config_path),
                "import",
                str(simple_history_file(tmp_path)),
                "--out",
                "no-llm",
            ]
        )
        == 0
    )
    assert (data_dir / "history" / f"no-llm{HISTORY_SUFFIX}").is_file()


def test_an_llm_section_without_a_model_is_absent_not_broken(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Settings alone name no model; the recommendation path says so plainly."""

    config_path = tmp_path / "sine.toml"
    data_dir = tmp_path / "d"
    config_path.write_text(
        f'[data]\ndata_dir = "{data_dir}"\n\n[llm.settings]\ntemperature = 0.1\n',
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--config",
                str(config_path),
                "import",
                str(simple_history_file(tmp_path)),
                "--out",
                "partial",
            ]
        )
        == 0
    )
    capsys.readouterr()

    with pytest.raises(SystemExit):
        main(["--config", str(config_path), "recommend", "partial"])
    assert "no [llm] section configured" in capsys.readouterr().err


def test_recommend_sends_a_configured_request_to_the_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the recommendation path: real listening context, structured
    output, and the caller's constraints actually reaching the provider."""

    from sine.llm import registry as registry_module

    seen: dict = {}

    class RecordingProvider:
        provider_id = "recording"

        def capabilities(self, model_id: str):
            from sine.llm.capabilities import ModelCapabilities, StructuredOutputDialect

            return ModelCapabilities(
                structured_output=StructuredOutputDialect.JSON_OBJECT
            )

        def list_models(self):
            return ()

        def generate(self, model_id, request):
            seen["request"] = request
            from sine.llm.generation import FinishReason, GenerationResponse

            return GenerationResponse(
                text=json.dumps(
                    {
                        "recommendations": [
                            {
                                "track": {
                                    "title": "Something New",
                                    "artists": [{"name": "Someone Else"}],
                                },
                                "rationale": "extends the ambient texture",
                                "confidence": "medium",
                                "novelty": "new_artist",
                            }
                        ]
                    }
                ),
                provider="recording",
                model=model_id,
                finish_reason=FinishReason.STOP,
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        registry_module, "build_provider", lambda *a, **k: RecordingProvider()
    )

    data_dir = tmp_path / "d"
    assert (
        main(
            [
                "--data-dir",
                str(data_dir),
                "import",
                str(simple_history_file(tmp_path)),
                "--out",
                "rec",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "--data-dir",
                str(data_dir),
                "recommend",
                "rec",
                "--provider",
                "openai",
                "--model",
                "gpt-4o",
                "--limit",
                "5",
                "--focus",
                "deepening",
                "--exclude-artist",
                "Artist 0",
                "--guidance",
                "avoid anything with a beat under 100bpm",
            ]
        )
        == 0
    )

    request = seen["request"]
    rendered = " ".join(message.content for message in request.messages)
    # Observed listening behaviour reaches the model...
    assert "Track 0" in rendered
    # ...along with the caller's constraints...
    assert "Artist 0" in rendered
    assert "100bpm" in rendered
    # ...and the format is constrained as far as this dialect allows.
    assert request.response_format.kind == "json_object"
    assert "recommendations" in rendered

    out = capsys.readouterr().out
    assert "Something New" in out
    assert "extends the ambient texture" in out


def test_recommend_can_print_machine_readable_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sine.llm import registry as registry_module

    class JsonProvider:
        provider_id = "json-provider"

        def capabilities(self, model_id: str):
            from sine.llm.capabilities import ModelCapabilities, StructuredOutputDialect

            return ModelCapabilities(
                structured_output=StructuredOutputDialect.JSON_OBJECT
            )

        def generate(self, model_id, request):
            from sine.llm.generation import FinishReason, GenerationResponse

            return GenerationResponse(
                text=json.dumps(
                    {
                        "recommendations": [
                            {
                                "track": {"title": "T", "artists": [{"name": "A"}]},
                                "rationale": "r",
                                "confidence": "low",
                                "novelty": "new_artist",
                            }
                        ]
                    }
                ),
                provider="json-provider",
                model=model_id,
                finish_reason=FinishReason.STOP,
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        registry_module, "build_provider", lambda *a, **k: JsonProvider()
    )

    data_dir = tmp_path / "d"
    main(
        [
            "--data-dir",
            str(data_dir),
            "import",
            str(simple_history_file(tmp_path)),
            "--out",
            "j",
        ]
    )
    capsys.readouterr()

    assert (
        main(
            [
                "--data-dir",
                str(data_dir),
                "recommend",
                "j",
                "--provider",
                "openai",
                "--model",
                "gpt-4o",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommendations"][0]["track"]["title"] == "T"


def test_a_provider_failure_does_not_crash_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sine.llm import registry as registry_module
    from sine.llm.errors import ProviderUnavailableError

    class FailingProvider:
        provider_id = "failing"

        def capabilities(self, model_id: str):
            from sine.llm.capabilities import ModelCapabilities, StructuredOutputDialect

            return ModelCapabilities(
                structured_output=StructuredOutputDialect.JSON_OBJECT
            )

        def generate(self, model_id, request):
            raise ProviderUnavailableError("failing", "503 from upstream")

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        registry_module, "build_provider", lambda *a, **k: FailingProvider()
    )

    data_dir = tmp_path / "d"
    main(
        [
            "--data-dir",
            str(data_dir),
            "import",
            str(simple_history_file(tmp_path)),
            "--out",
            "f",
        ]
    )
    capsys.readouterr()

    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--data-dir",
                str(data_dir),
                "recommend",
                "f",
                "--provider",
                "openai",
                "--model",
                "gpt-4o",
            ]
        )
    assert caught.value.code == 2
    assert "503 from upstream" in capsys.readouterr().err


def test_recommending_an_empty_history_is_refused_before_a_request_is_sent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sine.llm import registry as registry_module
    from sine.profile.builder import build_profile

    def explode(*args, **kwargs):
        raise AssertionError("no provider should be built for an empty history")

    monkeypatch.setattr(registry_module, "build_provider", explode)

    data_dir = tmp_path / "d"
    store = Store(DataConfig(data_dir=data_dir))

    # A profile file that records no listening at all: asking a model to recommend
    # from nothing would only invite invention.
    empty_profile = build_profile(ListeningHistory.from_events(()), now=NOW)
    store.save_history("void", ListeningHistory.from_events(()))
    store.save_profile("void", empty_profile)

    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--data-dir",
                str(data_dir),
                "recommend",
                "void",
                "--provider",
                "openai",
                "--model",
                "gpt-4o",
            ]
        )
    assert caught.value.code == 2
    assert "has no plays" in capsys.readouterr().err
