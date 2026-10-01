"""Command-line interface for Sine.

A thin shell over the library: ``import`` reads a listening history, ``profile``
computes the deterministic profile, ``recommend`` asks a configured model for
recommendations. Global flags can override configuration so that a one-off
provider or model can be selected without editing a file.

Commands are only present where there is real behaviour behind them.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from sine import __version__
from sine.config import AppConfig, DataConfig, LLMConfig, load_data_config
from sine.ingestion import IngestOptions, ingest
from sine.ingestion.sources import (
    JSON_PRESETS,
    describe_sources,
    open_source,
    resolve_source_name,
)
from sine.llm.adapters.http import RetryPolicy
from sine.llm.errors import ProviderConfigurationError, ProviderError
from sine.llm.registry import known_providers, profile_for
from sine.models.recommendation import (
    Artist,
    RecommendationFocus,
    RecommendationRequest,
    RecommendationSet,
)
from sine.profile.builder import build_profile
from sine.profile.context import render_profile_context
from sine.recommend.engine import RecommendationEngine, RecommendationError
from sine.storage import Store

PROG = "sine"


def main(argv: list[str] | None = None) -> int:
    """Run the Sine command-line interface."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    handlers = {
        "providers": _cmd_providers,
        "config": _cmd_config,
        "models": _cmd_models,
        "import": _cmd_import,
        "profile": _cmd_profile,
        "recommend": _cmd_recommend,
    }
    try:
        return handlers[args.command](args)
    except (
        ProviderConfigurationError,
        RecommendationError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        _fail(str(exc))
    except ProviderError as exc:  # pragma: no cover - defensive
        _fail(str(exc))
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        _fail("interrupted")
    return 0


# --------------------------------------------------------------------------- parser


def _add_global_arguments(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Add the global overrides.

    They are attached both to the top-level parser and to every subcommand. On the
    subparsers the defaults are suppressed, so an unspecified subcommand flag does
    not clobber a value given before the subcommand.
    """

    def default(value: Any) -> Any:
        return argparse.SUPPRESS if suppress else value

    parser.add_argument(
        "--config",
        metavar="PATH",
        default=default(None),
        help="Path to a Sine TOML config file.",
    )
    parser.add_argument(
        "--provider", default=default(None), help="Override the configured provider id."
    )
    parser.add_argument(
        "--model", default=default(None), help="Override the configured model id."
    )
    parser.add_argument(
        "--base-url", default=default(None), help="Override the provider endpoint."
    )
    parser.add_argument(
        "--data-dir", default=default(None), help="Override Sine's data directory."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="LLM-powered music recommendations from your own listening history.",
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    _add_global_arguments(parser, suppress=False)

    common = argparse.ArgumentParser(add_help=False)
    _add_global_arguments(common, suppress=True)

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "providers", parents=[common], help="List the providers Sine knows about."
    )
    subparsers.add_parser(
        "config",
        parents=[common],
        help="Show the resolved configuration, without secrets.",
    )

    models = subparsers.add_parser(
        "models", parents=[common], help="List the models a provider reports."
    )
    models.add_argument(
        "--limit", type=int, default=50, help="Maximum models to print (default: 50)."
    )

    importer = subparsers.add_parser(
        "import",
        parents=[common],
        help="Import a listening history from a service export or JSON file.",
        epilog="sources: " + "; ".join(f"{n} = {d}" for n, d in describe_sources()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    importer.add_argument("path", help="Path to the export file.")
    importer.add_argument(
        "--source",
        default="json",
        metavar="NAME",
        help=(
            "Export format to read: "
            + ", ".join(name for name, _ in describe_sources())
            + " (default: json)."
        ),
    )
    importer.add_argument(
        "--preset",
        default="sine",
        choices=JSON_PRESETS,
        help="Field mapping for --source json (default: sine).",
    )
    importer.add_argument(
        "--label",
        help="Label recorded on each event (default: the source's own id).",
    )
    importer.add_argument("--out", help="Name to save the history under.")
    importer.add_argument(
        "--assume-timezone",
        metavar="ZONE",
        help="Timezone for source timestamps that carry no offset, e.g. Europe/London.",
    )
    importer.add_argument(
        "--timestamp-format",
        help="strptime format, when the source is neither ISO 8601 nor epoch seconds.",
    )

    profiler = subparsers.add_parser(
        "profile", parents=[common], help="Build the listening profile for a history."
    )
    profiler.add_argument("history", help="Name of an imported history.")
    profiler.add_argument(
        "--out", help="Name to save the profile under (default: same as history)."
    )
    profiler.add_argument(
        "--recent-days",
        type=int,
        default=28,
        help="Recent-listening window (default: 28).",
    )
    profiler.add_argument(
        "--now",
        help="Treat this ISO 8601 instant as the present. For reproducible output only.",
    )
    profiler.add_argument(
        "--context",
        action="store_true",
        help="Print the model-facing context and exit.",
    )

    recommender = subparsers.add_parser(
        "recommend",
        parents=[common],
        help="Ask the configured model for recommendations.",
    )
    recommender.add_argument("history", help="Name of an imported history.")
    _add_request_arguments(recommender)
    recommender.add_argument(
        "--json", action="store_true", help="Print JSON instead of text."
    )

    return parser


def _add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--limit", type=int, default=10, help="Tracks to request (default: 10)."
    )
    parser.add_argument(
        "--focus",
        choices=[focus.value for focus in RecommendationFocus],
        default=RecommendationFocus.DISCOVERY.value,
        help="What kind of recommendations to ask for (default: discovery).",
    )
    parser.add_argument(
        "--allow-replays",
        action="store_true",
        help="Permit tracks already present in the history.",
    )
    parser.add_argument(
        "--seed-artist",
        action="append",
        default=[],
        metavar="NAME",
        help="Anchor recommendations on this artist. Repeatable.",
    )
    parser.add_argument(
        "--exclude-artist",
        action="append",
        default=[],
        metavar="NAME",
        help="Never recommend this artist. Repeatable.",
    )
    parser.add_argument(
        "--guidance", help="Free-text steer, passed to the model verbatim."
    )


# ------------------------------------------------------------------------- commands


def _cmd_providers(args: argparse.Namespace) -> int:
    print(f"{'ID':<20} {'FAMILY':<20} {'KEY ENVIRONMENT':<22} SELF-HOSTED")
    for profile in known_providers():
        # Naming the variable matters more than a yes/no column: a user who needs a
        # key has to know which one to export.
        key = profile.api_key_env if profile.api_key_env else "-"
        print(
            f"{profile.provider_id:<20} {profile.family!s:<20} {key:<22} "
            f"{'yes' if profile.self_hosted else 'no'}"
        )
        print(f"{'':<20} {profile.base_url or '(base_url required)'}")
        if profile.notes:
            print(f"{'':<20} note: {profile.notes}")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    config = _load_config(args)
    print(json.dumps(config.describe(), indent=2))
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    config = _load_config(args)
    config.require_llm()
    provider = _build_provider(config)
    try:
        models = provider.list_models()
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    if not models:
        print(f"{config.llm.provider} reported no models.")
        return 1
    dialect = (
        config.llm.structured_output
        or profile_for(config.llm.provider).structured_output
    )
    print(
        f"{config.llm.provider}: {len(models)} model(s); structured output: {dialect}"
    )
    for info in models[: args.limit]:
        context = f", {info.context_window} ctx" if info.context_window else ""
        print(f"  {info.model_id}{context}")
    if len(models) > args.limit:
        print(f"  ... and {len(models) - args.limit} more")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    store = _store(args)

    source_name = resolve_source_name(args.source)
    if args.label and source_name != "json":
        _fail("--label only applies to --source json; a service source names itself")

    if source_name == "json":
        source = open_source(
            source_name, args.path, source_id=args.label or "json", preset=args.preset
        )
    else:
        if args.preset != "sine":
            _fail(
                f"--preset does not apply to --source {source_name}; that format is fixed"
            )
        source = open_source(source_name, args.path)

    options = IngestOptions(
        default_timezone=args.assume_timezone,
        played_at_format=args.timestamp_format,
    )
    result = ingest(
        source, options=options, notes=f"imported from {Path(args.path).name}"
    )

    if result.accepted == 0:
        detail = ""
        if result.rejected:
            reasons: dict[str, int] = {}
            for rejection in result.rejected:
                reasons[rejection.reason] = reasons.get(rejection.reason, 0) + 1
            summary = "; ".join(
                f"{reason} ({count})" if count > 1 else reason
                for reason, count in sorted(reasons.items(), key=lambda item: -item[1])
            )
            detail = f": {summary}"
        _fail(
            f"no usable records in {args.path} "
            f"({len(result.rejected)} rejected{detail})"
        )

    name = args.out or source.source_id
    path = store.save_history(name, result.history)

    print(f"imported {result.accepted} play(s) from {Path(args.path).name}")
    print(f"  source: {result.source}")
    window = result.history.window
    if window is not None:
        print(
            f"  span: {window.start.date().isoformat()} to {window.end.date().isoformat()}"
            f" ({window.days:.0f} days)"
        )
    if result.rejected:
        print(f"  rejected {len(result.rejected)} record(s):")
        for rejection in result.rejected[:10]:
            print(f"    {rejection.describe()}")
        if len(result.rejected) > 10:
            print(f"    ... and {len(result.rejected) - 10} more")
    print(f"  saved as {path}")
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    store = _store(args)
    history = store.load_history(args.history)
    now = datetime.fromisoformat(args.now) if args.now else datetime.now(UTC)
    if now.tzinfo is None:
        _fail("--now must include a timezone offset, e.g. 2026-01-31T00:00:00+00:00")

    profile = build_profile(history, now=now, recent_window_days=args.recent_days)

    if args.context:
        print(render_profile_context(profile))
        return 0

    path = store.save_profile(args.out or args.history, profile)
    stats = profile.statistics
    print(f"profile for {args.history!r} saved to {path}")
    print(
        f"  {stats.event_count} play(s), {stats.unique_artists} artist(s), "
        f"{stats.unique_tracks} track(s) over {stats.window.days:.0f} days"
    )
    print(f"  {len(profile.signals)} signal(s), {len(profile.gaps)} gap(s)")
    for gap in profile.gaps:
        print(f"  gap: {gap.detail}")
    return 0


def _cmd_recommend(args: argparse.Namespace) -> int:
    store = _store(args)
    history = store.load_history(args.history)
    profile_path = store.profile_path(args.history)
    if profile_path.is_file():
        from sine.models.profile import ListeningProfile

        profile = ListeningProfile.model_validate(
            json.loads(profile_path.read_text(encoding="utf-8"))
        )
    else:
        profile = build_profile(history, now=datetime.now(UTC))

    if profile.statistics.event_count == 0:
        _fail(f"history {args.history!r} has no plays; import a history first")

    request = RecommendationRequest(
        limit=args.limit,
        focus=RecommendationFocus(args.focus),
        allow_replays=args.allow_replays,
        seed_artists=tuple(Artist(name=name) for name in args.seed_artist),
        exclude_artists=tuple(Artist(name=name) for name in args.exclude_artist),
        guidance=args.guidance,
    )

    config = _load_config(args)
    config.require_llm()
    provider = _build_provider(config)
    engine = RecommendationEngine(
        _model_for(config, provider),
        temperature=config.llm.settings.temperature,
        max_output_tokens=config.llm.settings.max_output_tokens,
    )

    print(
        f"asking {engine.model.qualified_id} for {request.limit} recommendation(s)…",
        file=sys.stderr,
    )
    result = engine.recommend(profile, request)
    _print_recommendations(result, engine.model.qualified_id, as_json=args.json)
    return 0


# ------------------------------------------------------------------------- plumbing


def _load_config(args: argparse.Namespace) -> AppConfig:
    """Load configuration and apply command-line overrides."""

    try:
        config = AppConfig.load(args.config)
    except ProviderConfigurationError:
        if not (args.provider and args.model):
            raise
        config = AppConfig(
            llm=LLMConfig(provider=args.provider, model=args.model), data=DataConfig()
        )

    llm_updates: dict[str, Any] = {
        key: value
        for key, value in (
            ("provider", args.provider),
            ("model", args.model),
            ("base_url", args.base_url),
        )
        if value
    }
    if llm_updates:
        base = config.llm or LLMConfig(provider="openai", model="placeholder")
        config = config.model_copy(update={"llm": base.model_copy(update=llm_updates)})

    if args.data_dir:
        config = config.model_copy(
            update={"data": DataConfig(data_dir=Path(args.data_dir))}
        )
    return config


def _store(args: argparse.Namespace) -> Store:
    """Build the local store, honouring --data-dir over the config file's [data]."""

    data = load_data_config(args.config)
    if args.data_dir:
        data = DataConfig(data_dir=Path(args.data_dir))
    return Store(data)


def _build_provider(config: AppConfig) -> Any:
    from sine.llm.registry import build_provider

    llm = config.require_llm()
    settings = llm.settings
    return build_provider(
        llm.provider,
        base_url=llm.base_url,
        api_key=llm.resolve_api_key(),
        structured_output=llm.structured_output,
        timeout=settings.timeout_seconds,
        retry=RetryPolicy(attempts=settings.retry_attempts),
    )


def _model_for(config: AppConfig, provider: Any) -> Any:
    from sine.llm.provider import Model

    model_id = config.require_llm().model
    return Model(
        provider=provider,
        model_id=model_id,
        capabilities=provider.capabilities(model_id),
    )


def _print_recommendations(
    result: RecommendationSet, qualified_id: str, *, as_json: bool
) -> None:
    if as_json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return

    if not result.recommendations:
        print("no recommendations were returned.")
    for index, recommendation in enumerate(result.recommendations, start=1):
        print(f"\n{index}. {recommendation.track.display_name}")
        print(f"   {recommendation.rationale}")
        print(
            f"   [{recommendation.novelty.value}, confidence {recommendation.confidence.value}]"
            + (
                f" genre hints: {', '.join(recommendation.genre_hints)}"
                if recommendation.genre_hints
                else ""
            )
        )
        for evidence in recommendation.evidence:
            references = (
                f" (refers to: {', '.join(evidence.references)})"
                if evidence.references
                else ""
            )
            print(f"   - {evidence.kind.value}: {evidence.statement}{references}")
    if result.notes:
        print(f"\nnotes from {qualified_id}:\n{result.notes}")


def _fail(message: str) -> NoReturn:
    print(f"{PROG}: error: {message}", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
