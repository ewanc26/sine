# AGENTS.md

Guidance for AI coding agents working in **sine**.

## Project overview

Sine is an LLM-powered music recommendation system built around listening history from music services and imported sources.

- Language: Python 3.12
- Package manager: uv
- Default branch: main

## Working rules

- Inspect the README, manifests, CI workflows, and nearby code before editing.
- Preserve existing architecture, naming, formatting, and error-handling conventions.
- Use project scripts for validation; never claim checks you did not run.
- Keep changes scoped and update tests or documentation when behavior changes.
- Use feature branches and pull requests.
- Treat generated files, credentials, deployment configuration, and release metadata as sensitive.

## Project-specific rules

- Keep music-service and listening-history ingestion separate from recommendation logic.
- Keep each music-service integration behind a narrow ingestion interface.
- Keep provider-specific LLM code behind narrow interfaces.
- Use Pydantic models for data crossing module or provider boundaries where practical.
- Prefer deterministic processing for parsing, normalisation, filtering, and profile construction.
- Do not invent track, artist, album, or listening-event metadata.
- Distinguish observed listening behaviour from inferred preferences.
- Treat listening history as user-owned data and do not add telemetry or external transmission without a clear reason and documentation.
- Do not introduce a machine-learning framework unless the project actually requires model training or inference beyond the configured LLM APIs.
- Do not couple the core system to Apple's private or undocumented APIs when the official Apple Music API can provide the required data.
- Keep music-service integrations, imported historical scrobbles, and LLM providers independently replaceable.
- Treat music services as interchangeable sources, not as the canonical domain model or storage format.
- Keep source-specific identifiers and metadata at the ingestion boundary; normalised domain models must not depend on a particular service.

## Python conventions

- Target Python 3.12.
- Manage dependencies and virtual environments with uv.
- Use the `src/sine/` package layout.
- Use type annotations for public functions and meaningful internal boundaries.
- Prefer standard-library functionality when it is sufficient.
- Use Ruff for formatting and linting.
- Use pytest for tests.
- Keep asynchronous I/O asynchronous where it materially improves the workflow; do not introduce async complexity for purely local processing.

## Repository changes

Before changing the project:

1. Inspect the existing repository structure and relevant source files.
2. Check `pyproject.toml`, CI workflows, and project configuration before introducing dependencies.
3. Follow existing naming, formatting, testing, and module conventions.
4. Keep unrelated refactors out of feature changes.
5. Update documentation when behaviour, configuration, data formats, or public interfaces change.

## LLM behaviour

LLMs are recommendation engines, not authoritative music databases.

When constructing LLM context:

- give the model structured and relevant listening information;
- distinguish observed listening behaviour from inferred preferences;
- avoid presenting uncertain metadata as fact;
- constrain structured outputs where machine-readable recommendations are required;
- validate model output before using it downstream;
- make provider failures and malformed responses recoverable.

Prompts should optimise for personalised recommendations grounded in the listener's history rather than generic popularity-based suggestions.

## Testing

Tests should cover:

- listening-history parsing and normalisation across multiple source shapes;
- duplicate and malformed events;
- profile/context construction;
- recommendation parsing and validation;
- provider failures and unavailable model responses;
- privacy-sensitive data handling.

Tests should not depend on a live LLM or external music service unless explicitly marked and isolated as integration tests.

## Git and commits

Use focused commits with clear conventional-style messages where the repository's history supports them.

Do not rewrite history or force-push branches unless explicitly requested.

## Documentation

The README should describe the current implementation, not an aspirational architecture.

When adding an integration or configuration option, document:

- what it does;
- what data it receives;
- what data leaves the local environment;
- required credentials or configuration;
- relevant limitations.

Keep the distinction between music-service sources, historical imports, and LLM providers clear. Sine is an independent project and must not imply affiliation with any music service.
