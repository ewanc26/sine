# AGENTS.md

## Project

Sine is an LLM-powered music recommendation system. It takes Apple Music-derived scrobble or listening-history data, builds useful musical context from it, and uses a language model to generate personalised recommendations.

The repository is currently at an early stage. Do not assume that an architecture, framework, provider, data schema, or runtime has been chosen unless it is present in the codebase.

## Development principles

- Keep the implementation focused on music recommendation from listening history.
- Prefer small, composable components over a large monolithic recommendation pipeline.
- Keep data ingestion, listening-history processing, LLM interaction, and recommendation presentation separable.
- Treat listening history as sensitive user-owned data. Do not add telemetry, analytics, or external data transmission without an explicit reason and clear documentation.
- Avoid coupling the core recommendation logic to a single LLM provider where practical.
- Keep provider-specific integrations behind narrow interfaces.
- Make prompts and model configuration inspectable rather than hiding important behaviour in opaque helpers.
- Preserve deterministic processing wherever possible. LLM output may be probabilistic, but parsing, normalisation, filtering, and data preparation should be predictable and testable.
- Do not invent metadata for tracks, artists, albums, or listening events when the source data does not provide it.
- Handle missing, duplicated, malformed, and incomplete listening-history records explicitly.
- Prefer precise terminology: distinguish a scrobble/listening event from a track, an artist, an album, and a recommendation.

## Repository changes

Before changing the project:

1. Inspect the existing repository structure and relevant source files.
2. Check the package/runtime configuration before introducing dependencies.
3. Follow existing naming, formatting, testing, and module conventions.
4. Keep unrelated refactors out of feature changes.
5. Update documentation when behaviour, configuration, data formats, or public interfaces change.

Do not introduce a framework or large dependency solely for convenience when the existing project can support the feature without it.

## LLM behaviour

LLMs should be treated as recommendation engines, not authoritative music databases.

When constructing LLM context:

- give the model structured, relevant listening information;
- distinguish observed listening behaviour from inferred preferences;
- avoid presenting uncertain metadata as fact;
- constrain structured outputs where the implementation requires machine-readable recommendations;
- validate model output before using it downstream;
- make failures and malformed responses recoverable.

Prompts should explain what Sine is trying to optimise for. Avoid prompts that encourage generic "top songs you might like" recommendations when the available listening history can support more personalised reasoning.

## Testing

Tests should cover:

- listening-history parsing and normalisation;
- duplicate and malformed events;
- profile/context construction;
- recommendation parsing and validation;
- provider failures and unavailable model responses;
- privacy-sensitive data handling.

Do not make tests depend on a live LLM or external music service unless the test is explicitly an integration test and is clearly isolated from the normal test suite.

## Git and commits

Use focused commits with clear conventional-style messages where the repository's history supports them.

Examples:

- `feat: add listening profile builder`
- `feat: add recommendation provider`
- `fix: handle duplicate scrobbles`
- `docs: document recommendation context`

Do not rewrite history or force-push branches unless explicitly requested.

## Documentation

The README should describe the current implementation, not an aspirational architecture.

When adding a new integration or configuration option, document:

- what it does;
- what data it receives;
- what data leaves the local environment;
- required credentials or configuration;
- relevant limitations.

Keep the distinction between Apple Music, scrobbling services, and LLM providers clear. Sine is an independent project and must not imply affiliation with Apple.
