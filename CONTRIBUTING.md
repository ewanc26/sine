# Contributing to Sine

Sine is an LLM-powered music recommendation system built around your listening history: a local-first pipeline where ingestion adapters normalise history from any service, a deterministic profile records what was measured, and an LLM — behind one provider interface — recommends from it. Thank you for wanting to help!

## Getting started

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

`sine profile --context` shows exactly what would be sent to a model — use it to test brief changes deterministically.

## Notes for this project

- **Observation and inference are kept apart, and that is the project's central rule.** Everything up to the profile is deterministic: parsing, normalisation, deduplication, statistics — no model involved. Interpretations are marked as interpretations, and thin history produces explicit gaps, never confident nonsense. Do not let model code creep above the profile boundary, or determinism creep below it.
- The brief adapts to the listener and **cites a number from the history for every adjustment** it makes, stating what the data cannot show. Keep it a set of observations, not a personality assessment.
- The model is a recommender, not a database: output is schema-validated, repaired at most once, then post-processed (replay/exclusion/duplicate filtering, track labelling, contiguous renumbering) — trust the pipeline over the model.
- Providers are HTTP adapters behind one interface; no vendor SDKs. Adding a provider means implementing that interface; `sine providers`/`sine models` must keep working.
- Source-specific limitations live at the ingestion boundary: never make one service the canonical source.

## Before opening a PR

- The documented build and tests pass, and you ran them yourself.
- Conventional commits; one focused PR per change.
- AI-assisted PRs are welcome; stand behind every line.

## Licence

AGPL-3.0 (see [LICENSE](LICENSE)). Contributions are accepted under the same terms.