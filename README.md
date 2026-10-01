# Sine

Sine is an LLM-powered music recommendation system built around your listening history.

It takes Apple Music-derived scrobble data, uses that history as context, and asks a language model to reason about listening patterns and suggest music that fits them. The goal is not simply to reproduce a conventional similarity-based recommendation engine, but to give an LLM enough structured listening context to make recommendations with a sense of musical continuity and discovery.

## How it works

At a high level, Sine is intended to sit between a listening-history source and an LLM:

```
Apple Music
    │
    ▼
Scrobble / listening history
    │
    ▼
Sine
    │
    ├── listening profile
    ├── recent listening
    └── relevant musical context
    │
    ▼
LLM
    │
    ▼
Music recommendations
```

The exact ingestion, modelling, prompting, and recommendation pipeline is still being developed.

## Goals

Sine is intended to:

- understand a listener from their actual listening history rather than a static set of preferences;
- use both long-term patterns and recent listening when making recommendations;
- recommend music that is familiar enough to make sense while still providing genuine discovery;
- make the reasoning and context supplied to the LLM explicit and reproducible;
- remain useful with a listener's own data rather than depending on a proprietary recommendation profile.

Sine is not intended to replace Apple Music's own recommendation systems or reproduce their internal algorithms.

## Project status

Sine is an early-stage project. The repository currently contains the project documentation and licensing, while the implementation is being developed.

Interfaces, data formats, model providers, and internal architecture should therefore be considered subject to change.

## Data and privacy

Listening history can reveal highly personal information about someone's tastes and habits. Sine should treat imported listening data as user-owned input and avoid collecting or transmitting it anywhere that is not explicitly required for the configured recommendation workflow.

When an external LLM provider is used, users should understand what listening data is sent to that provider and under which terms.

## Apple Music

Sine is an independent project and is not affiliated with or endorsed by Apple.

Apple Music is a trademark of Apple Inc.

## Licence

Sine is licensed under the GNU Affero General Public License, version 3 or any later version. See [LICENSE](LICENSE) for the full licence text.
