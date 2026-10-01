# Sine

Sine is an LLM-powered music recommendation system built around your listening history.

It takes Apple Music-derived listening data, builds a structured picture of what you listen to, and uses a language model to recommend music based on that history.

## How it works

Sine is designed as a local-first data pipeline:

```
Apple Music / listening-history source
              │
              ▼
        Ingestion adapter
              │
              ▼
       Normalised history
              │
              ▼
       Listening profile
              │
              ▼
        LLM provider
              │
              ▼
       Recommendations
```

The important distinction is between listening data and the recommendation model. Sine should be able to consume different sources of listening history without tying the recommendation logic to one ingestion service or LLM provider.

Apple's MusicKit and Apple Music API provide access to a user's recently played content and other personal music data with authorisation. They do not constitute a complete historical scrobble database, so Sine treats historical imports and live Apple Music access as separate ingestion paths.

## Tech Stack

- **Language**: Python 3.12
- **Package manager**: uv
- **Data validation**: Pydantic
- **HTTP**: httpx
- **LLM integration**: provider-specific SDKs behind a Sine interface
- **Testing**: pytest
- **Linting and formatting**: Ruff

Python is used for the core because Sine is primarily a data-processing and LLM application. It provides strong support for structured data, API clients, validation, and model SDKs without introducing a separate language for the recommendation pipeline.

The project does not currently require a machine-learning framework. The initial recommendation system uses an LLM over structured listening context rather than training its own model.

## Project Structure

```
src/
└── sine/
    ├── ingestion/       # Listening-history source adapters
    ├── models/          # Typed domain and API models
    ├── profile/         # Listening-profile construction
    ├── recommend/       # Recommendation logic and LLM providers
    └── cli.py           # Command-line interface
tests/                   # Unit and integration tests
```

The structure is deliberately small while the project is being established. New modules should follow the existing boundaries rather than introducing framework-specific layers without a concrete need.

## Goals

Sine is intended to:

- understand a listener from their actual listening history;
- use both long-term patterns and recent listening;
- distinguish repeated listening from one-off behaviour;
- balance familiarity with discovery;
- give the LLM structured, relevant context rather than an unbounded history dump;
- keep recommendation logic independent of a single LLM provider;
- keep listening data local unless an external service is explicitly required.

Sine is not intended to replace Apple Music's recommendation systems or reproduce their internal algorithms.

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

### Installation

```sh
uv sync
```

### Development

```sh
uv run sine
```

The CLI and available commands are still under development.

### Checks

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Data and privacy

Listening history can reveal highly personal information about someone's tastes and habits. Sine treats imported listening data as user-owned input.

External LLM providers may receive the listening context supplied to them. Provider integrations must make that data flow explicit rather than hiding it behind generic recommendation code.

Credentials and user data must never be committed to the repository.

## Apple Music

Sine is an independent project and is not affiliated with or endorsed by Apple.

Apple Music and MusicKit are trademarks of Apple Inc.

## Contributing

Contributions are welcome. Please submit a Pull Request with a focused change and enough context to review it.

See [CONTRIBUTING.md](CONTRIBUTING.md) for repository guidance.

## Licence

See [LICENSE](LICENSE) for the GNU Affero General Public License, version 3 or any later version.
