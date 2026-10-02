# Sine

Sine is an LLM-powered music recommendation system built around your listening history.

It takes listening data from one or more music services or imports, builds a structured picture of what you listen to, and uses a language model to recommend music based on that history.

## How it works

Sine is designed as a local-first data pipeline:

```
Music service / listening-history source
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

The important distinction is between listening data and the recommendation model. Sine should be able to consume listening history from different music services and import formats without tying the recommendation logic to one service, data source, or LLM provider.

Music services expose different APIs, permissions, identifiers, and amounts of historical data. Sine therefore treats each service as an ingestion adapter and keeps source-specific limitations at the ingestion boundary. Historical imports and live service access can coexist without making any one service the canonical source.

Two boundaries matter for trusting the output:

**Observation and inference are kept apart.** Everything up to the profile is deterministic: parsing, normalisation, deduplication, and statistics are computed from the history alone, with no model involved. The profile records what was measured, marks every interpretation as an interpretation, and states what the data cannot support. A thin history produces explicit gaps rather than confident nonsense.

**The brief adapts to the listener.** `--focus` says what kind of answer was asked for, and the instruction that follows it is derived from that listener's own measurements. Asking for discovery from someone whose plays are concentrated asks the model to reach further out; asking the same of someone already playing thirty artists once each tells it that more of the same is not discovery. Every adjustment cites a number from the history and says what the data cannot show, so the brief stays a set of observations rather than a personality assessment.

**The model is a recommender, not a database.** Whatever it returns is validated against a schema before use, and one malformed reply is repaired once before being reported as an error. Sine then applies what the model cannot be trusted to do itself: drops tracks the listener has already played (unless `--allow-replays`), drops excluded artists and duplicates, labels each track as a replay, a new track by a known artist, or a new artist, and renumbers a playlist so its positions stay contiguous. The rendering is deterministic, so `sine profile --context` shows exactly what a model would be sent.

## Tech Stack

- **Language**: Python 3.12
- **Package manager**: uv
- **Data validation**: Pydantic
- **HTTP**: httpx
- **LLM integration**: HTTP adapters behind a common provider interface
- **Testing**: pytest
- **Linting and formatting**: Ruff

Python is used for the core because Sine is primarily a data-processing and LLM application. It provides strong support for structured data, API clients, validation, and model SDKs without introducing a separate language for the recommendation pipeline.

Providers are reached over their HTTP APIs through one interface, so no vendor SDK is required and adding a provider means implementing that interface. `sine providers` lists the known providers, and `sine models` lists what a provider actually reports.

The project does not currently require a machine-learning framework. The initial recommendation system uses an LLM over structured listening context rather than training its own model.

## Project Structure

```
src/
└── sine/
    ├── cli.py           # Command-line interface
    ├── config.py        # Typed configuration and environment overrides
    ├── storage.py       # Local history and profile storage
    ├── ingestion/       # Listening-history sources and normalisation
    ├── models/          # Typed domain models
    ├── profile/         # Listening-profile and statistics construction
    ├── recommend/       # Recommendation engine, prompts, and validation
    └── llm/             # Provider interface, adapters, and model registry
tests/                   # Unit and integration tests
```

The structure is deliberately small. New modules should follow the existing boundaries rather than introducing framework-specific layers without a concrete need.

## Goals

Sine is intended to:

- understand a listener from their actual listening history;
- use both long-term patterns and recent listening;
- distinguish repeated listening from one-off behaviour;
- balance familiarity with discovery;
- give the LLM structured, relevant context rather than an unbounded history dump;
- keep recommendation logic independent of a single LLM provider;
- keep listening data independent of a single music service;
- allow multiple listening sources to be combined where their data can be normalised safely;
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

### Configuration

Sine reads a TOML config file, or the environment. Credentials are never stored in
the config file: it names the environment variable to read.

```toml
# sine.toml
[llm]
provider = "anthropic"
model = "claude-sonnet-4-5"
api_key_env = "ANTHROPIC_API_KEY"

[llm.settings]
temperature = 0.6
max_output_tokens = 4096

[data]
data_dir = "~/.ewanc26/sine"
```

The file is looked for at `--config`, then `$SINE_CONFIG`, then `./sine.toml`, then
`~/.config/sine/sine.toml`, then `~/.ewanc26/sine/sine.toml`. The provider and model can
also come from `SINE_LLM_PROVIDER` and `SINE_LLM_MODEL`, and the data directory
from `SINE_DATA_DIR`. Importing and profiling need no model at all, so `[llm]`
may be absent; only `recommend` requires it.

`sine providers` lists the providers Sine knows about, the environment variable
each one reads, and whether it is self-hosted.

### Usage

```sh
# Import a listening history. --source picks the export format.
uv run sine import lastfm-export.csv --source lastfm --out lastfm

# Or a JSON/JSON Lines file, with a field mapping for its shape.
uv run sine import listens.json --source json --preset listenbrainz --out listens

# Build the listening profile, and see exactly what a model would be sent.
uv run sine profile listens --context

# Ask the configured model for recommendations.
uv run sine recommend listens --limit 10 --focus discovery --json

# Or for an ordered playlist, optionally with a length to aim for.
uv run sine recommend listens --playlist --minutes 90 --playlist-title "Late shift"

uv run sine config      # resolved configuration, never including the credential
uv run sine models      # models the configured provider reports
```

Supported `--source` values: `apple-music`, `json`, `lastfm`, `listenbrainz`,
`spotify`, `youtube-music`. The `json` source reads a JSON array or JSON Lines
file and takes a `--preset` field mapping: `sine` (or `generic`, the same shape),
`lastfm`, `listenbrainz`, `spotify`.

Timestamps that carry no UTC offset are refused unless `--assume-timezone` is
given; an unparseable timestamp is reported as a rejected record rather than
guessed at.

### Focus

`--focus` takes `discovery`, `deepening`, `recent_rotation`, `familiarity`, or
`surprise`. It names the kind of answer asked for; the instruction that follows it
is derived from the listener's own statistics, so the same focus produces a
different brief for a habitual listener and a habitual explorer. `sine profile
<history> --context` shows the measurements those adjustments are read from.

### Playlists

`--playlist`, `--playlist-title`, and `--minutes` ask for a sequence instead of a
set; the last two imply the first.

- **The order is the model's contribution.** Each track carries the reason it
  follows the one before it, so a sequence is more than a ranking. Sine prints and
  returns the order the model gave.
- **The positions are Sine's.** Anything the history rules out — a replay, an
  excluded artist, a duplicate — is removed and the sequence is renumbered, so what
  comes back is contiguous from 1. A gap or a short answer is reported, never
  padded with tracks of Sine's own.
- **A length target is an aim, not a measurement.** Sine has no track duration
  data, so `--minutes` only sizes the request and the model is forbidden from
  stating durations; the note under the output says the running time is unknown.
  With no `--limit`, a playlist asks for more tracks than a plain list does, and
  `--minutes` sizes that count by an average track length.

### Checks

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Data and privacy

Listening history can reveal highly personal information about someone's tastes and habits. Sine treats imported listening data as user-owned input.

Sine is local-first: histories, profiles, and configuration stay on disk under the configured data directory, and no telemetry is collected. The only data that leaves the machine is the listening context Sine sends to the provider you configured, when you run `recommend`. That context is a derived summary — measurements, hedges, and stated gaps — not a raw event dump, and `sine profile --context` prints exactly what would be sent. No file paths or credentials are included in it.

Credentials and user data must never be committed to the repository.

## Listening sources

Sine is service-agnostic. A service is an ingestion source, not the canonical
representation of a listener's history, and Sine is not dependent on any one
ecosystem.

Each source below reads a file export only. None of them requires an account, a
token, or network access, and none of them sends anything anywhere; only
`recommend` talks to a provider.

| `--source` | Input | Notes |
| --- | --- | --- |
| `apple-music` | Play Activity CSV, optionally the daily-playlists CSV alongside it | Several column generations are handled. A file with no recognised title column is rejected outright. Where the per-play export omits the artist, it is recovered from `Track Description` only when that is unambiguous. |
| `spotify` | Account-data JSON export | Podcast episodes and audiobooks are reported as rejected records rather than dropped. |
| `youtube-music` | Google Takeout archive | Ordinary video watches are told apart from music plays and rejected as such. |
| `listenbrainz` | JSON, JSON Lines, or a full-export ZIP | The richest source of MusicBrainz identifiers, which is how the same recording reported by two services is recognised as one play. |
| `lastfm` | CSV export, or the `lastfm` JSON preset | Epoch seconds and milliseconds are both accepted. |
| `json` | JSON array or JSON Lines | Any shape, via a `--preset` field mapping. |

Records Sine cannot read are reported with a reason, never dropped silently.

Individual integrations must document their authentication requirements, available data, identifiers, historical limitations, and any data sent to external services.

## Contributing

Contributions are welcome. Please submit a Pull Request with a focused change and enough context to review it.

See [CONTRIBUTING.md](CONTRIBUTING.md) for repository guidance.

## Licence

See [LICENSE](LICENSE) for the GNU Affero General Public License, version 3 or any later version.
