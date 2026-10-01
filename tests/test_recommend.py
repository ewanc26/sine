"""Recommendation parsing, validation, and prompt construction.

The model is treated as a recommender, not a database: whatever it returns is
validated before use, and malformed output is repaired once rather than trusted.
"""

import json
from datetime import UTC, datetime, timedelta

import pytest

from sine.llm.capabilities import StructuredOutputDialect
from sine.llm.errors import ProviderError, ProviderUnavailableError
from sine.llm.generation import (
    FinishReason,
    GenerationRequest,
    GenerationResponse,
    TokenUsage,
)
from sine.llm.provider import Model
from sine.llm.structured import describe_schema
from sine.models import (
    Artist,
    ListeningEvent,
    ListeningHistory,
    Playlist,
    PlaylistRequest,
    RecommendationFocus,
    RecommendationRequest,
    Track,
)
from sine.models.profile import ListeningProfile
from sine.profile.builder import build_profile
from sine.profile.statistics import DEFAULT_TOP_N
from sine.recommend.engine import (
    RecommendationEngine,
    RecommendationError,
    playlist_schema,
    recommendation_schema,
)
from sine.recommend.prompts import build_system_prompt, build_user_prompt
from sine.recommend.validation import (
    ResponseParseError,
    extract_json_object,
    parse_model_payload,
)

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def profile() -> ListeningProfile:
    events = [
        ListeningEvent(
            track=Track(
                title=f"Track {index}", artists=(Artist(name=f"Artist {index % 4}"),)
            ),
            played_at=NOW - timedelta(days=index * 0.5),
            source="test",
            play_count=2,
        )
        for index in range(40)
    ]
    return build_profile(ListeningHistory.from_events(events), now=NOW)


# ------------------------------------------------------------------- JSON shape


def test_a_plain_json_object_is_extracted() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_json_wrapped_in_a_code_fence_is_extracted() -> None:
    text = 'Here you go:\n```json\n{"recommendations": []}\n```\nHope that helps!'
    assert extract_json_object(text) == {"recommendations": []}


def test_json_with_surrounding_prose_is_extracted() -> None:
    text = 'I think {"track": {"title": "Roygbiv"}} is a good fit. Enjoy!'
    assert extract_json_object(text) == {"track": {"title": "Roygbiv"}}


def test_nested_objects_and_escaped_quotes_survive_extraction() -> None:
    text = 'blah {"a": {"b": [1, {"c": "say \\"hi\\""}]}} tail'
    assert extract_json_object(text) == {"a": {"b": [1, {"c": 'say "hi"'}]}}


def test_braces_inside_strings_do_not_end_the_object_early() -> None:
    assert extract_json_object('{"note": "a } b"}') == {"note": "a } b"}


@pytest.mark.parametrize("text", ["", "no json at all", "{unclosed"])
def test_unparseable_output_raises_a_clear_error(text: str) -> None:
    with pytest.raises(ResponseParseError):
        extract_json_object(text)


def test_a_top_level_array_is_not_a_recommendation_object() -> None:
    """JSON parses, but the shape is wrong; only the object branch is accepted."""

    assert extract_json_object("[]") == []
    from sine.models import RecommendationSet

    with pytest.raises(ResponseParseError, match="expected a JSON object"):
        parse_model_payload("[]", RecommendationSet)


def test_payloads_are_validated_against_the_schema() -> None:
    from sine.models import RecommendationSet

    parsed = parse_model_payload(
        '{"recommendations": [{"track": {"title": "Roygbiv", "artists": [{"name": "BoC"}]},'
        ' "rationale": "similar ambience", "confidence": "high",'
        ' "novelty": "new_artist"}]}',
        RecommendationSet,
    )
    assert isinstance(parsed, RecommendationSet)
    assert parsed.recommendations[0].track.title == "Roygbiv"


def test_a_payload_missing_required_fields_is_rejected() -> None:
    from sine.models import RecommendationSet

    with pytest.raises(ResponseParseError, match="did not match the required schema"):
        parse_model_payload(
            '{"recommendations": [{"rationale": "no track"}]}', RecommendationSet
        )


def test_the_recommendation_schema_is_strict() -> None:
    """Strict mode is what makes structured output trustworthy."""

    schema = recommendation_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"], "every field must be required so the model must fill it"


# ----------------------------------------------------------------------- prompts


def test_the_system_prompt_distinguishes_observation_from_inference() -> None:
    prompt = build_system_prompt().content.casefold()
    assert "observ" in prompt
    assert "infer" in prompt or "prefer" in prompt
    assert "recommend" in prompt


def test_the_user_prompt_carries_the_listening_context() -> None:
    rendered = build_user_prompt(profile(), RecommendationRequest(limit=5)).content
    assert "Artist 0" in rendered
    assert "REQUEST" in rendered


def test_the_schema_is_described_in_the_prompt_when_not_enforced_by_the_api() -> None:
    without = build_system_prompt().content
    with_schema = build_system_prompt(
        include_schema=describe_schema(recommendation_schema())
    ).content
    assert len(with_schema) > len(without)
    # The schema reaches the model in a form it can actually act on.
    assert "recommendations" in with_schema
    assert "JSON Schema" in with_schema


def test_exclusions_reach_the_prompt() -> None:
    rendered = build_user_prompt(
        profile(),
        RecommendationRequest(exclude_artists=(Artist(name="Artist 0"),), limit=3),
    ).content
    assert "Artist 0" in rendered


def test_seed_artists_reach_the_prompt() -> None:
    rendered = build_user_prompt(
        profile(),
        RecommendationRequest(seed_artists=(Artist(name="Talk Talk"),), limit=3),
    ).content
    assert "Talk Talk" in rendered


# ------------------------------------------------------------------- the engine


class StubProvider:
    """A provider that returns canned responses, recording what it was sent."""

    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[GenerationRequest] = []

    provider_id = "stub"

    def capabilities(self, model_id: str):
        from sine.llm.capabilities import ModelCapabilities

        return ModelCapabilities(
            structured_output=StructuredOutputDialect.JSON_OBJECT,
            supports_system_messages=False,
        )

    def generate(self, model_id: str, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        outcome = self.responses.pop(0) if self.responses else "{}"
        if isinstance(outcome, Exception):
            raise outcome
        return GenerationResponse(
            text=outcome,
            provider="stub",
            model="stub",
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(input_tokens=10, output_tokens=20),
        )

    def list_models(self):
        return ()


def engine_for(provider: StubProvider) -> RecommendationEngine:
    return RecommendationEngine(
        Model(
            provider=provider,
            model_id="stub",
            capabilities=provider.capabilities("stub"),
        )
    )


VALID = json.dumps(
    {
        "recommendations": [
            {
                "track": {
                    "title": "Roygbiv",
                    "artists": [{"name": "Boards of Canada"}],
                },
                "rationale": "shares the ambient, tape-saturated texture you return to",
                "confidence": "high",
                "novelty": "new_artist",
                "genre_hints": ["ambient"],
            }
        ],
        "notes": "grounded in your recent plays",
    }
)


def test_a_valid_response_is_returned() -> None:
    result = engine_for(StubProvider(VALID)).recommend(
        profile(), RecommendationRequest(limit=1)
    )
    assert len(result.recommendations) == 1
    assert result.recommendations[0].track.title == "Roygbiv"
    assert "grounded in your recent plays" in result.notes


def test_a_shortfall_is_reported_rather_than_hidden() -> None:
    """Asking for 3 and receiving 1 should be visible, not silently accepted."""

    result = engine_for(StubProvider(VALID)).recommend(
        profile(), RecommendationRequest(limit=3)
    )
    assert "2 short" in result.notes


def test_malformed_output_is_repaired_with_a_second_request() -> None:
    provider = StubProvider("sorry, I cannot do that", VALID)
    result = engine_for(provider).recommend(profile(), RecommendationRequest(limit=1))
    assert len(result.recommendations) == 1
    assert len(provider.requests) == 2
    # The repair attempt tells the model what shape was expected.
    assert "json" in provider.requests[1].messages[-1].content.casefold()


def test_output_that_stays_malformed_raises_a_recoverable_error() -> None:
    provider = StubProvider("nope", "still nope")
    with pytest.raises(RecommendationError, match="usable recommendation set"):
        engine_for(provider).recommend(profile(), RecommendationRequest(limit=3))


def test_provider_failures_surface_as_recommendation_errors() -> None:
    provider = StubProvider(ProviderUnavailableError("openai", "connection reset"))
    with pytest.raises(RecommendationError, match="openai"):
        engine_for(provider).recommend(profile(), RecommendationRequest(limit=3))


def test_recommendations_the_listener_has_played_are_dropped_by_default() -> None:
    heard = profile().statistics.top_tracks[0].item.display_name
    known = profile().statistics.top_tracks[0].item
    payload = json.dumps(
        {
            "recommendations": [
                {
                    "track": {
                        "title": known.title,
                        "artists": [{"name": artist.name} for artist in known.artists],
                    },
                    "rationale": "more of the same",
                    "confidence": "medium",
                    "novelty": "new_artist",
                }
            ]
        }
    )
    result = engine_for(StubProvider(payload)).recommend(
        profile(), RecommendationRequest(limit=3)
    )
    assert result.recommendations == ()
    assert "removed by Sine" in result.notes
    assert heard


def test_replays_are_allowed_when_requested() -> None:
    known = profile().statistics.top_tracks[0].item
    payload = json.dumps(
        {
            "recommendations": [
                {
                    "track": {
                        "title": known.title,
                        "artists": [{"name": artist.name} for artist in known.artists],
                    },
                    "rationale": "a deliberate revisit",
                    "confidence": "medium",
                    "novelty": "new_artist",
                }
            ]
        }
    )
    result = engine_for(StubProvider(payload)).recommend(
        profile(), RecommendationRequest(limit=3, allow_replays=True)
    )
    assert len(result.recommendations) == 1
    assert result.recommendations[0].novelty.value == "replay"


def test_a_played_track_outside_the_top_list_is_still_known() -> None:
    """Novelty is judged on the full tally, not the truncated top-N summary."""

    events = [
        ListeningEvent(
            track=Track(title=f"Filler {index}", artists=(Artist(name="Filler"),)),
            played_at=NOW - timedelta(days=index),
            source="test",
            play_count=5,
        )
        for index in range(DEFAULT_TOP_N + 5)
    ]
    events.append(
        ListeningEvent(
            track=Track(title="Obscure", artists=(Artist(name="Rarely Played"),)),
            played_at=NOW - timedelta(days=90),
            source="test",
            play_count=1,
        )
    )
    history_profile = build_profile(ListeningHistory.from_events(events), now=NOW)
    payload = json.dumps(
        {
            "recommendations": [
                {
                    "track": {
                        "title": "Obscure",
                        "artists": [{"name": "Rarely Played"}],
                    },
                    "rationale": "you will not have seen this lately",
                    "confidence": "medium",
                    "novelty": "new_artist",
                }
            ]
        }
    )
    result = engine_for(StubProvider(payload)).recommend(
        history_profile, RecommendationRequest(limit=3, allow_replays=True)
    )
    assert result.recommendations[0].novelty.value == "replay"
    assert result.recommendations[0].known_plays == 1

    dropped = engine_for(StubProvider(payload)).recommend(
        history_profile, RecommendationRequest(limit=3)
    )
    assert dropped.recommendations == ()


def test_excluded_artists_are_never_recommended() -> None:
    payload = json.dumps(
        {
            "recommendations": [
                {
                    "track": {
                        "title": "Roygbiv",
                        "artists": [{"name": "Boards of Canada"}],
                    },
                    "rationale": "fits",
                    "confidence": "medium",
                    "novelty": "new_artist",
                }
            ]
        }
    )
    result = engine_for(StubProvider(payload)).recommend(
        profile(),
        RecommendationRequest(
            exclude_artists=(Artist(name="Boards of Canada"),), limit=3
        ),
    )
    assert result.recommendations == ()
    assert "removed by Sine" in result.notes


def test_duplicate_recommendations_are_collapsed() -> None:
    entry = {
        "track": {"title": "Roygbiv", "artists": [{"name": "Boards of Canada"}]},
        "rationale": "fits",
        "confidence": "medium",
        "novelty": "new_artist",
    }
    result = engine_for(
        StubProvider(json.dumps({"recommendations": [entry, entry, entry]}))
    ).recommend(profile(), RecommendationRequest(limit=5))
    assert len(result.recommendations) == 1
    assert "2 suggestion(s) were removed" in result.notes


def test_the_requested_limit_is_respected() -> None:
    entries = [
        {
            "track": {
                "title": f"Track {index}",
                "artists": [{"name": f"Artist {index}"}],
            },
            "rationale": "fits",
            "confidence": "medium",
            "novelty": "new_artist",
        }
        for index in range(10)
    ]
    result = engine_for(
        StubProvider(json.dumps({"recommendations": entries}))
    ).recommend(profile(), RecommendationRequest(limit=3))
    assert len(result.recommendations) == 3


def test_novelty_is_determined_from_the_history_not_the_model() -> None:
    """The model guessing "new" for a track already played must not be believed."""

    known = profile().statistics.top_tracks[0].item
    payload = json.dumps(
        {
            "recommendations": [
                {
                    "track": {
                        "title": known.title,
                        "artists": [{"name": artist.name} for artist in known.artists],
                    },
                    "rationale": "fits",
                    "confidence": "medium",
                    "novelty": "new_artist",
                }
            ]
        }
    )
    result = engine_for(StubProvider(payload)).recommend(
        profile(), RecommendationRequest(limit=3, allow_replays=True)
    )
    assert result.recommendations[0].novelty.value == "replay"


def test_an_empty_response_is_reported_rather_than_crashing() -> None:
    result = engine_for(StubProvider('{"recommendations": []}')).recommend(
        profile(), RecommendationRequest(limit=3)
    )
    assert result.recommendations == ()
    assert "3 short" in result.notes


def test_a_model_with_no_structured_output_still_gets_the_schema_in_prompt() -> None:
    provider = StubProvider(VALID)
    engine_for(provider).recommend(profile(), RecommendationRequest(limit=3))
    request = provider.requests[0]
    # The stub advertises JSON_OBJECT only, so the schema must also be in the prompt.
    assert request.response_format.kind == "json_object"
    assert "recommendations" in request.messages[0].content


def test_instructions_are_sent_as_a_system_message_for_adapters_to_place() -> None:
    """Where a system message ends up is the adapter's job, not the engine's."""

    provider = StubProvider(VALID)
    engine_for(provider).recommend(profile(), RecommendationRequest(limit=1))
    roles = [message.role.value for message in provider.requests[0].messages]
    assert roles[0] == "system"
    assert roles[-1] == "user"


def test_the_engine_refuses_a_provider_that_cannot_structure_output() -> None:
    """Without JSON support there is nothing to validate, so Sine declines."""

    class NoStructure(StubProvider):
        def capabilities(self, model_id: str):
            from sine.llm.capabilities import ModelCapabilities

            return ModelCapabilities(structured_output=StructuredOutputDialect.NONE)

    with pytest.raises(RecommendationError, match="structured output"):
        RecommendationEngine(
            Model(
                provider=NoStructure(),
                model_id="x",
                capabilities=NoStructure().capabilities("x"),
            )
        )


def test_focus_reaches_the_provider() -> None:
    provider = StubProvider(VALID)
    engine_for(provider).recommend(
        profile(), RecommendationRequest(limit=1, focus=RecommendationFocus.SURPRISE)
    )
    text = " ".join(
        message.content for message in provider.requests[0].messages
    ).casefold()
    # The focus guidance, not just the focus label, reaches the model.
    assert "reach past its edge" in text


def test_the_guidance_is_adapted_to_this_listener_not_only_the_focus() -> None:
    """The same focus produces a different brief for a different history."""

    provider = StubProvider(VALID)
    engine_for(provider).recommend(
        profile(), RecommendationRequest(limit=1, focus=RecommendationFocus.DISCOVERY)
    )
    text = " ".join(message.content for message in provider.requests[0].messages)
    # profile() is four artists played repeatedly: a measured reason to reach out.
    assert "plays are concentrated" in text
    assert "Reach further out" in text


def test_a_playlist_brief_carries_the_same_adapted_guidance() -> None:
    provider = StubProvider(
        json.dumps(
            {
                "title": "Late Shift",
                "tracks": [
                    {
                        "position": 1,
                        "track": {"title": "T", "artists": [{"name": "A"}]},
                        "rationale": "fits",
                        "confidence": "low",
                        "novelty": "new_artist",
                    }
                ],
            }
        )
    )
    engine_for(provider).playlist(
        profile(), PlaylistRequest(limit=3, focus=RecommendationFocus.DISCOVERY)
    )
    text = " ".join(message.content for message in provider.requests[0].messages)
    assert "plays are concentrated" in text
    assert "Build one ordered sequence" in text


def test_a_generation_request_carries_no_secrets() -> None:
    """Only listening context goes to a provider, never local paths."""

    provider = StubProvider(VALID)
    engine_for(provider).recommend(profile(), RecommendationRequest(limit=3))
    text = " ".join(message.content for message in provider.requests[0].messages)
    assert "/Users/" not in text
    assert "Artist 0" in text


def test_provider_errors_remain_provider_neutral() -> None:
    assert issubclass(ProviderUnavailableError, ProviderError)


# --------------------------------------------------------------------- playlists


def playlist_entry(
    title: str, artist: str, position: int, transition: str | None = None
):
    """A model-shaped playlist entry, as a playlist request expects one."""

    entry = {
        "position": position,
        "track": {"title": title, "artists": [{"name": artist}]},
        "rationale": "fits the sequence",
        "confidence": "medium",
        "novelty": "new_artist",
    }
    if transition is not None:
        entry["transition"] = transition
    return entry


def playlist_payload(*entries: dict, title: str = "Late Shift") -> str:
    return json.dumps({"title": title, "intent": "unwind slowly", "tracks": entries})


def test_a_playlist_keeps_the_order_the_model_asked_for() -> None:
    payload = playlist_payload(
        playlist_entry("Second", "Artist B", 2, "the tempo lifts"),
        playlist_entry("First", "Artist A", 1),
        playlist_entry("Third", "Artist C", 3, "the drums drop away"),
    )
    result = engine_for(StubProvider(payload)).playlist(
        profile(), PlaylistRequest(limit=5)
    )
    assert result.title == "Late Shift"
    assert [track.track.title for track in result.tracks] == [
        "First",
        "Second",
        "Third",
    ]
    assert [track.position for track in result.tracks] == [1, 2, 3]
    assert result.tracks[0].transition is None
    assert result.tracks[1].transition == "the tempo lifts"


def test_playlist_positions_are_renumbered_after_sine_drops_a_track() -> None:
    """A gap in the sequence would be a broken playlist, so Sine closes it."""

    heard = profile().statistics.top_tracks[0].item
    payload = playlist_payload(
        playlist_entry("First", "Artist A", 1),
        playlist_entry(heard.title, heard.artists[0].name, 2, "already known"),
        playlist_entry("Third", "Artist C", 3, "after that"),
    )
    result = engine_for(StubProvider(payload)).playlist(
        profile(), PlaylistRequest(limit=5)
    )
    assert [track.track.title for track in result.tracks] == ["First", "Third"]
    assert [track.position for track in result.tracks] == [1, 2]
    assert "renumbered" in result.notes


def test_playlist_screening_is_the_same_as_for_a_set() -> None:
    """Exclusions, duplicates, and replays are dropped from a playlist too."""

    entry = playlist_entry("Roygbiv", "Boards of Canada", 1)
    excluded = playlist_entry("Xtal", "Aphex Twin", 2)
    payload = playlist_payload(
        entry, playlist_entry("Roygbiv", "Boards of Canada", 3), excluded
    )
    result = engine_for(StubProvider(payload)).playlist(
        profile(),
        PlaylistRequest(limit=5, exclude_artists=(Artist(name="Aphex Twin"),)),
    )
    assert [track.track.title for track in result.tracks] == ["Roygbiv"]
    assert "2 track(s) were removed" in result.notes


def test_a_playlist_without_transitions_says_so() -> None:
    """An unexplained sequence is still a sequence, but the gap is reported."""

    payload = playlist_payload(
        playlist_entry("First", "Artist A", 1),
        playlist_entry("Second", "Artist B", 2),
        playlist_entry("Third", "Artist C", 3),
    )
    result = engine_for(StubProvider(payload)).playlist(
        profile(), PlaylistRequest(limit=5)
    )
    assert "not every placement was explained" in result.notes


def test_a_length_target_is_stated_as_unverified() -> None:
    """Sine has no duration data, so it must not imply that it hit the target."""

    payload = playlist_payload(playlist_entry("First", "Artist A", 1))
    result = engine_for(StubProvider(payload)).playlist(
        profile(), PlaylistRequest(limit=5, target_minutes=45)
    )
    assert "running time is unknown" in result.notes


def test_playlist_prompts_the_model_not_to_invent_durations() -> None:
    provider = StubProvider(playlist_payload(playlist_entry("First", "Artist A", 1)))
    engine_for(provider).playlist(
        profile(), PlaylistRequest(limit=3, target_minutes=30, title="Rainy Window")
    )
    text = " ".join(message.content for message in provider.requests[0].messages)
    assert "Do not state or estimate track durations" in text
    assert "30 minutes" in text
    assert "Rainy Window" in text
    # The ordering instruction is about the shape of the answer, not just the label.
    assert "one ordered sequence, not a ranked list" in text


def test_the_playlist_schema_asks_for_a_sequence_not_a_set() -> None:
    schema = playlist_schema()
    tracks = schema["$defs"]["PlaylistTrack"]["properties"]
    assert "position" in tracks
    assert "transition" in tracks
    assert schema["properties"]["title"]["type"] == "string"


def test_a_playlist_reply_that_stays_malformed_raises_a_recoverable_error() -> None:
    provider = StubProvider("not json", "still not json")
    with pytest.raises(RecommendationError, match="usable playlist"):
        engine_for(provider).playlist(profile(), PlaylistRequest(limit=3))


def test_a_playlist_is_asked_for_under_the_playlist_schema_name() -> None:
    provider = StubProvider(playlist_payload(playlist_entry("First", "Artist A", 1)))
    engine_for(provider).playlist(profile(), PlaylistRequest(limit=3))
    assert provider.requests[0].response_format.kind == "json_object"


def test_a_short_playlist_is_reported_rather_than_padded() -> None:
    payload = playlist_payload(playlist_entry("First", "Artist A", 1))
    result = engine_for(StubProvider(payload)).playlist(
        profile(), PlaylistRequest(limit=8)
    )
    assert len(result.tracks) == 1
    assert "7 short" in result.notes
    assert "will not pad the gap" in result.notes


def test_a_playlist_is_validated_into_the_domain_model() -> None:
    """The model-facing shape and the validated shape are the same model."""

    playlist = Playlist.model_validate(
        json.loads(playlist_payload(playlist_entry("First", "Artist A", 1)))
    )
    assert playlist.tracks[0].track.artist.name == "Artist A"
