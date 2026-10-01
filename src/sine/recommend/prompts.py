"""Prompts for the recommendation step.

The prompt states what the model is and is not: it is being asked to reason over a
listener's recorded history and to propose tracks, not to act as a music database.
It is told explicitly that frequency does not establish preference, that it must
not invent metadata it cannot support, and that it should say so when the data is
thin.
"""

from __future__ import annotations

from collections.abc import Mapping

from sine.llm.messages import Message
from sine.llm.structured import describe_schema
from sine.models.profile import ListeningProfile
from sine.models.recommendation import (
    PlaylistRequest,
    RecommendationFocus,
    RecommendationRequest,
)
from sine.profile.context import render_profile_context, render_request_context

SYSTEM_PROMPT = """\
You are the reasoning component of Sine, a music recommendation tool.

You receive a listener's own listening history, summarised by Sine. You propose \
tracks that listener might genuinely want to hear. You are not a music database, \
and you do not have authoritative metadata.

Rules you must follow:

- Ground every recommendation in the listening data you are given. Cite the \
specific artists, albums, or habits in the data that led you to it.
- Play frequency is a measurement, not proof of taste. Say "was played N times", \
never "loves" or "favourite".
- Do not invent genres, release years, album names, or release dates. If you are \
not confident about a detail, omit it.
- Do not repeat the purpose back to the listener; recommend music.
- Mark each piece of supporting evidence as either "observed" (a fact from the \
data) or "inferred" (your reading of it). Never present an inference as an \
observation.
- If the data is thin, say so in your notes and still give your best suggestions.
- Prefer real, specific recordings over vague genre descriptions.
- Rate your own confidence honestly. Use "low" when the data gives you little to \
work from.
"""

PLAYLIST_RULES = """

When you are asked for a playlist rather than a list:

- The order is the answer. Choose tracks because of where they sit next to each \
other, not because each is independently the best pick.
- Give every track a "position", counting from 1, and a "transition" saying why it \
follows the one before it. Omit the transition for the first track; there is \
nothing to transition from.
- Describe connections in musical terms the listener can hear — mood, tempo, \
texture, vocal style, the ground shifting — and ground them in the history. Do not \
invent BPMs, key changes, release years, or credits to justify a placement.
- Do not state or estimate track durations. Sine has no duration data, so any \
length you state would be invented. Pick a count that plausibly fits the requested \
length and let the listener judge it.
- Give the sequence a title and say in one paragraph what it is trying to do.
"""

FOCUS_GUIDANCE: Mapping[RecommendationFocus, str] = {
    RecommendationFocus.DISCOVERY: (
        "Recommend tracks by artists the listener has not played before, drawn from "
        "the musical neighbourhood of what they do play."
    ),
    RecommendationFocus.DEEPENING: (
        "Recommend deeper cuts by artists the listener already plays, including "
        "rarer material, and work that sits alongside their established listening."
    ),
    RecommendationFocus.RECENT_ROTATION: (
        "Weight the listener's most recent listening most heavily. They are asking "
        "what suits the current phase, not the long run."
    ),
    RecommendationFocus.FAMILIARITY: (
        "Favour well-known, accessible tracks in the listener's own territory. "
        "Reliability beats novelty here."
    ),
    RecommendationFocus.SURPRISE: (
        "Stretch beyond the obvious. Take the listener's listening seriously but \
reach past its edge, and explain the connection you are making."
    ),
}


def build_system_prompt(
    *, include_schema: str | None = None, playlist: bool = False
) -> Message:
    """Build the system message, optionally appending the required JSON schema."""

    rules = SYSTEM_PROMPT + PLAYLIST_RULES if playlist else SYSTEM_PROMPT
    if include_schema is None:
        return Message.system(rules)
    return Message.system(
        f"{rules}\n\n"
        "Reply with a single JSON object and nothing else. No prose before or "
        "after it, no Markdown code fences. It must satisfy this JSON Schema:\n\n"
        f"{include_schema}"
    )


def build_user_prompt(
    profile: ListeningProfile, request: RecommendationRequest
) -> Message:
    """Build the user message from rendered profile and request context."""

    sections = [
        render_profile_context(profile),
        "",
        render_request_context(request),
        "",
        "TASK",
        FOCUS_GUIDANCE[request.focus],
    ]
    if request.focus is RecommendationFocus.DISCOVERY and not request.seed_artists:
        sections.append(
            "Balance familiarity and discovery: the listener has not asked to be "
            "surprised, so most recommendations should sit near what they already "
            "play rather than far from it."
        )
    return Message.user("\n".join(sections))


def build_playlist_prompt(
    profile: ListeningProfile, request: PlaylistRequest
) -> Message:
    """Build the user message for an ordered sequence.

    A playlist needs the same grounding as a set of recommendations and more: the
    listener is asking what follows what, so the task section says so explicitly
    rather than leaving the model to treat the list as a ranking.
    """

    sections = [
        render_profile_context(profile),
        "",
        render_request_context(request),
        "",
        "TASK",
        "Build one ordered sequence, not a ranked list. Every position must earn \
its place by what it does after the track before it.",
        FOCUS_GUIDANCE[request.focus],
    ]
    if request.title:
        sections.append(f"The listener asked for a playlist called: {request.title}")
    return Message.user("\n".join(sections))


def build_json_reminder(schema: Mapping[str, object]) -> Message:
    """Instruction used when the provider cannot be constrained at the API level."""

    return Message.user(
        "Your previous reply could not be read as the required JSON object. "
        "Reply again with one JSON object and nothing else, satisfying this "
        f"schema exactly:\n\n{describe_schema(schema)}"
    )


__all__ = [
    "FOCUS_GUIDANCE",
    "PLAYLIST_RULES",
    "SYSTEM_PROMPT",
    "build_json_reminder",
    "build_playlist_prompt",
    "build_system_prompt",
    "build_user_prompt",
]
