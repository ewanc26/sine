"""Recommendation generation and validation."""

from sine.recommend.engine import (
    MINUTES_PER_TRACK,
    RecommendationEngine,
    RecommendationError,
    playlist_schema,
    recommendation_schema,
)
from sine.recommend.focus import BASE_GUIDANCE, focus_guidance
from sine.recommend.prompts import (
    SYSTEM_PROMPT,
    build_playlist_prompt,
    build_system_prompt,
    build_user_prompt,
)
from sine.recommend.validation import (
    ResponseParseError,
    extract_json_object,
    parse_model_payload,
)

__all__ = [
    "BASE_GUIDANCE",
    "MINUTES_PER_TRACK",
    "SYSTEM_PROMPT",
    "RecommendationEngine",
    "RecommendationError",
    "ResponseParseError",
    "build_playlist_prompt",
    "build_system_prompt",
    "build_user_prompt",
    "extract_json_object",
    "focus_guidance",
    "parse_model_payload",
    "playlist_schema",
    "recommendation_schema",
]
