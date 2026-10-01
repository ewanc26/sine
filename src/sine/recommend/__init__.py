"""Recommendation generation and validation."""

from sine.recommend.engine import (
    RecommendationEngine,
    RecommendationError,
    recommendation_schema,
)
from sine.recommend.prompts import SYSTEM_PROMPT, build_system_prompt, build_user_prompt
from sine.recommend.validation import (
    ResponseParseError,
    extract_json_object,
    parse_model_payload,
)

__all__ = [
    "SYSTEM_PROMPT",
    "RecommendationEngine",
    "RecommendationError",
    "ResponseParseError",
    "build_system_prompt",
    "build_user_prompt",
    "extract_json_object",
    "parse_model_payload",
    "recommendation_schema",
]
