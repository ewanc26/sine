"""Parsing and validating model output.

A model is not a database and not a parser, so its reply is treated as untrusted
input. Extraction is deliberately lenient — models wrap JSON in prose and code
fences even when told not to — and validation is deliberately strict.

The distinction matters: leniency here means "find the JSON the model meant",
never "guess at the structure of the reply". If the payload does not match the
schema, validation fails and the engine's repair path runs.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError


class ResponseParseError(Exception):
    """A model reply could not be read as the expected structured object."""


def extract_json_object(text: str) -> Any:
    """Extract the first complete JSON value from a model reply.

    Handles the common failure modes in order: the reply is already JSON, it is
    fenced, it is fenced with a language tag, or it has prose around the object.
    A balanced-brace scan is used for the last case so that braces inside strings
    do not end the object early.
    """

    candidate = text.strip()
    if not candidate:
        raise ResponseParseError("the reply was empty")

    try:
        return json.loads(candidate)
    except ValueError:
        pass

    fenced = _strip_code_fence(candidate)
    if fenced is not None:
        try:
            return json.loads(fenced)
        except ValueError:
            candidate = fenced

    for opener, closer in (("{", "}"), ("[", "]")):
        block = _first_balanced(candidate, opener, closer)
        if block is not None:
            try:
                return json.loads(block)
            except ValueError:
                continue

    raise ResponseParseError("no complete JSON value was found in the reply")


def _strip_code_fence(text: str) -> str | None:
    if not text.startswith("```"):
        return None
    lines = text.splitlines()
    if len(lines) < 3:
        return None
    if not lines[0].startswith("```"):
        return None
    for index in range(len(lines) - 1, 0, -1):
        if lines[index].strip().startswith("```"):
            return "\n".join(lines[1:index]).strip()
    return None


def _first_balanced(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for position in range(start, len(text)):
        char = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : position + 1]
    return None


def parse_model_payload(text: str, model: type[BaseModel]) -> BaseModel:
    """Extract and validate a model reply against ``model``.

    Raises :class:`ResponseParseError` with a reason the engine can act on, so
    that a repair attempt is possible.
    """

    try:
        payload = extract_json_object(text)
    except ResponseParseError as exc:
        raise ResponseParseError(str(exc)) from exc

    if not isinstance(payload, dict):
        raise ResponseParseError(
            f"expected a JSON object at the top level, got {type(payload).__name__}"
        )

    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise ResponseParseError(_describe_validation_error(exc)) from exc


def _describe_validation_error(error: ValidationError) -> str:
    """Summarise a validation failure in a form worth showing a model."""

    problems: list[str] = []
    for detail in error.errors(include_url=False):
        location = ".".join(str(part) for part in detail["loc"]) or "(root)"
        problems.append(f"{location}: {detail['msg']}")
    return "output did not match the required schema — " + "; ".join(problems[:8])


__all__ = [
    "ResponseParseError",
    "extract_json_object",
    "parse_model_payload",
]
