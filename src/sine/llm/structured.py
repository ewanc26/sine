"""JSON Schema helpers shared by every adapter.

Most providers accept a subset of JSON Schema and several reject Pydantic's
output as-is. Pydantic omits ``additionalProperties: false`` unless extra fields
are forbidden, and leaves fields with defaults out of ``required``; the strict
dialects require both. Rather than making every recommendation model carry
``ConfigDict(extra="forbid")`` and a field layout chosen to satisfy four
providers, the mismatch is fixed in one place.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _is_object(node: Mapping[str, Any]) -> bool:
    declared = node.get("type")
    if declared == "object":
        return True
    # Nullable objects are expressed as a type array in Pydantic output.
    return isinstance(declared, list) and "object" in declared


def _type_list(declared: Any) -> list[str]:
    if isinstance(declared, list):
        return [item for item in declared if isinstance(item, str)]
    if isinstance(declared, str):
        return [declared]
    return []


def _make_nullable(schema: dict[str, Any]) -> None:
    """Widen a property schema so ``null`` is an acceptable value.

    Strict dialects require every property to be present. Rather than forcing a
    model to invent a value for a field that may genuinely be unknown, the field
    becomes nullable and its default becomes ``None``, so validation downstream is
    unaffected.
    """

    variants = schema.get("anyOf")
    if isinstance(variants, list):
        if not any(
            isinstance(item, dict) and _type_list(item.get("type")) == ["null"]
            for item in variants
        ):
            variants.append({"type": "null"})
        schema["default"] = None
        return

    if "type" not in schema:
        return
    if "null" in _type_list(schema.get("type")):
        schema["default"] = None
        return

    original = {key: value for key, value in schema.items() if key != "default"}
    schema.clear()
    schema["anyOf"] = [original, {"type": "null"}]
    schema["default"] = None


def _promote_to_required(node: dict[str, Any]) -> None:
    """List every declared property as required, as strict dialects demand."""

    properties = node.get("properties")
    if not isinstance(properties, dict):
        return

    required = set(node.get("required") or [])
    for name, schema in properties.items():
        if not isinstance(schema, dict):
            continue
        if name not in required:
            required.add(name)
        if "default" in schema:
            _make_nullable(schema)
    node["required"] = sorted(required)


def ensure_strict(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` tightened for strict JSON-Schema dialects.

    Recursively, for every object node: sets ``additionalProperties: false`` and
    lists every property as required, widening defaulted properties to accept
    ``null``. Key order, titles, and descriptions are preserved. Non-schema
    keywords such as ``$defs`` are carried through untouched; all four researched
    API families resolve local ``$defs``.
    """

    return _tighten(dict(schema))


def _tighten(node: Any) -> Any:
    if isinstance(node, list):
        return [_tighten(item) for item in node]
    if not isinstance(node, dict):
        return node

    result = {key: _tighten(value) for key, value in node.items()}

    if _is_object(node):
        result["additionalProperties"] = False
        _promote_to_required(result)

    return result


def describe_schema(schema: Mapping[str, Any]) -> str:
    """Render a schema for inclusion in a prompt.

    Used when a model cannot be constrained at the API level. Deterministic, so
    that prompts are reproducible.
    """

    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False)
