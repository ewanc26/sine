"""Shared HTTP transport, error mapping, and retry for provider adapters.

Centralising this is what keeps vendor behaviour out of the rest of Sine. Each
adapter supplies a URL, headers, and a body; this module turns transport and
status-code failures into :mod:`sine.llm.errors` types and applies a retry policy
that only retries failures worth retrying.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from sine.llm.errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

#: Error bodies are truncated before they reach an exception message. Validation
#: errors from several providers echo the request body, which for Sine means
#: listening context. Long bodies are dropped rather than propagated.
MAX_ERROR_BODY_CHARS = 300

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 529})


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded exponential backoff for transient provider failures."""

    attempts: int = 3
    initial_delay: float = 0.5
    max_delay: float = 8.0
    backoff: float = 2.0
    respect_retry_after: bool = True

    def delay_for(self, attempt: int, retry_after: float | None) -> float:
        if self.respect_retry_after and retry_after is not None:
            return min(max(retry_after, 0.0), self.max_delay)
        return min(self.initial_delay * (self.backoff**attempt), self.max_delay)


def _truncate(text: str) -> str:
    if len(text) <= MAX_ERROR_BODY_CHARS:
        return text
    return f"{text[:MAX_ERROR_BODY_CHARS]}… (truncated)"


def _extract_message(payload: Any) -> str | None:
    """Pull a human-readable message out of the error shapes in common use.

    Handles the OpenAI-style ``{"error": {"message": ...}}`` used by most
    OpenAI-compatible providers, Anthropic's ``{"type": "error", "error": {...}}``,
    Gemini's gRPC-style ``{"error": {"code", "message", "status"}}``, and
    OpenRouter's numeric ``error.code`` with ``error.metadata.error_type``.
    """

    if not isinstance(payload, dict):
        return None

    error = payload.get("error")
    if isinstance(error, dict):
        candidates = [error.get("message")]
        metadata = error.get("metadata")
        if isinstance(metadata, dict):
            candidates.extend(
                [metadata.get("error_type"), metadata.get("provider_code")]
            )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        if isinstance(error.get("type"), str):
            return error["type"]
        if error.get("code") is not None:
            return f"error code {error['code']}"
    elif isinstance(error, str) and error.strip():
        return error

    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message

    return None


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        return (parsedate_to_datetime(raw) - time.time()).total_seconds()
    except (TypeError, ValueError):
        return None


def error_for_response(provider_id: str, response: httpx.Response) -> ProviderError:
    """Map a non-2xx response onto a provider-neutral error."""

    try:
        payload = response.json()
    except ValueError:
        payload = None

    detail = _extract_message(payload)
    if detail is None:
        detail = _truncate(response.text.strip())
    # A provider's error message is truncated on every path, not just the raw-body
    # one: several providers echo the rejected request back in ``error.message``,
    # which for Sine means the listener's own history.
    detail = _truncate(detail) or f"HTTP {response.status_code}"

    if response.status_code in (401, 403):
        return ProviderAuthenticationError(provider_id, detail)
    if response.status_code == 429:
        return ProviderRateLimitError(
            provider_id, detail, retry_after_seconds=_retry_after_seconds(response)
        )
    if response.status_code in (408, 504):
        return ProviderTimeoutError(provider_id, detail)
    if response.status_code >= 500:
        return ProviderUnavailableError(
            provider_id, detail, status_code=response.status_code
        )
    return ProviderRequestError(provider_id, detail, status_code=response.status_code)


class HttpTransport:
    """Retrying JSON transport bound to one provider."""

    def __init__(
        self,
        *,
        provider_id: str,
        client: httpx.Client,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._provider_id = provider_id
        self._client = client
        self._retry = retry or RetryPolicy()
        self._sleep = sleep

    def post_json(
        self, url: str, *, headers: dict[str, str], payload: dict[str, Any]
    ) -> Any:
        return self._send("POST", url, headers=headers, json_body=payload)

    def get_json(self, url: str, *, headers: dict[str, str]) -> Any:
        return self._send("GET", url, headers=headers)

    def _send(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        last: ProviderError | None = None

        for attempt in range(self._retry.attempts):
            try:
                response = self._client.request(
                    method, url, headers=headers, json=json_body
                )
            except httpx.TimeoutException as exc:
                last = ProviderTimeoutError(self._provider_id, _truncate(str(exc)))
            except httpx.HTTPError as exc:
                last = ProviderUnavailableError(self._provider_id, _truncate(str(exc)))
            else:
                if response.is_success:
                    return self._decode(response)
                last = error_for_response(self._provider_id, response)
                if response.status_code not in RETRYABLE_STATUS:
                    raise last
                if attempt < self._retry.attempts - 1:
                    self._sleep(
                        self._retry.delay_for(attempt, _retry_after_seconds(response))
                    )
                    continue
            if attempt < self._retry.attempts - 1:
                self._sleep(self._retry.delay_for(attempt, None))

        raise last or ProviderUnavailableError(self._provider_id, "request failed")

    def close(self) -> None:
        self._client.close()

    def _decode(self, response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderResponseError(
                self._provider_id,
                f"response was not JSON: {_truncate(str(exc))}",
                status_code=response.status_code,
                payload=_truncate(response.text),
            ) from exc


def build_client(timeout: float) -> httpx.Client:
    """Create the HTTP client used by adapters when none is injected."""

    return httpx.Client(timeout=timeout, follow_redirects=True)


def require_mapping(payload: Any, provider_id: str, what: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProviderResponseError(
            provider_id,
            f"expected a JSON object for {what}, got {type(payload).__name__}",
        )
    return payload
