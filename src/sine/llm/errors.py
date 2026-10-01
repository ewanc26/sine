"""Provider-neutral errors.

Every adapter maps its own failure modes onto these types. Nothing provider
specific — an HTTP status code shape, a vendor error enum, an SDK exception —
may cross into the recommendation engine. Adapters must also never leak a
credential into an error message.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for every LLM failure surfaced to the core."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.message = message


class ProviderConfigurationError(ProviderError):
    """The provider is not configured well enough to be called."""


class ProviderAuthenticationError(ProviderError):
    """Credentials were missing, rejected, or insufficient."""


class ProviderRateLimitError(ProviderError):
    """The provider refused the request because of a rate or quota limit."""

    def __init__(
        self, provider: str, message: str, *, retry_after_seconds: float | None = None
    ) -> None:
        super().__init__(provider, message)
        self.retry_after_seconds = retry_after_seconds


class ProviderRequestError(ProviderError):
    """The provider rejected the request as invalid."""

    def __init__(
        self, provider: str, message: str, *, status_code: int | None = None
    ) -> None:
        super().__init__(provider, message)
        self.status_code = status_code


class ProviderTimeoutError(ProviderError):
    """The provider did not respond in time."""


class ProviderUnavailableError(ProviderError):
    """The provider is temporarily failing."""

    def __init__(
        self, provider: str, message: str, *, status_code: int | None = None
    ) -> None:
        super().__init__(provider, message)
        self.status_code = status_code


class ProviderResponseError(ProviderError):
    """The provider responded with something Sine could not interpret.

    This covers malformed payloads, empty completions, in-band error objects, and
    refusals. It is the error the recommendation engine's repair path exists for.
    """

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        payload: str | None = None,
    ) -> None:
        super().__init__(provider, message)
        self.status_code = status_code
        self.payload = payload


class ProviderUnsupportedError(ProviderError):
    """The request asked for a capability this model does not have."""


#: Failures worth retrying, keyed to the exception type. A 400 is never retried:
#: the request is wrong, and repeating it will not fix it.
RETRYABLE: tuple[type[ProviderError], ...] = (
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
