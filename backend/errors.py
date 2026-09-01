"""Structured exception hierarchy for ArenaHub.

All errors raised by ArenaHub derive from :class:`ArenaHubError` so callers
(CLI, gateway) can handle them with a single except clause while still
distinguishing specific failure modes.
"""

from __future__ import annotations


class ArenaHubError(Exception):
    """Base class for every ArenaHub error."""


class ArenaConfigError(ArenaHubError):
    """Configuration is missing or invalid (e.g. no ``ARENA_API_KEY``)."""


class ArenaValidationError(ArenaHubError):
    """A request failed local validation before ever hitting the API."""


class ArenaNotSupportedError(ArenaHubError):
    """The requested compatibility feature is not supported end-to-end.

    Used when a client asks for something that ArenaHub cannot honestly
    translate (e.g. an unsupported content type) so we return a clear
    compatibility error instead of pretending it worked.
    """


class GatewayRateLimitError(ArenaHubError):
    """Local (gateway) rate limit tripped before the request went upstream."""

    def __init__(self, message: str = "Rate limit exceeded.", *, retry_after: int = 60) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after = str(retry_after)


class ArenaConnectionError(ArenaHubError):
    """The Arena API could not be reached (DNS, network, connection refused)."""


class ArenaTimeoutError(ArenaHubError):
    """The Arena API did not respond within the configured timeout."""


class ArenaAPIError(ArenaHubError):
    """The Arena API responded with an HTTP error status.

    Subclasses refine specific status codes; the raw response is always kept
    for debugging (but never contains credentials — we never log headers).
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str | None = None,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after

    def __str__(self) -> str:
        if self.status_code is not None:
            return f"[HTTP {self.status_code}] {self.message}"
        return self.message


class ArenaAuthError(ArenaAPIError):
    """401/403 from the Arena API — the API key is missing/invalid/unauthorized."""


class ArenaRateLimitError(ArenaAPIError):
    """429 from the Arena API — rate limited. ``retry_after`` may be set."""


class ArenaModelError(ArenaAPIError):
    """The requested model id is invalid or unknown to the Arena API."""
