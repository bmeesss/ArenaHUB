"""HTTP security middleware for ArenaHub.

A single middleware enforces, in order:

1. **Request IDs** — every request gets an ``X-Request-ID`` (honouring an
   inbound one), included in error envelopes for support/debugging.
2. **Authentication** — every non-health endpoint requires the ArenaHub
   gateway key. The Arena API key is never involved here and never leaves the
   server. Error envelopes match the calling surface (OpenAI vs Anthropic vs
   web) via the request path.
3. **Request size limit** — oversized JSON bodies are rejected with 413
   before they are buffered (a hard cap protects the process).
4. **Rate limiting** — a simple in-memory per-client token window; the
   limiter is behind :class:`RateLimiter` so it can be swapped for Redis etc.

CORS is added separately (as the outermost layer) so OPTIONS preflight
requests are answered without authentication.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import Message

# Paths that never require a gateway key.
PUBLIC_PATHS = frozenset({"/health", "/", "/docs", "/redoc", "/openapi.json"})


def error_envelope(path: str, message: str, *, error_type: str, code: str | None = None,
                   status: int = 400, request_id: str | None = None) -> JSONResponse:
    """Build a surface-appropriate JSON error (OpenAI / Anthropic / web)."""
    if path.startswith("/v1/messages"):
        body: dict[str, Any] = {
            "type": "error",
            "error": {"type": error_type, "message": message},
        }
    elif path.startswith("/v1/"):
        err: dict[str, Any] = {"message": message, "type": error_type}
        if code:
            err["code"] = code
        body = {"error": err}
    else:
        body = {"error": {"message": message, "type": error_type, "code": code}}
    if request_id:
        body["request_id"] = request_id
    return JSONResponse(status_code=status, content=body)


class RateLimiter:
    """Sliding-window per-client limiter.

    Replaceable with a Redis-backed implementation for multi-process
    deployments; the interface is just :meth:`check`.
    """

    def __init__(self, max_per_minute: int) -> None:
        self.max_per_minute = max_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, *, now: float | None = None) -> bool:
        if self.max_per_minute <= 0:
            return True
        now = time.monotonic() if now is None else now
        window = self._hits[key]
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self.max_per_minute:
            return False
        window.append(now)
        return True


class ArenaHubMiddleware(BaseHTTPMiddleware):
    """Auth + request-id + size-limit + rate-limit enforcement."""

    def __init__(self, app, *, settings, rate_limiter: RateLimiter | None = None) -> None:
        super().__init__(app)
        self._settings = settings
        self._limiter = rate_limiter or RateLimiter(settings.rate_limit_per_minute)

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
        request.state.request_id = request_id
        path = request.url.path

        # CORS preflight is handled by CORSMiddleware (outer layer); let it pass.
        if request.method == "OPTIONS":
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        # 1. Authentication (except public paths).
        if path not in PUBLIC_PATHS and not path.startswith("/docs") and not path.startswith("/redoc"):
            provided = self._extract_key(request)
            expected = self._settings.ensure_gateway_api_key()
            if not provided or not _safe_equals(provided, expected):
                response = error_envelope(
                    path,
                    "Missing or invalid ArenaHub API key. Provide it via 'Authorization: Bearer"
                    " <ARENAHUB_API_KEY>', 'X-API-Key' or 'X-Api-Key'.",
                    error_type="authentication_error",
                    code="invalid_api_key",
                    status=401,
                    request_id=request_id,
                )
                response.headers["X-Request-ID"] = request_id
                return response

        # 2. Rate limiting (keyed on client IP; loopback dev users share it).
        client_key = request.client.host if request.client else "local"
        if not self._limiter.check(client_key):
            response = error_envelope(
                path,
                "Rate limit exceeded. Slow down and retry shortly.",
                error_type="rate_limit_error",
                code="rate_limit_exceeded",
                status=429,
                request_id=request_id,
            )
            response.headers["Retry-After"] = "60"
            response.headers["X-Request-ID"] = request_id
            return response

        # 3. Body size limit. Multipart uploads (files) are capped per-file in
        # the upload route; every other body is rejected up-front when it
        # declares (or streams) more than the configured maximum.
        if request.method in POSTISH:
            max_bytes = self._settings.request_max_body_bytes
            is_multipart = request.headers.get("content-type", "").startswith("multipart/form-data")
            if not is_multipart:
                declared = request.headers.get("content-length")
                if declared and int(declared) > max_bytes:
                    response = error_envelope(
                        path,
                        f"Request body too large (limit {max_bytes} bytes).",
                        error_type="invalid_request_error",
                        code="request_too_large",
                        status=413,
                        request_id=request_id,
                    )
                    response.headers["X-Request-ID"] = request_id
                    return response
                if declared is None:
                    # Chunked body without a length: bound it while streaming.
                    oversized = await self._exceeds_limit(request, max_bytes)
                    if oversized:
                        response = error_envelope(
                            path,
                            f"Request body too large (limit {max_bytes} bytes).",
                            error_type="invalid_request_error",
                            code="request_too_large",
                            status=413,
                            request_id=request_id,
                        )
                        response.headers["X-Request-ID"] = request_id
                        return response

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @staticmethod
    def _extract_key(request: Request) -> str | None:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[len("bearer ") :].strip()
            if token:
                return token
        x_key = request.headers.get("x-api-key")
        return x_key.strip() if x_key else None

    @staticmethod
    async def _exceeds_limit(request: Request, max_bytes: int) -> bool:
        """True if a length-less request body streams more than ``max_bytes``.

        Consumes the body and reinstalls a cached receive so downstream code
        can read it normally.
        """
        total = 0
        chunks: list[bytes] = []
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_bytes:
                return True
            chunks.append(chunk)
        body = b"".join(chunks)

        async def _receive() -> Message:
            return {"type": "http.request", "body": body, "more_body": False}

        request._receive = _receive  # type: ignore[attr-defined]
        return False


POSTISH = frozenset({"POST", "PUT", "PATCH"})


def _safe_equals(a: str, b: str) -> bool:
    import hmac

    return hmac.compare_digest(a.encode(), b.encode())
