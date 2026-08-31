"""ArenaHub platform entry point.

Run with::

    python -m backend.main
    arenahub serve            # convenience wrapper (same server)

The server binds to 127.0.0.1 by default and requires an ArenaHub gateway key
for every non-health endpoint. It is intentionally NOT exposed publicly; see
README "Deployment" for HTTPS / reverse-proxy / PostgreSQL guidance.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .anthropic import anthropic_error_body
from .config import Settings, load_settings
from .errors import ArenaHubError, ArenaNotSupportedError, ArenaValidationError
from .db import ConversationRepository, build_repository
from .middleware import ArenaHubMiddleware, RateLimiter
from .model_router import ModelRouter
from .routes import (
    map_arena_error_to_http,
    openai_error_body,
    router as openai_router,
)
from .routes_anthropic import router as anthropic_router
from .routes_api import router as api_router

# Only loopback browser origins may call the gateway from front-end code
# unless extra origins are explicitly configured (ARENAHUB_ALLOW_ORIGINS).
_LOCALHOST_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


def create_app(
    settings: Settings | None = None,
    client_factory: Callable[[], Any] | None = None,
    *,
    repository: ConversationRepository | None = None,
    model_router: ModelRouter | None = None,
) -> FastAPI:
    """Build the FastAPI application (used by both the server and tests)."""
    settings = settings or load_settings()

    repository = repository if repository is not None else build_repository(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Create tables/connections on startup.
        if hasattr(repository, "initialize"):
            await repository.initialize()
        yield

    app = FastAPI(
        lifespan=lifespan,
        title="ArenaHub",
        version=__version__,
        summary="Unified AI gateway over the official Arena API "
        "(OpenAI, Anthropic, web and coding-agent compatible).",
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )
    app.state.settings = settings
    app.state.client_factory = client_factory
    app.state.repository = repository if repository is not None else build_repository(settings)
    app.state.model_router = (
        model_router
        if model_router is not None
        else ModelRouter(settings, client_factory=client_factory)
    )

    # CORS outermost: preflight (OPTIONS) must pass without authentication.
    cors_kwargs: dict[str, Any] = {
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Api-Key",
            "X-Arena-Model",
            "X-Request-ID",
            "Anthropic-Version",
        ],
    }
    if settings.allow_origins:
        cors_kwargs["allow_origins"] = list(settings.allow_origins)
    else:
        cors_kwargs["allow_origin_regex"] = _LOCALHOST_ORIGIN_REGEX
    app.add_middleware(CORSMiddleware, **cors_kwargs)

    # Auth / request-id / rate-limit / size-limit (inside CORS).
    app.add_middleware(
        ArenaHubMiddleware,
        settings=settings,
        rate_limiter=RateLimiter(settings.rate_limit_per_minute),
    )

    app.include_router(openai_router)
    app.include_router(anthropic_router)
    app.include_router(api_router)

    # ------------------------------------------------------------------
    # Error handlers — surface-appropriate error envelopes.
    # ------------------------------------------------------------------

    def _is_anthropic(request: Request) -> bool:
        return request.url.path.startswith("/v1/messages")

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        status = exc.status_code
        request_id = getattr(request.state, "request_id", None)

        if _is_anthropic(request):
            if status == 401:
                err_type = "authentication_error"
            elif status == 404:
                err_type = "not_found_error"
            elif status == 429:
                err_type = "rate_limit_error"
            elif status == 413:
                err_type = "invalid_request_error"
            elif 400 <= status < 500:
                err_type = "invalid_request_error"
            else:
                err_type = "api_error"
            content: dict[str, Any] = anthropic_error_body(message, error_type=err_type)
        else:
            if status == 401:
                error_type, code = "authentication_error", "invalid_api_key"
            elif status == 404:
                error_type, code = "not_found_error", "not_found"
            elif status == 429:
                error_type, code = "rate_limit_error", "rate_limit_exceeded"
            elif status == 413:
                error_type, code = "invalid_request_error", "request_too_large"
            elif 400 <= status < 500:
                error_type, code = "invalid_request_error", "bad_request"
            else:
                error_type, code = "api_error", "upstream_error"
            content = openai_error_body(
                str(message), error_type=error_type, code=code, request_id=request_id
            )
            if not request.url.path.startswith("/v1/"):
                content = {"error": {"message": str(message), "type": error_type, "code": code}}
                if request_id:
                    content["request_id"] = request_id

        headers = dict(getattr(exc, "headers", None) or {})
        if request_id:
            headers["X-Request-ID"] = request_id
        return JSONResponse(status_code=status, content=content, headers=headers)

    @app.exception_handler(ArenaHubError)
    async def _arena_error_handler(request: Request, exc: ArenaHubError) -> JSONResponse:
        """Map structured ArenaHub errors (raised anywhere) to HTTP responses."""
        request_id = getattr(request.state, "request_id", None)
        if request.url.path.startswith("/v1/messages"):
            if isinstance(exc, ArenaValidationError):
                status, err_type = 400, "invalid_request_error"
            elif isinstance(exc, ArenaNotSupportedError):
                status, err_type = 400, "invalid_request_error"
            else:
                http_exc = map_arena_error_to_http(exc)
                status, err_type = http_exc.status_code, "api_error"
            content = anthropic_error_body(str(exc), error_type=err_type)
        else:
            http_exc = map_arena_error_to_http(exc)
            status = http_exc.status_code
            message = str(http_exc.detail)
            if request.url.path.startswith("/v1/"):
                content = openai_error_body(
                    message,
                    error_type="invalid_request_error" if 400 <= status < 500 else "api_error",
                    request_id=request_id,
                )
            else:
                content = {"error": {"message": message, "type": "arena_error"}}
                if request_id:
                    content["request_id"] = request_id
        headers = {"X-Request-ID": request_id} if request_id else None
        return JSONResponse(status_code=status, content=content, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        messages = [
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        ]
        detail = "Invalid request: " + "; ".join(messages)
        request_id = getattr(request.state, "request_id", None)
        if _is_anthropic(request):
            content: dict[str, Any] = anthropic_error_body(detail)
        else:
            content = openai_error_body(
                detail, error_type="invalid_request_error", code="invalid_request",
                request_id=request_id,
            )
            if not request.url.path.startswith("/v1/"):
                content = {"error": {"message": detail, "type": "invalid_request_error"}}
                if request_id:
                    content["request_id"] = request_id
        headers = {"X-Request-ID": request_id} if request_id else None
        return JSONResponse(status_code=400, content=content, headers=headers)

    return app


# Module-level app for `uvicorn backend.main:app`.
app = create_app()


async def _initialize(app: FastAPI) -> None:
    repository = app.state.repository
    if hasattr(repository, "initialize"):
        await repository.initialize()


def main() -> None:
    """Start the gateway with uvicorn (loopback by default)."""
    settings = load_settings()
    gateway_key = settings.ensure_gateway_api_key()
    key_was_generated = not os.environ.get("ARENAHUB_API_KEY", "").strip()

    application = create_app(settings)

    print()
    print("=" * 68)
    print("  ArenaHub gateway")
    print(f"  Listening on   http://{settings.gateway_host}:{settings.gateway_port}")
    print(f"  Upstream       {settings.arena_base_url}")
    print(f"  Arena API key  {settings.masked_arena_key()}")
    print(f"  Database       {settings.db_path}")
    if key_was_generated:
        print()
        print("  No ARENAHUB_API_KEY set — generated an ephemeral gateway key:")
        print(f"    {gateway_key}")
        print("  Clients must send it as: Authorization: Bearer <key>")
        print("  Set ARENAHUB_API_KEY in .env for a stable key.")
    else:
        print("  Gateway key    configured (set ARENAHUB_API_KEY to rotate)")
    print("=" * 68)
    print()

    uvicorn.run(
        application,
        host=settings.gateway_host,
        port=settings.gateway_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
