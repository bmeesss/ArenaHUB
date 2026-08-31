"""HTTP routes for the ArenaHub OpenAI-compatible gateway.

Endpoints
---------
``GET  /health``               - liveness probe (no key required)
``GET  /v1/models``            - model list incl. ArenaHub aliases (key required)
``POST /v1/chat/completions``  - OpenAI-compatible chat, incl. SSE streaming,
                                 function/tool calling and alias resolution

The gateway authenticates local clients with its own key (``ARENAHUB_API_KEY``)
via ``Authorization: Bearer <key>``, ``X-API-Key`` or Anthropic-style
``X-Api-Key``. The Arena API key stays server-side and is never exposed.

Coding-agent clients (VS Code, Claude Code-style) may select the model with
the ``X-Arena-Model`` header (or the ``ARENA_DEFAULT_MODEL`` environment
variable) instead of the request body.
"""

from __future__ import annotations

import json
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .arena_client import ArenaClient
from .config import Settings
from .errors import (
    ArenaAuthError,
    ArenaHubError,
    ArenaModelError,
    ArenaNotSupportedError,
    ArenaRateLimitError,
    ArenaTimeoutError,
    ArenaValidationError,
)
from .model_router import ModelRouter
from .models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    normalize_chunk,
    normalize_completion,
)

router = APIRouter()

# A factory is used so tests can inject clients backed by a mock transport.
ClientFactory = Callable[[], ArenaClient]


def get_client_factory(request: Request) -> ClientFactory:
    factory: ClientFactory | None = getattr(request.app.state, "client_factory", None)
    if factory is not None:
        return factory
    settings: Settings = request.app.state.settings
    return lambda: ArenaClient(settings)


def get_model_router(request: Request) -> ModelRouter:
    model_router: ModelRouter | None = getattr(request.app.state, "model_router", None)
    if model_router is not None:
        return model_router
    # Lazy fallback (e.g. lightweight ASGI mounts without create_app wiring).
    model_router = ModelRouter(request.app.state.settings, client_factory=get_client_factory(request))
    request.app.state.model_router = model_router
    return model_router


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def extract_gateway_key(request: Request) -> str | None:
    """Accept OpenAI-style Bearer, X-API-Key and Anthropic-style X-Api-Key."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[len("bearer ") :].strip()
        if token:
            return token
    x_key = request.headers.get("x-api-key")
    return x_key.strip() if x_key else None


def check_gateway_key(request: Request) -> None:
    """Raise 401 unless the request carries the local gateway key."""
    settings: Settings = request.app.state.settings
    expected = settings.ensure_gateway_api_key()
    provided = extract_gateway_key(request)
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid ArenaHub gateway API key. "
            "Send 'Authorization: Bearer <ARENAHUB_API_KEY>', 'X-API-Key' or 'X-Api-Key'.",
        )


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def openai_error_body(
    message: str, *, error_type: str, code: str | None = None, request_id: str | None = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"message": message, "type": error_type}
    if code:
        error["code"] = code
    body: dict[str, Any] = {"error": error}
    if request_id:
        body["request_id"] = request_id
    return body


def map_arena_error_to_http(exc: ArenaHubError) -> HTTPException:
    """Translate Arena/client errors into OpenAI-style HTTP failures."""
    if isinstance(exc, ArenaValidationError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ArenaNotSupportedError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, ArenaModelError):
        return HTTPException(
            status_code=404,
            detail=f"Model not available through ArenaHub: {exc.message}",
        )
    if isinstance(exc, ArenaAuthError):
        # The upstream key is a server-side secret problem — never leak it.
        return HTTPException(
            status_code=502,
            detail="The gateway could not authenticate with the upstream Arena API. "
            "Check the server's ARENA_API_KEY configuration.",
        )
    if isinstance(exc, ArenaRateLimitError):
        return HTTPException(
            status_code=429,
            detail="Rate limited by the upstream Arena API. Please retry shortly.",
            headers={"Retry-After": exc.retry_after or "30"},
        )
    if isinstance(exc, ArenaTimeoutError):
        return HTTPException(status_code=504, detail="The upstream Arena API timed out.")
    return HTTPException(status_code=502, detail=f"Upstream Arena API error: {exc}")


def select_model(request: Request, body: ChatCompletionRequest) -> str:
    """Effective model: body, then ``X-Arena-Model`` header, then server default."""
    header_model = request.headers.get("x-arena-model", "").strip()
    settings: Settings = request.app.state.settings
    return body.model or header_model or settings.arena_default_model or ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — intentionally reveals no configuration details."""
    return {"status": "ok", "service": "arenahub-gateway"}


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    check_gateway_key(request)
    model_router = get_model_router(request)
    try:
        models = await model_router.list_all()
    except ArenaHubError as exc:
        raise map_arena_error_to_http(exc) from exc
    return JSONResponse(models.model_dump(exclude_none=True))


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest) -> Response:
    check_gateway_key(request)

    requested_model = select_model(request, body)
    if not requested_model:
        raise HTTPException(
            status_code=400,
            detail="No model specified (set 'model', the X-Arena-Model header, or ARENA_DEFAULT_MODEL).",
        )

    model_router = get_model_router(request)
    try:
        resolved_model = await model_router.resolve(requested_model)
    except ArenaHubError as exc:
        raise map_arena_error_to_http(exc) from exc

    # Send the resolved Arena id upstream; echo the requested id to clients.
    upstream_request = body.model_copy(update={"model": resolved_model})
    factory = get_client_factory(request)

    if body.stream:
        client = factory()
        first_chunk: dict[str, Any] | None = None
        agen = client.stream_chat_completion(upstream_request)
        try:
            # Prime the stream so upstream errors surface as proper HTTP
            # status codes instead of dying mid-stream after 200 OK.
            try:
                first_chunk = await agen.__anext__()
            except StopAsyncIteration:
                first_chunk = None
        except ArenaHubError as exc:
            await client.aclose()
            raise map_arena_error_to_http(exc) from exc

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        return StreamingResponse(
            _stream_payload(client, agen, requested_model, first_chunk, completion_id, created),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    client = factory()
    try:
        try:
            raw = await client.chat_completion(upstream_request)
        except ArenaHubError as exc:
            raise map_arena_error_to_http(exc) from exc
        response = normalize_completion(raw, requested_model)
        # Echo the model the client asked for (OpenAI behavior), even though
        # we sent the resolved Arena id upstream.
        response.model = requested_model
        # OpenAI omits unset optional fields (e.g. message.name).
        return JSONResponse(response.model_dump(exclude_none=True))
    finally:
        await client.aclose()


async def _stream_payload(
    client: ArenaClient,
    agen: AsyncIterator[dict[str, Any]],
    requested_model: str,
    first_chunk: dict[str, Any] | None,
    completion_id: str,
    created: int,
) -> AsyncIterator[bytes]:
    """Serialize normalized chunks as OpenAI-style SSE events."""
    done = b"data: [DONE]\n\n"
    role_sent = False
    try:
        pending = first_chunk
        while True:
            if pending is None:
                try:
                    raw_chunk = await agen.__anext__()
                except StopAsyncIteration:
                    break
            else:
                raw_chunk = pending
                pending = None

            chunk: ChatCompletionChunk = normalize_chunk(
                raw_chunk,
                requested_model,
                fallback_id=completion_id,
                fallback_created=created,
            )
            chunk.model = requested_model  # echo requested alias/id, not upstream id
            # Guarantee the first delta carries the assistant role.
            if not role_sent and chunk.choices:
                delta = chunk.choices[0].delta
                if "role" not in delta:
                    chunk.choices[0].delta = {"role": "assistant", **delta}
                role_sent = True

            yield f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n".encode("utf-8")

        if not role_sent:
            # Empty upstream stream — still emit a well-formed first chunk.
            empty = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=requested_model,
                choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            )
            yield f"data: {json.dumps(empty.model_dump(exclude_none=True))}\n\n".encode("utf-8")
        yield done
    except ArenaHubError as exc:
        http_exc = map_arena_error_to_http(exc)
        error_payload = openai_error_body(
            str(http_exc.detail),
            error_type="upstream_error",
            code=str(http_exc.status_code),
        )
        yield f"data: {json.dumps(error_payload)}\n\n".encode("utf-8")
        yield done
    finally:
        await client.aclose()
