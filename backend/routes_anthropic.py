"""Anthropic-compatible Messages endpoint: ``POST /v1/messages``.

Accepts the Anthropic Messages wire format (system prompts, user/assistant
messages, tool use, tool results, streaming), translates it to the OpenAI
shape the official Arena API speaks, and translates the response (and SSE
events) back to Anthropic format.

Auth: Anthropic clients send ``x-api-key`` (with optional
``anthropic-version`` header). The ArenaHub gateway key is expected; the
Arena API key never leaves the server.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from . import anthropic as anthropic_compat
from .anthropic import openai_stream_to_anthropic_events
from .errors import (
    ArenaAuthError,
    ArenaHubError,
    ArenaModelError,
    ArenaNotSupportedError,
    ArenaRateLimitError,
    ArenaTimeoutError,
    ArenaValidationError,
)
from .models import ChatCompletionRequest
from .routes import check_gateway_key, get_client_factory, get_model_router

router = APIRouter()


def _anthropic_http_error(exc: ArenaHubError) -> HTTPException:
    """Map ArenaHub errors to Anthropic-style HTTP status + error envelope."""
    if isinstance(exc, ArenaValidationError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ArenaNotSupportedError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ArenaModelError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ArenaAuthError):
        return HTTPException(
            status_code=502,
            detail="The gateway could not authenticate with the upstream Arena API.",
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


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "") or f"req_{uuid.uuid4().hex[:16]}"


async def _resolve(request: Request, requested_model: str) -> str:
    model_router = get_model_router(request)
    try:
        return await model_router.resolve(requested_model)
    except ArenaHubError as exc:
        raise _anthropic_http_error(exc) from exc


@router.post("/v1/messages")
async def messages(request: Request) -> Response:
    check_gateway_key(request)

    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001 - malformed JSON
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    requested_model = payload.get("model") or request.headers.get("x-arena-model", "")
    if not requested_model:
        raise HTTPException(status_code=400, detail="'model' is required.")

    resolved_model = await _resolve(request, requested_model)

    try:
        openai_request: ChatCompletionRequest = anthropic_compat.build_openai_request(
            payload, resolved_model
        )
    except (ArenaValidationError, ArenaNotSupportedError) as exc:
        raise _anthropic_http_error(exc) from exc

    factory = get_client_factory(request)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    if openai_request.stream:
        client = factory()
        first_chunk: dict[str, Any] | None = None
        agen = client.stream_chat_completion(openai_request)
        try:
            try:
                first_chunk = await agen.__anext__()
            except StopAsyncIteration:
                first_chunk = None
        except ArenaHubError as exc:
            await client.aclose()
            raise _anthropic_http_error(exc) from exc

        message_id = f"msg_{uuid.uuid4().hex[:24]}"

        async def _event_stream() -> AsyncIterator[bytes]:
            async def _chunks() -> AsyncIterator[dict[str, Any]]:
                if first_chunk is not None:
                    yield first_chunk
                async for chunk in agen:
                    yield chunk

            try:
                async for event in openai_stream_to_anthropic_events(
                    _chunks(), requested_model=requested_model, message_id=message_id
                ):
                    yield event
            except ArenaHubError as exc:
                http_exc = _anthropic_http_error(exc)
                yield anthropic_compat.sse(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": str(http_exc.detail),
                        },
                    },
                )
            finally:
                await client.aclose()

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------
    client = factory()
    try:
        try:
            raw = await client.chat_completion(openai_request)
        except ArenaHubError as exc:
            raise _anthropic_http_error(exc) from exc
    finally:
        await client.aclose()

    result = anthropic_compat.openai_to_anthropic_response(raw, requested_model)
    return JSONResponse(result)
