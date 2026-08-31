"""Tests for the Arena API client: listing, auth, chat, streaming, errors."""

from __future__ import annotations

import httpx
import pytest

from backend.arena_client import ArenaClient
from backend.config import Settings
from backend.errors import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaConfigError,
    ArenaConnectionError,
    ArenaModelError,
    ArenaRateLimitError,
    ArenaTimeoutError,
    ArenaValidationError,
)
from backend.models import ChatCompletionRequest, ChatMessage

from .conftest import SSE_CHUNKS, TEST_ARENA_KEY


def _chat_request(model: str = "claude-sonnet-4-6", *, stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role="user", content="Hello")],
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------


async def test_list_models_returns_dynamic_models(arena_client: ArenaClient) -> None:
    models = await arena_client.list_models()
    assert models.object == "list"
    ids = [m.id for m in models.data]
    assert "claude-sonnet-4-6" in ids
    assert "gpt-4o" in ids
    assert all(m.object == "model" for m in models.data)
    assert models.data[0].owned_by == "anthropic"
    await arena_client.aclose()


async def test_list_models_sends_bearer_token(
    arena_client: ArenaClient, recorded_requests: list[httpx.Request]
) -> None:
    await arena_client.list_models()
    assert len(recorded_requests) == 1
    request = recorded_requests[0]
    assert request.method == "GET"
    assert request.url.path == "/v1/models"
    assert request.headers["Authorization"] == f"Bearer {TEST_ARENA_KEY}"
    await arena_client.aclose()


# ---------------------------------------------------------------------------
# Authentication / configuration
# ---------------------------------------------------------------------------


async def test_missing_api_key_raises_config_error() -> None:
    client = ArenaClient(Settings(arena_api_key=None))
    with pytest.raises(ArenaConfigError, match="ARENA_API_KEY"):
        await client.list_models()


async def test_upstream_rejects_bad_key(recorded_requests: list[httpx.Request]) -> None:
    def bad_auth(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    client = ArenaClient(
        Settings(arena_api_key="wrong-key"), transport=httpx.MockTransport(bad_auth)
    )
    with pytest.raises(ArenaAuthError) as exc:
        await client.list_models()
    assert exc.value.status_code == 401
    await client.aclose()


# ---------------------------------------------------------------------------
# Successful chat completion
# ---------------------------------------------------------------------------


async def test_chat_completion_success(arena_client: ArenaClient) -> None:
    data = await arena_client.chat_completion(_chat_request())
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello there"
    assert data["usage"]["total_tokens"] == 7
    await arena_client.aclose()


async def test_chat_completion_posts_payload(
    arena_client: ArenaClient, recorded_requests: list[httpx.Request]
) -> None:
    import json

    await arena_client.chat_completion(_chat_request("gpt-4o"))
    request = recorded_requests[-1]
    assert request.url.path == "/v1/chat/completions"
    payload = json.loads(request.content)
    assert payload["model"] == "gpt-4o"
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]
    assert payload["stream"] is False
    await arena_client.aclose()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_streaming_yields_chunks_and_assembles_text(arena_client: ArenaClient) -> None:
    chunks = []
    async for chunk in arena_client.stream_chat_completion(_chat_request(stream=True)):
        chunks.append(chunk)

    assert len(chunks) == len(SSE_CHUNKS)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks
    )
    assert text == "Hello there"
    # [DONE] must be consumed, not yielded
    assert all("DONE" not in str(c) for c in chunks)
    await arena_client.aclose()


async def test_streaming_sends_stream_true(
    arena_client: ArenaClient, recorded_requests: list[httpx.Request]
) -> None:
    import json

    async for _ in arena_client.stream_chat_completion(_chat_request(stream=True)):
        pass
    payload = json.loads(recorded_requests[-1].content)
    assert payload["stream"] is True
    await arena_client.aclose()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


async def test_invalid_model_raises_model_error(arena_client: ArenaClient) -> None:
    with pytest.raises(ArenaModelError) as exc:
        await arena_client.chat_completion(_chat_request("bad-model"))
    assert exc.value.status_code == 404
    assert "bad-model" in str(exc.value) or "model" in str(exc.value).lower()
    await arena_client.aclose()


async def test_forbidden_model_raises_auth_error(arena_client: ArenaClient) -> None:
    with pytest.raises(ArenaAuthError) as exc:
        await arena_client.chat_completion(_chat_request("forbidden-model"))
    assert exc.value.status_code == 403
    await arena_client.aclose()


async def test_rate_limit_raises_rate_limit_error(arena_client: ArenaClient) -> None:
    with pytest.raises(ArenaRateLimitError) as exc:
        await arena_client.chat_completion(_chat_request("rate-limited-model"))
    assert exc.value.status_code == 429
    assert exc.value.retry_after == "42"
    await arena_client.aclose()


async def test_server_error_raises_api_error(arena_client: ArenaClient) -> None:
    with pytest.raises(ArenaAPIError) as exc:
        await arena_client.chat_completion(_chat_request("boom-model"))
    assert exc.value.status_code == 500
    await arena_client.aclose()


async def test_timeout_maps_to_timeout_error(settings: Settings) -> None:
    def slow(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never returns
        raise httpx.ReadTimeout("read timed out")

    client = ArenaClient(settings, transport=httpx.MockTransport(slow))
    with pytest.raises(ArenaTimeoutError):
        await client.chat_completion(_chat_request())
    await client.aclose()


async def test_connection_failure_maps_to_connection_error(settings: Settings) -> None:
    def broken(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never returns
        raise httpx.ConnectError("connection refused")

    client = ArenaClient(settings, transport=httpx.MockTransport(broken))
    with pytest.raises(ArenaConnectionError):
        await client.list_models()
    await client.aclose()


# ---------------------------------------------------------------------------
# Local validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_model",
    ["", "   ", "model with spaces", "model\tinjected", "x" * 250, "../../etc/passwd"],
)
async def test_invalid_model_ids_rejected_locally(
    arena_client: ArenaClient, bad_model: str, recorded_requests: list[httpx.Request]
) -> None:
    with pytest.raises(ArenaValidationError):
        await arena_client.chat_completion(_chat_request(bad_model))
    # Nothing was sent upstream
    assert recorded_requests == []
    await arena_client.aclose()
