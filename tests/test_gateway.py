"""Tests for the OpenAI-compatible local gateway: auth, formats, streaming."""

from __future__ import annotations

import json

import httpx
import pytest

from .conftest import TEST_GATEWAY_KEY

AUTH_HEADERS = {"Authorization": f"Bearer {TEST_GATEWAY_KEY}"}
CHAT_URL = "/v1/chat/completions"
MODELS_URL = "/v1/models"


# ---------------------------------------------------------------------------
# Health & gateway authentication
# ---------------------------------------------------------------------------


async def test_health_is_open(gateway_client: httpx.AsyncClient) -> None:
    response = await gateway_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_models_without_key_is_unauthorized(gateway_client: httpx.AsyncClient) -> None:
    response = await gateway_client.get(MODELS_URL)
    assert response.status_code == 401
    body = response.json()
    assert "error" in body
    assert "message" in body["error"]


async def test_models_with_wrong_key_is_unauthorized(gateway_client: httpx.AsyncClient) -> None:
    response = await gateway_client.get(
        MODELS_URL, headers={"Authorization": "Bearer nope-wrong-key"}
    )
    assert response.status_code == 401


async def test_models_with_x_api_key_header(gateway_client: httpx.AsyncClient) -> None:
    response = await gateway_client.get(MODELS_URL, headers={"X-API-Key": TEST_GATEWAY_KEY})
    assert response.status_code == 200


async def test_chat_without_key_is_unauthorized(gateway_client: httpx.AsyncClient) -> None:
    response = await gateway_client.post(
        CHAT_URL,
        json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


async def test_gateway_models_is_openai_shaped(gateway_client: httpx.AsyncClient) -> None:
    response = await gateway_client.get(MODELS_URL, headers=AUTH_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
    # Real Arena models come first; aliases follow.
    assert body["data"][0]["object"] == "model"
    ids = [entry["id"] for entry in body["data"]]
    assert "claude-sonnet-4-6" in ids
    assert "arena/claude" in ids  # aliases are exposed too
    real_models = [entry for entry in body["data"] if entry.get("owned_by") != "arenahub"]
    assert real_models  # concrete models precede aliases


# ---------------------------------------------------------------------------
# /v1/chat/completions (non-streaming)
# ---------------------------------------------------------------------------


async def test_chat_completion_openai_format(gateway_client: httpx.AsyncClient) -> None:
    response = await gateway_client.post(
        CHAT_URL,
        headers=AUTH_HEADERS,
        json={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert body["model"] == "claude-sonnet-4-6"
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hello there"
    assert choice["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 7


async def test_invalid_model_returns_404_openai_error(
    gateway_client: httpx.AsyncClient,
) -> None:
    response = await gateway_client.post(
        CHAT_URL,
        headers=AUTH_HEADERS,
        json={"model": "bad-model", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 404
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "not_found"


async def test_malformed_body_returns_400(gateway_client: httpx.AsyncClient) -> None:
    response = await gateway_client.post(
        CHAT_URL, headers=AUTH_HEADERS, json={"model": "claude-sonnet-4-6"}
    )
    assert response.status_code == 400
    assert "error" in response.json()


async def test_empty_messages_returns_400(gateway_client: httpx.AsyncClient) -> None:
    response = await gateway_client.post(
        CHAT_URL, headers=AUTH_HEADERS, json={"model": "claude-sonnet-4-6", "messages": []}
    )
    assert response.status_code == 400
    assert "error" in response.json()


async def test_locally_invalid_model_id_returns_400(gateway_client: httpx.AsyncClient) -> None:
    response = await gateway_client.post(
        CHAT_URL,
        headers=AUTH_HEADERS,
        json={
            "model": "no spaces allowed",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# /v1/chat/completions (streaming / SSE)
# ---------------------------------------------------------------------------


async def test_chat_streaming_sse_format(gateway_client: httpx.AsyncClient) -> None:
    async with gateway_client.stream(
        "POST",
        CHAT_URL,
        headers=AUTH_HEADERS,
        json={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events: list[dict] = []
        done_seen = False
        async for line in response.aiter_lines():
            if not line:
                continue
            assert line.startswith("data:")
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                done_seen = True
                continue
            events.append(json.loads(payload))

    assert done_seen
    assert len(events) == 3
    for event in events:
        assert event["object"] == "chat.completion.chunk"
        assert event["id"].startswith("chatcmpl-")
        assert event["model"] == "claude-sonnet-4-6"
        assert event["choices"][0]["index"] == 0

    # First chunk carries the assistant role; later chunks carry content.
    assert events[0]["choices"][0]["delta"].get("role") == "assistant"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in events)
    assert text == "Hello there"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


async def test_streaming_with_bad_model_fails_before_stream_starts(
    gateway_client: httpx.AsyncClient,
) -> None:
    async with gateway_client.stream(
        "POST",
        CHAT_URL,
        headers=AUTH_HEADERS,
        json={
            "model": "bad-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    ) as response:
        # Upstream error is surfaced as an HTTP status, not mid-SSE garbage.
        assert response.status_code == 404
        body = await response.aread()
        assert b"error" in body


# ---------------------------------------------------------------------------
# Security: the gateway key must never be forwarded upstream
# ---------------------------------------------------------------------------


async def test_gateway_key_not_forwarded_to_arena(
    gateway_client: httpx.AsyncClient, recorded_requests: list[httpx.Request]
) -> None:
    response = await gateway_client.post(
        CHAT_URL,
        headers=AUTH_HEADERS,
        json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 200
    upstream = recorded_requests[-1]
    assert upstream.headers["Authorization"] == "Bearer test-arena-key"
    assert TEST_GATEWAY_KEY not in upstream.headers.get("authorization", "")
    assert TEST_GATEWAY_KEY.encode() not in upstream.content


# ---------------------------------------------------------------------------
# CORS is restricted to loopback origins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "origin,allowed",
    [
        ("http://localhost:3000", True),
        ("http://127.0.0.1:5173", True),
        ("https://evil.example.com", False),
        ("http://192.168.1.10:3000", False),
    ],
)
async def test_cors_restricted_to_localhost(
    gateway_client: httpx.AsyncClient, origin: str, allowed: bool
) -> None:
    response = await gateway_client.options(
        MODELS_URL,
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )
    allow_origin = response.headers.get("access-control-allow-origin")
    if allowed:
        assert allow_origin == origin
    else:
        assert allow_origin is None
