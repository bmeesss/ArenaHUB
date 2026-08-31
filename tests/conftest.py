"""Shared pytest fixtures.

Environment variables are set BEFORE any application import so that config
loading never picks up a real ``.env`` or a real API key from the host.
"""

from __future__ import annotations

import json
import os

os.environ["ARENA_API_KEY"] = "test-arena-key"
os.environ["ARENA_BASE_URL"] = "https://api.preview.arena.ai"
os.environ["ARENAHUB_API_KEY"] = "test-gateway-key"
os.environ["ARENAHUB_HOST"] = "127.0.0.1"
os.environ["ARENAHUB_PORT"] = "8000"
os.environ["ARENAHUB_RATE_LIMIT_PER_MINUTE"] = "100000"
os.environ["ARENAHUB_DB_PATH"] = ":memory:"

from collections.abc import Iterator  # noqa: E402
from pathlib import Path  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

from backend.arena_client import ArenaClient  # noqa: E402
from backend.config import Settings  # noqa: E402
from backend.db import SqliteConversationRepository  # noqa: E402
from backend.main import create_app  # noqa: E402

TEST_ARENA_KEY = "test-arena-key"
TEST_GATEWAY_KEY = "test-gateway-key"

MODELS_PAYLOAD = {
    "object": "list",
    "data": [
        {"id": "claude-opus-4-0", "object": "model", "created": 1710000000, "owned_by": "anthropic"},
        {"id": "claude-sonnet-4-6", "object": "model", "created": 1710000000, "owned_by": "anthropic"},
        {"id": "claude-haiku-4-5", "object": "model", "created": 1710000000, "owned_by": "anthropic"},
        {"id": "gpt-4o", "object": "model", "created": 1710000000, "owned_by": "openai"},
        {"id": "gemini-2.0-pro", "object": "model", "created": 1710000000, "owned_by": "google"},
    ],
}

SSE_CHUNKS = [
    {"id": "chatcmpl-upstream", "object": "chat.completion.chunk", "created": 1710000001,
     "model": "claude-sonnet-4-6",
     "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
    {"id": "chatcmpl-upstream", "object": "chat.completion.chunk", "created": 1710000001,
     "model": "claude-sonnet-4-6",
     "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
    {"id": "chatcmpl-upstream", "object": "chat.completion.chunk", "created": 1710000001,
     "model": "claude-sonnet-4-6",
     "choices": [{"index": 0, "delta": {"content": " there"}, "finish_reason": "stop"}]},
]

TOOL_SSE_CHUNKS = [
    {"id": "chatcmpl-tool", "object": "chat.completion.chunk", "created": 1710000002,
     "model": "claude-sonnet-4-6",
     "choices": [{"index": 0, "delta": {"role": "assistant",
        "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": ""}}]},
        "finish_reason": None}]},
    {"id": "chatcmpl-tool", "object": "chat.completion.chunk", "created": 1710000002,
     "model": "claude-sonnet-4-6",
     "choices": [{"index": 0, "delta": {
        "tool_calls": [{"index": 0, "function": {"arguments": '{"city": "Paris"}'}}]},
        "finish_reason": None}]},
    {"id": "chatcmpl-tool", "object": "chat.completion.chunk", "created": 1710000002,
     "model": "claude-sonnet-4-6",
     "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
]


def make_arena_handler(recorded: list[httpx.Request]):
    """Build an httpx mock transport emulating the official Arena API."""

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)

        # The upstream must only ever see the Arena key — never a gateway key.
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {TEST_ARENA_KEY}":
            return httpx.Response(
                401, json={"error": {"message": "invalid api key", "type": "authentication_error"}}
            )

        if request.url.path == "/v1/models":
            return httpx.Response(200, json=MODELS_PAYLOAD)

        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content.decode("utf-8"))
            model = body.get("model", "")

            if model == "bad-model":
                return httpx.Response(
                    404,
                    json={"error": {"message": f"model '{model}' not found",
                                    "type": "invalid_request_error"}},
                )
            if model == "forbidden-model":
                return httpx.Response(
                    403, json={"error": {"message": "not allowed to use this model"}}
                )
            if model == "rate-limited-model":
                return httpx.Response(
                    429,
                    headers={"retry-after": "42"},
                    json={"error": {"message": "rate limit exceeded"}},
                )
            if model == "boom-model":
                return httpx.Response(500, json={"error": {"message": "internal server error"}})

            # Tool-calling responses (used by Anthropic/coding-agent tests).
            if model == "tool-model":
                return httpx.Response(200, json={
                    "id": "chatcmpl-tool", "object": "chat.completion", "created": 1710000002,
                    "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                        "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "get_weather",
                                         "arguments": '{"city": "Paris"}'}}]},
                        "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}})

            if body.get("stream"):
                chunks = TOOL_SSE_CHUNKS if model == "tool-stream-model" else SSE_CHUNKS
                lines = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) \
                    + "data: [DONE]\n\n"
                return httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, text=lines
                )

            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-upstream",
                    "object": "chat.completion",
                    "created": 1710000001,
                    "model": model,
                    "choices": [
                        {"index": 0,
                         "message": {"role": "assistant", "content": "Hello there"},
                         "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                },
            )

        return httpx.Response(404, json={"error": {"message": "unknown endpoint"}})

    return handler


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        arena_api_key=TEST_ARENA_KEY,
        arena_base_url="https://api.preview.arena.ai",
        gateway_api_key=TEST_GATEWAY_KEY,
        model_aliases={"arena/my-sonnet": "claude-sonnet-4-6"},
        database_path=tmp_path / "test.db",
    )


@pytest.fixture
def recorded_requests() -> list[httpx.Request]:
    return []


@pytest.fixture
def mock_transport(recorded_requests: list[httpx.Request]) -> httpx.MockTransport:
    return httpx.MockTransport(make_arena_handler(recorded_requests))


@pytest.fixture
def arena_client(settings: Settings, mock_transport: httpx.MockTransport) -> ArenaClient:
    return ArenaClient(settings, transport=mock_transport)


@pytest.fixture
async def gateway_client(
    settings: Settings, mock_transport: httpx.MockTransport
) -> Iterator[httpx.AsyncClient]:
    """HTTP client wired to the gateway ASGI app with a mocked Arena upstream."""

    def client_factory() -> ArenaClient:
        return ArenaClient(settings, transport=mock_transport)

    repository = SqliteConversationRepository(settings.db_path)
    app = create_app(settings, client_factory=client_factory, repository=repository)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.local") as client:
        yield client
