"""Tests for the Anthropic-compatible /v1/messages endpoint."""

from __future__ import annotations

import json

import httpx
import pytest

from .conftest import TEST_GATEWAY_KEY

MESSAGES_URL = "/v1/messages"
H = {"Authorization": f"Bearer {TEST_GATEWAY_KEY}", "anthropic-version": "2023-06-01"}


async def test_anthropic_requires_key(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.post(MESSAGES_URL, json={"model": "x", "messages": []})
    assert r.status_code == 401
    body = r.json()
    # Anthropic error envelope.
    assert body["type"] == "error"
    assert "message" in body["error"]


async def test_anthropic_accepts_x_api_key(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.post(
        MESSAGES_URL,
        headers={"x-api-key": TEST_GATEWAY_KEY},
        json={"model": "claude-sonnet-4-6", "max_tokens": 100,
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200


async def test_anthropic_non_stream_shape(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.post(
        MESSAGES_URL, headers=H,
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "system": "You are concise.",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["model"] == "claude-sonnet-4-6"
    assert data["content"][0]["type"] == "text"
    assert data["stop_reason"] == "end_turn"
    assert set(data["usage"]) == {"input_tokens", "output_tokens"}
    assert data["id"].startswith("msg_")


async def test_anthropic_requires_max_tokens(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.post(
        MESSAGES_URL, headers=H,
        json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 400
    assert r.json()["type"] == "error"
    assert "max_tokens" in r.json()["error"]["message"]


async def test_anthropic_forwards_system_prompt(
    gateway_client: httpx.AsyncClient, recorded_requests: list[httpx.Request]
) -> None:
    r = await gateway_client.post(
        MESSAGES_URL, headers=H,
        json={"model": "claude-sonnet-4-6", "max_tokens": 100,
              "system": "Be terse.",
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    upstream = json.loads(recorded_requests[-1].content)
    assert upstream["messages"][0] == {"role": "system", "content": "Be terse."}
    assert upstream["max_tokens"] == 100
    assert "temperature" not in upstream  # unset fields are omitted


async def test_anthropic_tool_call_non_stream(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.post(
        MESSAGES_URL, headers=H,
        json={
            "model": "tool-model", "max_tokens": 200,
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get the weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
    assert tool_blocks, data
    block = tool_blocks[0]
    assert block["name"] == "get_weather"
    assert block["input"] == {"city": "Paris"}
    assert block["id"].startswith(("toolu_", "call_"))
    assert data["stop_reason"] == "tool_use"


async def test_anthropic_tool_result_roundtrip(
    gateway_client: httpx.AsyncClient, recorded_requests: list[httpx.Request]
) -> None:
    """tool_result user blocks translate to OpenAI tool messages."""
    r = await gateway_client.post(
        MESSAGES_URL, headers=H,
        json={
            "model": "claude-sonnet-4-6", "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                     "input": {"city": "Paris"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1",
                     "content": [{"type": "text", "text": "18C and sunny"}]}]},
            ],
        },
    )
    assert r.status_code == 200, r.text
    upstream = json.loads(recorded_requests[-1].content)
    roles = [m["role"] for m in upstream["messages"]]
    assert "tool" in roles
    tool_msg = [m for m in upstream["messages"] if m["role"] == "tool"][0]
    assert tool_msg["tool_call_id"] == "toolu_1"
    assert "18C" in tool_msg["content"]
    assistant_msg = [m for m in upstream["messages"] if m["role"] == "assistant"][0]
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"


async def test_anthropic_streaming(gateway_client: httpx.AsyncClient) -> None:
    async with gateway_client.stream(
        "POST", MESSAGES_URL, headers=H,
        json={"model": "claude-sonnet-4-6", "max_tokens": 100, "stream": True,
              "messages": [{"role": "user", "content": "hi"}]},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = (await r.aread()).decode()

    events = [line for line in body.splitlines() if line.startswith("event:")]
    event_types = [e.split(":", 1)[1].strip() for e in events]
    assert "message_start" in event_types
    assert "content_block_start" in event_types
    assert "content_block_delta" in event_types
    assert "content_block_stop" in event_types
    assert "message_delta" in event_types
    assert event_types[-1] == "message_stop"
    # Text deltas carry text_delta.
    assert "text_delta" in body
    # The streamed text.
    assert "Hello" in body and "there" in body


async def test_anthropic_streaming_tool_calls(gateway_client: httpx.AsyncClient) -> None:
    async with gateway_client.stream(
        "POST", MESSAGES_URL, headers=H,
        json={"model": "tool-stream-model", "max_tokens": 200, "stream": True,
              "messages": [{"role": "user", "content": "weather?"}],
              "tools": [{"name": "get_weather", "description": "w",
                         "input_schema": {"type": "object", "properties": {}}}]},
    ) as r:
        assert r.status_code == 200
        body = (await r.aread()).decode()

    assert "tool_use" in body
    assert "input_json_delta" in body
    # finish mapping -> tool_use stop reason in message_delta
    assert '"stop_reason": "tool_use"' in body


async def test_anthropic_alias_resolution(
    gateway_client: httpx.AsyncClient, recorded_requests: list[httpx.Request]
) -> None:
    r = await gateway_client.post(
        MESSAGES_URL, headers=H,
        json={"model": "arena/claude-sonnet", "max_tokens": 50,
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    # Upstream receives the resolved Arena id.
    upstream = json.loads(recorded_requests[-1].content)
    assert upstream["model"] == "claude-sonnet-4-6"
    # Client sees the requested alias echoed back.
    assert r.json()["model"] == "arena/claude-sonnet"


async def test_anthropic_invalid_model_404(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.post(
        MESSAGES_URL, headers=H,
        json={"model": "bad-model", "max_tokens": 10,
              "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 404
    assert r.json()["type"] == "error"


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},  # no model
        {"model": "x", "messages": [], "max_tokens": 10},  # empty messages
        {"model": "x", "max_tokens": 10,
         "messages": [{"role": "weird", "content": "hi"}]},  # bad role
    ],
)
async def test_anthropic_invalid_requests(
    gateway_client: httpx.AsyncClient, payload: dict
) -> None:
    r = await gateway_client.post(MESSAGES_URL, headers=H, json=payload)
    assert r.status_code in (400, 422)
    assert r.json()["type"] == "error"
