"""Tests for the web/Android conversation API and model catalogue."""

from __future__ import annotations

import httpx

from .conftest import TEST_GATEWAY_KEY

H = {"Authorization": f"Bearer {TEST_GATEWAY_KEY}"}


async def test_api_requires_auth(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.get("/api/conversations")
    assert r.status_code == 401


async def test_model_catalogue(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.get("/api/models", headers=H)
    assert r.status_code == 200
    data = r.json()
    ids = [m["id"] for m in data["models"]]
    assert "claude-sonnet-4-6" in ids
    assert "arena/claude" in ids
    alias = [m for m in data["models"] if m["id"] == "arena/my-sonnet"][0]
    assert alias["is_alias"] is True
    assert alias["alias_for"] == "claude-sonnet-4-6"


async def test_conversation_crud(gateway_client: httpx.AsyncClient) -> None:
    # Create
    r = await gateway_client.post(
        "/api/conversations", headers=H, json={"title": "Test", "model": "arena/claude-sonnet"}
    )
    assert r.status_code == 201, r.text
    conv = r.json()
    cid = conv["id"]
    assert conv["title"] == "Test"
    assert conv["model"] == "arena/claude-sonnet"

    # List
    r = await gateway_client.get("/api/conversations", headers=H)
    assert r.status_code == 200
    assert cid in [c["id"] for c in r.json()["conversations"]]

    # Get (empty)
    r = await gateway_client.get(f"/api/conversations/{cid}", headers=H)
    assert r.status_code == 200
    assert r.json()["messages"] == []

    # Rename
    r = await gateway_client.patch(f"/api/conversations/{cid}", headers=H, json={"title": "New"})
    assert r.status_code == 200
    assert r.json()["title"] == "New"

    # Delete
    r = await gateway_client.delete(f"/api/conversations/{cid}", headers=H)
    assert r.status_code == 204
    r = await gateway_client.get(f"/api/conversations/{cid}", headers=H)
    assert r.status_code == 404


async def test_get_missing_conversation_404(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.get("/api/conversations/conv_nope", headers=H)
    assert r.status_code == 404


async def test_post_message_streams_and_persists(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.post(
        "/api/conversations", headers=H, json={"model": "claude-sonnet-4-6"}
    )
    cid = r.json()["id"]

    async with gateway_client.stream(
        "POST", f"/api/conversations/{cid}/messages", headers=H,
        json={"content": "Hello", "stream": True},
    ) as resp:
        assert resp.status_code == 200
        body = (await resp.aread()).decode()

    assert "event: user_message" in body
    assert "event: delta" in body
    assert "event: done" in body
    assert "Hello" in body  # streamed content from mock upstream

    # Persisted: user + assistant
    r = await gateway_client.get(f"/api/conversations/{cid}", headers=H)
    messages = r.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Hello"


async def test_post_message_non_stream(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.post(
        "/api/conversations", headers=H, json={"model": "claude-sonnet-4-6"}
    )
    cid = r.json()["id"]
    r = await gateway_client.post(
        f"/api/conversations/{cid}/messages", headers=H,
        json={"content": "hi", "stream": False},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["user_message"]["role"] == "user"
    assert data["assistant_message"]["role"] == "assistant"


async def test_regenerate_replaces_assistant(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.post(
        "/api/conversations", headers=H, json={"model": "claude-sonnet-4-6"}
    )
    cid = r.json()["id"]
    async with gateway_client.stream(
        "POST", f"/api/conversations/{cid}/messages", headers=H,
        json={"content": "q", "stream": True},
    ) as resp:
        await resp.aread()

    r = await gateway_client.get(f"/api/conversations/{cid}", headers=H)
    assert len(r.json()["messages"]) == 2
    first_assistant = r.json()["messages"][1]["id"]

    async with gateway_client.stream(
        "POST", f"/api/conversations/{cid}/regenerate", headers=H, json={},
    ) as resp:
        assert resp.status_code == 200
        await resp.aread()

    r = await gateway_client.get(f"/api/conversations/{cid}", headers=H)
    messages = r.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["id"] != first_assistant  # assistant turn replaced


async def test_edit_message_branches(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.post(
        "/api/conversations", headers=H, json={"model": "claude-sonnet-4-6"}
    )
    cid = r.json()["id"]
    async with gateway_client.stream(
        "POST", f"/api/conversations/{cid}/messages", headers=H,
        json={"content": "first question", "stream": True},
    ) as resp:
        await resp.aread()
    r = await gateway_client.get(f"/api/conversations/{cid}", headers=H)
    user_msg_id = r.json()["messages"][0]["id"]

    async with gateway_client.stream(
        "POST", f"/api/conversations/{cid}/messages/{user_msg_id}/edit",
        headers=H, json={"content": "edited question"},
    ) as resp:
        assert resp.status_code == 200
        body = (await resp.aread()).decode()
    assert "event: done" in body

    r = await gateway_client.get(f"/api/conversations/{cid}", headers=H)
    messages = r.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "edited question"


async def test_message_to_missing_conversation_404(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.post(
        "/api/conversations/conv_nope/messages", headers=H,
        json={"content": "hi", "stream": False},
    )
    assert r.status_code == 404


async def test_empty_message_rejected(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.post(
        "/api/conversations", headers=H, json={"model": "claude-sonnet-4-6"}
    )
    cid = r.json()["id"]
    r = await gateway_client.post(
        f"/api/conversations/{cid}/messages", headers=H,
        json={"content": "   ", "stream": False},
    )
    assert r.status_code == 400


async def test_file_upload(gateway_client: httpx.AsyncClient) -> None:
    files = {"file": ("notes.txt", b"hello world", "text/plain")}
    r = await gateway_client.post("/api/files", headers={"Authorization": f"Bearer {TEST_GATEWAY_KEY}"}, files=files)
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["filename"] == "notes.txt"
    assert data["size"] == len(b"hello world")
    assert data["id"].startswith("file_")


async def test_openai_tools_passthrough(
    gateway_client: httpx.AsyncClient, recorded_requests
) -> None:
    r = await gateway_client.post(
        "/v1/chat/completions", headers=H,
        json={
            "model": "tool-model",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    msg = data["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    # Tools were forwarded upstream.
    import json
    upstream = json.loads(recorded_requests[-1].content)
    assert upstream["tools"][0]["function"]["name"] == "get_weather"


async def test_upstream_error_surface(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.post(
        "/v1/chat/completions", headers=H,
        json={"model": "boom-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 502
    assert "error" in r.json()


async def test_request_id_header(gateway_client: httpx.AsyncClient) -> None:
    r = await gateway_client.get("/api/conversations", headers=H)
    assert r.headers.get("x-request-id")
    # Honours inbound request id.
    r = await gateway_client.get(
        "/api/conversations", headers={**H, "X-Request-ID": "my-trace-id"}
    )
    assert r.headers.get("x-request-id") == "my-trace-id"


async def test_oversized_body_rejected(gateway_client: httpx.AsyncClient) -> None:
    big = "x" * (3 * 1024 * 1024)  # default limit is 2 MiB
    r = await gateway_client.post(
        "/api/conversations", headers=H,
        json={"title": big, "model": "m"},
    )
    assert r.status_code == 413
