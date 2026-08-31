"""Tests for the model router: caching, aliases, resolution."""

from __future__ import annotations

import httpx
import pytest

from backend.arena_client import ArenaClient
from backend.config import Settings
from backend.errors import ArenaModelError
from backend.model_router import ModelRouter

from .conftest import make_arena_handler


def _router(settings: Settings, recorded: list) -> ModelRouter:
    return ModelRouter(
        settings,
        client_factory=lambda: ArenaClient(
            settings, transport=httpx.MockTransport(make_arena_handler(recorded))
        ),
    )


async def test_resolves_builtin_aliases(settings: Settings) -> None:
    recorded: list[httpx.Request] = []
    router = _router(settings, recorded)
    assert await router.resolve("arena/claude-sonnet") == "claude-sonnet-4-6"
    assert await router.resolve("arena/claude-opus") == "claude-opus-4-0"
    assert await router.resolve("arena/gpt") == "gpt-4o"
    assert await router.resolve("arena/gemini") == "gemini-2.0-pro"
    # arena/claude prefers sonnet (balanced default).
    assert await router.resolve("arena/claude") == "claude-sonnet-4-6"


async def test_custom_alias(settings: Settings) -> None:
    recorded: list[httpx.Request] = []
    router = _router(settings, recorded)
    assert await router.resolve("arena/my-sonnet") == "claude-sonnet-4-6"


async def test_real_ids_pass_through(settings: Settings) -> None:
    router = _router(settings, [])
    assert await router.resolve("gpt-4o") == "gpt-4o"
    # Unknown plain ids pass through for the upstream to validate.
    assert await router.resolve("some-new-model") == "some-new-model"


async def test_unresolvable_arena_alias_raises(settings: Settings) -> None:
    settings.model_aliases = {}
    # Catalogue with no claude models at all.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "gpt-4o", "object": "model"}]})
        return httpx.Response(500, json={"error": {"message": "x"}})

    router = ModelRouter(
        settings, client_factory=lambda: ArenaClient(settings, transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ArenaModelError):
        await router.resolve("arena/claude")


async def test_list_all_includes_aliases(settings: Settings) -> None:
    router = _router(settings, [])
    catalogue = await router.list_all()
    by_id = {m.id: m for m in catalogue.data}
    assert "arena/claude-sonnet" in by_id
    alias = by_id["arena/claude-sonnet"]
    assert alias.alias_for == "claude-sonnet-4-6"
    assert alias.owned_by == "arenahub"
    # Underlying ids are present too.
    assert "claude-sonnet-4-6" in by_id


async def test_model_cache_avoids_refetch(settings: Settings) -> None:
    calls = {"n": 0}
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            calls["n"] += 1
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "gpt-4o", "object": "model"}]})
        return httpx.Response(500, json={"error": {"message": "x"}})

    router = ModelRouter(
        settings,
        client_factory=lambda: ArenaClient(settings, transport=httpx.MockTransport(handler)),
    )
    settings.model_cache_ttl = 300
    await router.list_arena_models()
    await router.list_arena_models()
    await router.resolve("gpt-4o")
    assert calls["n"] == 1  # cached: single upstream fetch
    router.invalidate()
    await router.list_arena_models()
    assert calls["n"] == 2


async def test_is_known(settings: Settings) -> None:
    router = _router(settings, [])
    assert await router.is_known("arena/gpt")
    assert await router.is_known("gpt-4o")
    assert not await router.is_known("arena/nonexistent-family")


async def test_alias_resolution_over_openai_endpoint(
    gateway_client, recorded_requests
) -> None:
    r = await gateway_client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-gateway-key"},
        json={"model": "arena/gpt", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    # Upstream receives the resolved id.
    import json
    upstream = json.loads(recorded_requests[-1].content)
    assert upstream["model"] == "gpt-4o"
    # Client sees the requested alias.
    assert r.json()["model"] == "arena/gpt"
