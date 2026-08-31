"""Unified model router over the official Arena API.

Responsibilities
----------------
* Fetch the available model list from Arena and cache it for a short,
  configurable TTL (``ARENA_MODEL_CACHE_TTL``).
* Resolve ArenaHub model **aliases** (e.g. ``arena/claude-sonnet``) to real
  Arena model ids.
* Expose a merged catalogue (real ids + aliases) for model pickers and the
  OpenAI/Anthropic ``models`` surfaces.

Aliases come in two flavours:

* **Built-in smart aliases** — matched against the live model catalogue
  (``arena/claude``, ``arena/claude-opus``, ``arena/claude-sonnet``,
  ``arena/gpt``, ``arena/gemini``). They resolve to the best available model
  at call time, so they keep working as Arena refreshes its catalogue.
* **Custom aliases** — explicit ``alias=model-id`` mappings from
  ``ARENA_MODEL_ALIASES``.

The router never fabricates a model that Arena does not offer: smart aliases
that cannot be resolved raise a clear error instead of guessing.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from .arena_client import ArenaClient
from .config import Settings, load_settings
from .errors import ArenaModelError
from .models import ModelInfo, ModelList

# Built-in aliases. Each requires *all* ``required`` substrings (case
# insensitive) and scores candidates by ``preferred`` weights; the highest
# scoring candidate wins, ties broken by id for determinism.
BUILTIN_ALIASES: dict[str, dict[str, object]] = {
    "arena/claude-opus": {
        "required": ("claude", "opus"),
        "preferred": (("opus", 10),),
    },
    "arena/claude-sonnet": {
        "required": ("claude", "sonnet"),
        "preferred": (("sonnet", 10),),
    },
    "arena/claude": {
        "required": ("claude",),
        # Balanced default: sonnet first, then opus, then haiku.
        "preferred": (("sonnet", 10), ("opus", 5), ("haiku", -5), ("opus", 1)),
    },
    "arena/gpt": {
        "required": ("gpt",),
        "preferred": (("gpt-4o", 10), ("gpt-4", 6), ("4o", 8), ("o1", 4), ("gpt-3.5", -5)),
    },
    "arena/gemini": {
        "required": ("gemini",),
        "preferred": (("pro", 10), ("ultra", 8), ("flash", 2), ("nano", -5)),
    },
}


class ModelRouter:
    """Caches Arena model metadata and resolves model identifiers/aliases."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client_factory: Callable[[], ArenaClient] | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._client_factory = client_factory or (lambda: ArenaClient(self.settings))
        self._cache: ModelList | None = None
        self._cache_expires_at = 0.0
        self._lock = asyncio.Lock()

    # -- caching -----------------------------------------------------------

    def invalidate(self) -> None:
        self._cache = None
        self._cache_expires_at = 0.0

    async def _fetch_models(self, *, force: bool = False) -> ModelList:
        now = time.monotonic()
        if not force and self._cache is not None and now < self._cache_expires_at:
            return self._cache
        async with self._lock:
            # Re-check after acquiring the lock (only one refresh at a time).
            now = time.monotonic()
            if not force and self._cache is not None and now < self._cache_expires_at:
                return self._cache
            client = self._client_factory()
            try:
                models = await client.list_models()
            finally:
                await client.aclose()
            self._cache = models
            self._cache_expires_at = now + max(1.0, self.settings.model_cache_ttl)
            return models

    # -- listing -----------------------------------------------------------

    async def list_arena_models(self) -> list[ModelInfo]:
        return (await self._fetch_models()).data

    async def list_all(self) -> ModelList:
        """Real Arena models (in catalogue order) plus resolvable aliases.

        Real models are always listed first (clients that take ``data[0]``
        get a concrete model); aliases follow with ``owned_by="arenahub"``.
        """
        models = await self._fetch_models()
        data = list(models.data)
        for alias in await self.list_aliases():
            data.append(
                ModelInfo(
                    id=alias, object="model", owned_by="arenahub",
                    alias_for=await self.resolve(alias),
                )
            )
        return ModelList(data=data)

    async def list_aliases(self) -> list[str]:
        """All alias names that currently resolve to a real Arena model."""
        ids = {m.id for m in (await self._fetch_models()).data}
        aliases: list[str] = []
        for alias in BUILTIN_ALIASES:
            if self._match_builtin(alias, ids) is not None:
                aliases.append(alias)
        for alias, target in self.settings.model_aliases.items():
            aliases.append(alias)  # explicit user aliases are always offered
            _ = ids, target
        return sorted(set(aliases))

    # -- resolution --------------------------------------------------------

    async def resolve(self, identifier: str) -> str:
        """Resolve a model id or ArenaHub alias to a real Arena model id.

        Exact Arena ids pass through. Unknown plain ids also pass through so
        the upstream API can validate them; unresolvable ``arena/`` aliases
        raise :class:`ArenaModelError` with a clear message.
        """
        if identifier in self.settings.model_aliases:
            return self.settings.model_aliases[identifier]

        if identifier in BUILTIN_ALIASES:
            ids = {m.id for m in (await self._fetch_models()).data}
            match = self._match_builtin(identifier, ids)
            if match is None:
                available = sorted(ids)
                raise ArenaModelError(
                    f"Alias {identifier!r} could not be resolved: no matching model is currently "
                    f"available through Arena. Available models: {', '.join(available[:20])}"
                )
            return match

        # Exact id or unknown id — let Arena validate unknown ones.
        return identifier

    async def is_known(self, identifier: str) -> bool:
        """True for aliases or ids present in the live catalogue."""
        if identifier in BUILTIN_ALIASES or identifier in self.settings.model_aliases:
            return True
        ids = {m.id for m in (await self._fetch_models()).data}
        return identifier in ids

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _match_builtin(alias: str, ids: set[str]) -> str | None:
        spec = BUILTIN_ALIASES[alias]
        required = spec["required"]  # type: ignore[assignment]
        preferred = spec["preferred"]  # type: ignore[assignment]

        def score(model_id: str) -> int | None:
            lowered = model_id.lower()
            if not all(token in lowered for token in required):  # type: ignore[union-attr]
                return None
            total = 0
            for token, weight in preferred:  # type: ignore[union-attr]
                if token in lowered:
                    total += weight
            return total

        candidates: list[tuple[int, str]] = []
        for model_id in ids:
            s = score(model_id)
            if s is not None:
                candidates.append((s, model_id))
        if not candidates:
            return None
        # Highest score first; deterministic tie-break on id.
        candidates.sort(key=lambda pair: (-pair[0], pair[1]))
        return candidates[0][1]
