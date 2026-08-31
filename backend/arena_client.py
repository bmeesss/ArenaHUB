"""Async client for the official Arena API.

Only the documented public endpoints are used:

* ``GET  /v1/models``
* ``POST /v1/chat/completions`` (including SSE streaming)

The client never logs credentials, applies timeouts, validates model ids
locally, and maps every failure mode to a structured exception from
:mod:`backend.errors`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import USER_AGENT, Settings, load_settings
from .errors import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaConnectionError,
    ArenaModelError,
    ArenaRateLimitError,
    ArenaTimeoutError,
)
from .models import (
    ChatCompletionRequest,
    ModelList,
    normalize_models,
    validate_model_id,
)

MODELS_PATH = "/v1/models"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


def _extract_error_message(body: str) -> str:
    """Best-effort extraction of a human-readable error message."""
    if not body:
        return "The Arena API returned an error with no response body."
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return body[:500]
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or body[:500])
    if isinstance(error, str) and error:
        return error
    message = payload.get("message") if isinstance(payload, dict) else None
    return str(message or body[:500])


class ArenaClient:
    """Thin async wrapper around the official Arena API."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle ---------------------------------------------------------

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            api_key = self.settings.require_arena_api_key()
            self._client = httpx.AsyncClient(
                base_url=self.settings.arena_base_url.rstrip("/"),
                timeout=httpx.Timeout(self.settings.arena_timeout, connect=10.0),
                transport=self._transport,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> ArenaClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- public API --------------------------------------------------------

    async def list_models(self) -> ModelList:
        """Fetch the list of models available through the Arena API."""
        data = await self._request_json("GET", MODELS_PATH)
        return normalize_models(data)

    async def chat_completion(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Non-streaming chat completion; returns the raw (OpenAI-shaped) JSON."""
        validate_model_id(request.model)
        return await self._request_json(
            "POST", CHAT_COMPLETIONS_PATH, json=request.to_arena_payload()
        )

    async def stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming chat completion.

        Yields each raw SSE ``data:`` payload as a parsed dict. The terminal
        ``[DONE]`` marker is consumed and not yielded.
        """
        validate_model_id(request.model)
        payload = request.to_arena_payload()
        payload["stream"] = True

        client = self._http()
        try:
            req = client.build_request("POST", CHAT_COMPLETIONS_PATH, json=payload)
            response = await client.send(req, stream=True)
        except httpx.TimeoutException as exc:
            raise ArenaTimeoutError(
                f"Timed out while connecting to the Arena API ({self.settings.arena_base_url})."
            ) from exc
        except httpx.TransportError as exc:
            raise ArenaConnectionError(
                f"Could not reach the Arena API at {self.settings.arena_base_url}: {exc}"
            ) from exc

        try:
            await self._raise_for_status(response)
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                try:
                    parsed = json.loads(data_str)
                except json.JSONDecodeError:
                    # Ignore keep-alive comments / malformed lines.
                    continue
                if isinstance(parsed, dict):
                    yield parsed
        finally:
            await response.aclose()

    # -- internals ---------------------------------------------------------

    async def _request_json(
        self, method: str, path: str, **kwargs: Any
    ) -> dict[str, Any]:
        client = self._http()
        try:
            response = await client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise ArenaTimeoutError(
                f"The Arena API did not respond within {self.settings.arena_timeout:.0f}s."
            ) from exc
        except httpx.TransportError as exc:
            raise ArenaConnectionError(
                f"Could not reach the Arena API at {self.settings.arena_base_url}: {exc}"
            ) from exc
        await self._raise_for_status(response)
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ArenaAPIError(
                f"The Arena API returned a non-JSON response (HTTP {response.status_code}).",
                status_code=response.status_code,
                body=response.text[:500],
            ) from exc

    @staticmethod
    async def _raise_for_status(response: httpx.Response) -> None:
        """Map HTTP error responses onto the structured error hierarchy."""
        if response.is_success:
            return

        # For streamed responses the body may not have been read yet.
        try:
            await response.aread()
        except Exception:  # pragma: no cover - defensive
            pass
        body = response.text
        message = _extract_error_message(body)
        status = response.status_code
        lowered = message.lower()

        if status in (401, 403):
            raise ArenaAuthError(
                "Arena API rejected the request (invalid or missing API key, or insufficient "
                f"permissions): {message}",
                status_code=status,
                body=body,
            )
        if status == 429:
            raise ArenaRateLimitError(
                f"Rate limited by the Arena API. Retry after {response.headers.get('retry-after', 'a short wait')}."
                f" ({message})",
                status_code=status,
                body=body,
                retry_after=response.headers.get("retry-after"),
            )
        mentions_model = "model" in lowered and (
            "not found" in lowered or "invalid" in lowered or "unknown" in lowered or "does not exist" in lowered
        )
        if status == 404 or mentions_model:
            raise ArenaModelError(
                f"Invalid or unknown model: {message}",
                status_code=status,
                body=body,
            )
        raise ArenaAPIError(
            f"The Arena API returned an error: {message}",
            status_code=status,
            body=body,
        )
