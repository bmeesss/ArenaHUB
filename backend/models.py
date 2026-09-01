"""Pydantic schemas and response normalisation for the OpenAI-compatible API.

The gateway speaks the OpenAI Chat Completions / Models wire format so that
existing OpenAI-compatible clients can point at ArenaHub unchanged. Raw
responses from the Arena API are normalised through these models so the
local gateway always returns a well-formed OpenAI-shaped payload.

The same shapes are reused by the Anthropic compatibility layer
(:mod:`backend.anthropic`) and the web/coding-agent APIs.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from .errors import ArenaValidationError

# Model ids may contain letters, digits, dots, dashes, underscores, colons,
# forward slashes and an "arena/" alias prefix. This rejects whitespace,
# control chars and header-injection attempts.
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")


def validate_model_id(model: str) -> None:
    """Validate a model id locally before sending it upstream."""
    if not isinstance(model, str) or not model.strip():
        raise ArenaValidationError("Model id must be a non-empty string.")
    if not _MODEL_ID_RE.match(model):
        raise ArenaValidationError(
            f"Invalid model id {model!r}: use letters, digits and . _ : / - only."
        )


def _new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Content blocks (multimodal / tool messages)
# ---------------------------------------------------------------------------

# A message's ``content`` may be a plain string (the common case) or a list of
# OpenAI-style content parts: text, image_url, or tool results.
ContentPart = dict[str, Any]
MessageContent = str | list[ContentPart]


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON-encoded arguments, per the OpenAI wire format


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: MessageContent | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def to_arena(self) -> dict[str, Any]:
        """Payload forwarded to the Arena API (omits empty/None fields)."""
        return self.model_dump(exclude_none=True)


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    """OpenAI-style chat completion request (subset we forward upstream)."""

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | str | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    n: int | None = None
    seed: int | None = None
    user: str | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None

    def to_arena_payload(self) -> dict[str, Any]:
        """JSON body sent to the Arena API (only set fields are included)."""
        return self.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int | None = None
    owned_by: str | None = None
    # Set on entries that are ArenaHub aliases pointing at a real Arena model.
    alias_for: str | None = None


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo] = Field(default_factory=list)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ChunkChoice(BaseModel):
    index: int = 0
    delta: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalisation helpers (raw Arena payload -> OpenAI-compatible models)
# ---------------------------------------------------------------------------


def normalize_models(data: dict[str, Any]) -> ModelList:
    """Normalise a raw ``/v1/models`` payload into an OpenAI model list."""
    raw_items = data.get("data") if isinstance(data, dict) else None
    items: list[ModelInfo] = []
    for entry in raw_items or []:
        if isinstance(entry, str):
            items.append(ModelInfo(id=entry))
        elif isinstance(entry, dict) and entry.get("id"):
            items.append(
                ModelInfo(
                    id=str(entry["id"]),
                    created=entry.get("created"),
                    owned_by=entry.get("owned_by") or entry.get("owner") or entry.get("provider"),
                )
            )
    return ModelList(data=items)


def normalize_completion(data: dict[str, Any], fallback_model: str) -> ChatCompletionResponse:
    """Normalise a raw non-streaming chat completion into OpenAI shape."""
    raw_choices = data.get("choices") or []
    choices: list[Choice] = []
    for i, raw_choice in enumerate(raw_choices):
        raw_message = raw_choice.get("message") or {}
        raw_content = raw_message.get("content")
        if isinstance(raw_content, list):
            # Collapse text parts to a plain string for OpenAI compatibility;
            # non-text parts (images/tool artifacts) are passed through.
            text = "".join(
                part.get("text", "")
                for part in raw_content
                if isinstance(part, dict) and part.get("type") == "text"
            )
            has_non_text = any(
                not (isinstance(part, dict) and part.get("type") == "text")
                for part in raw_content
            )
            content: MessageContent | None = raw_content if has_non_text else text
        else:
            content = raw_content
        message = ChatMessage(
            role=raw_message.get("role") or "assistant",
            content=content if content is not None else "",
            tool_calls=raw_message.get("tool_calls"),
        )
        choices.append(
            Choice(
                index=raw_choice.get("index", i) or i,
                message=message,
                finish_reason=raw_choice.get("finish_reason") or "stop",
            )
        )
    if not choices:
        choices = [
            Choice(index=0, message=ChatMessage(role="assistant", content=""), finish_reason="stop")
        ]

    raw_usage = data.get("usage") or {}
    usage = UsageInfo(
        prompt_tokens=raw_usage.get("prompt_tokens", 0) or 0,
        completion_tokens=raw_usage.get("completion_tokens", 0) or 0,
        total_tokens=raw_usage.get("total_tokens", 0) or 0,
    )
    return ChatCompletionResponse(
        id=data.get("id") or _new_completion_id(),
        created=int(data.get("created") or time.time()),
        model=data.get("model") or fallback_model,
        choices=choices,
        usage=usage,
    )


def normalize_chunk(
    data: dict[str, Any],
    fallback_model: str,
    *,
    fallback_id: str | None = None,
    fallback_created: int | None = None,
) -> ChatCompletionChunk:
    """Normalise a raw streaming SSE chunk into OpenAI chunk shape."""
    raw_choices = data.get("choices") or []
    choices: list[ChunkChoice] = []
    for i, raw_choice in enumerate(raw_choices):
        delta = raw_choice.get("delta") or {}
        if not isinstance(delta, dict):
            delta = {"content": str(delta)}
        choices.append(
            ChunkChoice(
                index=raw_choice.get("index", i) or i,
                delta=dict(delta),
                finish_reason=raw_choice.get("finish_reason"),
            )
        )
    return ChatCompletionChunk(
        id=data.get("id") or fallback_id or _new_completion_id(),
        created=int(data.get("created") or fallback_created or time.time()),
        model=data.get("model") or fallback_model,
        choices=choices,
    )


def extract_delta_text(chunk: dict[str, Any]) -> str:
    """Pull incremental text out of a raw streaming chunk (for the CLI)."""
    try:
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        return content if isinstance(content, str) else ""
    except (AttributeError, IndexError, TypeError):
        return ""
