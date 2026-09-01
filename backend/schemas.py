"""Pydantic schemas for the web/Android REST API (``/api/...``).

These contracts are designed for a ChatGPT-style frontend and for native
mobile clients: JSON request/response bodies, SSE for streaming, and stable
ids for optimistic UI updates.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Models surface for frontends (model selector)
# ---------------------------------------------------------------------------


class ModelOption(BaseModel):
    id: str
    owned_by: str | None = None
    is_alias: bool = False
    alias_for: str | None = None


class ModelCatalogue(BaseModel):
    models: list[ModelOption]
    default_model: str | None = None


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


class ConversationCreate(BaseModel):
    title: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationSummary(BaseModel):
    id: str
    title: str
    model: str
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationList(BaseModel):
    conversations: list[ConversationSummary]
    total: int | None = None


class MessageObject(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: Any
    model: str | None = None
    created_at: float
    parent_message_id: str | None = None


class ConversationDetail(BaseModel):
    id: str
    title: str
    model: str
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    messages: list[MessageObject] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sending / editing / regenerating messages
# ---------------------------------------------------------------------------


class ConversationMessageCreate(BaseModel):
    """Post a user message; the assistant reply is generated via Arena.

    ``content`` is a plain string or OpenAI-style content blocks (text /
    image_url / tool results) for coding-agent and multimodal clients.
    """

    content: Any
    model: str | None = None
    stream: bool = True
    files: list[str] = Field(default_factory=list)  # ids from POST /api/files


class MessageEdit(BaseModel):
    """Edit a prior message and regenerate everything after it."""

    content: Any


class RegenerateRequest(BaseModel):
    """Regenerate the assistant reply to the last user message."""

    model: str | None = None


class FileRecord(BaseModel):
    id: str
    filename: str
    content_type: str
    size: int
    created_at: float
