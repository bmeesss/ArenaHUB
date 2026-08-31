"""Web / Android REST API: conversations, messages, files, model catalogue.

All routes are mounted under ``/api`` and (like every non-health route) are
protected by the ArenaHub authentication middleware.

Streaming responses use Server-Sent Events so they work from browsers
(``EventSource`` / fetch streams) and native Android clients alike.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from .db import MessageRecord
from .errors import ArenaHubError, ArenaNotSupportedError, ArenaValidationError
from .model_router import ModelRouter
from .models import ChatCompletionRequest, ChatMessage
from .schemas import (
    ConversationCreate,
    ConversationDetail,
    ConversationMessageCreate,
    ConversationRename,
    FileRecord,
    MessageEdit,
    MessageObject,
    ModelCatalogue,
    ModelOption,
    RegenerateRequest,
)
from .web_events import web_error_event, web_message_event

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def _repo(request: Request):
    return request.app.state.repository


def _router(request: Request) -> ModelRouter:
    return request.app.state.model_router


def _client(request: Request):
    factory = getattr(request.app.state, "client_factory", None)
    if factory is not None:
        return factory()
    from .arena_client import ArenaClient

    return ArenaClient(request.app.state.settings)


def _settings(request: Request):
    return request.app.state.settings


def _conversation_or_404(record, conversation_id: str):
    if record is None:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id!r} not found.")
    return record


def _message_to_object(message: MessageRecord) -> MessageObject:
    return MessageObject(
        id=message.id,
        conversation_id=message.conversation_id,
        role=message.role,
        content=message.content,
        model=message.model,
        created_at=message.created_at,
        parent_message_id=message.parent_message_id,
    )


# ---------------------------------------------------------------------------
# Model catalogue (frontend model selector)
# ---------------------------------------------------------------------------


@router.get("/models")
async def list_model_catalogue(request: Request) -> ModelCatalogue:
    model_router = _router(request)
    catalogue = await model_router.list_all()
    options = [
        ModelOption(
            id=model.id,
            owned_by=model.owned_by,
            is_alias=model.alias_for is not None,
            alias_for=model.alias_for,
        )
        for model in catalogue.data
    ]
    return ModelCatalogue(models=options, default_model=_settings(request).arena_default_model)


# ---------------------------------------------------------------------------
# Conversation CRUD
# ---------------------------------------------------------------------------


@router.post("/conversations", status_code=201)
async def create_conversation(request: Request, body: ConversationCreate) -> dict[str, Any]:
    settings = _settings(request)
    model = body.model or settings.arena_default_model
    if not model:
        # Fall back to the first available model so "new chat" always works.
        try:
            arena_models = await _router(request).list_arena_models()
            model = arena_models[0].id if arena_models else ""
        except ArenaHubError:
            model = ""
    if not model:
        raise HTTPException(
            status_code=400,
            detail="No model specified and no default model is configured.",
        )
    record = await _repo(request).create_conversation(
        title=body.title or "New conversation", model=model, metadata=body.metadata
    )
    return record.to_dict()


@router.get("/conversations")
async def list_conversations(request: Request, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    limit = max(1, min(limit, 500))
    records = await _repo(request).list_conversations(limit=limit, offset=offset)
    return {"conversations": [record.to_dict() for record in records]}


@router.get("/conversations/{conversation_id}")
async def get_conversation(request: Request, conversation_id: str) -> ConversationDetail:
    repo = _repo(request)
    record = _conversation_or_404(await repo.get_conversation(conversation_id), conversation_id)
    messages = await repo.list_messages(conversation_id)
    detail = record.to_dict()
    detail["messages"] = [_message_to_object(m).model_dump() for m in messages]
    return ConversationDetail(**detail)


@router.patch("/conversations/{conversation_id}")
async def rename_conversation(
    request: Request, conversation_id: str, body: ConversationRename
) -> dict[str, Any]:
    record = await _repo(request).rename_conversation(conversation_id, body.title)
    _conversation_or_404(record, conversation_id)
    return record.to_dict()


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(request: Request, conversation_id: str):
    deleted = await _repo(request).delete_conversation(conversation_id)
    if not deleted:
        _conversation_or_404(None, conversation_id)
    return JSONResponse(status_code=204, content=None)


# ---------------------------------------------------------------------------
# Messages (with Arena-generated assistant replies)
# ---------------------------------------------------------------------------


def _history_to_chat_messages(messages: list[MessageRecord]) -> list[ChatMessage]:
    chat_messages: list[ChatMessage] = []
    for message in messages:
        if message.role == "tool":
            chat_messages.append(
                ChatMessage(role="tool", content=message.content, tool_call_id=message.parent_message_id)
            )
        else:
            chat_messages.append(ChatMessage(role=message.role, content=message.content))
    return chat_messages


def _content_from_body(content: Any) -> Any:
    if content is None:
        raise ArenaValidationError("'content' is required.")
    if isinstance(content, str):
        if not content.strip():
            raise ArenaValidationError("Message content must not be empty.")
        return content
    if isinstance(content, list):
        return content
    raise ArenaValidationError("'content' must be a string or a list of content blocks.")


async def _resolve_model(request: Request, requested: str | None, conversation_model: str) -> str:
    model_router = _router(request)
    target = requested or conversation_model
    if not target:
        raise ArenaValidationError("No model selected for this conversation.")
    return await model_router.resolve(target)


async def _stream_web_reply(
    request: Request,
    *,
    conversation_id: str,
    model: str,
    resolved_model: str,
    history: list[MessageRecord],
    user_message: MessageRecord,
) -> AsyncIterator[bytes]:
    """SSE stream for the web client: user_message -> delta(s) -> done/error."""
    client = _client(request)
    repo = _repo(request)
    chat_messages = _history_to_chat_messages(history)
    chat_request = ChatCompletionRequest(model=resolved_model, messages=chat_messages, stream=True)

    yield web_message_event("user_message", _message_to_object(user_message).model_dump())

    assistant_text = ""
    try:
        async for chunk in client.stream_chat_completion(chat_request):
            from .models import extract_delta_text

            delta = extract_delta_text(chunk)
            if delta:
                assistant_text += delta
                yield web_message_event("delta", {"content": delta})
    except ArenaHubError as exc:
        yield web_error_event(str(exc))
        return
    finally:
        await client.aclose()

    assistant_message = await repo.add_message(
        conversation_id, role="assistant", content=assistant_text, model=model
    )
    yield web_message_event(
        "done",
        {
            "message": _message_to_object(assistant_message).model_dump(),
            "conversation_id": conversation_id,
        },
    )


@router.post("/conversations/{conversation_id}/messages")
async def post_message(
    request: Request, conversation_id: str, body: ConversationMessageCreate
):
    repo = _repo(request)
    conversation = _conversation_or_404(
        await repo.get_conversation(conversation_id), conversation_id
    )

    content = _content_from_body(body.content)
    try:
        resolved_model = await _resolve_model(request, body.model, conversation.model)
    except ArenaHubError as exc:
        raise _web_http_error(exc) from exc

    user_message = await repo.add_message(conversation_id, role="user", content=content)
    history = await repo.list_messages(conversation_id)

    if body.stream:
        return StreamingResponse(
            _stream_web_reply(
                request,
                conversation_id=conversation_id,
                model=body.model or conversation.model,
                resolved_model=resolved_model,
                history=history,
                user_message=user_message,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: call Arena, persist the assistant turn, return both.
    client = _client(request)
    try:
        chat_request = ChatCompletionRequest(
            model=resolved_model, messages=_history_to_chat_messages(history), stream=False
        )
        try:
            raw = await client.chat_completion(chat_request)
        except ArenaHubError as exc:
            raise _web_http_error(exc) from exc
    finally:
        await client.aclose()

    reply_text = ""
    choices = raw.get("choices") or []
    if choices:
        reply_text = (choices[0].get("message") or {}).get("content") or ""
    assistant_message = await repo.add_message(
        conversation_id, role="assistant", content=reply_text, model=body.model or conversation.model
    )
    return {
        "user_message": _message_to_object(user_message).model_dump(),
        "assistant_message": _message_to_object(assistant_message).model_dump(),
    }


@router.post("/conversations/{conversation_id}/messages/{message_id}/edit")
async def edit_message(
    request: Request, conversation_id: str, message_id: str, body: MessageEdit
):
    """Edit a prior user message: drop it and everything after, then regenerate."""
    repo = _repo(request)
    _conversation_or_404(await repo.get_conversation(conversation_id), conversation_id)
    target = await repo.get_message(message_id)
    if target is None or target.conversation_id != conversation_id:
        raise HTTPException(status_code=404, detail=f"Message {message_id!r} not found.")
    if target.role != "user":
        raise HTTPException(status_code=400, detail="Only user messages can be edited.")

    content = _content_from_body(body.content)
    removed = await repo.truncate_messages_after(message_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Message {message_id!r} not found.")

    user_message = await repo.add_message(
        conversation_id, role="user", content=content,
        parent_message_id=target.parent_message_id,
    )
    history = await repo.list_messages(conversation_id)
    conversation = await repo.get_conversation(conversation_id)
    try:
        resolved_model = await _resolve_model(request, None, conversation.model)
    except ArenaHubError as exc:
        raise _web_http_error(exc) from exc

    return StreamingResponse(
        _stream_web_reply(
            request,
            conversation_id=conversation_id,
            model=conversation.model,
            resolved_model=resolved_model,
            history=history,
            user_message=user_message,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/conversations/{conversation_id}/regenerate")
async def regenerate_reply(
    request: Request, conversation_id: str, body: RegenerateRequest | None = None
):
    """Regenerate the assistant response to the most recent user message."""
    repo = _repo(request)
    conversation = _conversation_or_404(
        await repo.get_conversation(conversation_id), conversation_id
    )
    messages = await repo.list_messages(conversation_id)

    # Drop trailing assistant messages, then regenerate from the last user turn.
    while messages and messages[-1].role != "user":
        await repo.truncate_messages_after(messages[-1].id)
        messages.pop()
    if not messages:
        raise HTTPException(status_code=400, detail="No user message to regenerate from.")

    history = await repo.list_messages(conversation_id)
    user_message = history[-1]
    model = (body.model if body else None) or conversation.model
    try:
        resolved_model = await _resolve_model(request, body.model if body else None, conversation.model)
    except ArenaHubError as exc:
        raise _web_http_error(exc) from exc

    return StreamingResponse(
        _stream_web_reply(
            request,
            conversation_id=conversation_id,
            model=model,
            resolved_model=resolved_model,
            history=history,
            user_message=user_message,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# File uploads
# ---------------------------------------------------------------------------


@router.post("/files", status_code=201)
async def upload_file(request: Request, file: UploadFile = File(...)) -> FileRecord:
    """Store an uploaded file locally and return a reference id.

    Text/code files can be attached to later messages; image upload is
    accepted and stored, though image understanding depends on Arena model
    support and is surfaced as a compatibility error if unavailable.
    """
    settings = _settings(request)
    upload_dir = settings.uploads_path
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = f"file_{uuid.uuid4().hex}"
    suffix = ""
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1][:16]
    destination = upload_dir / f"{file_id}{suffix}"

    size = 0
    max_bytes = settings.request_max_body_bytes
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    handle.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large; limit is {max_bytes} bytes.",
                    )
                handle.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:  # pragma: no cover - filesystem dependent
        raise HTTPException(status_code=500, detail=f"Could not store file: {exc}") from exc

    return FileRecord(
        id=file_id,
        filename=file.filename or "upload",
        content_type=file.content_type or "application/octet-stream",
        size=size,
        created_at=time.time(),
    )


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def _web_http_error(exc: ArenaHubError) -> HTTPException:
    if isinstance(exc, ArenaValidationError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ArenaNotSupportedError):
        return HTTPException(status_code=422, detail=str(exc))
    # Upstream/rate-limit/auth errors reuse the shared mapping.
    from .routes import map_arena_error_to_http

    return map_arena_error_to_http(exc)
