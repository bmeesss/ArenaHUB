"""Anthropic Messages API <-> OpenAI/Arena translation layer.

The official Arena API speaks the OpenAI Chat Completions shape, so the
Anthropic-compatible endpoint translates requests **to** OpenAI format before
calling Arena, and translates responses (and SSE events) **back to** the
Anthropic Messages format documented at
https://docs.anthropic.com/en/api/messages.

Only features that can be faithfully translated are advertised; anything else
returns a clear ``invalid_request_error`` rather than silently misbehaving.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from .errors import ArenaNotSupportedError, ArenaValidationError
from .models import ChatCompletionRequest, ChatMessage, ToolDefinition


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------


def _anthropic_content_to_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [dict(part) for part in content if isinstance(part, dict)]
    raise ArenaValidationError("Message 'content' must be a string or a list of content blocks.")


def anthropic_to_openai_messages(
    system: Any, messages: list[dict[str, Any]]
) -> list[ChatMessage]:
    """Convert Anthropic ``system`` + ``messages`` to OpenAI chat messages."""
    openai_messages: list[ChatMessage] = []

    # System prompt becomes an OpenAI system message at the head.
    if system:
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
            texts = [b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"]
            system_text = "\n\n".join(t for t in texts if t)
        else:
            raise ArenaValidationError("'system' must be a string or a list of text blocks.")
        if system_text.strip():
            openai_messages.append(ChatMessage(role="system", content=system_text))

    for message in messages:
        role = message.get("role")
        if role not in ("user", "assistant"):
            raise ArenaValidationError(
                f"Unsupported message role {role!r}; expected 'user' or 'assistant'."
            )
        parts = _anthropic_content_to_parts(message.get("content"))

        if role == "user":
            text_parts: list[dict[str, Any]] = []
            for part in parts:
                ptype = part.get("type")
                if ptype == "text":
                    text_parts.append({"type": "text", "text": part.get("text", "")})
                elif ptype == "image":
                    source = part.get("source") or {}
                    if source.get("type") == "base64":
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        text_parts.append(
                            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
                        )
                    elif source.get("type") == "url":
                        text_parts.append(
                            {"type": "image_url", "image_url": {"url": source.get("url", "")}}
                        )
                    else:
                        raise ArenaNotSupportedError(
                            "Only base64 or URL image sources are supported for image blocks."
                        )
                elif ptype == "tool_result":
                    # Tool results become a separate OpenAI tool message.
                    tool_call_id = part.get("tool_use_id")
                    result_content = part.get("content")
                    if isinstance(result_content, list):
                        result_text = "\n".join(
                            b.get("text", "") for b in result_content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    else:
                        result_text = "" if result_content is None else str(result_content)
                    openai_messages.append(
                        ChatMessage(role="tool", content=result_text, tool_call_id=tool_call_id)
                    )
                else:
                    raise ArenaNotSupportedError(
                        f"Content block type {ptype!r} in a user message is not supported by ArenaHub."
                    )
            user_text = "".join(p.get("text", "") for p in text_parts if p["type"] == "text").strip()
            if text_parts:
                openai_messages.append(ChatMessage(role="user", content=text_parts))
            elif not user_text and not any(p.get("type") == "tool_result" for p in parts):
                # Empty user message — keep OpenAI happy with an empty string.
                openai_messages.append(ChatMessage(role="user", content=""))

        else:  # assistant
            text_chunks: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for part in parts:
                ptype = part.get("type")
                if ptype == "text":
                    text_chunks.append(part.get("text", ""))
                elif ptype == "tool_use":
                    tool_calls.append(
                        {
                            "id": part.get("id"),
                            "type": "function",
                            "function": {
                                "name": part.get("name", ""),
                                "arguments": json.dumps(part.get("input", {})),
                            },
                        }
                    )
                elif ptype in ("thinking", "redacted_thinking"):
                    # Extended thinking is not part of the OpenAI wire shape;
                    # drop it rather than forwarding unsupported content.
                    continue
                else:
                    raise ArenaNotSupportedError(
                        f"Content block type {ptype!r} in an assistant message is not supported."
                    )
            openai_messages.append(
                ChatMessage(
                    role="assistant",
                    content="\n".join(text_chunks) if text_chunks else "",
                    tool_calls=tool_calls or None,
                )
            )

    return openai_messages


def anthropic_tools_to_openai(tools: list[dict[str, Any]] | None) -> list[ToolDefinition] | None:
    """Convert Anthropic ``tools`` definitions to OpenAI function tools."""
    if not tools:
        return None
    converted: list[ToolDefinition] = []
    for tool in tools:
        if tool.get("name") and tool.get("input_schema") is not None:
            converted.append(
                ToolDefinition(
                    function={
                        "name": tool["name"],
                        "description": tool.get("description"),
                        "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                    }
                )
            )
        else:
            raise ArenaValidationError("Each tool requires 'name' and 'input_schema'.")
    return converted


def anthropic_tool_choice(tool_choice: Any) -> str | dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, dict):
        kind = tool_choice.get("type")
        if kind == "auto":
            return "auto"
        if kind == "any":
            return "required"
        if kind == "tool":
            name = (tool_choice.get("name") or "").strip()
            return {"type": "function", "function": {"name": name}} if name else "required"
    return "auto"


def build_openai_request(payload: dict[str, Any], resolved_model: str) -> ChatCompletionRequest:
    """Build the OpenAI-shaped request sent to Arena from an Anthropic payload."""
    if not payload.get("messages"):
        raise ArenaValidationError("'messages' must contain at least one message.")

    max_tokens = payload.get("max_tokens")
    if max_tokens is None:
        raise ArenaValidationError("'max_tokens' is required for the Anthropic Messages API.")

    openai_messages = anthropic_to_openai_messages(payload.get("system"), payload["messages"])
    if not openai_messages:
        raise ArenaValidationError("The request contains no usable messages.")

    return ChatCompletionRequest(
        model=resolved_model,
        messages=openai_messages,
        stream=bool(payload.get("stream", False)),
        max_tokens=max_tokens,
        temperature=payload.get("temperature"),
        top_p=payload.get("top_p"),
        stop=payload.get("stop_sequences"),
        tools=anthropic_tools_to_openai(payload.get("tools")),
        tool_choice=anthropic_tool_choice(payload.get("tool_choice")),
    )


# ---------------------------------------------------------------------------
# Response translation (non-streaming)
# ---------------------------------------------------------------------------

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "tool_use": "tool_use",
    "content_filter": "stop_sequence",
}


def openai_to_anthropic_response(data: dict[str, Any], requested_model: str) -> dict[str, Any]:
    """Translate an OpenAI completion JSON body to an Anthropic Message."""
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish = choice.get("finish_reason") or "stop"

    content: list[dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    elif isinstance(text, list):
        for part in text:
            if isinstance(part, dict) and part.get("type") == "text":
                content.append({"type": "text", "text": part.get("text", "")})

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        try:
            tool_input = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            tool_input = {"raw": function.get("arguments") or ""}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": function.get("name", ""),
                "input": tool_input,
            }
        )

    if not content:
        content.append({"type": "text", "text": ""})

    usage = data.get("usage") or {}
    stop_reason = _FINISH_REASON_MAP.get(finish, "end_turn")
    if finish == "stop" and any(c["type"] == "tool_use" for c in content):
        stop_reason = "tool_use"

    return {
        # Anthropic message ids are always generated by the gateway (the
        # upstream id is an OpenAI-shaped chatcmpl id and must not leak).
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0) or 0,
            "output_tokens": usage.get("completion_tokens", 0) or 0,
        },
    }


# ---------------------------------------------------------------------------
# Response translation (streaming / SSE)
# ---------------------------------------------------------------------------


def sse(event_type: str, data: dict[str, Any]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


async def openai_stream_to_anthropic_events(
    chunks: AsyncIterator[dict[str, Any]],
    *,
    requested_model: str,
    message_id: str,
) -> AsyncIterator[bytes]:
    """Yield Anthropic SSE events translated from OpenAI SSE chunks.

    Emits: ``message_start``, ``content_block_start`` (text + tool_use),
    ``content_block_delta`` (text/input-json), ``content_block_stop``,
    ``message_delta``, ``message_stop`` and ``ping``.
    """
    created = int(time.time())
    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": requested_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    text_block_open = False
    text_index: int | None = None
    tool_blocks: dict[int, dict[str, int]] = {}  # tool_call index -> {block, order}
    next_block = 0
    finish_reason: str | None = None
    usage: dict[str, int] = {}
    started = False

    def open_text() -> tuple[int, int]:
        nonlocal text_block_open, text_index, next_block
        idx = next_block
        next_block += 1
        text_index = idx
        text_block_open = True
        return idx, idx

    async for chunk in chunks:
        if not started:
            started = True
            yield sse("ping", {"type": "ping"})

        if chunk.get("usage"):
            usage = chunk["usage"]

        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            # Text deltas
            text = delta.get("content")
            if isinstance(text, str) and text:
                if not text_block_open:
                    idx, _ = open_text()
                    yield sse("content_block_start", {
                        "type": "content_block_start", "index": idx,
                        "content_block": {"type": "text", "text": ""},
                    })
                yield sse("content_block_delta", {
                    "type": "content_block_delta", "index": text_index,
                    "delta": {"type": "text_delta", "text": text},
                })

            # Tool call deltas
            for tool_call in delta.get("tool_calls") or []:
                tc_index = tool_call.get("index", 0)
                if tc_index not in tool_blocks:
                    block_index = next_block
                    next_block += 1
                    tool_blocks[tc_index] = {"block": block_index, "order": block_index}
                    function = tool_call.get("function") or {}
                    yield sse("content_block_start", {
                        "type": "content_block_start", "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                            "name": function.get("name", ""),
                            "input": {},
                        },
                    })
                else:
                    block_index = tool_blocks[tc_index]["block"]
                    function = tool_call.get("function") or {}
                    arguments_fragment = function.get("arguments")
                    if arguments_fragment:
                        yield sse("content_block_delta", {
                            "type": "content_block_delta", "index": block_index,
                            "delta": {"type": "input_json_delta", "partial_json": arguments_fragment},
                        })

    # Close open blocks in index order.
    open_indices = ([text_index] if text_block_open and text_index is not None else []) + [
        info["block"] for info in tool_blocks.values()
    ]
    for index in sorted(i for i in open_indices if i is not None):
        yield sse("content_block_stop", {"type": "content_block_stop", "index": index})

    has_tool_use = bool(tool_blocks)
    stop_reason = "tool_use" if has_tool_use else _FINISH_REASON_MAP.get(finish_reason or "stop", "end_turn")
    output_tokens = usage.get("completion_tokens", 0) or 0
    yield sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield sse("message_stop", {"type": "message_stop"})
    _ = created


def anthropic_error_body(message: str, *, error_type: str = "invalid_request_error") -> dict[str, Any]:
    """Anthropic-style error envelope: ``{"type": "error", "error": {...}}``."""
    return {
        "type": "error",
        "error": {"type": error_type, "message": message},
    }
